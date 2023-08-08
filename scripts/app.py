import os
import cv2
import glob
import time
import torch
import shutil
import argparse
import platform
import datetime
import subprocess
import insightface
import onnxruntime
import numpy as np
import gradio as gr
from PIL import Image
from tqdm import tqdm
import concurrent.futures
from moviepy.editor import VideoFileClip
import requests
from face_swapper import Inswapper, paste_to_whole, place_foreground_on_background
from face_analyser import detect_conditions, get_analysed_data, swap_options_list
from face_enhancer import get_available_enhancer_names, load_face_enhancer_model, cv2_interpolations
from face_parsing import init_parser, swap_regions, mask_regions, mask_regions_to_list, SoftErosion
from utils import trim_video, StreamerThread, ProcessBar, open_directory, split_list_by_lengths, merge_img_sequence_from_ref
from modules import generation_parameters_copypaste as params_copypaste
import modules.scripts as scripts
from modules import script_callbacks


root_path = os.getcwd()
outputs_dir = os.path.join(root_path, "Outputs", "swap-mukham")
   
USE_COLAB = False
try:
    from google.colab import files
    USE_COLAB = True
except ImportError:
    USE_COLAB = False

outputs_dir = '/content/outputs' if USE_COLAB else os.path.join(root_path, "Outputs", "swap-mukham")

if not os.path.exists(outputs_dir):
    # Si no existe, crea el directorio
    os.makedirs(outputs_dir)
    print(f"{outputs_dir} folder created.")



DEF_OUTPUT_PATH = outputs_dir
USE_CUDA = True
USE_COLAB = False
BATCH_SIZE = 32
WORKSPACE = None
OUTPUT_FILE = None
CURRENT_FRAME = None
STREAMER = None
DETECT_CONDITION = "best detection"
DETECT_SIZE = 640
DETECT_THRESH = 0.6
NUM_OF_SRC_SPECIFIC = 10
MASK_INCLUDE = [
    "Skin",
    "R-Eyebrow",
    "L-Eyebrow",
    "L-Eye",
    "R-Eye",
    "Nose",
    "Mouth",
    "L-Lip",
    "U-Lip"
]
MASK_SOFT_KERNEL = 17
MASK_SOFT_ITERATIONS = 7
MASK_BLUR_AMOUNT = 20

FACE_SWAPPER = None
FACE_ANALYSER = None
FACE_ENHANCER = None
FACE_PARSER = None
FACE_ENHANCER_LIST = ["NONE"]
FACE_ENHANCER_LIST.extend(get_available_enhancer_names())
FACE_ENHANCER_LIST.extend(cv2_interpolations)

## _________________________ SET EXECUTION PROVIDER _________________________

PROVIDER = ["CPUExecutionProvider"]

if USE_CUDA:
    available_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        print("\nSwap-Mukham: Running on CUDA \n")
        PROVIDER = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        USE_CUDA = False
        print("\n*** CUDA unavailable running on CPU ***\n")
else:
    USE_CUDA = False
    print("\n*** Running on CPU ***\n")

device = "cuda" if USE_CUDA else "cpu"
EMPTY_CACHE = lambda: torch.cuda.empty_cache() if device == "cuda" else None

## _________________________ LOAD MODELS _________________________

base_dir = scripts.basedir()
models_dir = os.path.join(base_dir, "assets", "pretrained_models")

def load_face_analyser_model(name="buffalo_l"):
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name=name, providers=PROVIDER)
        FACE_ANALYSER.prepare(
            ctx_id=0, det_size=(DETECT_SIZE, DETECT_SIZE), det_thresh=DETECT_THRESH
        )

def load_face_swapper_model(path=os.path.join(models_dir, "inswapper_128.onnx")):
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        batch = int(BATCH_SIZE) if device == "cuda" else 1
        FACE_SWAPPER = Inswapper(model_file=path, batch_size=batch, providers=PROVIDER)

def load_face_parser_model(path=os.path.join(models_dir, "79999_iter.pth")):
    global FACE_PARSER
    if FACE_PARSER is None:
        FACE_PARSER = init_parser(path, mode=device)

def unload_models():
    global FACE_SWAPPER, FACE_PARSER, FACE_ENHANCER, FACE_ANALYSER

    # Opcional: Llamar a los métodos de cierre o liberación explícita si existen
    if FACE_ANALYSER is not None:
        FACE_ANALYSER.close()
    if FACE_SWAPPER is not None:
        FACE_SWAPPER.close()
    if FACE_PARSER is not None:
        FACE_PARSER.close()
    if FACE_ENHANCER is not None:
        FACE_ENHANCER.close()

    # Liberar la memoria VRAM
    FACE_ANALYSER = None
    FACE_SWAPPER = None
    FACE_PARSER = None
    FACE_ENHANCER = None
    
    yield "🤖 Models Unloaded from GPU"
    time.sleep(5)
    yield ""
    return


## _________________________ MAIN PROCESS _________________________

def process(
    input_type,
    image_path,
    video_path,
    directory_path,
    source_path,
    output_path,
    output_name,
    keep_output_sequence,
    condition,
    age,
    distance,
    face_enhancer_name,
    enable_face_parser,
    mask_includes,
    mask_soft_kernel,
    mask_soft_iterations,
    blur_amount,
    face_scale,
    enable_laplacian_blend,
    crop_top,
    crop_bott,
    crop_left,
    crop_right,
    *specifics,
):
    global WORKSPACE
    global OUTPUT_FILE
    global PREVIEW
    WORKSPACE, OUTPUT_FILE, PREVIEW = None, None, None

## _________________________ GUI UPDATE FUNC _________________________

    def ui_before():
        return (
            gr.update(visible=True, value=PREVIEW),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(visible=False),
        )

    def ui_after():
        return (
            gr.update(visible=True, value=PREVIEW),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(visible=False),
        )

    def ui_after_vid():
        return (
            gr.update(visible=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=OUTPUT_FILE, visible=True),
        )

    start_time = time.time()
    total_exec_time = lambda start_time: divmod(time.time() - start_time, 60)
    get_finsh_text = lambda start_time: f"&nbsp;&nbsp;&nbsp;&nbsp;✔️ Completed in {int(total_exec_time(start_time)[0])} min {int(total_exec_time(start_time)[1])} sec."

## _________________________ PREPARE INPUTS & LOAD MODELS _________________________

    yield "⌛️ Loading Face Analyser...", *ui_before()
    load_face_analyser_model()

    yield "⌛ Loading Face Swapper...", *ui_before()
    load_face_swapper_model()

    if face_enhancer_name != "NONE":
        yield f"⌛ Loading {face_enhancer_name} ...", *ui_before()
        FACE_ENHANCER = load_face_enhancer_model(name=face_enhancer_name, device=device)
    else:
        FACE_ENHANCER = None

    if enable_face_parser:
        yield "⌛ Loading Face Parsing...", *ui_before()
        load_face_parser_model()

    includes = mask_regions_to_list(mask_includes)
    smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=int(mask_soft_iterations)).to(device) if mask_soft_iterations > 0 else None
    specifics = list(specifics)
    half = len(specifics) // 2
    sources = specifics[:half]
    specifics = specifics[half:]

## _________________________ ANALYSE & SWAP FUNC _________________________

    def swap_process(image_sequence):
        yield "⌛ Analysing Face Data...", *ui_before()
        if condition != "Specific Face":
            source_data = source_path, age
        else:
            source_data = ((sources, specifics), distance)
        analysed_targets, analysed_sources, whole_frame_list, num_faces_per_frame = get_analysed_data(
            FACE_ANALYSER,
            image_sequence,
            source_data,
            swap_condition=condition,
            detect_condition=DETECT_CONDITION,
            scale=face_scale
        )

        yield "⌛ Swapping Faces...", *ui_before()
        preds, aimgs, matrs = FACE_SWAPPER.batch_forward(whole_frame_list, analysed_targets, analysed_sources)
        EMPTY_CACHE()

        if enable_face_parser:
            yield "⌛ Applying Face-parsing mask...", *ui_before()
            for idx, (pred, aimg) in tqdm(enumerate(zip(preds, aimgs)), total=len(preds), desc="Face parsing"):
                preds[idx] = swap_regions(pred, aimg, FACE_PARSER, smooth_mask, includes=includes, blur=int(blur_amount))
        EMPTY_CACHE()

        if face_enhancer_name != "NONE":
            yield f"⌛ Enhancing Faces with {face_enhancer_name}...", *ui_before()
            for idx, pred in tqdm(enumerate(preds), total=len(preds), desc=f"{face_enhancer_name}"):
                enhancer_model, enhancer_model_runner = FACE_ENHANCER
                pred = enhancer_model_runner(pred, enhancer_model)
                preds[idx] = cv2.resize(pred, (512,512))
                aimgs[idx] = cv2.resize(aimgs[idx], (512,512))
                matrs[idx] /= 0.25

        EMPTY_CACHE()

        split_preds = split_list_by_lengths(preds, num_faces_per_frame)
        del preds
        split_aimgs = split_list_by_lengths(aimgs, num_faces_per_frame)
        del aimgs
        split_matrs = split_list_by_lengths(matrs, num_faces_per_frame)
        del matrs

        yield "⌛ Post-processing...", *ui_before()
        def post_process(frame_idx, frame_img, split_preds, split_aimgs, split_matrs, enable_laplacian_blend, crop_top, crop_bott, crop_left, crop_right):
            whole_img_path = frame_img
            whole_img = cv2.imread(whole_img_path)
            for p, a, m in zip(split_preds[frame_idx], split_aimgs[frame_idx], split_matrs[frame_idx]):
                whole_img = paste_to_whole(p, a, m, whole_img, laplacian_blend=enable_laplacian_blend, crop_mask=(crop_top, crop_bott, crop_left, crop_right))
            cv2.imwrite(whole_img_path, whole_img)

        def concurrent_post_process(image_sequence, split_preds, split_aimgs, split_matrs, enable_laplacian_blend, crop_top, crop_bott, crop_left, crop_right):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for idx, frame_img in enumerate(image_sequence):
                    future = executor.submit(
                        post_process,
                        idx,
                        frame_img,
                        split_preds,
                        split_aimgs,
                        split_matrs,
                        enable_laplacian_blend,
                        crop_top,
                        crop_bott,
                        crop_left,
                        crop_right
                    )
                    futures.append(future)

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Post-Processing"):
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"An error occurred: {e}")

        concurrent_post_process(
            image_sequence,
            split_preds,
            split_aimgs,
            split_matrs,
            enable_laplacian_blend,
            crop_top,
            crop_bott,
            crop_left,
            crop_right
        )
## _________________________ IMAGE _________________________

    if input_type == "Image":
        target = cv2.imread(image_path)
        # Get date and time
        now = datetime.datetime.now()
        # Format time
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(output_path, output_name + "_" + date_str + ".png")
        cv2.imwrite(output_file, target)

        for info_update in swap_process([output_file]):
            yield info_update

        OUTPUT_FILE = output_file
        WORKSPACE = output_path
        PREVIEW = cv2.imread(output_file)[:, :, ::-1]

        yield get_finsh_text(start_time), *ui_after()

## _________________________ VIDEO _________________________

    elif input_type == "Video":
        temp_path = os.path.join(output_path, output_name, "sequence")
        os.makedirs(temp_path, exist_ok=True)

        yield "⌛ Extracting video frames...", *ui_before()
        image_sequence = []
        cap = cv2.VideoCapture(video_path)
        curr_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:break
            frame_path = os.path.join(temp_path, f"frame_{curr_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            image_sequence.append(frame_path)
            curr_idx += 1
        cap.release()
        cv2.destroyAllWindows()

        for info_update in swap_process(image_sequence):
            yield info_update

        yield "⌛ Merging sequence...", *ui_before()
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_video_path = os.path.join(output_path, output_name + "_" + date_str + ".mp4")
        merge_img_sequence_from_ref(video_path, image_sequence, output_video_path)

        if os.path.exists(temp_path) and not keep_output_sequence:
            yield "⌛ Removing temporary files...", *ui_before()
            shutil.rmtree(temp_path)
        last_generated_video = output_video_path
        WORKSPACE = output_path
        OUTPUT_FILE = output_video_path

        yield get_finsh_text(start_time), *ui_after_vid()

## _________________________ DIRECTORY _________________________

    elif input_type == "Directory":
        extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "ico", "webp"]
        temp_path = os.path.join(output_path, output_name)
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)

        file_paths =[]
        for file_path in glob.glob(os.path.join(directory_path, "*")):
            if any(file_path.lower().endswith(ext) for ext in extensions):
                img = cv2.imread(file_path)
                new_file_path = os.path.join(temp_path, os.path.basename(file_path))
                cv2.imwrite(new_file_path, img)
                file_paths.append(new_file_path)

        for info_update in swap_process(file_paths):
            yield info_update

        PREVIEW = cv2.imread(file_paths[-1])[:, :, ::-1]
        WORKSPACE = temp_path
        OUTPUT_FILE = file_paths[-1]

        yield get_finsh_text(start_time), *ui_after()

## _________________________ STREAM _________________________

    elif input_type == "Stream":
        pass


## _________________________ GRADIO FUNC _________________________


def update_radio(value):
    if value == "Image":
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif value == "Video":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    elif value == "Directory":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    elif value == "Stream":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )


def swap_option_changed(value):
    if value.startswith("Age"):
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    elif value == "Specific Face":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def video_changed(video_path):
    sliders_update = gr.Slider.update
    button_update = gr.Button.update
    number_update = gr.Number.update

    if video_path is None:
        return (
            sliders_update(minimum=0, maximum=0, value=0),
            sliders_update(minimum=1, maximum=1, value=1),
            number_update(value=1),
        )
    try:
        clip = VideoFileClip(video_path)
        fps = clip.fps
        total_frames = clip.reader.nframes
        clip.close()
        return (
            sliders_update(minimum=0, maximum=total_frames, value=0, interactive=True),
            sliders_update(
                minimum=0, maximum=total_frames, value=total_frames, interactive=True
            ),
            number_update(value=fps),
        )
    except:
        return (
            sliders_update(value=0),
            sliders_update(value=0),
            number_update(value=1),
        )

def analyse_settings_changed(detect_condition, detection_size, detection_threshold):
    yield "⌛ Applying new values..."
    global FACE_ANALYSER
    global DETECT_CONDITION
    DETECT_CONDITION = detect_condition
    FACE_ANALYSER = insightface.app.FaceAnalysis(name="buffalo_l", providers=PROVIDER)
    FACE_ANALYSER.prepare(
        ctx_id=0,
        det_size=(int(detection_size), int(detection_size)),
        det_thresh=float(detection_threshold),
    )
    yield f"🆗 Applied detect condition:{detect_condition}, detection size: {detection_size}, detection threshold: {detection_threshold}"

def stop_running():
    global STREAMER
    if hasattr(STREAMER, "stop"):
        STREAMER.stop()
        STREAMER = None
    yield "💢 Process Stopped."
    time.sleep(5)
    yield ""

def slider_changed(show_frame, video_path, frame_index):
    if not show_frame:
        return None, None
    if video_path is None:
        return None, None
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(frame_index / clip.fps)
    frame_array = np.array(frame)
    clip.close()
    return gr.Image.update(value=frame_array, visible=True), gr.Video.update(
        visible=False
    )

def trim_and_reload(video_path, output_path, output_name, start_frame, stop_frame):
    yield video_path, f"⌛ Trimming video frame {start_frame} to {stop_frame} ..."
    try:
        output_path = os.path.join(output_path, output_name)
        trimmed_video = trim_video(video_path, output_path, start_frame, stop_frame)
        yield trimmed_video, "🆗 Video trimmed and reloaded."
    except Exception as e:
        print(e)
        yield video_path, "💢 Video trimming failed. See console for more info."

## _________________________ GRADIO GUI _________________________

class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Swap"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()
    
    def on_ui_tabs(self):
        return [(interface, "Swap", "swap")]


def remove_showing_image():
    if OUTPUT_FILE is None:
        yield "💬 No Image or Video generated."
    else:
        try:
            base_filename = os.path.basename(OUTPUT_FILE)
            os.remove(OUTPUT_FILE)
            yield f"🖼️ {base_filename}: has been deleted..."
        except FileNotFoundError:
            yield f"💢 {base_filename}: does not exist..."
    time.sleep(5)
    yield ""

def download_models_bt():
    models_download_dir = os.path.join(base_dir, "assets", "pretrained_models")
    num_urls = len(urls)
    num_downloaded = 0
    for url in urls:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            filename = os.path.basename(url)
            file_path = os.path.join(models_download_dir, filename)

            if os.path.exists(file_path):
                yield f"💬 {filename} already exists. Skipping download."
                num_downloaded += 1
                continue

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1 MB
            downloaded_size = 0
            prev_progress = -1

            with open(file_path, "wb") as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    downloaded_size += len(data)
                    percentage = round(downloaded_size / total_size * 100, 1)
                    current_progress = int(percentage // 5)
                    
                    if current_progress > prev_progress:
                        progress_bar = "🟩" * current_progress + "" * (50 - current_progress)
                        total_size_mb = total_size / (1024 * 1024)  # Convert total_size to megabytes
                        yield f"🔽 Downloading: {filename}({total_size_mb:.2f}) MB {progress_bar} Completed: {percentage:.1f}%"
                        prev_progress = current_progress

            num_downloaded += 1

        except requests.exceptions.RequestException as e:
            yield f"💢 Couldn't download {filename}. Error: {e}"
        except Exception as e:
            yield f"💢 Unknown error downloading {filename}. Error: {e}"

    if num_downloaded == num_urls:
        yield "🆗 All Models downloaded."
    time.sleep(5)
    yield ""

urls = [
    "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "https://huggingface.co/bluefoxcreation/Codeformer-ONNX/resolve/main/codeformer.onnx",
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth",
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth",
    "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x8.pth",
    "https://github.com/zllrunning/face-makeup.PyTorch/raw/master/cp/79999_iter.pth",
]


with gr.Blocks() as interface:

    with gr.Row():
        with gr.Row():
            with gr.Column(scale=0.4):
                with gr.Tab("🌟 Swap"):
                    swap_option = gr.Dropdown(swap_options_list, label="Face to swap", multiselect=False, show_label=True, value=swap_options_list[0], interactive=True)
                    age = gr.Number(value=25, label="Age", info="Im not sure if this work", interactive=True, visible=False)
                    face_enhancer_name = gr.Dropdown(FACE_ENHANCER_LIST, label="Face Enhancer", value="NONE", multiselect=False,interactive=True)
                    keep_output_sequence = gr.Checkbox(value=False, label="Keep Temp Sequence", interactive=True)
                    
                with gr.Tab("🔍 Detection"):
                    detect_condition_dropdown = gr.Dropdown( detect_conditions, label="Condition", value=DETECT_CONDITION, interactive=True, info="This condition is only used when multiple faces are detected on source or specific image.")
                    detection_size = gr.Number(label="Detection Size", value=DETECT_SIZE, interactive=True)
                    detection_threshold = gr.Number(label="Detection Threshold", value=DETECT_THRESH, interactive=True)
                    apply_detection_settings = gr.Button("Apply settings", variant="primary")

                with gr.Tab("🎭 Masks"):
                    with gr.Accordion("Advanced Mask", open=False):
                        enable_face_parser_mask = gr.Checkbox(label="Enable Face Parsing", value=False, interactive=True)

                        mask_include = gr.Dropdown(mask_regions.keys(), value=MASK_INCLUDE, multiselect=True, label="Include", interactive=True)
                        mask_soft_kernel = gr.Number(label="Soft Erode Kernel", value=MASK_SOFT_KERNEL, minimum=3, interactive=True, visible = False)
                        mask_soft_iterations = gr.Number(label="Soft Erode Iterations", value=MASK_SOFT_ITERATIONS, minimum=0, interactive=True)
                        blur_amount = gr.Number(label="Mask Blur", value=MASK_BLUR_AMOUNT, minimum=0, interactive=True)

                    face_scale = gr.Slider(label="Face Scale", minimum=0, maximum=2, value=1, interactive=True)

                    with gr.Accordion("Crop Mask", open=False):
                        crop_top = gr.Number(label="Top", value=0, minimum=0, interactive=True)
                        crop_bott = gr.Number(label="Bottom", value=0, minimum=0, interactive=True)
                        crop_left = gr.Number(label="Left", value=0, minimum=0, interactive=True)
                        crop_right = gr.Number(label="Right", value=0, minimum=0, interactive=True)

                    enable_laplacian_blend = gr.Checkbox(label="Laplacian Blending", value=True, interactive=True)
                
                with gr.Tab("Ⓜ️ Models"):
                    button_models_download = gr.Button(value="🔽 Download Models", label="Download Models", visible=not USE_COLAB)
                    output_directory = gr.Text(value=outputs_dir, label="Output Directory",  interactive=True, visible=False)
                    output_name = gr.Text(label="Output Name", value="Image", interactive=True, visible=False)

                with gr.Box():
                    gr.HTML("""&nbsp;&nbsp;&nbsp;Input Face""", elem_id="input_face")
                    source_image_input = gr.Image(label="Input Face", elem_id="input_face", type="filepath", interactive=True)

                with gr.Box(visible=False) as specific_face:
                    for i in range(NUM_OF_SRC_SPECIFIC):
                        idx = i + 1
                        code = "\n"
                        code += f"with gr.Tab(label='({idx})'):"
                        code += "\n\twith gr.Row():"
                        code += f"\n\t\tsrc{idx} = gr.Image(interactive=True, type='numpy', label='Source Face {idx}')"
                        code += f"\n\t\ttrg{idx} = gr.Image(interactive=True, type='numpy', label='Specific Face {idx}')"
                        exec(code)

                    distance_slider = gr.Slider(minimum=0, maximum=2, value=0.6, interactive=True, label="Distance", info="Lower distance is more similar and higher distance is less similar to the target face.")

                with gr.Group():
                    with gr.Box():
                        gr.HTML("""&nbsp;&nbsp;&nbsp;Target Face</span>""", elem_id="target_face")
                        input_type = gr.Radio(["Image", "Video", "Directory"], label="", value="Video")

                    with gr.Box(visible=False) as input_image_group:
                        image_input = gr.Image(interactive=True, type="filepath", elem_id="preview_img_back")
                        
                    with gr.Box(visible=True) as input_video_group:
                        vid_widget = gr.Text if USE_COLAB else gr.Video
                        video_input = vid_widget(
                            label="Target Video Path", interactive=True
                        )
                        with gr.Accordion("✂️ Video Edit", open=False):
                            with gr.Column():
                                with gr.Row():
                                    set_slider_range_btn = gr.Button("Get Frames Range", interactive=True)
                                    show_trim_preview_btn = gr.Checkbox(label="Show frame when slider change", value=True, interactive=True)

                                video_fps = gr.Number(value=30, interactive=False, label="Fps", visible=False)
                                start_frame = gr.Slider(minimum=0, maximum=1, value=0, step=1, interactive=True, label="◀️ Start Frame", info="")
                                end_frame = gr.Slider(minimum=0, maximum=1, value=1, step=1, interactive=True, label="▶️ End Frame", info="")
                            trim_and_reload_btn = gr.Button("Cut and Reload", variant="primary", interactive=True)

                    with gr.Box(visible=False) as input_directory_group:
                        direc_input = gr.Text(label="Path", interactive=True)

            with gr.Column():
                info = gr.HTML(value="", elem_id="info", interactive=False)            
                with gr.Row():
                    swap_button = gr.Button("✨ Swap", variant="primary")
                    cancel_button = gr.Button("⛔ Cancel", variant="stop")
                    remove_image = gr.Button("🗑️ Remove", interactive=True, variant="stop")
                    
                with gr.Row():
                    output_directory_button = gr.Button("📂 Open Results", visible=not USE_COLAB)
                    output_video_button = gr.Button("📦 Open File", visible=not USE_COLAB)
                    unload_models_button = gr.Button(value="🆓 Unload Models", label="Unload Models", visible=not USE_COLAB)

                with gr.Box():
                    gr.HTML("""&nbsp;&nbsp;&nbsp;Preview""", elem_id="preview_img")
                    preview_image = gr.Image(label="Output", interactive=False)
                    preview_video = gr.Video(label="Output", interactive=False, visible=False)
                        
                with gr.Box():
                    with gr.Row():
                        gr.Markdown("[🧩 Extension](https://github.com/rauldlnx10/sd-webui-swap-mukham)", elem_id="extension")
                        gr.Markdown("[📝 Official version](https://github.com/harisreedhar/Swap-Mukham)", elem_id="oficial")
                        gr.Markdown("[📜 Acknowledgements](https://github.com/harisreedhar/Swap-Mukham#acknowledgements)", elem_id="thanks")
                        gr.Markdown("[☁️ Run in Colab](https://colab.research.google.com/github/harisreedhar/Swap-Mukham/blob/main/swap_mukham_colab.ipynb)", elem_id="colab")

## _________________________ GRADIO EVENTS _________________________

    unload_models_button.click(unload_models, outputs=info)
    button_models_download.click(fn=download_models_bt, outputs=info, show_progress=True,)
    remove_image.click(fn=remove_showing_image, outputs=info)
    #send_to_img2img.click()
 
    set_slider_range_event = set_slider_range_btn.click(video_changed, inputs=[video_input], outputs=[start_frame, end_frame, video_fps],)

    trim_and_reload_event = trim_and_reload_btn.click(
        fn=trim_and_reload,
        inputs=[video_input, output_directory, output_name, start_frame, end_frame],
        outputs=[video_input, info],
    )

    start_frame_event = start_frame.release(
        fn=slider_changed,
        inputs=[show_trim_preview_btn, video_input, start_frame],
        outputs=[preview_image, preview_video],
        show_progress=True,
    )

    end_frame_event = end_frame.release(
        fn=slider_changed,
        inputs=[show_trim_preview_btn, video_input, end_frame],
        outputs=[preview_image, preview_video],
        show_progress=True,
    )

    input_type.change(
        update_radio,
        inputs=[input_type],
        outputs=[input_image_group, input_video_group, input_directory_group],
    )
    swap_option.change(
        swap_option_changed,
        inputs=[swap_option],
        outputs=[age, specific_face, source_image_input],
    )

    apply_detection_settings.click(
        analyse_settings_changed,
        inputs=[detect_condition_dropdown, detection_size, detection_threshold],
        outputs=[info],
    )

    src_specific_inputs = []
    gen_variable_txt = ",".join(
        [f"src{i+1}" for i in range(NUM_OF_SRC_SPECIFIC)]
        + [f"trg{i+1}" for i in range(NUM_OF_SRC_SPECIFIC)]
    )
    exec(f"src_specific_inputs = ({gen_variable_txt})")
    swap_inputs = [
        input_type,
        image_input,
        video_input,
        direc_input,
        source_image_input,
        output_directory,
        output_name,
        keep_output_sequence,
        swap_option,
        age,
        distance_slider,
        face_enhancer_name,
        enable_face_parser_mask,
        mask_include,
        mask_soft_kernel,
        mask_soft_iterations,
        blur_amount,
        face_scale,
        enable_laplacian_blend,
        crop_top,
        crop_bott,
        crop_left,
        crop_right,
        *src_specific_inputs,
    ]

    swap_outputs = [
        info,
        preview_image,
        output_directory_button,
        output_video_button,
        preview_video,
    ]

    swap_event = swap_button.click(
        fn=process, inputs=swap_inputs, outputs=swap_outputs, show_progress=True
    )

    cancel_button.click(
        fn=stop_running,
        inputs=None,
        outputs=[info],
        cancels=[
            swap_event,
            trim_and_reload_event,
            set_slider_range_event,
            start_frame_event,
            end_frame_event,
        ],
        show_progress=True,
    )
    output_directory_button.click(lambda: open_directory(path=outputs_dir), inputs=None, outputs=None)
    output_video_button.click(lambda: open_directory(path=OUTPUT_FILE), inputs=None, outputs=None)

if __name__ == "__main__":
    if USE_COLAB:
        print("Running in colab mode")

    interface.queue(concurrency_count=2, max_size=20).launch(share=USE_COLAB)

script = Script()
script_callbacks.on_ui_tabs(script.on_ui_tabs)
