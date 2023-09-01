import launch
import os
import requests

current_dir = os.path.dirname(os.path.realpath(__file__))
req_file = os.path.join(current_dir, "requirements.txt")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not launch.is_installed(lib):
            launch.run_pip(
                f"install {lib}",
                f"Swap-Mukham requirement: {lib}")

models_dir = os.path.abspath("models/SwapMuhkan")
model_url = "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir, model_name)

def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(model_path):
    download(model_url, model_path)
