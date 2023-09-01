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

# Download the ONNX file
model_url = "https://huggingface.co/henryruhs/roop/resolve/main/inswapper_128.onnx"
model_filename = "inswapper_128.onnx"
model_save_path = os.path.join(current_dir, "assets", "pretrained_models", model_filename)

response = requests.get(model_url)
if response.status_code == 200:
    with open(model_save_path, "wb") as model_file:
        model_file.write(response.content)
    print(f"Model downloaded and saved at {model_save_path}")
else:
    print(f"Failed to download the model from {model_url}")
