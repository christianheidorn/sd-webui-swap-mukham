import os
import torch
import gfpgan
from PIL import Image
from upscaler.RealESRGAN import RealESRGAN


def gfpgan_runner(img, model):
    _, imgs, _ = model.enhance(img, paste_back=True, has_aligned=True)
    return imgs[0]


def realesrgan_runner(img, model):
    img = model.predict(img)
    return img

supported_enhancers = {
    "GFPGAN": ("./assets/pretrained_models/GFPGANv1.4.pth", gfpgan_runner),
    "GFPGANx2": ("./assets/pretrained_models/GFPGANv1.4.pth", gfpgan_runner),
    "REAL-ESRGAN 2x": ("./assets/pretrained_models/RealESRGAN_x2.pth", realesrgan_runner),
    "REAL-ESRGAN 4x": ("./assets/pretrained_models/RealESRGAN_x4.pth", realesrgan_runner),
    "REAL-ESRGAN 8x": ("./assets/pretrained_models/RealESRGAN_x8.pth", realesrgan_runner),
}

def get_available_enhancer_names():
    available = []
    for name, data in supported_enhancers.items():
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), data[0])
        if os.path.exists(path):
            available.append(name)
    return available

def load_face_enhancer_model(name='GFPGAN', device="cpu"):
    assert name in get_available_enhancer_names(), f"Face enhancer {name} unavailable."
    model_path, model_runner = supported_enhancers.get(name)
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_path)
    if name == 'GFPGAN':
        model = gfpgan.GFPGANer(model_path=model_path, upscale=1, device=device)
    elif name == 'GFPGANx2':
        model = gfpgan.GFPGANer(model_path=model_path, upscale=2, device=device)
    elif name == 'REAL-ESRGAN 2x':
        model = RealESRGAN(device, scale=2)
        model.load_weights(model_path, download=False)
    elif name == 'REAL-ESRGAN 4x':
        model = RealESRGAN(device, scale=4)
        model.load_weights(model_path, download=False)
    elif name == 'REAL-ESRGAN 8x':
        model = RealESRGAN(device, scale=8)
        model.load_weights(model_path, download=False)
    else:
        model = None
    return (model, model_runner)
