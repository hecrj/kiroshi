from .upscaler import Upscaler
from .image import Image

import PIL.Image
import threading
import torch

semaphor = threading.Semaphore()
pipe = None
last = None


def upscale(
    upscaler: Upscaler,
    image: Image,
    on_progress=None,
) -> PIL.Image.Image:
    from RealESRGAN import RealESRGAN

    global semaphor, pipe, last
    semaphor.acquire()

    try:
        if last != upscaler:
            weight = upscaler.model.weight()
            scale = upscaler.model.scale()

            print(f"Loading {weight} upscaler ({scale}x)...")

            device = torch.device("cuda")
            pipe = RealESRGAN(device, scale=scale)
            pipe.load_weights(f"weights/{weight}.pth")
            last = upscaler

        return pipe.predict(
            image.raw, patches_size=upscaler.tile_size, padding=upscaler.tile_padding
        )
    finally:
        semaphor.release()
