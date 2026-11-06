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

        src = image.raw

        # The RealESRGAN weights are RGB-only, and transparent pixels
        # carry no meaningful color underneath, so an image with
        # transparency is processed as RGB + alpha: the RGB channels
        # are run through the upscaler while the alpha channel is
        # resampled separately, then the two are combined again.
        alpha = None

        if src.mode == "RGBA":
            if src.getchannel("A").getextrema()[0] < 255:
                alpha = src.getchannel("A")
            src = src.convert("RGB")

        rgb = pipe.predict(
            src, patches_size=upscaler.tile_size, padding=upscaler.tile_padding
        )

        if alpha is not None:
            alpha = alpha.resize(rgb.size, PIL.Image.Resampling.LANCZOS)
            rgb.putalpha(alpha)

        return rgb
    finally:
        semaphor.release()
