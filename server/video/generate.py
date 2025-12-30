from typing import Callable

import numpy as np
import PIL.Image
import torch
import gc

from image import Image, Pipe as ImagePipe

from .pipe import Pipe
from .recipe import Recipe
from .video import Video


def generate(
    recipe: Recipe,
    first_frame: Image,
    last_frame: Image | None,
    on_progress: Callable[[float, PIL.Image.Image], None] | None = None,
) -> Video:
    ImagePipe.clear()
    gc.collect()
    torch.cuda.empty_cache()

    with Pipe.acquire(recipe) as pipe:
        aspect_ratio = first_frame.raw.height / first_frame.raw.width
        mod_value = (
            pipe.generate.vae_scale_factor_spatial
            * pipe.generate.transformer.config.patch_size[1]
        )

        width = round(np.sqrt(recipe.max_area / aspect_ratio)) // mod_value * mod_value
        height = round(np.sqrt(recipe.max_area * aspect_ratio)) // mod_value * mod_value

        framerate = 16
        generator = torch.Generator(device="cuda").manual_seed(recipe.seed)
        image = first_frame.raw.resize((width, height))
        last_image = None

        if last_frame is not None:
            last_image = last_frame.raw.resize((width, height))

        frames = pipe.generate(
            image=image,
            last_image=last_image,
            prompt=recipe.prompt,
            negative_prompt=recipe.negative_prompt,
            width=width,
            height=height,
            num_frames=recipe.duration * framerate + 1,
            guidance_scale=recipe.guidance,
            num_inference_steps=recipe.steps,
            generator=generator,
            output_type="pil",
        ).frames[0]

        frames = [
            Image(raw=frame, hash=hash((recipe, first_frame, i)))
            for i, frame in enumerate(frames)
        ]

        return Video(width, height, framerate=framerate, frames=frames)
