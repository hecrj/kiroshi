from .recipe import Recipe
from .pipe import Pipe
from .prompt import Prompt
from .latent import notify


from typing import Callable

import PIL.Image
import torch


def generate(
    recipe: Recipe,
    cpu_offload: bool = False,
    on_progress: Callable[[float, PIL.Image.Image], None] | None = None,
) -> PIL.Image.Image:
    with Pipe.acquire(recipe, cpu_offload) as pipe:
        prompt = Prompt(pipe.compel, recipe.prompt, recipe.negative_prompt)
        generator = torch.Generator(device="cuda").manual_seed(recipe.seed)

        return pipe.generate(
            num_inference_steps=recipe.steps,
            guidance_scale=recipe.guidance,
            prompt_embeds=prompt.embeds,
            pooled_prompt_embeds=prompt.pooled,
            negative_prompt_embeds=prompt.negative_embeds,
            negative_pooled_prompt_embeds=prompt.negative_pooled,
            width=recipe.size.width,
            height=recipe.size.height,
            generator=generator,
            callback_on_step_end=notify(recipe, on_progress),
            callback_on_step_end_tensor_inputs=["latents"],
            pag_scale=recipe.pag.scale if recipe.pag else 0,
        ).images[0]
