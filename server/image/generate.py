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

        from diffusers import Krea2Pipeline

        from .anima import AnimaPipeline

        if isinstance(pipe.generate, AnimaPipeline):
            # Flow-matching pipeline: single fixed-length conditioning tensor
            # (Qwen3-0.6B + T5 -> LLM adapter), no pooled embeddings or PAG.
            kwargs = dict(
                prompt_embeds=prompt.embeds,
                sampler=pipe.sampler,
            )
            if prompt.negative_embeds is not None:
                kwargs["negative_prompt_embeds"] = prompt.negative_embeds
        elif isinstance(pipe.generate, Krea2Pipeline):
            # Flow-matching pipeline: multi-layer text embeddings with an
            # attention mask, no pooled embeddings or PAG.
            kwargs = dict(
                prompt_embeds=prompt.embeds,
                prompt_embeds_mask=prompt.embeds_mask,
            )
            if prompt.negative_embeds is not None:
                kwargs["negative_prompt_embeds"] = prompt.negative_embeds
                kwargs["negative_prompt_embeds_mask"] = prompt.negative_embeds_mask
        else:
            kwargs = dict(
                prompt_embeds=prompt.embeds,
                pooled_prompt_embeds=prompt.pooled,
                negative_prompt_embeds=prompt.negative_embeds,
                negative_pooled_prompt_embeds=prompt.negative_pooled,
                pag_scale=recipe.pag.scale if recipe.pag else 0,
            )

        return pipe.generate(
            num_inference_steps=recipe.steps,
            guidance_scale=recipe.guidance,
            width=recipe.size.width,
            height=recipe.size.height,
            generator=generator,
            callback_on_step_end=notify(recipe, on_progress),
            callback_on_step_end_tensor_inputs=["latents"],
            **kwargs,
        ).images[0]
