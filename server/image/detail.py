from .recipe import Recipe
from .pipe import Pipe
from .prompt import Prompt
from .latent import notify
from .image import Image
from .rectangle import Rectangle
from .anima import AnimaPipeline, inpaint_steps

from diffusers.image_processor import VaeImageProcessor

from dataclasses import dataclass
import torch
import gc
import PIL.Image
import PIL.ImageFilter

initialized = False


@dataclass(frozen=True)
class Detail:
    strength: int
    padding: int
    max_area: int | None = None

    def from_dict(detail: dict):
        return Detail(
            strength=detail["strength"],
            padding=detail["padding"],
            max_area=detail.get("max_area"),
        )


def detail_faces(
    recipe: Recipe,
    detail: Detail,
    image: Image,
    on_progress=None,
) -> tuple[PIL.Image.Image, dict]:
    with Pipe.acquire(recipe) as pipe:
        output, faces = increase_detail(
            "face",
            "weights/face_yolov8n.pt",
            recipe,
            detail,
            pipe,
            image.raw,
            max_amount=1,
            on_progress=on_progress,
        )

        return output, {"faces": faces}


def detail_hands(
    recipe: Recipe,
    detail: Detail,
    image: Image,
    on_progress=None,
) -> tuple[PIL.Image.Image, dict]:
    with Pipe.acquire(recipe) as pipe:
        output, hands = increase_detail(
            "hand",
            "weights/hand_yolov9c.pt",
            recipe,
            detail,
            pipe,
            image.raw,
            max_amount=2,
            on_progress=on_progress,
        )

    return output, {"hands": hands}


def increase_detail(
    label: str,
    model: str,
    recipe: Recipe,
    detail: Detail,
    pipe: Pipe,
    image: PIL.Image.Image,
    max_amount: int | None = None,
    on_progress=None,
) -> tuple[PIL.Image.Image, list[dict]]:
    from adetailer.mask import mask_preprocess, bbox_area
    from adetailer import ultralytics_predict

    global initialized

    is_anima = isinstance(pipe.generate, AnimaPipeline)

    if pipe.inpaint is None and not is_anima:
        raise ValueError("detailing is not supported for this model")

    if not initialized:
        gc.collect()
        torch.cuda.empty_cache()
        initialized = True

    prompt = Prompt(pipe.compel, recipe.prompt, recipe.negative_prompt)
    generator = torch.Generator(device="cuda").manual_seed(recipe.seed)
    prediction = ultralytics_predict(
        model,
        image,
        confidence=0.1,
    )

    if not (prediction.masks):
        return (image, [])

    n = len(prediction.masks)
    print(f"{n} {label}(s) detected")

    max_amount = min(max_amount, n) or n
    i = 0
    processed = 0

    while i < n and processed < max_amount:
        area = bbox_area(prediction.bboxes[i])

        if detail.max_area is not None and area > detail.max_area:
            i += 1
            continue

        print(f"Detailing {label} with area: {area}")

        mask = prediction.masks[i]
        mask = mask_preprocess([mask], 4)[0]

        if is_anima:
            # Anima native inpainting: the white region is regenerated
            # through a strength-fraction of the schedule while the rest
            # of the image is preserved exactly. Like the SDXL
            # `padding_mask_crop`, the masked area is cropped with
            # `detail.padding` of context and both the crop and its mask
            # are upscaled to the processing resolution before the
            # generation, so the region is generated at a higher
            # resolution than it occupied in the image; the result is
            # then overlaid back on the original image.
            strength = detail.strength / 100.0

            mask = mask.filter(PIL.ImageFilter.GaussianBlur(4))

            crops_coords = VaeImageProcessor.get_crop_region(
                mask,
                recipe.size.width,
                recipe.size.height,
                pad=detail.padding,
            )

            generated = pipe.generate(
                image=image.crop(crops_coords).resize(
                    (recipe.size.width, recipe.size.height),
                    PIL.Image.Resampling.BILINEAR,
                ),
                mask=mask.crop(crops_coords).resize(
                    (recipe.size.width, recipe.size.height),
                    PIL.Image.Resampling.BILINEAR,
                ),
                strength=strength,
                num_inference_steps=recipe.steps,
                guidance_scale=recipe.guidance,
                width=recipe.size.width,
                height=recipe.size.height,
                generator=generator,
                prompt_embeds=prompt.embeds,
                negative_prompt_embeds=prompt.negative_embeds,
                sampler=pipe.sampler,
                pag_scale=recipe.pag.scale if recipe.pag else 0,
                callback_on_step_end=notify(
                    recipe, on_progress, inpaint_steps(recipe.steps, strength)
                ),
                callback_on_step_end_tensor_inputs=["latents"],
            ).images[0]

            image = pipe.generate.image_processor.apply_overlay(
                mask, image, generated, crops_coords
            )
        else:
            mask = pipe.inpaint.mask_processor.blur(mask, blur_factor=4)

            image = pipe.inpaint(
                image=image,
                mask_image=mask,
                strength=detail.strength / 100.0,
                padding_mask_crop=detail.padding,
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
            ).images[0]

        i += 1
        processed += 1

    regions = [
        Rectangle.from_list(bbox).to_dict()
        for bbox in prediction.bboxes[i - processed : i]
    ]

    return (image, regions)
