from .recipe import Recipe
from .pipe import Pipe
from .prompt import Prompt
from .latent import notify
from .image import Image
from .rectangle import Rectangle

from multiprocessing import Process, Queue
from dataclasses import dataclass
import torch
import gc
import PIL.Image

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


def adetailer(input, output):
    while True:
        try:
            (model, image) = input.get()
        except:
            return

        from adetailer import ultralytics_predict

        prediction = ultralytics_predict(
            model,
            image,
            confidence=0.1,
        )
        output.put(prediction)


AdetailerInput = Queue()
AdetailerOutput = Queue()
Process(target=adetailer, args=(AdetailerInput, AdetailerOutput)).start()


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

    global initialized

    if not initialized:
        gc.collect()
        torch.cuda.empty_cache()
        initialized = True

    AdetailerInput.put((model, image))

    prompt = Prompt(pipe.compel, recipe.prompt, recipe.negative_prompt)
    generator = torch.Generator(device="cuda").manual_seed(recipe.seed)
    prediction = AdetailerOutput.get()

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
