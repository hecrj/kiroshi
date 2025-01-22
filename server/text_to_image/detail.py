from text_to_image.configuration import Configuration

from multiprocessing import Process, Queue
from dataclasses import dataclass
from PIL import Image
import torch
import gc

initialized = False


@dataclass
class Detail:
    strength: int
    padding: int
    max_area: int | None = None

    def from_dict(detail: dict):
        return Detail(strength=detail['strength'],
                      padding=detail['padding'],
                      max_area=detail.get('max_area'))


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


def increase_face_detail(detail: Detail, configuration: Configuration,
                         image: Image,
                         inpainting) -> (Image, list[list[float]]):
    return increase_detail("face",
                           "weights/face_yolov8n.pt",
                           detail,
                           configuration,
                           image,
                           inpainting,
                           max_amount=1)


def increase_hand_detail(detail: Detail, configuration: Configuration,
                         image: Image,
                         inpainting) -> (Image, list[list[float]]):
    return increase_detail("hand",
                           "weights/hand_yolov9c.pt",
                           detail,
                           configuration,
                           image,
                           inpainting,
                           max_amount=2)


def increase_detail(
        label: str,
        model: str,
        detail: Detail,
        configuration: Configuration,
        image: Image,
        inpainting,
        max_amount: int | None = None) -> (Image, list[list[float]]):
    from adetailer.mask import mask_preprocess, bbox_area

    global initialized

    if not initialized:
        gc.collect()
        torch.cuda.empty_cache()
        initialized = True

    AdetailerInput.put((model, image))
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

        if not detail.max_area is None and area > detail.max_area:
            i += 1
            continue

        print(f"Detailing {label} with area: {area}")

        mask = prediction.masks[i]
        mask = mask_preprocess([mask], 4)[0]
        mask = inpainting.mask_processor.blur(mask, blur_factor=4)

        image = inpainting(
            image=image,
            mask_image=mask,
            strength=detail.strength / 100.0,
            padding_mask_crop=detail.padding,
            num_inference_steps=configuration.steps,
            guidance_scale=configuration.guidance,
            prompt_embeds=configuration.prompt_embeds,
            pooled_prompt_embeds=configuration.prompt_pooled,
            negative_prompt_embeds=configuration.negative_prompt_embeds,
            negative_pooled_prompt_embeds=configuration.negative_prompt_pooled,
            width=configuration.width,
            height=configuration.height,
            generator=configuration.generator,
            callback_on_step_end=configuration.on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

        i += 1
        processed += 1

    return (image, prediction.bboxes[i - processed:i])
