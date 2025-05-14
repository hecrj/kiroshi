from text_to_image.configuration import Configuration
from text_to_image.detail import Detail, increase_face_detail, increase_hand_detail

from enum import Enum
from PIL import Image
from typing import Callable
from dataclasses import dataclass, field
import torch
import threading
import gc
import time
from pathlib import Path

class Quality(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    ULTRA = 3
    INSANE = 4

class Upscaling(Enum):
    REAL_ESRGAN_2X = 0
    ULTRASHARP_4X = 2

    def scale(self):
        match self:
            case Upscaling.REAL_ESRGAN_2X:
                return 2
            case Upscaling.ULTRASHARP_4X:
                return 4

    def weight(self):
        match self:
            case Upscaling.REAL_ESRGAN_2X:
                return "RealESRGAN_x2plus"
            case Upscaling.ULTRASHARP_4X:
                return "4x-UltraSharp"


class Sampler(Enum):
    EULER_A = 0
    DPM_SDE_KARRAS = 1
    DPM_2M_KARRAS = 2
    DPM_2M_SDE_KARRAS = 3

last_parameters = None
last_image = None
last_face = None
last_model = None
last_loras = None
last_sampler = None
last_cpu_offload = None
pipe = None
inpainting_pipe = None
upscaler_pipe = None
last_upscaling = None
semaphore = threading.Semaphore()


@dataclass
class Lora:
    path: str
    strength: int

    def from_dict(lora: dict):
        return Lora(path=lora['path'], strength=lora['strength'])

    def name(self):
        return Path(self.path).stem.replace('.', '')


@dataclass
class Parameters:
    model: str
    prompt: str
    width: int
    height: int
    negative_prompt: str = ""
    steps: int = 30
    guidance: float = 5.0
    seed: int | None = None
    quality: Quality = Quality.NORMAL
    loras: list[Lora] = field(default_factory=list)
    sampler: Sampler = Sampler.EULER_A


@dataclass
class Upscaler:
    model: Upscaling = Upscaling.ULTRASHARP_4X
    tile_size: int = 192
    tile_padding: int = 24


@dataclass
class Cache:
    value: any
    generator: torch.Tensor
    key: any = None


@dataclass
class Generation:
    image: Image
    faces: list[list[float]]
    hands: list[list[float]]


def generate(parameters: Parameters,
             upscaler: Upscaler | None = None,
             face_detail: Detail | None = None,
             hand_detail: Detail | None = None,
             on_progress: Callable[float, Image] | None = None,
             cpu_offload: bool = False) -> Generation:
    global last_parameters, last_image, last_face, last_generator, last_model, last_loras, last_sampler, last_cpu_offload
    global pipe, inpainting_pipe, last_upscaling, upscaler_pipe, compel_proc, semaphore
    semaphore.acquire()

    from diffusers import AutoPipelineForInpainting, StableDiffusionXLPipeline
    from compel import Compel, ReturnedEmbeddingsType
    from RealESRGAN import RealESRGAN
    import torch

    if parameters.loras != last_loras:
        last_model = None

    if last_model != parameters.model or last_cpu_offload != cpu_offload:
        del pipe, inpainting_pipe
        gc.collect()
        torch.cuda.empty_cache()

        pipe = StableDiffusionXLPipeline.from_single_file(
            parameters.model,
            config="sdxl-1.0",
            use_safetensors=True,
            torch_dtype=torch.float16,
            local_files_only=True)
        pipe = pipe.to("cuda")
        pipe.safety_checker = None

        last_model = parameters.model
        last_loras = parameters.loras
        last_sampler = None
        last_image = None
        last_face = None
        last_cpu_offload = cpu_offload

        if parameters.loras:
            print("Fusing LoRAs...")
            start = time.time()

            for lora in parameters.loras:
                pipe.load_lora_weights(
                    lora.path,
                    adapter_name=lora.name(),
                )

            pipe.set_adapters([lora.name() for lora in parameters.loras],
                              adapter_weights=[
                                  lora.strength / 100.0
                                  for lora in parameters.loras
                              ])
            pipe.fuse_lora()

            print(f"LoRAs fused: {time.time() - start}s")

        if hasattr(pipe, 'tokenizer_2'):
            compel_proc = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.
                PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False)
        else:
            compel_proc = Compel(tokenizer=pipe.tokenizer,
                                 text_encoder=pipe.text_encoder,
                                 truncate_long_prompts=False)

        # TODO: Expose setting
        # pipe.unet = torch.compile(pipe.unet,
        #                           mode="reduce-overhead",
        #                           fullgraph=True)

        if cpu_offload:
            pipe.enable_model_cpu_offload()

    if last_sampler != parameters.sampler:
        match parameters.sampler:
            case Sampler.EULER_A:
                from diffusers import EulerAncestralDiscreteScheduler

                pipe.scheduler = EulerAncestralDiscreteScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    timestep_spacing="leading",
                    steps_offset=1,
                )

            case Sampler.DPM_SDE_KARRAS:
                from diffusers import DPMSolverSinglestepScheduler

                pipe.scheduler = DPMSolverSinglestepScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    use_karras_sigmas=True,
                    algorithm_type="sde-dpmsolver++")

            case Sampler.DPM_2M_KARRAS | Sampler.DPM_2M_SDE_KARRAS:
                from diffusers import DPMSolverMultistepScheduler

                pipe.scheduler = DPMSolverMultistepScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    timestep_spacing="leading",
                    steps_offset=1,
                    euler_at_final=True,
                    use_karras_sigmas=True,
                    algorithm_type="dpmsolver++" if parameters.sampler
                    == Sampler.DPM_2M_KARRAS else "sde-dpmsolver++",
                )

        inpainting_pipe = AutoPipelineForInpainting.from_pipe(pipe)
        last_sampler = parameters.sampler

    if not upscaler is None and last_upscaling != upscaler.model:
        weight = upscaler.model.weight()
        scale = upscaler.model.scale()

        print(f"Loading {weight} upscaler ({scale}x)...")

        device = torch.device('cuda')
        upscaler_pipe = RealESRGAN(device, scale=scale)
        upscaler_pipe.load_weights(f'weights/{weight}.pth')
        last_upscaling = upscaler.model

    def on_step_end(pipe, step, timestep, callback_kwargs):
        nonlocal on_progress
        latents = callback_kwargs["latents"]

        on_progress(step / parameters.steps, latents_to_rgb(latents))

        return callback_kwargs

    if hasattr(pipe, 'tokenizer_2'):
        prompt_embeds, prompt_pooled = compel_proc(parameters.prompt)
        negative_prompt_embeds, negative_prompt_pooled = compel_proc(
            parameters.negative_prompt)
    else:
        prompt_embeds, prompt_pooled = [compel_proc(parameters.prompt), None]
        negative_prompt_embeds, negative_prompt_pooled = [
            compel_proc(parameters.negative_prompt), None
        ]

    [prompt_embeds, negative_prompt_embeds
     ] = compel_proc.pad_conditioning_tensors_to_same_length(
         [prompt_embeds, negative_prompt_embeds])

    if not parameters.seed is None:
        generator = torch.Generator(device="cuda").manual_seed(parameters.seed)
    else:
        generator = None

    match parameters.quality:
        case Quality.LOW:
            quality_factor = 1.0
        case Quality.NORMAL:
            quality_factor = 1.25
        case Quality.HIGH:
            quality_factor = 1.5
        case Quality.ULTRA:
            quality_factor = 1.75
        case Quality.INSANE:
            quality_factor = 2.0

    configuration = Configuration(
        steps=parameters.steps,
        guidance=parameters.guidance,
        width=int(parameters.width * quality_factor),
        height=int(parameters.height * quality_factor),
        prompt_embeds=prompt_embeds,
        prompt_pooled=prompt_pooled,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_pooled=negative_prompt_pooled,
        generator=generator,
        on_step_end=on_step_end)

    is_new = last_image is None or parameters.seed is None or parameters != last_parameters

    try:
        if is_new:
            image = pipe(
                num_inference_steps=configuration.steps,
                guidance_scale=configuration.guidance,
                prompt_embeds=configuration.prompt_embeds,
                pooled_prompt_embeds=configuration.prompt_pooled,
                negative_prompt_embeds=configuration.negative_prompt_embeds,
                negative_pooled_prompt_embeds=configuration.
                negative_prompt_pooled,
                width=configuration.width,
                height=configuration.height,
                generator=configuration.generator,
                callback_on_step_end=configuration.on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images[0]

            if parameters.seed is None:
                last_parameters = None
                last_image = None
                last_face = None
            else:
                last_parameters = parameters
                last_image = Cache(
                    value=image, generator=configuration.generator.get_state())
                last_face = None
        else:
            image = last_image.value
            configuration.generator.set_state(last_image.generator)

        faces = []
        hands = []

        if not face_detail is None:
            face_detail.padding *= quality_factor

            if not face_detail.max_area is None:
                face_detail.max_area *= quality_factor

            if is_new or last_face is None or face_detail != last_face.key:
                (image, faces) = increase_face_detail(face_detail,
                                                      configuration, image,
                                                      inpainting_pipe)

                if parameters.seed is None:
                    last_face = None
                else:
                    last_face = Cache(
                        key=face_detail,
                        value=(image, faces),
                        generator=configuration.generator.get_state())
            else:
                (image, faces) = last_face.value
                configuration.generator.set_state(last_face.generator)

        if not hand_detail is None:
            hand_detail.padding *= quality_factor

            if not hand_detail.max_area is None:
                hand_detail.max_area *= quality_factor

            (image, hands) = increase_hand_detail(hand_detail, configuration,
                                                  image, inpainting_pipe)

        if upscaler is None:
            image = image.copy()
        else:
            print(f"Upscaling: {upscaler}")
            on_progress(1.0, image.copy())

            start = time.time()
            image = upscaler_pipe.predict(image, patches_size=upscaler.tile_size, padding=upscaler.tile_padding)

            scale = upscaler.model.scale()
            faces = [[point * scale for point in face] for face in faces]
            hands = [[point * scale for point in hand] for hand in hands]
            print(f"Upscaled: {time.time() - start}s")

    finally:
        semaphore.release()

    return Generation(image, faces, hands)


def latents_to_rgb(latents):
    weights = ((60, -60, 25, -70), (60, -5, 15, -50), (60, 10, -5, -35))

    weights_tensor = torch.t(
        torch.tensor(weights, dtype=latents.dtype).to(latents.device))

    biases_tensor = torch.tensor((150, 140, 130),
                                 dtype=latents.dtype).to(latents.device)

    rgb_tensor = torch.einsum(
        "...lxy,lr -> ...rxy", latents,
        weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)

    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)

    return Image.fromarray(image_array)
