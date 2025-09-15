from .precision import Precision
from .sampler import Sampler
from .recipe import Recipe
from .lora import Lora

from typing import Any, Iterator
from compel import Compel, ReturnedEmbeddingsType
from contextlib import contextmanager

import torch
import gc
import time
import threading

semaphore = threading.Semaphore()
pipe: type["Pipe"] | None = None


class Pipe:
    model: str
    precision: Precision
    sampler: Sampler
    pag: bool
    cpu: bool
    loras: tuple[Lora]
    generate: Any
    inpaint: Any
    compel: Compel

    @contextmanager
    def acquire(recipe: Recipe, cpu: bool = False) -> Iterator["Pipe"]:
        global semaphore, pipe
        semaphore.acquire()

        try:
            if pipe is None:
                pipe = Pipe(
                    recipe.model,
                    recipe.precision,
                    recipe.sampler,
                    recipe.pag is not None,
                    recipe.loras,
                    cpu,
                )
            else:
                pipe.update(
                    recipe.model,
                    recipe.precision,
                    recipe.sampler,
                    recipe.pag,
                    recipe.loras,
                    cpu,
                )

            yield pipe

        finally:
            semaphore.release()

    def __init__(
        self,
        model: str,
        precision: Precision,
        sampler: Sampler,
        pag: bool,
        loras: tuple[Lora] = tuple([]),
        cpu: bool = False,
    ):
        from diffusers import (
            AutoPipelineForText2Image,
            StableDiffusionXLPipeline,
        )

        gc.collect()
        torch.cuda.empty_cache()

        self.model = model
        self.precision = precision
        self.sampler = sampler
        self.pag = pag
        self.loras = loras
        self.cpu = cpu

        self.generate = StableDiffusionXLPipeline.from_single_file(
            model,
            config="sdxl-1.0",
            use_safetensors=True,
            torch_dtype=precision.dtype(),
            local_files_only=True,
        )

        if pag:
            self.generate = AutoPipelineForText2Image.from_pipe(
                self.generate, enable_pag=True, pag_applied_layers=["mid"]
            )

        self.generate = self.generate.to("cuda")
        self.generate.safety_checker = None

        if loras:
            print("Fusing LoRAs...")
            start = time.time()

            for lora in loras:
                self.generate.load_lora_weights(
                    lora.path,
                    adapter_name=lora.name(),
                )

            self.generate.set_adapters(
                [lora.name() for lora in loras],
                adapter_weights=[lora.strength / 100.0 for lora in loras],
            )
            self.generate.fuse_lora()

            print(f"LoRAs fused: {time.time() - start}s")

        if cpu:
            self.generate.enable_model_cpu_offload()

        if hasattr(self.generate, "tokenizer_2"):
            self.compel = Compel(
                tokenizer=[self.generate.tokenizer, self.generate.tokenizer_2],
                text_encoder=[self.generate.text_encoder, self.generate.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False,
            )
        else:
            self.compel = Compel(
                tokenizer=self.generate.tokenizer,
                text_encoder=self.generate.text_encoder,
                truncate_long_prompts=False,
            )

        self.resample(sampler)

    def resample(self, sampler: Sampler):
        from diffusers import AutoPipelineForInpainting

        match sampler:
            case Sampler.EULER_A:
                from diffusers import EulerAncestralDiscreteScheduler

                self.generate.scheduler = EulerAncestralDiscreteScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    timestep_spacing="leading",
                    steps_offset=1,
                )

            case Sampler.DPM_SDE_KARRAS:
                from diffusers import DPMSolverSinglestepScheduler

                self.generate.scheduler = DPMSolverSinglestepScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    use_karras_sigmas=True,
                    algorithm_type="sde-dpmsolver++",
                )

            case Sampler.DPM_2M_KARRAS | Sampler.DPM_2M_SDE_KARRAS:
                from diffusers import DPMSolverMultistepScheduler

                self.generate.scheduler = DPMSolverMultistepScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    timestep_spacing="leading",
                    steps_offset=1,
                    euler_at_final=True,
                    use_karras_sigmas=True,
                    algorithm_type="dpmsolver++"
                    if sampler == Sampler.DPM_2M_KARRAS
                    else "sde-dpmsolver++",
                )

        self.sampler = sampler
        self.inpaint = AutoPipelineForInpainting.from_pipe(self.generate)

    def update(
        self,
        model: str,
        precision: Precision,
        sampler: Sampler,
        pag: bool,
        loras: list[Lora] = [],
        cpu: bool = False,
    ):
        is_new = (
            model != self.model
            or precision != self.precision
            or loras != self.loras
            or pag != self.pag
            or cpu != self.cpu
        )

        if is_new:
            self.__init__(model, precision, sampler, pag, loras, cpu)
        elif sampler != self.sampler:
            self.resample(sampler)
