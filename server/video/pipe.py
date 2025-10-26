from image import Precision
from .recipe import Recipe

from typing import Any, Iterator
from contextlib import contextmanager

import torch
import gc
import threading

semaphore = threading.Semaphore()
pipe: type["Pipe"] | None = None


class Pipe:
    model: str
    precision: Precision
    generate: Any
    inpaint: Any

    @contextmanager
    def acquire(recipe: Recipe) -> Iterator["Pipe"]:
        global semaphore, pipe
        semaphore.acquire()

        try:
            if pipe is None:
                pipe = Pipe(recipe.model, recipe.precision)
            else:
                pipe.update(recipe.model, recipe.precision)

            yield pipe

        finally:
            semaphore.release()

    def clear():
        global semaphore, pipe
        semaphore.acquire()

        try:
            pipe = None

            gc.collect()
            torch.cuda.empty_cache()

        finally:
            semaphore.release()

    def __init__(self, model: str, precision: Precision):
        from diffusers import WanImageToVideoPipeline

        gc.collect()
        torch.cuda.empty_cache()

        self.model = model
        self.precision = precision

        self.generate = WanImageToVideoPipeline.from_single_file(
            model,
            config="Wan2.2-I2V-A14B",
            torch_dtype=precision.dtype(),
            transformer_2=None,
            boundary_ratio=None,
            local_files_only=True,
        )

        onload_device = torch.device("cuda")
        offload_device = torch.device("cpu")

        self.generate.transformer.enable_group_offload(
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=True,
            low_cpu_mem_usage=True,
        )

        self.generate = self.generate.to(onload_device)
        self.generate.safety_checker = None

    def update(
        self,
        model: str,
        precision: Precision,
    ):
        is_new = model != self.model or precision != self.precision

        if is_new:
            self.__init__(model, precision)
