from enum import Enum
from dataclasses import dataclass


class Model(Enum):
    REAL_ESRGAN_2X = 0
    ULTRASHARP_4X = 2

    def parse(upscaler: str) -> "Model":
        return {
            "2x-real_esrgan": Model.REAL_ESRGAN_2X,
            "4x-ultrasharp": Model.ULTRASHARP_4X,
        }[upscaler]

    def scale(self) -> int:
        match self:
            case Model.REAL_ESRGAN_2X:
                return 2
            case Model.ULTRASHARP_4X:
                return 4
            case _:
                raise Exception(f"invalid upscaling: {self}")

    def weight(self) -> str:
        match self:
            case Model.REAL_ESRGAN_2X:
                return "RealESRGAN_x2plus"
            case Model.ULTRASHARP_4X:
                return "4x-UltraSharp"
            case _:
                raise Exception(f"invalid upscaling: {self}")


@dataclass(frozen=True)
class Upscaler:
    model: Model = Model.ULTRASHARP_4X
    tile_size: int = 192
    tile_padding: int = 24

    def from_dict(upscaler: dict) -> "Upscaler":
        return Upscaler(
            model=Model.parse(upscaler["model"]),
            tile_size=upscaler["tile_size"],
            tile_padding=upscaler["tile_padding"],
        )
