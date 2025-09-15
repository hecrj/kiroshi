import torch

from enum import Enum


class Precision(Enum):
    FLOAT16 = 0
    BFLOAT16 = 1
    FLOAT32 = 2

    def parse(precision: str) -> "Precision":
        return {
            "float16": Precision.FLOAT16,
            "bfloat16": Precision.BFLOAT16,
            "float32": Precision.FLOAT32,
        }[precision]

    def dtype(self):
        match self:
            case Precision.FLOAT16:
                return torch.float16
            case Precision.BFLOAT16:
                return torch.bfloat16
            case Precision.FLOAT32:
                return torch.float32
