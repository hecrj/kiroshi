import PIL.Image

from dataclasses import dataclass


@dataclass
class Image:
    raw: PIL.Image.Image
    hash: int

    def __init__(self, raw: PIL.Image.Image, hash: int):
        self.raw = raw
        self.hash = hash

    def __eq__(self, other) -> bool:
        if isinstance(other, Image):
            return self.hash == other.hash
        return NotImplemented

    def __hash__(self) -> int:
        return self.hash
