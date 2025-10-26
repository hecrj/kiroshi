from image import Image
from dataclasses import dataclass


@dataclass
class Video:
    width: int
    height: int
    framerate: int
    frames: list[Image]
