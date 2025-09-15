from dataclasses import dataclass


@dataclass(frozen=True)
class Size:
    width: int
    height: int

    def from_dict(size: dict) -> "Size":
        return Size(width=size["width"], height=size["height"])
