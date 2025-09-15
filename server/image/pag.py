from dataclasses import dataclass


@dataclass(frozen=True)
class Pag:
    scale: float = 3.0

    def from_dict(pag: dict):
        return Pag(scale=pag["scale"])
