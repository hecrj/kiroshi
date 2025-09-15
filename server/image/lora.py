from dataclasses import dataclass

from pathlib import Path


@dataclass(frozen=True)
class Lora:
    path: str
    strength: int

    def from_dict(lora: dict):
        return Lora(path=lora["path"], strength=lora["strength"])

    def name(self):
        return Path(self.path).stem.replace(".", "")
