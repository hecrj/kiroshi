from .precision import Precision
from .sampler import Sampler
from .lora import Lora
from .pag import Pag
from .size import Size

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Recipe:
    model: str
    prompt: str
    seed: int
    size: Size
    negative_prompt: str = ""
    precision: Precision = Precision.BFLOAT16
    sampler: Sampler = Sampler.EULER_A
    steps: int = 30
    guidance: float = 5.0
    loras: tuple[Lora] = field(default_factory=lambda: tuple([]))
    pag: Pag | None = None

    def from_dict(data: dict) -> "Recipe":
        pag = data.get("pag")

        if pag is not None:
            pag = Pag.from_dict(pag)

        return Recipe(
            model=f"/models/image/{data['model']}.safetensors",
            prompt=data["prompt"],
            seed=data["seed"],
            size=Size.from_dict(data["size"]),
            negative_prompt=data["negative_prompt"],
            precision=Precision.parse(data["precision"]),
            sampler=Sampler.parse(data["sampler"]),
            steps=data["steps"],
            guidance=data["guidance"],
            loras=tuple([Lora.from_dict(lora) for lora in data.get("loras", [])]),
            pag=pag,
        )
