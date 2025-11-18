from image import Size, Precision

from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    model: str
    seed: int
    prompt: str
    size: Size
    duration: int
    max_area: int = 720 * 1280
    negative_prompt: str = ""
    precision: Precision = Precision.BFLOAT16
    steps: int = 4
    guidance: float = 1.0

    def from_dict(data: dict) -> "Recipe":
        return Recipe(
            model=f"/models/video/{data['model']}.safetensors",
            seed=data["seed"],
            prompt=data["prompt"],
            size=Size.from_dict(data["size"]),
            duration=data["duration"],
            max_area=data["max_area"],
            negative_prompt=data["negative_prompt"],
            precision=Precision.parse(data["precision"]),
            steps=data["steps"],
            guidance=data["guidance"],
        )
