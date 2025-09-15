from dataclasses import dataclass


@dataclass(frozen=True)
class Rectangle:
    x: int
    y: int
    width: int
    height: int

    def from_list(coords: list[float]) -> "Rectangle":
        [left, top, right, bottom] = coords

        return Rectangle(
            x=int(round(left)),
            y=int(round(top)),
            width=int(round(right - left)),
            height=int(round(bottom - top)),
        )

    def to_dict(self) -> dict:
        return vars(self)
