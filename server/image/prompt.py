import torch
from compel import Compel, CompelForSDXL


def _split(result) -> tuple[torch.Tensor, torch.Tensor | None]:
    # compel only returns (embeds, pooled) when pooled is requested;
    # otherwise it returns the embeds tensor alone.
    if isinstance(result, tuple):
        return result
    return result, None


class Prompt:
    embeds: torch.Tensor
    pooled: torch.Tensor | None
    negative_embeds: torch.Tensor | None
    negative_pooled: torch.Tensor | None

    def __init__(
        self, compel: "Compel | CompelForSDXL", positive: str, negative: str | None
    ):
        if isinstance(compel, CompelForSDXL):
            conditioning = compel(positive, negative_prompt=negative)

            self.embeds = conditioning.embeds
            self.pooled = conditioning.pooled_embeds
            self.negative_embeds = conditioning.negative_embeds
            self.negative_pooled = conditioning.negative_pooled_embeds

            return

        self.embeds, self.pooled = _split(compel(positive))

        if negative is not None:
            self.negative_embeds, self.negative_pooled = _split(compel(negative))

            self.embeds, self.negative_embeds = (
                compel.pad_conditioning_tensors_to_same_length(
                    [self.embeds, self.negative_embeds]
                )
            )
