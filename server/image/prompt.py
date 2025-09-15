import torch
from compel import Compel


class Prompt:
    embeds: torch.Tensor
    pooled: torch.Tensor | None
    negative_embeds: torch.Tensor | None
    negative_pooled: torch.Tensor | None

    def __init__(self, compel: Compel, positive: str, negative: str | None):
        embeds, pooled = compel(positive)

        if negative is not None:
            negative_embeds, negative_pooled = compel(negative)

        if negative is not None:
            [embeds, negative_embeds] = compel.pad_conditioning_tensors_to_same_length(
                [embeds, negative_embeds]
            )

            self.negative_embeds = negative_embeds
            self.negative_pooled = negative_pooled

        self.embeds = embeds
        self.pooled = pooled
