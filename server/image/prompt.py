from typing import Any

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
    embeds_mask: torch.Tensor | None
    negative_embeds: torch.Tensor | None
    negative_pooled: torch.Tensor | None
    negative_embeds_mask: torch.Tensor | None

    def __init__(
        self,
        compel: "Compel | CompelForSDXL | Any",
        positive: str,
        negative: str | None,
    ):
        if isinstance(compel, CompelForSDXL):
            conditioning = compel(positive, negative_prompt=negative)

            self.embeds = conditioning.embeds
            self.pooled = conditioning.pooled_embeds
            self.negative_embeds = conditioning.negative_embeds
            self.negative_pooled = conditioning.negative_pooled_embeds
            self.embeds_mask = None
            self.negative_embeds_mask = None

            return

        # Krea 2 (Qwen3-VL text encoder): the ``compel`` slot holds the
        # pipeline itself. The pipeline encodes fixed-length
        # (prefix | prompt | padding | suffix) blocks and taps a stack of
        # hidden states, so there is no Compel and no pooled embeddings.
        from diffusers import Krea2Pipeline

        if isinstance(compel, Krea2Pipeline):
            embeds, embeds_mask = compel.encode_prompt([positive])

            self.embeds = embeds
            self.pooled = None
            self.embeds_mask = embeds_mask

            if negative is not None:
                negative_embeds, negative_embeds_mask = compel.encode_prompt([negative])

                self.negative_embeds = negative_embeds
                self.negative_pooled = None
                self.negative_embeds_mask = negative_embeds_mask
            else:
                self.negative_embeds = None
                self.negative_pooled = None
                self.negative_embeds_mask = None

            return

        self.embeds, self.pooled = _split(compel(positive))

        if negative is not None:
            self.negative_embeds, self.negative_pooled = _split(compel(negative))

            self.embeds, self.negative_embeds = (
                compel.pad_conditioning_tensors_to_same_length(
                    [self.embeds, self.negative_embeds]
                )
            )

        self.embeds_mask = None
        self.negative_embeds_mask = None
