from dataclasses import dataclass


@dataclass
class Configuration:
    steps: int
    guidance: float
    width: int
    height: int
    prompt_embeds: any
    prompt_pooled: any
    negative_prompt_embeds: str
    negative_prompt_pooled: str
    generator: any
    on_step_end: any
