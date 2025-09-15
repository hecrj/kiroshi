from enum import Enum


class Sampler(Enum):
    EULER_A = 0
    DPM_SDE_KARRAS = 1
    DPM_2M_KARRAS = 2
    DPM_2M_SDE_KARRAS = 3

    def parse(sampler: str) -> "Sampler":
        return {
            "euler_a": Sampler.EULER_A,
            "dpm++_sde_karras": Sampler.DPM_SDE_KARRAS,
            "dpm++_2m_karras": Sampler.DPM_2M_KARRAS,
            "dpm++_2m_sde_karras": Sampler.DPM_2M_SDE_KARRAS,
        }[sampler]
