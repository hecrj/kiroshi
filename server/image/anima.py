"""Anima 1.0 (``circlestone-labs/Anima``) single-file support.

Anima is a Cosmos-Predict2-based text-to-image model: a 2B DiT with the
ComfyUI ``net.`` key layout plus an "LLM adapter" (``llm_adapter.*``) that
cross-attends learned T5 token embeddings over Qwen3-0.6B hidden states.
The single-file checkpoints contain exactly those two parts; the Qwen3-0.6B
text encoder, the Qwen + T5 tokenizers and the Qwen-Image VAE (the same VAE
as Krea 2) come from the local ``anima-1.0`` reference config dir.

Sampling follows the reference implementations (ComfyUI's k-diffusion
sigma-domain samplers and the diffusers Anima pipeline):

- the flow-matching sigmas live in ``[0, 1]`` with the shift=3.0 schedule
  (``sigma' = 3s / (1 + 2s)``), and the transformer receives the raw sigma
  as its timestep;
- latents are 5D flow latents ``[B, 16, 1, H/8, W/8]``;
- a zero pixel-space padding mask ``[1, 1, H, W]`` is concatenated to the
  latents (the DiT sees 16+1 input channels);
- classifier-free guidance is ``uncond + guidance * (cond - uncond)``;
- VAE decoding applies ``latents * latents_std + latents_mean``.

Only Anima 1.0-generation (2B, 28-block) checkpoints are supported. The
expanded-layer "29B" fine-tunes (40 blocks) are rejected with a clear error.
"""

import json
import math
import re
import struct
import time
from pathlib import Path

import numpy as np
import PIL.Image
import safetensors
import torch
import torch.nn.functional as F
import torchsde

from diffusers import (
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    DiffusionPipeline,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import AnimaLoraLoaderMixin
from diffusers.loaders.single_file_utils import (
    convert_cosmos_transformer_checkpoint_to_diffusers,
)
from diffusers.utils import BaseOutput
from transformers import Qwen2Tokenizer, Qwen3Model, T5TokenizerFast

from .sampler import Sampler

#: Marker keys identifying an Anima checkpoint (raw ``net.`` layout or
#: ``model.diffusion_model.``-prefixed ComfyUI unified layout). The
#: ``llm_adapter`` is Anima-specific (Cosmos Predict2 files lack it).
ANIMA_MARKER_KEYS = (
    "net.llm_adapter.blocks.0.cross_attn.q_proj.weight",
    "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight",
)

_ANIMA_PREFIXES = ("net.", "model.diffusion_model.")

#: Flow-matching schedule constants (Anima ``sampling_settings``:
#: ``shift=3.0``, ``multiplier=1.0``).
_SIGMA_SHIFT = 3.0
_SIGMA_GRID = 1000

#: Anima 1.0-generation architecture (2B). Anything else is an
#: expanded-layer fine-tune and out of scope.
_ANIMA_MODEL_CHANNELS = 2048
_ANIMA_HEAD_DIM = 128
_ANIMA_NUM_LAYERS = 28


def _timestep_start(steps: int, strength: float) -> int:
    """Index of the first sigma of a ``strength``-fraction trajectory.

    ``strength`` is the fraction of the denoising schedule to run (the
    diffusers ``get_timesteps`` convention): ``1.0`` starts at full noise,
    ``0.5`` at the middle of the grid, and ``0`` degenerates to the
    conditioning image alone.
    """
    init_timestep = min(steps * strength, steps)
    return int(max(steps - init_timestep, 0))


def inpaint_steps(steps: int, strength: float) -> int:
    """Number of denoising steps an Anima inpainting run at ``strength``
    actually executes (the ``strength``-fraction tail of the schedule).
    Used as the progress denominator of the step callback."""
    return steps - _timestep_start(steps, strength)


def _mask_tensor(mask: PIL.Image.Image, device: torch.device) -> torch.Tensor:
    """A pixel-space regeneration mask as float32 ``[1, 1, H, W]``.

    ``1.0`` marks the region to generate; ``0.0`` marks the conditioning
    region. The mask is the L-mode image as provided (already blurred),
    so the alpha gradient becomes the soft boundary between the two.
    """
    array = torch.from_numpy(np.asarray(mask.convert("L"), dtype=np.float32) / 255.0)
    return array[None, None].to(device)


def _header(path: str) -> dict:
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(length))


def is_anima_file(path: str) -> bool:
    """Whether a single-file checkpoint is an Anima diffusion model."""
    return any(key in ANIMA_MARKER_KEYS for key in _header(path) if key != "__metadata__")


# ---------------------------------------------------------------------------
# Schedules (ComfyUI "simple" + karras, in the shifted sigma domain)
# ---------------------------------------------------------------------------


def _time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    return alpha * t / (1 + (alpha - 1) * t)


def _simple_sigmas(steps: int, device: torch.device) -> torch.Tensor:
    """ComfyUI ``simple`` scheduler with the Anima shift (multiplier 1.0).

    Samples ``steps`` points evenly from the 1000-entry shifted grid, like
    ``comfy.samplers.simple_scheduler`` does with
    ``ModelSamplingDiscreteFlow(shift=3.0, multiplier=1.0)``.
    """
    t = torch.linspace(1.0 / _SIGMA_GRID, 1.0, _SIGMA_GRID, device=device)
    sigmas = _time_snr_shift(_SIGMA_SHIFT, t)
    stride = _SIGMA_GRID / steps
    out = torch.stack([sigmas[_SIGMA_GRID - 1 - int(x * stride)] for x in range(steps)])
    return torch.cat([out, torch.zeros(1, device=device)])


def _karras_sigmas(steps: int, device: torch.device) -> torch.Tensor:
    """Karras schedule over the Anima sigma range (k_diffusion)."""
    sigma_min = _time_snr_shift(_SIGMA_SHIFT, torch.tensor(1.0 / _SIGMA_GRID)).item()
    sigma_max = 1.0
    ramp = torch.linspace(0, 1, steps, device=device)
    min_inv_rho = sigma_min ** (1 / 7)
    max_inv_rho = sigma_max ** (1 / 7)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** 7
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def _get_ancestral_step(sigma_from: float, sigma_to: float, eta: float = 1.0) -> tuple[float, float]:
    sigma_up = min(sigma_to, eta * math.sqrt(sigma_to**2 - sigma_to**4 / sigma_from**2))
    sigma_down = math.sqrt(max(sigma_to**2 - sigma_up**2, 0.0))
    return sigma_down, sigma_up


# ---------------------------------------------------------------------------
# Samplers (ComfyUI / k-diffusion sigma-domain implementations,
# flow-model variants).
#
# ``predict(x, sigma)`` returns the x0 (clean-image) estimate: ComfyUI's
# FLOW models combine ``ModelSamplingDiscreteFlow`` with the ``CONST``
# class, whose ``calculate_denoised`` is ``x0 = x - sigma * v`` (``v`` is
# the DiT velocity output), and every sampler below consumes that x0
# estimate — exactly ``comfy.k_diffusion.sampling`` for CONST models.
#
# ``noise`` returns the SDE increment for a step. ``dpmpp_sde`` and
# ``dpmpp_2m_sde`` use a Brownian-tree sampler (``torchsde.BrownianTree``),
# a port of ComfyUI's ``BrownianTreeNoiseSampler``; ``euler_ancestral``
# uses independent per-step Gaussian noise, like ComfyUI's
# ``default_noise_sampler``.
#
# ``callback(i, x, sigma)`` reports each step before the update.
# ---------------------------------------------------------------------------


def _flow_lambda(sigma: torch.Tensor) -> torch.Tensor:
    """``sigma_to_half_log_snr`` for CONST (flow) models: ``-logit(sigma)``."""
    return sigma.logit().neg()


def _flow_sigma(half_log_snr: torch.Tensor) -> torch.Tensor:
    """``half_log_snr_to_sigma`` for CONST (flow) models."""
    return half_log_snr.neg().sigmoid()


def _offset_first_sigma_for_snr(sigmas: torch.Tensor) -> torch.Tensor:
    """``offset_first_sigma_for_snr``: ``-logit(1.0)`` is -inf, so when the
    first sigma is >= 1 it is nudged to ``percent_to_sigma(1e-4)``."""
    if len(sigmas) > 1 and sigmas[0] >= 1.0:
        sigmas = sigmas.clone()
        sigmas[0] = _time_snr_shift(_SIGMA_SHIFT, torch.tensor(1.0 - 1e-4)).item()
    return sigmas


class _BrownianTreeNoise:
    """Port of ComfyUI's ``BrownianTreeNoiseSampler`` (single seed).

    ``torchsde.BrownianTree`` gives temporally correlated Gaussian
    increments over the ``(sigma_min, sigma_max)`` interval, so the SDE
    noise of consecutive steps is consistent — matching ComfyUI.
    """

    def __init__(self, x: torch.Tensor, sigma_min: float, sigma_max: float, seed: int):
        self.sign = 1.0 if sigma_min < sigma_max else -1.0
        t0, t1 = (sigma_min, sigma_max) if self.sign > 0 else (sigma_max, sigma_min)
        # ComfyUI builds the tree from ``w0 = zeros_like(x)`` (latent dtype)
        # on CPU: mirror that construction exactly.
        w0 = torch.zeros_like(x).detach().cpu()
        self.tree = torchsde.BrownianTree(torch.as_tensor(t0), w0, torch.as_tensor(t1), entropy=seed)
        self.device = x.device

    def __call__(self, sigma: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
        t0, t1 = float(sigma), float(sigma_next)
        if t0 < t1:
            sign = 1.0
        else:
            sign = -1.0
            t0, t1 = t1, t0
        w = self.tree(t0, t1).to(device=self.device, dtype=torch.float32)
        return w * (self.sign * sign) / math.sqrt(abs(t1 - t0))


def _sample_euler_ancestral(predict, x: torch.Tensor, sigmas: torch.Tensor, noise, callback) -> torch.Tensor:
    """``sample_euler_ancestral_RF`` (the CONST/flow variant)."""
    for i in range(len(sigmas) - 1):
        callback(i, x, sigmas[i])
        denoised = predict(x, sigmas[i])

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            eta = 1.0
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
            # Euler method
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
            if eta > 0:
                x = (alpha_ip1 / alpha_down) * x + noise(x) * renoise_coeff
    return x


def _sample_dpmpp_2m(predict, x: torch.Tensor, sigmas: torch.Tensor, callback) -> torch.Tensor:
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in range(len(sigmas) - 1):
        callback(i, x, sigmas[i])
        denoised = predict(x, sigmas[i])
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


def _sample_dpmpp_2m_sde(predict, x: torch.Tensor, sigmas: torch.Tensor, noise, callback) -> torch.Tensor:
    sigmas = _offset_first_sigma_for_snr(sigmas)
    old_denoised = None
    h, h_last = None, None

    for i in range(len(sigmas) - 1):
        callback(i, x, sigmas[i])
        denoised = predict(x, sigmas[i])
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            lambda_s, lambda_t = _flow_lambda(sigmas[i]), _flow_lambda(sigmas[i + 1])
            h = lambda_t - lambda_s
            h_eta = h * 2  # eta = 1
            alpha_t = sigmas[i + 1] * lambda_t.exp()

            x = sigmas[i + 1] / sigmas[i] * (-h * 1.0).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                # solver_type = 'midpoint'
                x = x + 0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / r) * (denoised - old_denoised)

            x = x + noise(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * 1.0).expm1().neg().sqrt()
        old_denoised = denoised
        h_last = h
    return x


def _sample_dpmpp_sde(predict, x: torch.Tensor, sigmas: torch.Tensor, noise, callback) -> torch.Tensor:
    r = 1 / 2
    sigmas = _offset_first_sigma_for_snr(sigmas)

    for i in range(len(sigmas) - 1):
        callback(i, x, sigmas[i])
        denoised = predict(x, sigmas[i])
        if sigmas[i + 1] == 0:
            x = denoised
            continue

        lambda_s, lambda_t = _flow_lambda(sigmas[i]), _flow_lambda(sigmas[i + 1])
        h = lambda_t - lambda_s
        lambda_s_1 = lambda_s + r * h
        fac = 1 / (2 * r)

        sigma_s_1 = _flow_sigma(lambda_s_1)
        alpha_s = sigmas[i] * lambda_s.exp()
        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_t = sigmas[i + 1] * lambda_t.exp()

        # Step 1 (ancestral sub-step in half-log-SNR sigma, as ComfyUI does)
        sd, su = _get_ancestral_step(float(lambda_s.neg().exp()), float(lambda_s_1.neg().exp()))
        lambda_s_1_ = torch.as_tensor(sd).log().neg()
        h_ = lambda_s_1_ - lambda_s
        x_2 = (alpha_s_1 / alpha_s) * (-h_).exp() * x - alpha_s_1 * (-h_).expm1() * denoised
        x_2 = x_2 + alpha_s_1 * noise(sigmas[i], sigma_s_1) * su
        denoised_2 = predict(x_2, sigma_s_1)

        # Step 2
        sd, su = _get_ancestral_step(float(lambda_s.neg().exp()), float(lambda_t.neg().exp()))
        lambda_t_ = torch.as_tensor(sd).log().neg()
        h_ = lambda_t_ - lambda_s
        denoised_d = (1 - fac) * denoised + fac * denoised_2
        x = (alpha_t / alpha_s) * (-h_).exp() * x - alpha_t * (-h_).expm1() * denoised_d
        x = x + alpha_t * noise(sigmas[i], sigmas[i + 1]) * su
    return x


_SAMPLERS = {
    Sampler.EULER_A: ("euler_ancestral", _simple_sigmas, _sample_euler_ancestral),
    Sampler.DPM_SDE_KARRAS: ("dpmpp_sde", _karras_sigmas, _sample_dpmpp_sde),
    Sampler.DPM_2M_KARRAS: ("dpmpp_2m", _karras_sigmas, _sample_dpmpp_2m),
    Sampler.DPM_2M_SDE_KARRAS: ("dpmpp_2m_sde", _karras_sigmas, _sample_dpmpp_2m_sde),
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AnimaOutput(BaseOutput):
    images: list[PIL.Image.Image]


class AnimaPipeline(DiffusionPipeline, AnimaLoraLoaderMixin):
    """Anima 1.0 text-to-image pipeline (with native inpainting).

    Built from a single-file Anima checkpoint (DiT + LLM adapter) plus the
    ``anima-1.0`` reference components. The denoising loop is implemented
    directly in the sigma domain (see the module docstring), so no
    diffusers scheduler is involved. LoRA support follows
    ``AnimaLoraLoaderMixin`` (``transformer`` + ``text_conditioner``).

    Inpainting (``image`` + ``mask``): the mask's white region is painted
    1 in the DiT's padding-mask channel (the Cosmos convention, where 1
    marks the region whose content is not real) while the rest of the
    image is conditioned on through a ``strength``-fraction tail of the
    sigma schedule with the known region held exactly on its noise
    trajectory at every step.
    """

    model_cpu_offload_seq = "text_encoder->text_conditioner->transformer->vae"

    def __init__(
        self,
        transformer: CosmosTransformer3DModel,
        text_conditioner: AnimaTextConditioner,
        text_encoder: Qwen3Model,
        tokenizer: Qwen2Tokenizer,
        t5_tokenizer: T5TokenizerFast,
        vae: AutoencoderKLQwenImage,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            text_conditioner=text_conditioner,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            t5_tokenizer=t5_tokenizer,
            vae=vae,
        )
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @property
    def num_channels_latents(self) -> int:
        return self.transformer.config.in_channels

    @torch.no_grad()
    def encode_prompt(
        self,
        prompts: list[str],
        max_sequence_length: int = 512,
    ) -> torch.Tensor:
        """Encode prompts into Anima text conditioning.

        Runs the Qwen3-0.6B encoder and the T5 tokenizer, then the text
        conditioner (LLM adapter), which returns ``[B, 512, D]`` embeddings
        (padded to 512, padding zeroed).
        """
        device = self._execution_device
        dtype = self.text_encoder.dtype

        text_inputs = self.tokenizer(
            prompts,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        if input_ids.shape[-1] == 0:
            input_ids = input_ids.new_zeros((input_ids.shape[0], 1))
            attention_mask = attention_mask.new_zeros((attention_mask.shape[0], 1))

        prompt_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        prompt_embeds = prompt_embeds * attention_mask.to(prompt_embeds).unsqueeze(-1)

        t5_inputs = self.t5_tokenizer(
            prompts,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        t5_ids = t5_inputs.input_ids.to(device)
        t5_mask = t5_inputs.attention_mask.to(device)
        if t5_ids.shape[-1] == 0:
            t5_ids = t5_ids.new_zeros((t5_ids.shape[0], 1))
            t5_mask = t5_mask.new_zeros((t5_mask.shape[0], 1))

        return self.text_conditioner(
            prompt_embeds,
            t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=attention_mask,
        )

    @torch.no_grad()
    def _encode_image(self, image: PIL.Image.Image, width: int, height: int) -> torch.Tensor:
        """Encode a PIL image into normalized float32 latents.

        The exact inverse of ``_decode_latents`` (``(z - mean) / std`` of
        the ``vae.encode`` posterior mean), so the encoded conditioning
        round-trips through the same normalization the sampling loop
        operates in. The posterior mean (``mode()``) is used so the
        conditioning is deterministic for a given image.
        """
        z_dim = self.vae.config.z_dim
        device = self._execution_device
        dtype = self.vae.dtype

        pixels = self.image_processor.preprocess(image, height=height, width=width)
        encoded = self.vae.encode(pixels.to(device=device, dtype=dtype)[:, :, None]).latent_dist.mode()

        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(device)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(device)
        return ((encoded - latents_mean) / latents_std).float()

    def _decode_latents(self, latents: torch.Tensor) -> PIL.Image.Image:
        z_dim = self.vae.config.z_dim
        dtype = self.vae.dtype
        latents = latents.to(dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents.device, dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(latents.device, dtype)
        )
        latents = latents * latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return self.image_processor.postprocess(image.detach(), output_type="pil")[0]

    @torch.no_grad()
    def __call__(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        width: int = 1024,
        height: int = 1024,
        generator: torch.Generator | None = None,
        sampler: Sampler = Sampler.EULER_A,
        image: PIL.Image.Image | None = None,
        mask: PIL.Image.Image | None = None,
        strength: float = 1.0,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: tuple[str, ...] = ("latents",),
        **kwargs,
    ) -> AnimaOutput:
        width = (width // 16) * 16
        height = (height // 16) * 16
        if width == 0 or height == 0:
            raise ValueError(f"Width/height must be at least 16 pixels, got {width}x{height}")

        device = self._execution_device
        dtype = self.transformer.dtype
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor

        if generator is None:
            generator = torch.Generator(device=device)
        # ComfyUI's ``prepare_noise`` draws the initial noise on CPU in
        # float32 and casts it to the latent dtype — mirror that so the
        # initial latent is bit-identical for a given seed. One noise
        # tensor is drawn for the whole trajectory (the ComfyUI
        # convention); inpainting additionally reuses it for its
        # conditioning-region renoising.
        gen_device = generator.device if generator.device.type in ("cpu", "cuda") else device
        init_noise = torch.randn(
            (1, self.num_channels_latents, 1, latent_h, latent_w),
            dtype=torch.float32,
            generator=generator,
            device=gen_device,
        )

        if image is not None:
            # Inpainting: ``mask`` marks the region to regenerate (white);
            # everything else is conditioned on ``image``.
            if mask is None:
                raise ValueError("Anima inpainting requires a mask (white = region to regenerate)")
            if not 0.0 < strength <= 1.0:
                raise ValueError(f"Anima inpainting strength must be in (0, 1], got {strength}")

            original_size = (image.width, image.height)
            if original_size != (width, height):
                image = image.resize((width, height), PIL.Image.Resampling.BILINEAR)
                mask = mask.resize((width, height), PIL.Image.Resampling.BILINEAR)

            # Conditioning image as float32 normalized latents.
            image_latents = self._encode_image(image, width, height)

            # Pixel-space regeneration mask (float32, 1.0 = regenerate).
            gen = _mask_tensor(mask, device)

            # The per-step re-blend and the final snap operate on latents,
            # while the DiT resizes its padding mask from pixel space
            # itself — keep one mask per resolution.
            blend = F.interpolate(
                gen, size=(latent_h, latent_w), mode="bilinear", align_corners=False
            )
            keep = 1.0 - blend

            # The DiT's padding-mask channel follows the Cosmos training
            # convention, where 1 marks the region whose content is not
            # real (aspect-ratio padding), so the region to regenerate is
            # painted with 1 and the conditioning region with 0.
            padding_mask = gen.to(dtype)

            eps = init_noise.to(device=device)
            t_start = _timestep_start(num_inference_steps, strength)
        else:
            original_size = None
            image_latents = None
            gen = None
            blend = None
            keep = None
            eps = None
            t_start = 0

            # Zero pixel-space padding mask (T2I: no conditioning region).
            padding_mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)

        if negative_prompt_embeds is None or negative_prompt_embeds is prompt_embeds:
            guidance = 1.0
            negative_prompt_embeds = prompt_embeds
        else:
            guidance = guidance_scale

        # The k-diffusion samplers below operate on the x0 (clean-image)
        # estimate, not the raw velocity: ComfyUI's flow sampling combines
        # ``ModelSamplingDiscreteFlow`` with the ``CONST`` class, whose
        # ``calculate_denoised`` is ``x0 = x - sigma * v``. So the DiT's
        # (CFG-guided) velocity output is converted here before the sampler
        # sees it.
        def predict(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            timestep = sigma.expand(x.shape[0]).to(dtype)
            # The sampler state can drift to float32 after SDE steps; the
            # model always sees its own dtype (ComfyUI ``apply_model`` does
            # ``xc = xc.to(dtype)`` per forward).
            x_model = x.to(dtype)
            if image_latents is not None:
                # Keep the conditioning region exactly on its noise
                # trajectory at the current sigma — ``(1 - s) * x0 + s *
                # eps`` is the exact distribution of the known pixels at
                # time ``s`` — while the region to regenerate keeps the
                # sampler state.
                s = float(sigma)
                x_model = (keep * ((1.0 - s) * image_latents + s * eps) + blend * x_model).to(dtype)
            if math.isclose(guidance, 1.0):
                v = self.transformer(
                    hidden_states=x_model,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
            else:
                x_doubled = torch.cat([x_model, x_model])
                context = torch.cat([negative_prompt_embeds, prompt_embeds])
                out = self.transformer(
                    hidden_states=x_doubled,
                    timestep=timestep,
                    encoder_hidden_states=context,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                neg, pos = out.chunk(2)
                v = neg + guidance * (pos - neg)
            # x0 = x - sigma * v, in float32 like ComfyUI's
            # ``calculate_denoised(sigma, model_output.float(), x)``.
            return x - v.float() * sigma

        name, schedule, sampler_fn = _SAMPLERS[sampler]
        # Inpainting runs only the ``strength``-fraction tail of the
        # schedule; text-to-image (``t_start == 0``) is unchanged.
        sigmas = schedule(num_inference_steps, device)[t_start:]

        seed = generator.initial_seed()

        if image_latents is not None:
            # The conditioning image noised to the first sigma — the
            # exact trajectory state of the encoded image at ``s0``
            # (diffusers' ``scale_noise`` convention). At full strength
            # this is pure noise (text-to-image in the masked region);
            # at lower strengths the region to regenerate starts close
            # to the image, so sampling refines it instead of
            # overwriting it.
            s0 = float(sigmas[0])
            latents = (1.0 - s0) * image_latents + s0 * eps
        else:
            latents = init_noise.to(dtype=dtype, device=device)

        def callback(i: int, x: torch.Tensor, sigma: torch.Tensor) -> None:
            if callback_on_step_end is not None:
                preview = x
                if image_latents is not None:
                    # Offer the true trajectory state: the conditioning
                    # region renoised to the current sigma, so previews
                    # do not show drift outside the painted region.
                    s = float(sigma)
                    preview = keep * ((1.0 - s) * image_latents + s * eps) + blend * x
                callback_on_step_end(self, i, float(sigma), {"latents": preview})

        if name == "euler_ancestral":
            # ComfyUI's ``default_noise_sampler``: independent per-step
            # Gaussian noise; the CPU generator seed is offset by +1.
            rng_seed = seed + 1 if (seed is not None and device.type == "cpu") else seed
            sde_gen = torch.Generator(device=device).manual_seed(rng_seed)
            noise = lambda x: torch.randn(x.shape, generator=sde_gen, device=x.device, dtype=x.dtype)
            latents = sampler_fn(predict, latents, sigmas, noise, callback)
        elif name == "dpmpp_2m":
            latents = sampler_fn(predict, latents, sigmas, callback)
        else:
            # ComfyUI's SDE samplers draw their noise from a Brownian tree
            # over ``(sigma_min, sigma_max)`` seeded with the raw seed.
            noise = _BrownianTreeNoise(
                latents, float(sigmas[sigmas > 0].min()), float(sigmas.max()), seed
            )
            latents = sampler_fn(predict, latents, sigmas, noise, callback)

        if image_latents is not None:
            # Snap the conditioning region to the exact image latents:
            # the final x0 estimate covers the whole image and would
            # otherwise leave a faint drift outside the painted region.
            latents = keep * image_latents + blend * latents

        image = self._decode_latents(latents)
        if original_size is not None and (image.width, image.height) != original_size:
            image = image.resize(original_size, PIL.Image.Resampling.BILINEAR)

        return AnimaOutput(images=[image])


# ---------------------------------------------------------------------------
# Single-file loading
# ---------------------------------------------------------------------------


def _block_count(state: dict, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}blocks\.(\d+)\.")
    return max(int(m.group(1)) for key in state for m in [pattern.match(key)] if m) + 1


def _transformer_config(state: dict) -> dict:
    """Derive the ``CosmosTransformer3DModel`` config from a DiT state dict
    (ComfyUI layout, prefix stripped)."""
    x_proj = state["x_embedder.proj.1.weight"]
    in_channels = x_proj.shape[1] // 4 - 1  # concat_padding_mask
    model_channels = x_proj.shape[0]

    if model_channels != _ANIMA_MODEL_CHANNELS:
        raise ValueError(
            f"Unsupported Anima checkpoint: {model_channels} model channels "
            f"({x_proj.shape}). Only the 1.0-generation 2B architecture "
            f"({_ANIMA_MODEL_CHANNELS}) is supported; expanded-layer fine-tunes "
            "are out of scope."
        )

    q_norm = state["blocks.0.self_attn.q_norm.weight"]
    if q_norm.numel() != _ANIMA_HEAD_DIM:
        raise ValueError(f"Unsupported Anima checkpoint: head dim {q_norm.numel()}")

    out_channels = state["final_layer.linear.weight"].shape[0] // 4
    adaln_lora_dim = state["blocks.0.adaln_modulation_self_attn.1.weight"].shape[0]
    text_embed_dim = state["blocks.0.cross_attn.k_proj.weight"].shape[1]
    num_layers = _block_count(state, "")

    if num_layers != _ANIMA_NUM_LAYERS:
        raise ValueError(
            f"Unsupported Anima checkpoint: {num_layers} transformer blocks. Only the "
            f"1.0-generation {_ANIMA_NUM_LAYERS}-block architecture is supported; "
            "expanded-layer (\"29B\") fine-tunes are out of scope."
        )

    # NTK RoPE factors per in_channels (ComfyUI `supported_models.Anima`
    # detection): (t, h, w).
    rope_scale = (1.0, 4.0, 4.0) if in_channels == 16 else (1.0, 3.0, 3.0)

    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "num_attention_heads": model_channels // _ANIMA_HEAD_DIM,
        "attention_head_dim": _ANIMA_HEAD_DIM,
        "num_layers": num_layers,
        "mlp_ratio": 4.0,
        "text_embed_dim": text_embed_dim,
        "adaln_lora_dim": adaln_lora_dim,
        "max_size": (128, 240, 240),
        "patch_size": (1, 2, 2),
        "rope_scale": rope_scale,
        "concat_padding_mask": True,
        "extra_pos_embed_type": None,
    }


def _conditioner_config(state: dict, source_dim: int, target_dim: int) -> dict:
    """Derive the ``AnimaTextConditioner`` config from an LLM-adapter state
    dict (``llm_adapter.`` prefix stripped)."""
    embed = state["embed.weight"]
    num_layers = _block_count(state, "")
    head_dim = state["blocks.0.self_attn.q_norm.weight"].numel()
    return {
        "source_dim": source_dim,
        "target_dim": target_dim,
        "model_dim": embed.shape[1],
        "num_layers": num_layers,
        "num_attention_heads": embed.shape[1] // head_dim,
        "mlp_ratio": state["blocks.0.mlp.0.weight"].shape[0] // embed.shape[1],
        "target_vocab_size": embed.shape[0],
        "use_self_attention": "blocks.0.norm_self_attn.weight" in state,
        "use_layer_norm": "blocks.0.norm_self_attn.bias" in state,
        "min_sequence_length": 512,
    }


def load_anima_pipeline(
    single_file: str,
    dtype: torch.dtype,
    config_dir: str = "templates/anima",
    lora: bool = False,
) -> AnimaPipeline:
    """Load an Anima 1.0 pipeline from a transformer-only single file.

    The single file contains the DiT and the LLM adapter (ComfyUI key
    layout); the Qwen3-0.6B text encoder, tokenizers and VAE come from the
    ``templates/anima`` reference dir (the hecrj/anima submodule).
    """
    from accelerate import init_empty_weights

    required = (
        Path(config_dir) / "text_encoder" / "model.safetensors",
        Path(config_dir) / "text_encoder" / "config.json",
        Path(config_dir) / "vae" / "diffusion_pytorch_model.safetensors",
        Path(config_dir) / "vae" / "config.json",
        Path(config_dir) / "tokenizer" / "tokenizer_config.json",
        Path(config_dir) / "tokenizer" / "vocab.json",
        Path(config_dir) / "tokenizer" / "merges.txt",
        Path(config_dir) / "t5_tokenizer" / "tokenizer.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Anima reference weights missing: {', '.join(missing)}. The server "
            "image was likely built with WITH_ANIMA=0 (SDXL only); rebuild with "
            "WITH_ANIMA=1 or clone hecrj/anima into templates/anima."
        )

    size = Path(single_file).stat().st_size
    print(f"Loading Anima model: {Path(single_file).name} ({size / 2**30:.1f} GiB)")
    start = time.time()

    header = _header(single_file)
    if any(key.endswith(".comfy_quant") for key in header if key != "__metadata__"):
        raise ValueError("Quantized Anima checkpoints are not supported")

    prefix = next(p for p in _ANIMA_PREFIXES if any(key.startswith(p) for key in header if key != "__metadata__"))

    state = list(safetensors.torch.load_file(single_file).items())
    print(f"  checkpoint loaded: {time.time() - start:.1f}s")
    adapter = {
        key.removeprefix(prefix + "llm_adapter."): value
        for key, value in state
        if key.startswith(prefix + "llm_adapter.")
    }
    dit = {
        key.removeprefix(prefix): value
        for key, value in state
        if not key.startswith(prefix + "llm_adapter.")
    }

    if not adapter or not dit:
        raise ValueError(f"Unrecognized Anima checkpoint layout in {single_file}")

    # -- DiT -> CosmosTransformer3DModel
    config = _transformer_config(dit)
    with init_empty_weights():
        transformer = CosmosTransformer3DModel.from_config(config, torch_dtype=dtype)

    converted = convert_cosmos_transformer_checkpoint_to_diffusers(dict(dit))
    expected = set(transformer.state_dict().keys())
    missing = expected - set(converted)
    unexpected = set(converted) - expected
    if missing or unexpected:
        raise ValueError(
            f"Anima DiT key mismatch: missing {sorted(missing)[:5]}, "
            f"unexpected {sorted(unexpected)[:5]}"
        )
    transformer.load_state_dict(converted, assign=True)

    # -- LLM adapter -> AnimaTextConditioner
    source_dim = config["text_embed_dim"]
    conditioner = AnimaTextConditioner.from_config(
        _conditioner_config(adapter, source_dim, source_dim),
        torch_dtype=dtype,
    )
    expected = set(conditioner.state_dict().keys())
    missing = expected - set(adapter)
    unexpected = set(adapter) - expected
    if missing or unexpected:
        raise ValueError(
            f"Anima LLM adapter key mismatch: missing {sorted(missing)[:5]}, "
            f"unexpected {sorted(unexpected)[:5]}"
        )
    conditioner.load_state_dict(adapter, assign=True)
    print(f"  transformer + adapter built: {time.time() - start:.1f}s")

    # -- Reference components
    vae = AutoencoderKLQwenImage.from_pretrained(
        config_dir, subfolder="vae", torch_dtype=dtype, local_files_only=True
    )
    text_encoder = Qwen3Model.from_pretrained(
        config_dir, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(config_dir, subfolder="tokenizer", local_files_only=True)
    t5_tokenizer = T5TokenizerFast.from_pretrained(config_dir, subfolder="t5_tokenizer", local_files_only=True)
    print(f"Anima model ready: {time.time() - start:.1f}s")

    return AnimaPipeline(
        transformer=transformer,
        text_conditioner=conditioner,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        t5_tokenizer=t5_tokenizer,
        vae=vae,
    )
