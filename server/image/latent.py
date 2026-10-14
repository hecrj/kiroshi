import PIL.Image
import torch
import torch.nn.functional as F

# A preview frame is a VAE decode at generation resolution — for the
# flow models (Anima / Krea 2) as expensive as the final image — so
# previews are throttled (one every Nth step) and downsampled (decoded
# at 1/div of the latent spatial size; decode cost scales with pixel
# count). Retune here if that ever feels too coarse or too fine.
PREVIEW_EVERY = 1
PREVIEW_DIV = 4


def notify(recipe, on_progress, steps=None):
    # ``steps`` is the number of steps the sampler actually runs; for
    # strength-sliced trajectories (inpainting) that is a tail of the
    # recipe's schedule, not the full one.
    if steps is None:
        steps = recipe.steps

    height = ((recipe.size.height + 15) // 16) * 16
    width = ((recipe.size.width + 15) // 16) * 16

    def on_step_end(pipeline, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]

        # Only offer a preview when the step warrants one: not the first
        # step (the latents are still pure noise), not every step
        # (PREVIEW_EVERY), and not the final step — the
        # full-resolution decode of the final image follows
        # immediately. The decode is handed over as a lambda and only
        # runs if on_progress passes its own `preview_after` check, so
        # a suppressed preview costs nothing.
        if (
            on_progress is not None
            and 0 < step < steps - 1
            and step % PREVIEW_EVERY == 0
        ):
            on_progress(
                step / steps,
                lambda: to_rgb(latents, pipeline, height, width, PREVIEW_DIV),
            )

        return callback_kwargs

    return on_step_end


def to_rgb(latents, pipeline=None, height=None, width=None, div=1) -> PIL.Image.Image:
    from .anima import AnimaPipeline

    if isinstance(pipeline, AnimaPipeline):
        # Anima latents are unpacked 5D flow latents [B, 16, 1, H/8, W/8];
        # decode through the VAE (same convention as the final output).
        return pipeline._decode_latents(_downsample(latents, div))

    if pipeline is not None and hasattr(pipeline, "_unpack_latents"):
        # Flow-matching (Krea 2) pipelines hold packed 16-channel latents;
        # the SDXL-style channel hack below would be nonsense, so decode
        # through the VAE instead.
        latents = _downsample(pipeline._unpack_latents(latents, height, width), div).to(
            pipeline.vae.dtype
        )

        latents_mean = (
            torch.tensor(pipeline.vae.config.latents_mean)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            (1.0 / torch.tensor(pipeline.vae.config.latents_std))
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean

        image = pipeline.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return pipeline.image_processor.postprocess(image, output_type="pil")[0]

    weights = ((60, -60, 25, -70), (60, -5, 15, -50), (60, 10, -5, -35))

    weights_tensor = torch.t(
        torch.tensor(weights, dtype=latents.dtype).to(latents.device)
    )

    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(
        latents.device
    )

    rgb_tensor = torch.einsum(
        "...lxy,lr -> ...rxy", latents, weights_tensor
    ) + biases_tensor.unsqueeze(-1).unsqueeze(-1)

    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)

    return PIL.Image.fromarray(image_array)


def _downsample(latents: torch.Tensor, div: int) -> torch.Tensor:
    """Spatially shrink a 5D flow latent [B, C, 1, H, W] by ``div`` so a
    preview VAE decode runs on 1/div**2 of the pixels (decode cost scales
    with pixel count). The mean/std normalization commutes with the
    average pooling, so this is equivalent to downsampling after it."""
    if div <= 1:
        return latents

    h, w = latents.shape[-2:]
    size = (max(1, h // div), max(1, w // div))
    return F.interpolate(latents[:, :, 0], size=size, mode="area").unsqueeze(2)
