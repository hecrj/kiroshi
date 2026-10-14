import PIL.Image
import torch


def notify(recipe, on_progress):
    height = ((recipe.size.height + 15) // 16) * 16
    width = ((recipe.size.width + 15) // 16) * 16

    def on_step_end(pipeline, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]

        if on_progress is not None:
            on_progress(step / recipe.steps, to_rgb(latents, pipeline, height, width))

        return callback_kwargs

    return on_step_end


def to_rgb(latents, pipeline=None, height=None, width=None) -> PIL.Image.Image:
    from .anima import AnimaPipeline

    if isinstance(pipeline, AnimaPipeline):
        # Anima latents are unpacked 5D flow latents [B, 16, 1, H/8, W/8];
        # decode through the VAE (same convention as the final output).
        return pipeline._decode_latents(latents)

    if pipeline is not None and hasattr(pipeline, "_unpack_latents"):
        # Flow-matching (Krea 2) pipelines hold packed 16-channel latents;
        # the SDXL-style channel hack below would be nonsense, so decode
        # through the VAE instead.
        latents = pipeline._unpack_latents(latents, height, width).to(
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
