import PIL.Image
import torch


def notify(recipe, on_progress):
    def on_step_end(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]

        if on_progress is not None:
            on_progress(step / recipe.steps, to_rgb(latents))

        return callback_kwargs

    return on_step_end


def to_rgb(latents) -> PIL.Image.Image:
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
