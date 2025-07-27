"""
Functions to perform DDPM denoising steps, iterative image denoising,
 and image generation via reverse diffusion.
 """

import torch

def denoise_step_ddpm(x, model, cur_t, next_t, alphas, betas, alphas_cumprod):
    """
    Single denoising step from x_t to x_{t-1}.

    Args:
        x (Tensor): Noisy input at timestep t.
        model (nn.Module): Noise prediction model.
        cur_t (int): Current timestep.
        next_t (int): Next timestep.
        alphas, betas, alphas_cumprod (Tensor): Diffusion schedules.

    Returns:
        Tensor: x_{t-1} after denoising.
    """
    device = x.device
    times = torch.full((x.shape[0],), cur_t, device=device, dtype=torch.long)
    eps = model(x, times)

    alpha_t = alphas[cur_t].to(device)
    alpha_cum_t = alphas_cumprod[cur_t].to(device)
    beta_t = betas[cur_t].to(device)

    x_prev_mean = (1 / torch.sqrt(alpha_t)) * (
        x - (beta_t / torch.sqrt(1 - alpha_cum_t)) * eps
    )

    if next_t == -1:
        x_prev = x_prev_mean
    else:
        noise = torch.randn_like(x)
        sigma_t = torch.sqrt(beta_t)
        x_prev = x_prev_mean + sigma_t * noise

    return x_prev


def denoise_img_ddpm(x_t, t_start, model, alphas, betas, alphas_cumprod):
    """
    Iteratively denoise x_t from timestep t_start to 0.

    Args:
        x_t (Tensor): Noisy image at timestep t_start.
        t_start (int): Starting timestep.
        model (nn.Module): Noise prediction model.
        alphas, betas, alphas_cumprod (Tensor): Diffusion schedules.

    Returns:
        Tensor: Denoised image x_0.
    """
    device = x_t.device
    x = x_t
    alphas = alphas.to(device)
    betas = betas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)

    for cur_t in reversed(range(t_start + 1)):
        next_t = cur_t - 1 if cur_t > 0 else -1
        x = denoise_step_ddpm(x, model, cur_t, next_t, alphas, betas, alphas_cumprod)

    return x


def generate_image_ddpm(model, image_shape, alphas, betas, alphas_cumprod, T):
    """
    Generate a denoised image by running reverse diffusion from noise.

    Args:
        model (nn.Module): Noise prediction model.
        image_shape (tuple): Output image shape.
        alphas, betas, alphas_cumprod (Tensor): Diffusion schedules.
        T (int): Total number of timesteps.

    Returns:
        Tensor: Generated image.
    """
    device = next(model.parameters()).device
    x_t = torch.randn(image_shape, device=device)
    x_0 = denoise_img_ddpm(x_t, T - 1, model, alphas, betas, alphas_cumprod)
    return x_0
