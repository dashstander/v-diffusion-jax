from pathlib import Path

from einops import repeat
import jax
import jax.numpy as jnp

from diffusion import utils

MODULE_DIR = Path(__file__).resolve().parent


def rl_sample_step(
    diff_model,
    diff_params,
    clip_loss_fn,
    image_fn,
    clip_params,
    clip_patch_size,
    clip_size,
    normalize_fn,
    policy,
    policy_params,
    key,
    x,
    t,
    target,
    extra_args
):
    extent = clip_patch_size // 2
    clip_in = jnp.pad(x, [(0, 0), (0, 0), (extent, extent), (extent, extent)], 'edge')
    clip_in = jax.image.resize(clip_in, (*x.shape[:2], clip_size, clip_size), 'cubic')
    x_embed = image_fn(clip_params, normalize_fn((clip_in + 1) / 2))
    del clip_in
    keys = jax.random.split(key, num=5)
    clip_loss, clip_grad = clip_loss_fn(x, target, image_fn, clip_params) # Doing two forward passes on CLIP :/
    control, timestep = policy.apply(policy_params, keys[1], x, clip_grad, t, x_embed, target, {})
    #print(f'control: {control.shape}')
    v = diff_model.apply(diff_params, keys[2], x, t, extra_args)
    #print(f'v: {v.shape}')
    alpha, sigma = utils.t_to_alpha_sigma(t)
    v = v - control * (sigma / alpha)
    #pred = x * alpha - v * sigma
    #eps = x * sigma + v * alpha
    t_next = t - jnp.abs((timestep + jax.random.normal(keys[3], ()) * 0.1))
    t_next = jnp.maximum(0, t_next)
    alpha_next, sigma_next = utils.t_to_alpha_sigma(t_next)
    adjusted_sigma = jnp.sqrt(sigma_next**2 - sigma**2)
    x = (x * alpha - v * sigma) * alpha_next + (x * sigma + v * alpha) * adjusted_sigma
    x = x + jax.random.normal(keys[4], x.shape) * adjusted_sigma
    control_loss = jnp.sum(jnp.square(control))/jnp.size(control)
    return x, t_next, clip_loss, control_loss
