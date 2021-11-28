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
    keys = jax.random.split(key, num=5)
    clip_loss, clip_grad = clip_loss_fn(x_embed, target)
    control, timestep = policy.apply(keys[1], policy_params, x, clip_grad, t, x_embed, target, {})
    v = diff_model.apply(diff_params, key[2], x, repeat(t, '-> n', n=x.shape[0]), extra_args)
    alpha, sigma = utils.t_to_alpha_sigma(t)
    v = v - control * (sigma / alpha)
    pred = x * alpha - v * sigma
    eps = x * sigma + v * alpha
    t_next = t - jnp.abs((timestep + jax.random.normal(keys[3], [1]) * 0.1))
    t_next = jnp.maximum(0, t_next)
    alpha_next, sigma_next = utils.t_to_alpha_sigma(t_next)
    adjusted_sigma = jnp.sqrt(sigma_next**2 - sigma**2)
    x = pred * alpha_next + eps * adjusted_sigma
    x = x + jax.random.normal(keys[4], x.shape) * adjusted_sigma
    control_loss = jnp.sum(jnp.square(control))/jnp.size(control)
    return x, pred, t_next, clip_loss, control_loss
