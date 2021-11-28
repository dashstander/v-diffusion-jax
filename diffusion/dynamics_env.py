from collections import namedtuple
from pathlib import Path
import sys

import haiku as hk
from einops import repeat
import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from diffusion import utils

MODULE_DIR = Path(__file__).resolve().parent
sys.path.append(str(MODULE_DIR / 'CLIP_JAX'))

import clip_jax


Observation = namedtuple('Observation', ['image', 'v', 'embedding', 'clip_grad', 'time', 'reward'])


def rl_sample_step(
    diff_model,
    diff_params,
    policy,
    policy_params,
    key,
    x,
    t,
    target,
    extra_args,
    clip_loss_fn
):
    keys = jax.random.split(key, num=4)
    value, clip_grad = clip_loss_fn(keys[0], x, target)
    clip_loss, x_embed = value
    control, t_next = policy.apply(keys[1], policy_params, x, clip_grad, t, x_embed, target, {})
    v = diff_model.apply(diff_params, key[2], x, repeat(t, '-> n', n=x.shape[0]), extra_args)
    alpha, sigma = utils.t_to_alpha_sigma(t)
    v = v - control * (sigma / alpha)
    pred = x * alpha - v * sigma
    eps = x * sigma + v * alpha
    alpha_next, sigma_next = utils.t_to_alpha_sigma(t_next)
    # ddim_sigma = eta * jnp.sqrt(sigma_next**2 / sigma**2) *  jnp.sqrt(1 - alpha**2 / alpha_next**2)
    adjusted_sigma = jnp.sqrt(sigma_next**2 - sigma**2)
    x = pred * alpha_next + eps * adjusted_sigma
    x = x + jax.random.normal(keys[3], x.shape) * adjusted_sigma
    control_loss = jnp.sum(jnp.square(control))/jnp.size(control)
    return x, pred, clip_loss + control_loss


class DynamicsEnv(hk.Module):

    def __init__(self, diffusion_model, diffusion_params, image_fn, text_fn, clip_params, patch_size, clip_size, eta):
        self.image_fn = Partial(image_fn, clip_params)
        self.patch_size = patch_size
        self.clip_size = clip_size
        self.eta = eta
        self.sample_step = jax.jit(Partial(
            rl_sample_step,
            diffusion_model,
            diffusion_params,
            eta=self.eta,
            extra_args={}
        ))

    def __call__(self, key, image, prev_v, timestep, prev_timestep, nudge):
        normalize = utils.make_normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        clip_loss_grad = jax.value_and_grad(clip_loss)
        key, subkey = jax.random.split(key)
        pred, v = self.sample_step(subkey, image, prev_v, timestep, prev_timestep, nudge)
        value, clip_grad = clip_loss_grad(pred)
        dist_loss, pred_embed = value
        nudge_loss = jnp.sum(jnp.square(nudge)) / jnp.size(nudge)
        return Observation(
            pred,
            v,
            pred_embed,
            clip_grad,
            timestep,
            dist_loss + nudge_loss
        )
