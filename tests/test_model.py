import jax
from jax._src.random import normal
import jax.numpy as jnp
import haiku as hk

import sys
import os

sys.path.append(os.getcwd()) # cursed hack

from diffusion.models.clip_latent import diffusion_model


def test_model():
    shape = (3, 256, 256)
    seed = 42
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    model = hk.transform(diffusion_model)
    params = model.init(
        subkey,
        jnp.zeros([1, *shape]),
        jnp.zeros([1]),
        jnp.zeros([1, 512]),
        {},
        jnp.array(0)
    )

    keys = jax.random.split(key, num=4)

    fake_image = jax.random.normal(keys[0], [3, 3, 256, 256])
    fake_embed = jax.random.normal(keys[1], [3, 512])
    fake_time = jax.random.normal(keys[2], [3])

    def f(rng, image, time, embed):
        v_emb = model.apply(params, rng, image, time, embed, {}, jnp.array(0))
        return v_emb
        
    #output = model.apply(params, keys[3], fake_image, fake_time, fake_embed, {}, jnp.array(0))
    f(keys[3], fake_image, fake_time, fake_embed)
    #print(output)
    print('Model parameters:', hk.data_structures.tree_size(params))



test_model()
