import jax
import jax.numpy as jnp
import haiku as hk
from einops import repeat

import sys
import os

sys.path.append(os.getcwd()) # cursed hack

from diffusion.models.clip_latent import diffusion_model
from diffusion.utils import t_to_alpha_sigma, log_snr_to_alpha_sigma, get_sampling_schedule


def make_forward_fn(model, opt, gamma):

    def compute_loss(params, key, images, embeds, extra_args, is_training):
        key, subkey = jax.random.split(key)
        t = jax.random.uniform(subkey, images.shape[:1])
        alphas, sigmas = t_to_alpha_sigma(t)
        key, subkey = jax.random.split(key)
        image_noise = jax.random.normal(subkey, images.shape)
        embed_noise = jax.random.normal(subkey, embeds.shape)
        noised_images = images * alphas[:, None, None, None] + image_noise * sigmas[:, None, None, None]
        noised_embeds = embeds * alphas[:, None] + embed_noise * sigmas[:, None]
        image_targets = image_noise * alphas[:, None, None, None] - images * sigmas[:, None, None, None]
        embed_targets = embed_noise * alphas[:, None] - embeds * sigmas[:, None]
        v_im, v_emb = model.apply(params, key, noised_images, t, noised_embeds, extra_args, is_training)
        im_loss = jnp.mean(jnp.square(v_im - image_targets))
        emb_loss = jnp.mean(jnp.square(v_emb - embed_targets))
        #im_loss, emb_loss = host_callback.id_print((im_loss, emb_loss), what='Image, Embed')
        return 0.5 * (im_loss + gamma * emb_loss), (im_loss, emb_loss)
        

    def train_step(params, key, inputs, embeddings, extra_args):
        loss_grads = jax.value_and_grad(compute_loss, has_aux=True)(params, key, inputs, embeddings, extra_args, jnp.array(1))
        loss_aux, grads = jax.tree_util.tree_map(jnp.mean, loss_grads)
        loss, aux_data = loss_aux
        #updates, opt_state = opt.update(grads, opt_state, params)
        #params = optax.apply_updates(params, updates)
        return loss, params, aux_data
    
    return train_step


def make_sample_fn(model):
    def sample_step(params, key, x_t, y_t, log_snr, log_snr_next, eta, extra_args):
        keys = jax.random.split(key, num=3)
        v_im, v_lat = model.apply(
            params,
            keys[0],
            x_t,
            repeat(log_snr, '-> n', n=x_t.shape[0]),
            y_t,
            extra_args,
            jnp.array(0.0)
        )
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)
        pred_im = x_t * alpha - v_im * sigma
        eps_im = x_t * sigma + v_im * alpha
        pred_lat = y_t * alpha - v_lat * sigma
        eps_lat = y_t * sigma - v_lat * alpha
        ddim_sigma = eta * jnp.sqrt(sigma_next**2 / sigma**2) * jnp.sqrt(1 - alpha**2 / alpha_next**2)
        adjusted_sigma = jnp.sqrt(sigma_next**2 - ddim_sigma**2)
        x_t = pred_im * alpha_next + eps_im * adjusted_sigma + jax.random.normal(keys[1], x_t.shape) * ddim_sigma
        y_t = pred_lat * alpha_next + eps_lat * adjusted_sigma + jax.random.normal(keys[2], y_t.shape) * ddim_sigma
        return x_t, y_t, pred_im

    def sample(params, key, x_t, y_t, steps, eta, extra_args):
        log_snrs = get_sampling_schedule(steps)
        sample_fn = jax.jit(sample_step)
        for i in range(steps-1):
            key, subkey = jax.random.split(key)
            x_t, y_t, pred = sample_fn(
                params,
                subkey,
                x_t,
                y_t,
                log_snrs[i],
                log_snrs[i + 1],
                eta,
                extra_args
            )
        return pred

    return sample


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
    forward = make_forward_fn(model, None, 1.9)

    #output = model.apply(params, keys[3], fake_image, fake_time, fake_embed, {}, jnp.array(0))
    forward(params, keys[3], fake_image, fake_embed, {})
    #print(output)
    print('Model parameters:', hk.data_structures.tree_size(params))


def test_sample():
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
    keys = jax.random.split(key, num=3)
    do_sample = make_sample_fn(model)
    fake_image = jax.random.normal(keys[0], [3, 3, 256, 256])
    fake_embed = jax.random.normal(keys[1], [3, 512])
    do_sample(params, keys[2], fake_image, fake_embed, 100, 2, {})

# test_model()
test_sample()