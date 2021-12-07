#!/usr/bin/env python3

import argparse
import pickle
import time
import clip_jax
from einops import rearrange, repeat
import haiku as hk
from jax.tree_util import Partial
import jax
import jax.numpy as jnp
import optax
import PIL
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm, trange
import warnings
import numpy as np

from diffusion.cloud_storage import BucketDataset
from diffusion.utils import (
    ema_update,
    get_ddpm_schedule,
    psplit, punsplit,
    log_snr_to_alpha_sigma,
    ToMode,
    unreplicate,
    to_pil_image,
    worker_init_fn
)
from diffusion.models.clip_latent import diffusion_model

warnings.simplefilter('error', PIL.Image.DecompressionBombWarning)

from jax.experimental import host_callback

bucket = 'clip-diffusion-01'

def ema_decay_schedule(decay, epoch):
    if epoch < 20:
        return 0.99
    return decay

def resume_training(checkpoint_file):
    with open(checkpoint_file, mode='rb') as param_file: 
        ckpt = pickle.load(param_file, 'rb')
    epoch = ckpt['epoch']
    params = jax.tree_map(jnp.array, ckpt['params'])
    params_ema = jax.tree_map(jnp.array, ckpt['params_ema'])
    opt_state = jax.tree_map(jnp.array, ckpt['opt_state'])
    key = jax.tree_map(jnp.array, ckpt['key'])
    del ckpt
    log_file = open('losses.csv', 'a+')
    return epoch, params, params_ema, opt_state, key, log_file


def get_dataloader(image_size, batch_size, num_processes, local_rank, seed, train_set_dir):
    tf = transforms.Compose([
        ToMode('RGB'),
        transforms.Resize(image_size, interpolation=PIL.Image.LANCZOS),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_set = BucketDataset(bucket, train_set_dir, transform_fn=tf)
    train_sampler = data.DistributedSampler(
        train_set,
        num_processes,
        local_rank,
        seed=seed,
        drop_last=True
    )
    train_dl = data.DataLoader(
        train_set,
        batch_size,
        sampler=train_sampler,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        num_workers=48,
        persistent_workers=True
    )
    return train_dl, train_sampler


def make_normalize(mean, std):
    mean = jnp.array(mean).reshape([3, 1, 1])
    std = jnp.array(std).reshape([3, 1, 1])

    def inner(image):
        return (image - mean) / std
    return inner


def make_clip_embed_fn(image_fn, params, normalize):
    clip_size = 224
    clip_patch_size = 16
    # extent = clip_patch_size // 2
    def f(batch):
        clip_in = jax.image.resize(batch, (*batch.shape[:2], clip_size, clip_size), 'cubic')
        #print(clip_in.shape)
        #clip_in = jnp.pad(clip_in, [(0, 0), (0, 0), (extent, extent), (extent, extent)], 'edge')
        #print(clip_in.shape)
        return image_fn(params, normalize((clip_in + 1) / 2))
    return f


def make_forward_fn(model, opt, gamma):
    def compute_loss(params, key, images, embeds, extra_args, is_training):
        key, subkey = jax.random.split(key)
        t = jax.random.uniform(subkey, images.shape[:1])
        log_snrs = get_ddpm_schedule(get_ddpm_schedule(t))
        alphas, sigmas = log_snr_to_alpha_sigma(log_snrs)
        key, subkey = jax.random.split(key)
        image_noise = jax.random.normal(subkey, images.shape)
        embed_noise = jax.random.normal(subkey, embeds.shape)
        noised_images = images * alphas[:, None, None, None] + image_noise * sigmas[:, None, None, None]
        noised_embeds = embeds * alphas[:, None] + embed_noise * sigmas[:, None]
        image_targets = image_noise * alphas[:, None, None, None] - images * sigmas[:, None, None, None]
        embed_targets = embed_noise * alphas[:, None] - embeds * sigmas[:, None]
        v_im, v_emb = model.apply(params, key, noised_images, log_snrs, noised_embeds, extra_args, is_training)
        im_loss = jnp.mean(jnp.square(v_im - image_targets)) * 0.280219 
        emb_loss = jnp.mean(jnp.square(v_emb - embed_targets))
        host_callback.id_print(jax.process_index(), image=im_loss, embed=emb_loss)
        return im_loss + gamma * emb_loss

    def train_step(params, opt_state, key, inputs, embeddings, extra_args, axis_name='i'):
        loss_grads = jax.value_and_grad(compute_loss)(params, key, inputs, embeddings, extra_args, jnp.array(1))
        loss, grads = jax.lax.pmean(loss_grads, axis_name)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state
    
    return train_step




def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', '-bs', type=int, default=64,
                   help='the batch size')
    p.add_argument('--ema-decay', type=float, default=0.999,
                   help='the EMA decay')
    p.add_argument('--gamma', type=float, default=2)
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    p.add_argument('--train-set', type=str, required=True,
                   help='the training set')
    args = p.parse_args()

    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    num_processes = jax.process_count()
    local_rank = jax.process_index()

    size = 256
    shape = (3, size, size)

    train_dl, train_sampler = get_dataloader(
        size,
        args.batch_size,
        num_processes,
        local_rank,
        args.seed,
        args.train_set
    )
    normalize = make_normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    image_fn, _, clip_params, _ = clip_jax.load('ViT-B/16')


    clip_embed = make_clip_embed_fn(image_fn, clip_params, normalize)

    p_ema_update = jax.pmap(ema_update, in_axes=(0, 0, None))


    model = hk.transform(diffusion_model)

    opt = optax.noisy_sgd(5e-5)
    key = jax.random.PRNGKey(args.seed)
    
    if not args.resume:
        key, subkey = jax.random.split(key)
        epoch = 0
        params = model.init(
            subkey,
            jnp.zeros([1, *shape]),
            jnp.zeros([1]),
            jnp.zeros([1, 512]),
            {},
            jnp.array(0.0)
        )
        params = jax.tree_map(lambda x: x / 2, params)
        params_ema = params
        opt_state = opt.init(params)
        log_file = open('losses.csv', 'w')
        print('epoch', 'i', 'time', 'loss', sep=',', file=log_file, flush=True)
    else:
        epoch, params, params_ema, opt_state, key, log_file = resume_training(args.resume)

    get_ema_decay = Partial(ema_decay_schedule, args.ema_decay)

        
    print('Model parameters:', hk.data_structures.tree_size(params))

    params = jax.device_put_replicated(params, jax.local_devices())
    params_ema = jax.device_put_replicated(params_ema, jax.local_devices())
    opt_state = jax.device_put_replicated(opt_state, jax.local_devices())

    key = jax.random.split(key, num_processes)[local_rank]

    train_step = make_forward_fn(model, opt, args.gamma)


    def train_one_epoch(params, params_ema, opt_state, key):
        pmap_train_step = jax.pmap(train_step, axis_name='i')
        for i, batch in enumerate(tqdm(train_dl)):
            batch_embeds = clip_embed(jnp.array(batch))
            images = jax.tree_map(lambda x: psplit(jnp.array(x), num_local_devices), batch)
            embeds = jax.tree_map(lambda x: psplit(x, num_local_devices), batch_embeds)
            key, subkey = jax.random.split(key)
            keys = jnp.stack(jax.random.split(subkey, num_local_devices))
            loss, params, opt_state = pmap_train_step(params, opt_state, keys, images, embeds, {})
            params_ema = p_ema_update(params, params_ema, get_ema_decay(epoch))
            print(epoch, i, time.time(), unreplicate(loss), sep=',', file=log_file, flush=True)
            if i % 50 == 0:
                tqdm.write(f'Epoch {epoch}, iteration {i}, loss {unreplicate(loss):g}')
        return params, params_ema, opt_state

    def sample_step(params, key, x_t, log_snr, log_snr_next, eta, extra_args):
        dummy_key = jax.random.PRNGKey(0)
        v = model.apply(
            params,
            dummy_key,
            x_t,
            repeat(log_snr, '-> n', n=x_t.shape[0]),
            extra_args,
            jnp.array(0)
        )
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        pred = x_t * alpha - v * sigma
        eps = x_t * sigma + v * alpha
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)
        ddim_sigma = eta * jnp.sqrt(sigma_next**2 / sigma**2) * jnp.sqrt(1 - alpha**2 / alpha_next**2)
        adjusted_sigma = jnp.sqrt(sigma_next**2 - ddim_sigma**2)
        x_t = pred * alpha_next + eps * adjusted_sigma
        x_t = x_t + jax.random.normal(key, x_t.shape) * ddim_sigma
        return x_t, pred

    def sample(params, key, x_t, steps, eta, extra_args):
        t = jnp.linspace(1, 0, steps + 1)[:-1]
        log_snrs = get_ddpm_schedule(t)
        pmap_sample_step = jax.pmap(sample_step, in_axes=(0, 0, 0, None, None, None, 0))
        for i in trange(steps):
            key, subkey = jax.random.split(key)
            keys = jnp.stack(jax.random.split(subkey, num_local_devices))
            if i < steps - 1:
                x_t, _ = pmap_sample_step(
                    params,
                    keys,
                    x_t,
                    log_snrs[i],
                    log_snrs[i + 1],
                    eta,
                    extra_args
                )
            else:
                _, pred = pmap_sample_step(
                    params,
                    keys,
                    x_t,
                    log_snrs[i],
                    log_snrs[i],
                    eta,
                    extra_args
                )
        return pred

    def save():
        if local_rank != 0:
            return
        obj = {
            'params': unreplicate(params),
            'params_ema': unreplicate(params_ema),
            'opt_state': unreplicate(opt_state),
            'epoch': epoch,
            'key': key
        }
        with open('model.pkl', 'wb') as f:
            pickle.dump(obj, f)

    def demo(key):
        if local_rank != 0:
            return
        tqdm.write('Sampling...')
        outs = []
        for _ in trange(1):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, [8, 8, *shape])
            key, subkey = jax.random.split(key)
            out = punsplit(sample(params_ema, subkey, noise, 1000, 1, {}))
            outs.append(out)
        out = jnp.concatenate(outs, axis=0)
        grid = rearrange(out, '(s1 s2) c h w -> c (s1 h) (s2 w)', s1=8)
        to_pil_image(grid).save(f'demo_{epoch:06}.png')

    try:
        key, subkey = jax.random.split(key)
        if epoch > 0:
            demo(subkey)
        while True:
            tqdm.write(f'Epoch {epoch}')
            key, subkey = jax.random.split(key)
            train_sampler.set_epoch(epoch)
            params, params_ema, opt_state = train_one_epoch(params, params_ema, opt_state, subkey)
            epoch += 1
            tqdm.write('')
            if epoch % 5 == 0:
                key, subkey = jax.random.split(key)
                demo(subkey)
            if epoch % 5 == 0:
                save()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()