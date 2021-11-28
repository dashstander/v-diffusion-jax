#!/usr/bin/env python3

"""CLIP guided sampling from a diffusion model."""

import argparse
from einops import rearrange, repeat
import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import optax
from pathlib import Path
import pickle
import time
from torch.utils import data
from datasets import load_dataset
from tqdm import tqdm, trange

from diffusion import get_model, load_params, utils
from diffusion.dynamics_env import rl_sample_step
from diffusion.utils import to_pil_image

MODULE_DIR = Path(__file__).resolve().parent
import clip_jax


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-bs', type=int, default=64,
                   help='the batch size')
parser.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
parser.add_argument('--seed', type=int, default=0,
                   help='the random seed')
parser.add_argument('--train_set', type=str, required=True,
                   help='the training set')
parser.add_argument('--eta', default=1.0,
                   help='the amount of noise to add during sampling (0-1)')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)


def get_dataset(train_set, batch_size, num_workers, seed):
    train_set = load_dataset(train_set, split='train')
    train_sampler = data.DistributedSampler(
        train_set,
        seed=seed,
        drop_last=True
    )
    train_dl = data.DataLoader(
        train_set,
        batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    return train_dl, train_sampler



def clip_loss_fn(image_fn, clip_params, patch_size, size, normalize_fn, pred, target):
    clip_in = jax.image.resize(pred, (*pred.shape[:2], size, size), 'cubic')
    extent = patch_size // 2
    clip_in = jnp.pad(clip_in, [(0, 0), (0, 0), (extent, extent), (extent, extent)], 'edge')
    image_embed = image_fn(clip_params, normalize_fn((clip_in + 1) / 2))
    return jnp.sum(utils.spherical_dist_loss(image_embed, target)), image_embed


def main():
    args, _ = parser.parse_known_args()
    shape = (3, args.size, args.size)

    train_dl, train_sampler = get_dataset(args.train_set, args.batch_size, args.num_workers, args.seed)

    diffusion_model = get_model('wikiart_128')
    checkpoint = MODULE_DIR / f'checkpoints/{args.model}.pkl'
    diffusion_params = load_params(checkpoint)

    clip_patch_size = 16 # Constant?
    clip_size = 224 # Constant?
    
    policy_model = get_model('policy')

    opt = optax.adam(5e-5)

    if not args.resume:
        epoch = 0
        params = policy_model.init(
            jax.random.PRNGKey(args.seed),
            jnp.zeros([1, *shape]), # Represents the image, 1x3x128x128
            jnp.zeros([1, *shape]), # Grad w/r/t to the image 1x3x128x128
            jnp.zeros([1]), # Time 
            jnp.zeros([1, 512]), # CLIP Embedding of the current image
            jnp.zeros([1, 512]), # CLIP Embedding of the target
            {}, # Extra args
        )
        params = jax.tree_map(lambda x: x / 2, params)
        opt_state = opt.init(params)
        #log_file = open('losses.csv', 'w')
        #print('epoch', 'i', 'time', 'loss', sep=',', file=log_file, flush=True)
    else:
        ckpt = pickle.load(open(args.resume, 'rb'))
        epoch = ckpt['epoch']
        params = jax.tree_map(jnp.array, ckpt['params'])
        opt_state = jax.tree_map(jnp.array, ckpt['opt_state'])
        del ckpt
        #log_file = open('losses.csv', 'a+')

    print('Model parameters:', hk.data_structures.tree_size(params))

    # params = jax.device_put_replicated(params, jax.local_devices())
    # opt_state = jax.device_put_replicated(opt_state, jax.local_devices())

    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)

    normalize = utils.make_normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

    image_fn, text_fn, clip_params, _ = clip_jax.load('ViT-B/16')

    clip_loss_grad = jax.value_and_grad(
        Partial(
            clip_loss_fn,
            image_fn,
            clip_params,
            clip_patch_size,
            clip_size,
            normalize
        )
    )

    sample_step = jax.jit(Partial(
        rl_sample_step,
        diffusion_model,
        diffusion_params,
        clip_loss_grad,
        extra_args={}
    ))
    
    def control_episode(params, key, target, min_time, max_steps):
        keys = jax.random.split(key, num=3)
        time = jax.random.uniform(keys[0], [1], minval=min_time, maxval=1.0)
        x = jax.random.normal(subkey, [1, *diffusion_model.shape])
        total_loss = jnp.zeros([1])
        keys = jax.random.split(key, num=max_steps)
        for i in jnp.arange(max_steps):
            if i < max_steps - 1 and time > 0.0:
                x, _, loss, time_step = sample_step(policy_model, params, keys[i], x, time, target)
                time -= jnp.abs(time_step)
            else:
                time = jnp.maximum(0.0, time)
                _, pred, loss, _ = sample_step(policy_model, params, keys[i], x, time, target)
            total_loss += loss
        return total_loss, pred


    def train_step(params, opt_state, key, target, min_time, max_steps):
        key, subkey = jax.random.split(key)
        loss, grads, pred = jax.value_and_grad(control_episode, has_aux=True)(params, subkey, target, min_time, max_steps)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, pred


    def train_one_epoch(params, opt_state, data_loader, key):
        keys = jax.random.split(key, num=len(data_loader))
        for i, record in enumerate(tqdm(data_loader)):
            target = text_fn(clip_params, clip_jax.tokenize([record['text']]))
            loss, params, opt_state = train_step(params, opt_state, keys[i], target, {})
            if i % 50 == 0:
                tqdm.write(f'Epoch {epoch}, iteration {i}, loss {loss:g}')
        return params, opt_state

    def save():
        obj = {
            'params': params,
            'opt_state': opt_state,
            'epoch': epoch
        }
        with open(f'model_{epoch}.pkl', 'wb') as f:
            pickle.dump(obj, f)

    #def demo(key):
    #    tqdm.write('Sampling...')
    #    outs = []
    #    for _ in trange(1):
    #        key, subkey = jax.random.split(key)
    #        noise = jax.random.normal(subkey, [8, 8, *shape])
    #        key, subkey = jax.random.split(key)
    #        out = sample(subkey, noise, 1000, 1, {})
    #        outs.append(out)
    #    out = jnp.concatenate(outs, axis=0)
    #    grid = rearrange(out, '(s1 s2) c h w -> c (s1 h) (s2 w)', s1=8)
    #    to_pil_image(grid).save(f'demo_{epoch:06}.png')

    try:
        key, subkey = jax.random.split(key)
        #if epoch > 0: 
        #    demo(subkey)
        while True:
            tqdm.write(f'Epoch {epoch}')
            key, subkey = jax.random.split(key)
            train_sampler.set_epoch(epoch)
            params, opt_state = train_one_epoch(params, opt_state, train_dl, subkey)
            epoch += 1
            tqdm.write('')
            if epoch % 5 == 0:
                key, subkey = jax.random.split(key)
                #demo(subkey)
            if epoch % 5 == 0:
                save()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
