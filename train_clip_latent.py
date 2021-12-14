#!/usr/bin/env python3

import argparse
import pickle
import time
import clip_jax
import haiku as hk
from jax.tree_util import Partial
import jax
import jax.numpy as jnp
import optax
import PIL
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import warnings

from diffusion.clip_tokenizer import tokenize
from diffusion.cloud_storage import ImageDataset
from diffusion.utils import (
    ema_update,
    ToMode,
    t_to_alpha_sigma,
    unreplicate,
    worker_init_fn,
    psplit
)
from diffusion.models.clip_latent import big_latent_model

warnings.simplefilter('ignore', PIL.Image.DecompressionBombWarning)

import wandb

bucket = 'clip-diffusion-01'


p = argparse.ArgumentParser()
p.add_argument('--batch-size', '-bs', type=int, default=64,
                   help='the batch size')
p.add_argument('--ema-decay', type=float, default=0.999,
                   help='the EMA decay')
p.add_argument('--gamma', type=float, default=1.0)
p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
p.add_argument('--train-set', type=str, required=True,
                   help='the training set')
p.add_argument('--epochs', type=int, default=10)
p.add_argument('--lr', type=float, default=0.00005)
p.add_argument('--grad-clip', type=float, default=1.0)
p.add_argument('--run-name', type=str, default='CLIP-Diffusion')


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
    train_set = ImageDataset(tf, train_set_dir)
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

def save(params, params_ema, opt_state, epoch, key):
    if jax.process_index() != 0:
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


def make_normalize(mean, std):
    mean = jnp.array(mean).reshape([3, 1, 1])
    std = jnp.array(std).reshape([3, 1, 1])

    def inner(image):
        return (image - mean) / std
    return inner


def make_clip_embed_fn(image_fn, text_fn, params, normalize):
    clip_size = 224
    # clip_patch_size = 16
    # extent = clip_patch_size // 2
    def f(batch, key):
        images = batch['image_tensor']
        texts = batch['text']
        clip_in = jax.image.resize(
            jnp.array(images),
            (*images.shape[:2], clip_size, clip_size),
            'cubic'
        )
        image_embeds = image_fn(params, normalize((clip_in + 1) / 2))
        text_embeds = text_fn(params, tokenize(texts))
        dice_roll = jnp.repeat(
            jax.random.uniform(key, [text_embeds.shape[0],]),
            text_embeds.shape[1]
        ).reshape(text_embeds.shape)
        return jax.lax.select(dice_roll > 0.5, image_embeds, text_embeds)
    return f


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
        embed_targets = embed_noise * alphas[:, None] - embeds * sigmas[:, None]
        v_emb = model.apply(params, key, noised_images, t, noised_embeds, extra_args, is_training)
        emb_loss = jnp.mean(jnp.square(v_emb - embed_targets))
        return emb_loss 
        

    def train_step(params, opt_state, key, inputs, embeddings, extra_args, axis_name='i'):
        loss_grads = jax.value_and_grad(compute_loss)(params, key, inputs, embeddings, extra_args, jnp.array(1))
        loss, grads = jax.lax.pmean(loss_grads, axis_name)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state
    
    return train_step


def main():
    args = p.parse_args()

    wandb.init(
        project="clip-diffusion",
        entity="dstander",
        config=args,
        name=args.run_name
    )

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
    image_fn, text_fn, clip_params, _ = clip_jax.load('ViT-B/16')
    clip_embed = make_clip_embed_fn(image_fn, text_fn, clip_params, normalize)
    p_ema_update = jax.pmap(ema_update, in_axes=(0, 0, None))
    model = hk.transform(big_latent_model)
    opt = optax.chain(
        optax.adam(args.lr),    
        optax.clip(args.grad_clip)
    )
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
    jit_start = time.time()
    train_step = jax.pmap(
        make_forward_fn(model, opt, args.gamma),
        axis_name='i'
    )
    jit_time = time.time() - jit_start
    print(f'It took {jit_time}s to compile the train_step function.')
    def train_one_epoch(params, params_ema, opt_state, key):
        for i, batch in enumerate(tqdm(train_dl)):
            key, subkey = jax.random.split(key)
            print('Getting CLIP embeddings')
            batch_embeds = clip_embed(batch, subkey)
            images = jax.tree_map(lambda x: psplit(jnp.array(x), num_local_devices), batch['image_tensor'])
            embeds = jax.tree_map(lambda x: psplit(x, num_local_devices), batch_embeds)
            key, subkey = jax.random.split(key)
            keys = jnp.stack(jax.random.split(subkey, num_local_devices))
            print('Doing forward and backward passes')
            loss, params, opt_state = train_step(
                params,
                opt_state,
                keys,
                images,
                embeds,
                {}
            )
            params_ema = p_ema_update(params, params_ema, get_ema_decay(epoch))
            batch_log = {'embedding_loss': unreplicate(loss)}
            wandb.log(batch_log)
            del batch_log
            if i % 50 == 0:
                tqdm.write(f'Epoch {epoch}, iteration {i}, loss {unreplicate(loss):g}')
        return params, params_ema, opt_state

    try:
        key, subkey = jax.random.split(key)
        while epoch < args.epochs:
            tqdm.write(f'Epoch {epoch}')
            key, subkey = jax.random.split(key)
            train_sampler.set_epoch(epoch)
            params, params_ema, opt_state = train_one_epoch(params, params_ema, opt_state, subkey)
            epoch += 1
            tqdm.write('')
            if epoch % 5 == 0:
                save(params, params_ema, opt_state, epoch, subkey)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
