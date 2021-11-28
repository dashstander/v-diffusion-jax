#!/usr/bin/env python3

import argparse
import pickle
import time

from einops import rearrange, repeat
import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
import optax
from PIL import Image, ImageFile
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm, trange


def to_pil_image(x):
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = x[0]
    if x.shape[0] == 1:
        x = x[0]
    else:
        x = x.transpose((1, 2, 0))
    arr = np.array(jnp.round(jnp.clip((x + 1) * 127.5, 0, 255)).astype(jnp.uint8))
    return Image.fromarray(arr)


def ema_update(params, averaged_params, decay):
    return jax.tree_map(lambda p, a: p * (1 - decay) + a * decay, params, averaged_params)


p_ema_update = jax.pmap(ema_update, in_axes=(0, 0, None))


def unreplicate(x):
    return jax.tree_map(lambda x: x[0], x)


def psplit(x, n):
    return jax.tree_map(lambda x: jnp.stack(jnp.split(x, n)), x)


def punsplit(x):
    return jax.tree_map(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])), x)


def get_alpha_sigma(log_snrs):
    """Returns the scaling factors for the clean image and for the noise, given
    the log SNR for a timestep."""
    alphas_squared = jax.nn.sigmoid(log_snrs)
    return jnp.sqrt(alphas_squared), jnp.sqrt(1 - alphas_squared)


def get_ddpm_schedule(t):
    """Returns log SNRs for the noise schedule from the DDPM paper."""
    return -jnp.log(jnp.expm1(1e-4 + 10 * t**2))


def get_ddpm_inverse_cdf(t):
    z = jax.scipy.special.erfinv(jax.scipy.special.erf(jnp.sqrt(10)))
    return jax.scipy.special.erfinv(t * jax.scipy.special.erf(jnp.sqrt(10))) / z


class FourierFeatures(hk.Module):
    def __init__(self, output_size, std=1., name=None):
        super().__init__(name=name)
        assert output_size % 2 == 0
        self.output_size = output_size
        self.std = std

    def __call__(self, x):
        w = hk.get_parameter('w', [self.output_size // 2, x.shape[1]],
                             init=hk.initializers.RandomNormal(self.std, 0))
        f = 2 * jnp.pi * x @ w.T
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class Dropout2d(hk.Module):
    def __init__(self, rate=0.5, name=None):
        super().__init__(name=name)
        self.rate = rate

    def __call__(self, x, enabled):
        rate = self.rate * enabled
        key = hk.next_rng_key()
        p = jax.random.bernoulli(key, 1.0 - rate, shape=x.shape[:2])[..., None, None]
        return x * p / (1.0 - rate)


@hk.remat
class SelfAttention2d(hk.Module):
    def __init__(self, n_head=1, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        self.n_head = n_head
        self.dropout_rate = dropout_rate

    def __call__(self, x, dropout_enabled):
        n, c, h, w = x.shape
        assert c % self.n_head == 0
        qkv_proj = hk.Conv2D(c * 3, 1, data_format='NCHW', name='qkv_proj')
        out_proj = hk.Conv2D(c, 1, data_format='NCHW', name='out_proj')
        dropout = Dropout2d(self.dropout_rate)
        qkv = qkv_proj(x)
        qkv = jnp.swapaxes(qkv.reshape([n, self.n_head * 3, c // self.n_head, h * w]), 2, 3)
        q, k, v = jnp.split(qkv, 3, axis=1)
        scale = k.shape[3]**-0.25
        att = jax.nn.softmax((q * scale) @ (jnp.swapaxes(k, 2, 3) * scale), axis=3)
        y = jnp.swapaxes(att @ v, 2, 3).reshape([n, c, h, w])
        return x + dropout(out_proj(y), dropout_enabled)


def res_conv_block(c_mid, c_out, dropout_last=True):
    @hk.remat
    def inner(x, is_training):
        x_skip_layer = hk.Conv2D(c_out, 1, with_bias=False, data_format='NCHW')
        x_skip = x if x.shape[1] == c_out else x_skip_layer(x)
        x = hk.Conv2D(c_mid, 3, data_format='NCHW')(x)
        x = jax.nn.relu(x)
        x = Dropout2d(0.1)(x, is_training)
        x = hk.Conv2D(c_out, 3, data_format='NCHW')(x)
        if dropout_last:
            x = jax.nn.relu(x)
            x = Dropout2d(0.1)(x, is_training)
        return x + x_skip
    return inner


def diffusion_model(x, log_snr, extra_args, is_training):
    c = 128
    timestep_embed = FourierFeatures(16, 0.2)(log_snr[:, None])
    te_planes = jnp.tile(timestep_embed[..., None, None], [1, 1, x.shape[2], x.shape[3]])
    x = jnp.concatenate([x, te_planes], axis=1)  # 256x256
    x = res_conv_block(c // 2, c // 2)(x, is_training)
    x = res_conv_block(c // 2, c // 2)(x, is_training)
    x = res_conv_block(c // 2, c // 2)(x, is_training)
    x = res_conv_block(c // 2, c // 2)(x, is_training)
    ##########################################
    x_2 = hk.AvgPool(2, 2, 'SAME', 1)(x)  # 128x128
    x_2 = res_conv_block(c, c)(x_2, is_training)
    x_2 = res_conv_block(c, c)(x_2, is_training)
    x_2 = res_conv_block(c, c)(x_2, is_training)
    x_2 = res_conv_block(c, c)(x_2, is_training)
    ##########################################
    x_3 = hk.AvgPool(2, 2, 'SAME', 1)(x_2)  # 64x64
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    ##########################################
    x_4 = hk.AvgPool(2, 2, 'SAME', 1)(x_3)  # 32x32
    x_4 = res_conv_block(c * 2, c * 2)(x_4, is_training)
    x_4 = res_conv_block(c * 2, c * 2)(x_4, is_training)
    x_4 = res_conv_block(c * 2, c * 2)(x_4, is_training)
    x_4 = res_conv_block(c * 2, c * 2)(x_4, is_training)
    ##########################################
    x_5 = hk.AvgPool(2, 2, 'SAME', 1)(x_4)  # 16x16
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = SelfAttention2d(c * 4 // 128)(x_5, is_training)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = SelfAttention2d(c * 4 // 128)(x_5, is_training)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = SelfAttention2d(c * 4 // 128)(x_5, is_training)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = SelfAttention2d(c * 4 // 128)(x_5, is_training)
    ##########################################
    x_6 = hk.AvgPool(2, 2, 'SAME', 1)(x_5)  # 8x8
    x_6 = res_conv_block(c * 4, c * 4)(x_6, is_training)
    x_6 = SelfAttention2d(c * 4 // 128)(x_6, is_training)
    x_6 = res_conv_block(c * 4, c * 4)(x_6, is_training)
    x_6 = SelfAttention2d(c * 4 // 128)(x_6, is_training)
    x_6 = res_conv_block(c * 4, c * 4)(x_6, is_training)
    x_6 = SelfAttention2d(c * 4 // 128)(x_6, is_training)
    x_6 = res_conv_block(c * 4, c * 4)(x_6, is_training)
    x_6 = SelfAttention2d(c * 4 // 128)(x_6, is_training)
    ##########################################
    x_7 = hk.AvgPool(2, 2, 'SAME', 1)(x_6)  # 4x4
    x_7 = res_conv_block(c * 8, c * 8)(x_7, is_training)
    x_7 = SelfAttention2d(c * 8 // 128)(x_7, is_training)
    x_7 = res_conv_block(c * 8, c * 8)(x_7, is_training)
    x_7 = SelfAttention2d(c * 8 // 128)(x_7, is_training)
    x_7 = res_conv_block(c * 8, c * 8)(x_7, is_training)
    x_7 = SelfAttention2d(c * 8 // 128)(x_7, is_training)
    x_7 = res_conv_block(c * 8, c * 8)(x_7, is_training)
    x_7 = SelfAttention2d(c * 8 // 128)(x_7, is_training)
    x_7 = res_conv_block(c * 8, c * 8)(x_7, is_training)
    x_7 = SelfAttention2d(c * 8 // 128)(x_7, is_training)
    x_7 = res_conv_block(c * 8, c * 8)(x_7, is_training)
    x_7 = SelfAttention2d(c * 8 // 128)(x_7, is_training)
    x_7 = res_conv_block(c * 8, c * 8)(x_7, is_training)
    x_7 = SelfAttention2d(c * 8 // 128)(x_7, is_training)
    x_7 = res_conv_block(c * 8, c * 4)(x_7, is_training)
    x_7 = SelfAttention2d(c * 4 // 128)(x_7, is_training)
    x_7 = jax.image.resize(x_7, [*x_7.shape[:2], *x_6.shape[2:]], 'nearest')
    ##########################################
    x_6 = jnp.concatenate([x_6, x_7], axis=1)
    x_6 = res_conv_block(c * 4, c * 4)(x_6, is_training)
    x_6 = SelfAttention2d(c * 4 // 128)(x_6, is_training)
    x_6 = res_conv_block(c * 4, c * 4)(x_6, is_training)
    x_6 = SelfAttention2d(c * 4 // 128)(x_6, is_training)
    x_6 = res_conv_block(c * 4, c * 4)(x_6, is_training)
    x_6 = SelfAttention2d(c * 4 // 128)(x_6, is_training)
    x_6 = res_conv_block(c * 4, c * 4)(x_6, is_training)
    x_6 = SelfAttention2d(c * 4 // 128)(x_6, is_training)
    x_6 = jax.image.resize(x_6, [*x_6.shape[:2], *x_5.shape[2:]], 'nearest')
    ##########################################
    x_5 = jnp.concatenate([x_5, x_6], axis=1)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = hk.remat(SelfAttention2d(c * 4 // 128))(x_5, is_training)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = SelfAttention2d(c * 4 // 128)(x_5, is_training)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = SelfAttention2d(c * 4 // 128)(x_5, is_training)
    x_5 = res_conv_block(c * 4, c * 2)(x_5, is_training)
    x_5 = SelfAttention2d(c * 2 // 128)(x_5, is_training)
    x_5 = jax.image.resize(x_5, [*x_5.shape[:2], *x_4.shape[2:]], 'nearest')
    ##########################################
    x_4 = jnp.concatenate([x_4, x_5], axis=1)
    x_4 = res_conv_block(c * 2, c * 2)(x_4, is_training)
    x_4 = res_conv_block(c * 2, c * 2)(x_4, is_training)
    x_4 = res_conv_block(c * 2, c * 2)(x_4, is_training)
    x_4 = res_conv_block(c * 2, c * 2)(x_4, is_training)
    x_4 = jax.image.resize(x_4, [*x_4.shape[:2], *x_3.shape[2:]], 'nearest')
    ##########################################
    x_3 = jnp.concatenate([x_3, x_4], axis=1)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = res_conv_block(c * 2, c)(x_3, is_training)
    x_3 = jax.image.resize(x_3, [*x_3.shape[:2], *x_2.shape[2:]], 'nearest')
    ##########################################
    x_2 = jnp.concatenate([x_2, x_3], axis=1)
    x_2 = res_conv_block(c, c)(x_2, is_training)
    x_2 = res_conv_block(c, c)(x_2, is_training)
    x_2 = res_conv_block(c, c)(x_2, is_training)
    x_2 = res_conv_block(c, c // 2)(x_2, is_training)
    x_2 = jax.image.resize(x_2, [*x_2.shape[:2], *x.shape[2:]], 'nearest')
    ##########################################
    x = jnp.concatenate([x, x_2], axis=1)
    x = res_conv_block(c // 2, c // 2)(x, is_training)
    x = res_conv_block(c // 2, c // 2)(x, is_training)
    x = res_conv_block(c // 2, c // 2)(x, is_training)
    x = res_conv_block(c // 2, 3, dropout_last=False)(x, is_training)
    return x


class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def worker_init_fn(worker_id):
    ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_ema_decay(default_decay, epoch):
    if epoch < 20:
        return 0.99
    return default_decay


def v_loss(model, params, key, inputs, extra_args, is_training):
    key, subkey = jax.random.split(key)
    t = jax.random.uniform(subkey, inputs.shape[:1])
    log_snrs = get_ddpm_schedule(get_ddpm_inverse_cdf(t))
    alphas, sigmas = get_alpha_sigma(log_snrs)
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, inputs.shape)
    noised_inputs = inputs * alphas[:, None, None, None] + noise * sigmas[:, None, None, None]
    targets = noise * alphas[:, None, None, None] - inputs * sigmas[:, None, None, None]
    v = model.apply(params, key, noised_inputs, log_snrs, extra_args, is_training)
    return jnp.mean(jnp.square(v - targets)) * 0.280219


def train_one_step(loss_fn, optimizer, params, opt_state, key, inputs, extra_args, axis_name='i'):
    loss_grads = jax.value_and_grad(loss_fn)(params, key, inputs, extra_args, jnp.array(1))
    loss, grads = jax.lax.pmean(loss_grads, axis_name)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    ema_decay_fn,
    num_devices,
    epoch,
    log_file,
    params,
    params_ema,
    opt_state,
    key
):
    def v_loss(params, loss_key, inputs, extra_args, is_training):
        key, subkey = jax.random.split(loss_key)
        t = jax.random.uniform(subkey, inputs.shape[:1])
        log_snrs = get_ddpm_schedule(get_ddpm_inverse_cdf(t))
        alphas, sigmas = get_alpha_sigma(log_snrs)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, inputs.shape)
        noised_inputs = inputs * alphas[:, None, None, None] + noise * sigmas[:, None, None, None]
        targets = noise * alphas[:, None, None, None] - inputs * sigmas[:, None, None, None]
        v = model.apply(params, key, noised_inputs, log_snrs, extra_args, is_training)
        return jnp.mean(jnp.square(v - targets)) * 0.280219
    
    def train_one_step(params, opt_state, key, inputs, extra_args, axis_name='i'):
        loss_grads = jax.value_and_grad(v_loss)(params, key, inputs, extra_args, jnp.array(1))
        loss, grads = jax.lax.pmean(loss_grads, axis_name)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state
    
    pmap_train_step = jax.pmap(train_one_step, axis_name='i')
    
    for i, batch in enumerate(data_loader):
        inputs, _ = jax.tree_map(lambda x: psplit(jnp.array(x), num_devices), batch)
        key, subkey = jax.random.split(key)
        keys = jnp.stack(jax.random.split(subkey, num_devices))
        loss, params, opt_state = pmap_train_step(params, opt_state, keys, inputs, {})
        params_ema = p_ema_update(params, params_ema, ema_decay_fn(epoch))
        print(epoch, i, time.time(), unreplicate(loss), sep=',', file=log_file, flush=True)
        if i % 50 == 0:
            tqdm.write(f'Epoch {epoch}, iteration {i}, loss {unreplicate(loss):g}')
    return params, params_ema, opt_state


def sample_step(model, params, key, x_t, log_snr, log_snr_next, eta, extra_args):
    dummy_key = jax.random.PRNGKey(0)
    v = model.apply(
        params,
        dummy_key,
        x_t,
        repeat(log_snr, '-> n', n=x_t.shape[0]),
        extra_args,
        jnp.array(0)
    )
    alpha, sigma = get_alpha_sigma(log_snr)
    pred = x_t * alpha - v * sigma
    eps = x_t * sigma + v * alpha
    alpha_next, sigma_next = get_alpha_sigma(log_snr_next)
    ddim_sigma = eta * jnp.sqrt(sigma_next**2 / sigma**2) * jnp.sqrt(1 - alpha**2 / alpha_next**2)
    adjusted_sigma = jnp.sqrt(sigma_next**2 - ddim_sigma**2)
    x_t = pred * alpha_next + eps * adjusted_sigma
    x_t = x_t + jax.random.normal(key, x_t.shape) * ddim_sigma
    return x_t, pred


def sample(sample_fn, num_devices, params, key, x_t, steps, eta, extra_args):
    t = jnp.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    pmap_sample_step = jax.pmap(sample_fn, in_axes=(0, 0, 0, None, None, None, 0))
    for i in trange(steps):
        key, subkey = jax.random.split(key)
        keys = jnp.stack(jax.random.split(subkey, num_devices))
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', '-bs', type=int, default=64,
                   help='the batch size')
    p.add_argument('--ema-decay', type=float, default=0.999,
                   help='the EMA decay')
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

    tf = transforms.Compose([
        ToMode('RGB'),
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_set = datasets.ImageFolder(args.train_set, transform=tf)
    train_sampler = data.DistributedSampler(
        train_set,
        num_processes,
        local_rank,
        seed=args.seed,
        drop_last=True
    )
    train_dl = data.DataLoader(
        train_set,
        args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        num_workers=48,
        persistent_workers=True
    )

    model = hk.transform(diffusion_model)

    opt = optax.adam(5e-5)

    if not args.resume:
        epoch = 0
        params = model.init(
            jax.random.PRNGKey(args.seed),
            jnp.zeros([1, *shape]),
            jnp.zeros([1]),
            {},
            jnp.array(0)
        )
        params = jax.tree_map(lambda x: x / 2, params)
        params_ema = params
        opt_state = opt.init(params)
        log_file = open('losses.csv', 'w')
        print('epoch', 'i', 'time', 'loss', sep=',', file=log_file, flush=True)
    else:
        ckpt = pickle.load(open(args.resume, 'rb'))
        epoch = ckpt['epoch']
        params = jax.tree_map(jnp.array, ckpt['params'])
        params_ema = jax.tree_map(jnp.array, ckpt['params_ema'])
        opt_state = jax.tree_map(jnp.array, ckpt['opt_state'])
        del ckpt
        log_file = open('losses.csv', 'a+')

    print('Model parameters:', hk.data_structures.tree_size(params))

    params = jax.device_put_replicated(params, jax.local_devices())
    params_ema = jax.device_put_replicated(params_ema, jax.local_devices())
    opt_state = jax.device_put_replicated(opt_state, jax.local_devices())

    key = jax.random.PRNGKey(args.seed)
    key = jax.random.split(key, num_processes)[local_rank]
    
    ema_decay = Partial(get_ema_decay, args.ema_decay)
    compute_loss = Partial(v_loss, model) # I think this will be equally as efficient?
    train_step = Partial(train_one_step, compute_loss, opt) # ??? Is this at all better / easier to read?
    train_epoch = Partial(
        train_one_epoch,
        train_dl,
        ema_decay,
        train_step,
        num_local_devices,
        epoch,
        log_file
    )
    model_sample_step = Partial(sample_step, model)
    model_sample = Partial(sample, model_sample_step, num_local_devices)

    def save():
        if local_rank == 0:
            obj = {
                'params': unreplicate(params),
                'params_ema': unreplicate(params_ema),
                'opt_state': unreplicate(opt_state),
                'epoch': epoch
            }
            with open('model.pkl', 'wb') as f:
                pickle.dump(obj, f)

    def demo(key):
        if local_rank == 0:
            tqdm.write('Sampling...')
            outs = []
            for i in trange(1):
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, [8, 8, *shape])
                key, subkey = jax.random.split(key)
                out = punsplit(model_sample(params_ema, subkey, noise, 1000, 1, {}))
                outs.append(out)
            out = jnp.concatenate(outs, axis=0)
            grid = rearrange(out, '(s1 s2) c h w -> c (s1 h) (s2 w)', s1=8)
            to_pil_image(grid).save(f'demo_{epoch:06}.png')

    try:
        key, subkey = jax.random.split(key)
        demo(subkey)
        while True:
            tqdm.write(f'Epoch {epoch}')
            key, subkey = jax.random.split(key)
            train_sampler.set_epoch(epoch)
            params, params_ema, opt_state = train_epoch(params, params_ema, opt_state, subkey)
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