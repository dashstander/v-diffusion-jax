#!/usr/bin/env python3

import argparse
import pickle
import time

from einops import rearrange, repeat
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from PIL import ImageFile
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm, trange

from diffusion.utils import *
from diffusion.models.clip_latent import diffusion_model


def ema_update(params, averaged_params, decay):
    return jax.tree_map(lambda p, a: p * (1 - decay) + a * decay, params, averaged_params)


p_ema_update = jax.pmap(ema_update, in_axes=(0, 0, None))


def unreplicate(x):
    return jax.tree_map(lambda x: x[0], x)


def psplit(x, n):
    return jax.tree_map(lambda x: jnp.stack(jnp.split(x, n)), x)


def punsplit(x):
    return jax.tree_map(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])), x)


class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def worker_init_fn(worker_id):
    ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    train_sampler = data.DistributedSampler(train_set, num_processes, local_rank,
                                            seed=args.seed, drop_last=True)
    train_dl = data.DataLoader(train_set, args.batch_size, sampler=train_sampler, drop_last=True,
                               worker_init_fn=worker_init_fn, num_workers=48,
                               persistent_workers=True)

    model = hk.transform(diffusion_model)

    opt = optax.adam(5e-5)

    if not args.resume:
        epoch = 0
        params = model.init(jax.random.PRNGKey(args.seed),
                            jnp.zeros([1, *shape]),
                            jnp.zeros([1]),
                            {},
                            jnp.array(0))
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

    def get_ema_decay(epoch):
        if epoch < 20:
            return 0.99
        return args.ema_decay

    def compute_loss(params, key, inputs, extra_args, is_training):
        key, subkey = jax.random.split(key)
        t = jax.random.uniform(subkey, inputs.shape[:1])
        log_snrs = get_ddpm_schedule(get_ddpm_schedule(t))
        alphas, sigmas = log_snr_to_alpha_sigma(log_snrs)
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, inputs.shape)
        noised_inputs = inputs * alphas[:, None, None, None] + noise * sigmas[:, None, None, None]
        targets = noise * alphas[:, None, None, None] - inputs * sigmas[:, None, None, None]
        v = model.apply(params, key, noised_inputs, log_snrs, extra_args, is_training)
        return jnp.mean(jnp.square(v - targets)) * 0.280219

    def train_step(params, opt_state, key, inputs, extra_args, axis_name='i'):
        loss_grads = jax.value_and_grad(compute_loss)(params, key, inputs, extra_args, jnp.array(1))
        loss, grads = jax.lax.pmean(loss_grads, axis_name)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    def train_one_epoch(params, params_ema, opt_state, key):
        pmap_train_step = jax.pmap(train_step, axis_name='i')
        for i, batch in enumerate(tqdm(train_dl)):
            inputs, _ = jax.tree_map(lambda x: psplit(jnp.array(x), num_local_devices), batch)
            key, subkey = jax.random.split(key)
            keys = jnp.stack(jax.random.split(subkey, num_local_devices))
            loss, params, opt_state = pmap_train_step(params, opt_state, keys, inputs, {})
            params_ema = p_ema_update(params, params_ema, get_ema_decay(epoch))
            print(epoch, i, time.time(), unreplicate(loss), sep=',', file=log_file, flush=True)
            if i % 50 == 0:
                tqdm.write(f'Epoch {epoch}, iteration {i}, loss {unreplicate(loss):g}')
        return params, params_ema, opt_state

    def sample_step(params, key, x_t, log_snr, log_snr_next, eta, extra_args):
        dummy_key = jax.random.PRNGKey(0)
        v = model.apply(params, dummy_key, x_t, repeat(log_snr, '-> n', n=x_t.shape[0]), extra_args,
                        jnp.array(0))
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
                x_t, _ = pmap_sample_step(params, keys, x_t, log_snrs[i], log_snrs[i + 1], eta,
                                          extra_args)
            else:
                _, pred = pmap_sample_step(params, keys, x_t, log_snrs[i], log_snrs[i], eta,
                                           extra_args)
        return pred

    def save():
        if local_rank == 0:
            obj = {'params': unreplicate(params),
                   'params_ema': unreplicate(params_ema),
                   'opt_state': unreplicate(opt_state),
                   'epoch': epoch}
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
                out = punsplit(sample(params_ema, subkey, noise, 1000, 1, {}))
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