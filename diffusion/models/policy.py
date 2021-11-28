import haiku as hk
import jax
from jax._src.tree_util import Partial
import jax.numpy as jnp
from diffusion import utils


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


def res_conv_block(c_mid, c_out, dropout_last=True):
    @hk.remat
    def inner(x, is_training):
        x_skip_layer = hk.Conv2D(c_out, 1, with_bias=False, data_format='NCHW')
        x_skip = jax.lax.cond(x.shape[1] == c_out, lambda y: y, x_skip_layer, x)
        x = hk.Conv2D(c_mid, 3, data_format='NCHW')(x)
        x = jax.nn.relu(x)
        x = Dropout2d(0.1)(x, is_training)
        x = hk.Conv2D(c_out, 3, data_format='NCHW')(x)
        x = jax.nn.relu(x)
        x = jax.lax.cond(dropout_last, Dropout2d(0.1), lambda y, _: y, x, is_training)
        return x + x_skip
    return inner


class CLIPEmbeddingLayer(hk.Module):

    def __init__(self, input_size, output_size, name=None):
        super().__init__(name=name)
        self.input_size = input_size
        self.output_size = output_size
    
    def __call__(self, key, x, target, timestep):
        key, subkey = jax.random.split(key)
        x = jnp.concatenate([x, target, timestep], axis=1)
        x = hk.Linear(self.input_size * 2)(x)
        x = hk.dropout(subkey, 0.1, x)
        x = jax.nn.relu(x)
        x = hk.MultiHeadAttention(2, self.input_size * 2, 1.0, self.input_size * 2, self.input_size * 2)(x)
        x = hk.Linear(self.input_size)(jnp.concatenate([x, timestep]))
        key, subkey = jax.random.split(key)
        x = hk.dropout(subkey, 0.1, x)
        x = jax.nn.relu(x)
        x = hk.MultiHeadAttention(1, self.input_size, 1.0, self.input_size, self.input_size)(x)
        return hk.Linear(self.output_size)(x)


def image_to_ts(x, time_embedding):
    keys = hk.next_rng_keys(5)
    x = hk.Flatten()(x)
    x = jnp.concatenate(x, time_embedding, axis=1)
    x = hk.Sequential(
        [
            hk.Linear(2048, with_bias=True),
            Partial(hk.dropout, rng=keys[0], rate=0.1),
            jax.nn.relu,
            hk.Linear(1024, with_bias=True),
            Partial(hk.dropout, rng=keys[1], rate=0.1),
            jax.nn.relu,
            hk.Linear(512, with_bias=True),
            Partial(hk.dropout, rng=keys[2], rate=0.1),
            jax.nn.relu,
            hk.Linear(256, with_bias=True),
            Partial(hk.dropout, rng=keys[3], rate=0.1),
            jax.nn.relu,
            hk.Linear(128, with_bias=True),
            Partial(hk.dropout, rng=keys[4], rate=0.1),
            jax.nn.relu,
            hk.Linear(64, with_bias=True),
            hk.Linear(1, with_bias=True)
        ]
    )(x)
    
    return {
        'mu': x,
    }


def pi_model(x, target_grad, t, x_embed, target, extra_args):
    c = 128
    is_training = jnp.array(0.)
    log_snr = utils.alpha_sigma_to_log_snr(*utils.t_to_alpha_sigma(t))
    time_embed = FourierFeatures(16, 0.2)(log_snr[:, None])
    state_embed = CLIPEmbeddingLayer(512, c, name='CLIP Dynamics State')(x_embed, target, time_embed)
    te_planes = jnp.tile(state_embed[..., None, None], [1, 1, x.shape[2], x.shape[3]])
    x = jnp.concatenate([x, te_planes, target_grad], axis=1)  # 128x128
    x = res_conv_block(c, c)(x, is_training)
    x = res_conv_block(c, c)(x, is_training)
    x = res_conv_block(c, c)(x, is_training)
    x = res_conv_block(c, c)(x, is_training)
    x_2 = hk.AvgPool(2, 2, 'SAME', 1)(x)  # 64x64
    x_2 = res_conv_block(c * 2, c * 2)(x_2, is_training)
    x_2 = res_conv_block(c * 2, c * 2)(x_2, is_training)
    x_2 = res_conv_block(c * 2, c * 2)(x_2, is_training)
    x_2 = res_conv_block(c * 2, c * 2)(x_2, is_training)
    x_3 = hk.AvgPool(2, 2, 'SAME', 1)(x_2)  # 32x32
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_4 = hk.AvgPool(2, 2, 'SAME', 1)(x_3)  # 16x16
    x_4 = res_conv_block(c * 4, c * 4)(x_4, is_training)
    x_4 = res_conv_block(c * 4, c * 4)(x_4, is_training)
    x_4 = res_conv_block(c * 4, c * 4)(x_4, is_training)
    x_4 = res_conv_block(c * 4, c * 4)(x_4, is_training)
    x_5 = hk.AvgPool(2, 2, 'SAME', 1)(x_4)  # 8x8
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_6 = hk.AvgPool(2, 2, 'SAME', 1)(x_5)  # 4x4
    x_6 = res_conv_block(c * 8, c * 8)(x_6, is_training)
    x_6 = res_conv_block(c * 8, c * 8)(x_6, is_training)
    x_6 = res_conv_block(c * 8, c * 8)(x_6, is_training)
    x_6 = res_conv_block(c * 8, c * 8)(x_6, is_training)
    x_6 = res_conv_block(c * 8, c * 8)(x_6, is_training)
    x_6 = res_conv_block(c * 8, c * 8)(x_6, is_training)
    x_6 = res_conv_block(c * 8, c * 8)(x_6, is_training)
    x_6 = res_conv_block(c * 8, c * 4)(x_6, is_training)
    x_6 = jax.image.resize(x_6, [*x_6.shape[:2], *x_5.shape[2:]], 'nearest')
    x_5 = jnp.concatenate([x_5, x_6], axis=1)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = res_conv_block(c * 4, c * 4)(x_5, is_training)
    x_5 = jax.image.resize(x_5, [*x_5.shape[:2], *x_4.shape[2:]], 'nearest')
    x_4 = jnp.concatenate([x_4, x_5], axis=1)
    x_4 = res_conv_block(c * 4, c * 4)(x_4, is_training)
    x_4 = res_conv_block(c * 4, c * 4)(x_4, is_training)
    x_4 = res_conv_block(c * 4, c * 4)(x_4, is_training)
    x_4 = res_conv_block(c * 4, c * 2)(x_4, is_training)
    x_4 = jax.image.resize(x_4, [*x_4.shape[:2], *x_3.shape[2:]], 'nearest')
    x_3 = jnp.concatenate([x_3, x_4], axis=1)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = res_conv_block(c * 2, c * 2)(x_3, is_training)
    x_3 = jax.image.resize(x_3, [*x_3.shape[:2], *x_2.shape[2:]], 'nearest')
    x_2 = jnp.concatenate([x_2, x_3], axis=1)
    x_2 = res_conv_block(c * 2, c * 2)(x_2, is_training)
    x_2 = res_conv_block(c * 2, c * 2)(x_2, is_training)
    x_2 = res_conv_block(c * 2, c * 2)(x_2, is_training)
    x_2 = res_conv_block(c * 2, c)(x_2, is_training)
    x_2 = jax.image.resize(x_2, [*x_2.shape[:2], *x.shape[2:]], 'nearest')
    x = jnp.concatenate([x, x_2], axis=1)
    x = res_conv_block(c, c)(x, is_training)
    x = res_conv_block(c, c)(x, is_training)
    x = res_conv_block(c, c)(x, is_training)
    x = res_conv_block(c, 3, dropout_last=False)(x, is_training)
    timestep_params = image_to_ts(x, state_embed)
    return x, timestep_params


class PiModel:
    init, apply = hk.transform(pi_model)
    shape = (3, 128, 128)
    min_t = float(utils.get_ddpm_schedule(jnp.array(0.)))
    max_t = float(utils.get_ddpm_schedule(jnp.array(1.)))