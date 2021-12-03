
from clip_jax.model import Transformer, VisualTransformer
import haiku as hk
from haiku._src.basic import Linear
import jax
import jax.numpy as jnp
from .. import utils



class IdentityLayer(hk.Module):
    def __call__(self, x):
        return x


class FourierFeatures(hk.Module):
    def __init__(self, output_size, std=1., name=None):
        super().__init__(name=name)
        assert output_size % 2 == 0
        self.output_size = output_size
        self.std = std

    def __call__(self, x):
        w = hk.get_parameter(
            'w',
            [self.output_size // 2, x.shape[1]],
            init=hk.initializers.RandomNormal(self.std, 0)
        )
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

@jax.remat
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


class ReluDropout(hk.Module):

    def __init__(self, dropout_rate: float):
        self.dr = Dropout2d(dropout_rate)

    def __call__(self, x, is_training):
        x = jax.nn.relu(x)
        return self.dr(x, is_training)



@hk.remat
class ResnetConvLayer(hk.Module):

    def __init__(self, c_mid: int, c_out: int, dropout_rate: float, dropout_last: bool):
        self.c_out = c_out
        self.do_last_dropout = dropout_last
        self.conv_skip = hk.Conv2D(c_out, 1, with_bias=False, data_format='NCHW')
        self.conv_1 = hk.Conv2D(c_mid, 3, data_format='NCHW')
        self.dr_1 = Dropout2d(dropout_rate)
        self.conv_2 = hk.Conv2D(c_out, 3, data_format='NCHW')
        if dropout_last:
            self.last = ReluDropout(dropout_rate)
        else:
            self.last = IdentityLayer()

    def __call__(self, x, is_training):
        x_skip = x if x.shape[1] == self.c_out else self.conv_skip(x) # TODO:s Why won't this JIT?
        x = self.conv_1(x)
        x = jax.nn.relu(x)
        x = self.dr_1(x, is_training)
        x = self.conv_2(x)
        return self.last(x) + x_skip


class ResConvBlock(hk.Module):

    def __init__(self, num_layers, c_mid, c_out, c_out_last=None, dropout_last=True, dropout_rate=0.1):
        self.layers = [ResnetConvLayer(c_mid, c_out, dropout_rate, dropout_last) for _ in range(num_layers)]
        if c_out_last is not None:
            self.layers.append(ResnetConvLayer(c_mid, c_out_last, dropout_rate, dropout_last))

    def __call__(self, x, is_training):
        for layer in self.layers:
            x = layer(x, is_training)
        return x


class ResConvBlockAtt(hk.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        c_mid,
        c_out,
        c_out_last=None,
        num_heads_last=None,
        dropout_last=True,
        dropout_rate=0.1
    ):
        self.layers = []
        for i in num_layers:
            if i == (num_layers - 1) and num_heads_last is not None:
                self.layers.append([ResnetConvLayer(c_mid, c_out_last, dropout_rate, dropout_last), SelfAttention2d(num_heads_last)])
            else:
                self.layers.append([ResnetConvLayer(c_mid, c_out, dropout_rate, dropout_last), SelfAttention2d(num_heads)])

    def __call__(self, x, is_training):
        for layer in self.layers:
            x = layer(x, is_training)
        return x



class CrossAttention(hk.Module):
    def __init__(self, num_heads, key_size, output_size, name=None):
        super().__init__(name=name)
        self.x_attention = hk.MultiHeadAttention(num_heads, key_size=key_size)
        self.y_attention = hk.MultiHeadAttention(num_heads, key_size=key_size)
        self.fuse = hk.Linear(output_size)
    
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray):
        x_atts = self.x_attention(y, x, x)
        y_atts = self.y_attention(x, y, y)
        xy = jax.nn.relu(jnp.concatenate([x_atts, y_atts], axis=1))
        return self.fuse(xy)


def diffusion_model(x, t, y, extra_args):
    c = 256
    is_training = jnp.array(0.)
    log_snr = utils.alpha_sigma_to_log_snr(*utils.t_to_alpha_sigma(t))
    f_feat = FourierFeatures(16, 0.2)(log_snr[:, None])
    timestep_embed = jnp.concatenate(f_feat, axis=1)

    y = jnp.concatenate([t, y], axis=1)
    y = Transformer(width=512, heads=2, layers=2, dropout_rate=0.1, name='EmbedTransformer01')(y, is_training=is_training)

    te_planes = jnp.tile(timestep_embed[..., None, None], [1, 1, x.shape[2], x.shape[3]])
    x = jnp.concatenate([x, te_planes], axis=1)  # 256x256
    x = ResConvBlock(4, c // 2, c // 2)(x, is_training)
    ########################################################
    x_2 = hk.AvgPool(2, 2, 'SAME', 1)(x)  # 128x128
    x_embed_2 = VisualTransformer(128, 16, 128, 2, 2, 512, "ViT01")(x_2)
    xy_1 = CrossAttention(2, 512, 1024, "CrossAtt01")(y, x_embed_2)
    x_2 = ResConvBlock(4, c, c)(x_2, is_training)
    ########################################################
    x_3 = hk.AvgPool(2, 2, 'SAME', 1)(x_2)  # 64x64
    x_3 = ResConvBlock(4, c * 2, c * 2)(x_3, is_training)
    ########################################################
    x_4 = hk.AvgPool(2, 2, 'SAME', 1)(x_3)  # 32x32
    x_4 = jnp.concatenate([x_4, xy_1.reshape(x_4.shape)], axis=1)
    y_2 = jax.relu(xy_1)
    y_2 = hk.Linear(512)(y_2)
    y_2 = Transformer(128, 3, 2, "EmbedTransformer2")(y_2)
    x_4 = ResConvBlock(4, c * 2, c * 2)(x_4, is_training)
    ########################################################
    x_5 = hk.AvgPool(2, 2, 'SAME', 1)(x_4)  # 16x16
    x_5 = ResConvBlockAtt(4, c * 4 // 128, c * 4, c * 4)(x_5, is_training)
    ########################################################
    x_6 = hk.AvgPool(2, 2, 'SAME', 1)(x_5)  # 8x8
    x_6 = ResConvBlockAtt(4, c * 4 // 128, c * 4, c * 4)(x_6, is_training)
    #########################################################
    x_7 = hk.AvgPool(2, 2, 'SAME', 1)(x_6)  # 4x4
    x_7 = ResConvBlockAtt(8, c * 8 // 128, c * 8, c * 8, c * 4, c * 4 // 128)(x_7, is_training)
    x_7 = jax.image.resize(x_7, [*x_7.shape[:2], *x_6.shape[2:]], 'nearest')
    ##########################################################
    x_6 = jnp.concatenate([x_6, x_7], axis=1)
    x_6 = ResConvBlockAtt(4, c * 4 // 128, c * 4, c * 4)(x_6, is_training)
    x_6 = jax.image.resize(x_6, [*x_6.shape[:2], *x_5.shape[2:]], 'nearest')
    x_embed_6 = VisualTransformer(x_6.shape[2], c * 4 // 128, x_6.shape[3], 2, 2, 128, "Vit6")
    xy_2 = CrossAttention(2, 128, 256, "CrossAtt6")(x_embed_6, y_2)
    y_3 = Transformer(256, 3, 2, "EmbedTransformer3")(xy_2)
    ############################################################
    x_5 = jnp.concatenate([x_5, x_6], axis=1)
    x_5 = ResConvBlockAtt(4, c * 4 // 128, c * 4, c * 4, c * 2, c * 2 // 128)(x_5, is_training)
    x_5 = jax.image.resize(x_5, [*x_5.shape[:2], *x_4.shape[2:]], 'nearest')
    ##############################################################
    x_4 = jnp.concatenate([x_4, x_5], axis=1)
    x_4 = ResConvBlock(4, c * 2, c * 2)(x_4, is_training)
    x_4 = jax.image.resize(x_4, [*x_4.shape[:2], *x_3.shape[2:]], 'nearest')
    ##############################################################
    x_3 = jnp.concatenate([x_3, x_4], axis=1)
    x_3 = ResConvBlock(4, c * 2, c * 2, c)(x_3, is_training)
    x_3 = jax.image.resize(x_3, [*x_3.shape[:2], *x_2.shape[2:]], 'nearest')
    ##############################################################
    x_2 = jnp.concatenate([x_2, x_3], axis=1)
    x_2 = ResConvBlock(4, c, c, c // 2)(x_2, is_training)
    x_2 = jax.image.resize(x_2, [*x_2.shape[:2], *x.shape[2:]], 'nearest')
    ##############################################################
    x = jnp.concatenate([x, x_2], axis=1)
    x = ResConvBlock(4, c // 2, c // 2, 3)(x, is_training)
    y = Transformer(512, 2, 2, "EmbedTransformerFinal")(y_3)
    return x, y


class CLIPWikiArt256:
    init, apply = hk.transform(diffusion_model)
    shape = (3, 256, 256)
    min_t = float(utils.get_ddpm_schedule(jnp.array(0.)))
    max_t = float(utils.get_ddpm_schedule(jnp.array(1.)))
