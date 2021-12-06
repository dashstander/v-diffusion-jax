import clip_jax.model as clip
import haiku as hk
import jax
import jax.numpy as jnp
from .. import utils


class EmbedTransformer(clip.Transformer):
    def __call__(self, x: jnp.ndarray):
        x = super().__call__(x[:, None].transpose((1, 0, 2)))
        x = x.transpose((1, 0, 2))
        return x.squeeze(1)


class FourierFeatures(hk.Module):
    def __init__(self, output_size, std=1.0, name=None):
        super().__init__(name=name)
        assert output_size % 2 == 0
        self.output_size = output_size
        self.std = std

    def __call__(self, x):
        w = hk.get_parameter(
            "w",
            [self.output_size // 2, x.shape[1]],
            init=hk.initializers.RandomNormal(self.std, 0),
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


class SelfAttention2d(hk.Module):
    def __init__(self, n_head=1, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        self.n_head = n_head
        self.dropout_rate = dropout_rate

    def __call__(self, x, dropout_enabled):
        n, c, h, w = x.shape
        assert c % self.n_head == 0
        qkv_proj = hk.Conv2D(c * 3, 1, data_format="NCHW", name="qkv_proj")
        out_proj = hk.Conv2D(c, 1, data_format="NCHW", name="out_proj")
        dropout = Dropout2d(self.dropout_rate)
        qkv = qkv_proj(x)
        qkv = jnp.swapaxes(
            qkv.reshape([n, self.n_head * 3, c // self.n_head, h * w]), 2, 3
        )
        q, k, v = jnp.split(qkv, 3, axis=1)
        scale = k.shape[3] ** -0.25
        att = jax.nn.softmax((q * scale) @ (jnp.swapaxes(k, 2, 3) * scale), axis=3)
        y = jnp.swapaxes(att @ v, 2, 3).reshape([n, c, h, w])
        return x + dropout(out_proj(y), dropout_enabled)


class ReluDropout(hk.Module):
    def __init__(self, dropout_rate: float, name=None):
        super().__init__(name)
        self.dr = Dropout2d(dropout_rate)

    def __call__(self, x, is_training):
        x = jax.nn.relu(x)
        return self.dr(x, is_training)


class ResnetConvLayer(hk.Module):
    def __init__(self, c_mid: int, c_out: int, dropout_rate: float, name=None):
        super().__init__(name=name)
        self.c_out = c_out
        self.conv_skip = hk.Conv2D(c_out, 1, with_bias=False, data_format="NCHW")
        self.conv_1 = hk.Conv2D(c_mid, 3, data_format="NCHW")
        self.dr_1 = Dropout2d(dropout_rate)
        self.conv_2 = hk.Conv2D(c_out, 3, data_format="NCHW")

    def __call__(self, x, is_training):
        x_skip = (
            x if x.shape[1] == self.c_out else self.conv_skip(x)
        )  # TODO:s Why won't this JIT?
        x = self.conv_1(x)
        x = jax.nn.relu(x)
        x = self.dr_1(x, is_training)
        x = self.conv_2(x)
        return x + x_skip


class ResConvBlock(hk.Module):
    def __init__(
        self,
        num_layers,
        c_mid,
        c_out,
        c_out_last=None,
        dropout_rate=0.1,
        name=None,
    ):
        super().__init__(name=name)
        self.num_layers = num_layers if c_out_last is None else num_layers - 1
        self.c_mid = c_mid
        self.c_out = c_out
        self.c_out_last=c_out_last
        self.dropout_rate = dropout_rate
        
    
    def make_layers(self):
        layers =  [
            ResnetConvLayer(self.c_mid, self.c_out, self.dropout_rate)
            for _ in range(self.num_layers)
        ]
        if self.c_out_last is not None:
            layers.append(
                ResnetConvLayer(self.c_mid, self.c_out_last, self.dropout_rate)
            )
        return layers

    def __call__(self, x, is_training):
        for layer in self.make_layers():
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
        dropout_rate=0.1,
        name=None,
    ):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.c_mid = c_mid
        self.c_out = c_out
        self.c_out_last=c_out_last
        self.num_heads_last = num_heads_last
        self.dropout_rate = dropout_rate
    
    def make_layers(self):
        layers = []
        for i in range(self.num_layers):
            if i == (self.num_layers - 1) and self.num_heads_last is not None:
                layers += [
                    ResnetConvLayer(self.c_mid, self.c_out_last, self.dropout_rate),
                    SelfAttention2d(self.num_heads_last),
                ]
            else:
                layers += [
                    ResnetConvLayer(self.c_mid, self.c_out, self.dropout_rate),
                    SelfAttention2d(self.num_heads),
                ]
        return layers

    def __call__(self, x, is_training):
        for layer in self.make_layers():
            x = layer(x, is_training)
        return x


class CrossAttention(hk.Module):
    def __init__(self, num_heads, key_size, output_size, name=None):
        super().__init__(name=name)
        self.x_attention = hk.MultiHeadAttention(num_heads, key_size, 1)
        self.y_attention = hk.MultiHeadAttention(num_heads, key_size, 1)
        self.fuse = hk.Linear(output_size)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray):
        x_atts = self.x_attention(y, x, x)
        y_atts = self.y_attention(x, y, y)
        xy = jax.nn.relu(jnp.concatenate([x_atts, y_atts], axis=1))
        return self.fuse(xy)


def reshape_embed_to_image(x, image_shape):
    assert x.ndim == 2
    target_shape = (image_shape[0], 1, *image_shape[2:])
    image_size = image_shape[2] * image_shape[3]
    assert image_size % x.shape[1] == 0
    repeats = image_size // x.shape[1]
    return x[:, None].repeat(repeats, axis=1).reshape(target_shape)        


def diffusion_model(x, t, y, extra_args, is_training):
    c = 256
    log_snr = utils.alpha_sigma_to_log_snr(*utils.t_to_alpha_sigma(t))
    timestep_embed = FourierFeatures(16, 0.2)(log_snr[:, None])

    y = jnp.concatenate([timestep_embed, y], axis=1)
    y = hk.Linear(256)(y)
    y = jax.nn.relu(y)
    y = EmbedTransformer(width=256, heads=1, layers=1, name="EmbedTransformer01")(y)

    te_planes = jnp.tile(
        timestep_embed[..., None, None], [1, 1, x.shape[2], x.shape[3]]
    )
    x = jnp.concatenate([x, te_planes], axis=1)  
    x = ResConvBlock(4, c // 2, c // 2, name="ResBlock1")(x, is_training) # Nx128x256x256
    #print(f'x: {x.shape}')
    ########################################################
    x_2 = hk.AvgPool(2, 2, "SAME", 1)(x)  # Nx128x128x128
    #print(f'x_2: {x_2.shape}')
    x_embed_2 = clip.VisualTransformer(128, 16, 128, 2, 2, 256, "ViT01")(x_2)
    xy_1 = CrossAttention(2, 256, 512, "CrossAtt01")(y, x_embed_2)
    x_2 = ResConvBlock(2, c, c, name="ResBlock2")(x_2, is_training)
    ########################################################
    x_3 = hk.AvgPool(2, 2, "SAME", 1)(x_2)  # Nx256x64x64
    #print(f'x_3: {x_3.shape}')
    x_3 = ResConvBlock(2, c * 2, (c * 2) - 1, name="ResBlock3")(x_3, is_training)
    ########################################################
    x_4 = hk.AvgPool(2, 2, "SAME", 1)(x_3)  
    x_4 = jnp.concatenate([
        x_4,
        reshape_embed_to_image(xy_1, x_4.shape)],
        axis=1
    ) # Nx512x32x32
    #print(f'x_4: {x_4.shape}')
    y_2 = jax.nn.relu(xy_1)
    y_2 = hk.Linear(64)(y_2)
    y_2 = EmbedTransformer(64, 1, 2, "EmbedTransformer2")(y_2)
    x_4 = ResConvBlock(2, c * 2, c * 2, name="ResBlock4")(x_4, is_training)
    ########################################################
    x_5 = hk.AvgPool(2, 2, "SAME", 1)(x_4)  # Nx256x16x16
    #print(f'x_5: {x_5.shape}')
    x_5 = ResConvBlockAtt(2, c * 4 // 128, c * 4, c * 4, name="ResBlock5")(
        x_5, is_training
    )
    ########################################################
    x_6 = hk.AvgPool(2, 2, "SAME", 1)(x_5)  # 8x8
    #print(f'x_6: {x_6.shape}')
    x_6 = ResConvBlockAtt(4, c * 4 // 128, c * 4, c * 4, name="ResBlock6")(
        x_6, is_training
    )
    #########################################################
    x_7 = hk.AvgPool(2, 2, "SAME", 1)(x_6)  # Nx1024x4x4
    #print(f'x_7: {x_7.shape}')
    x_embed_7 = hk.Linear(64)(jnp.mean(x_7, axis=(2 , 3))) # Nx1024x4x4 -> Nx1024 --> Nx128
    xy_2 = CrossAttention(2, 64, 64, "CrossAtt7")(x_embed_7, y_2)
    y_3 = EmbedTransformer(64, 1, 2, "EmbedTransformer3")(xy_2)
    x_7 = ResConvBlock(4, c * 8, c * 8, (c * 4) - 1, name="ResBlock7")(x_7, is_training)
    x_7 = jax.image.resize(x_7, [*x_7.shape[:2], *x_6.shape[2:]], "nearest")
    x_7 = jnp.concatenate([x_7, reshape_embed_to_image(y_3, x_7.shape)], axis=1)
    ##########################################################
    x_6 = jnp.concatenate([x_6, x_7], axis=1)
    x_6 = ResConvBlockAtt(2, c * 4 // 128, c * 4, c * 4, name="SecondResBlock6")(x_6, is_training)
    x_6 = jax.image.resize(x_6, [*x_6.shape[:2], *x_5.shape[2:]], "nearest")
    ############################################################
    x_5 = jnp.concatenate([x_5, x_6], axis=1)
    x_5 = ResConvBlockAtt(
        2, c * 4 // 128, c * 4, c * 4, c * 2, c * 2 // 128, name="SecondResBlock5"
    )(x_5, is_training)
    x_5 = jax.image.resize(x_5, [*x_5.shape[:2], *x_4.shape[2:]], "nearest")
    ##############################################################
    x_4 = jnp.concatenate([x_4, x_5], axis=1)
    x_4 = ResConvBlock(2, c * 2, c * 2, name='SecondResBlock4')(x_4, is_training)
    x_4 = jax.image.resize(x_4, [*x_4.shape[:2], *x_3.shape[2:]], "nearest")
    ##############################################################
    x_3 = jnp.concatenate([x_3, x_4], axis=1)
    x_3 = ResConvBlock(2, c * 2, c * 2, c, name='SecondResBlock3')(x_3, is_training)
    x_3 = jax.image.resize(x_3, [*x_3.shape[:2], *x_2.shape[2:]], "nearest")
    ##############################################################
    x_2 = jnp.concatenate([x_2, x_3], axis=1)
    x_2 = ResConvBlock(2, c, c, c // 2, name='SecondResBlock2')(x_2, is_training)
    x_2 = jax.image.resize(x_2, [*x_2.shape[:2], *x.shape[2:]], "nearest")
    y_4 = EmbedTransformer(64, 1, 2, "EmbedTransforme4")(y_3)
    y_4 = jax.nn.relu(y_4)
    y_4 = hk.Linear(512)(y_4)
    ##############################################################
    x = jnp.concatenate([x, x_2], axis=1)
    x = ResConvBlock(2, c // 2, c // 2, 3, name='SecondResBlock1')(x, is_training)
    y = EmbedTransformer(512, 1, 2, "EmbedTransformerFinal")(y_4)
    return x, y


class CLIPWikiArt256:
    init, apply = hk.transform(diffusion_model)
    shape = (3, 256, 256)
    min_t = float(utils.get_ddpm_schedule(jnp.array(0.0)))
    max_t = float(utils.get_ddpm_schedule(jnp.array(1.0)))

