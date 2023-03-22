from warnings import formatwarning
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, Layer, Flatten
from keras.layers import Concatenate, Multiply, Add, LayerNormalization, GlobalAveragePooling2D
from keras.models import Model
from keras.initializers.initializers_v2 import LecunNormal, Zeros, TruncatedNormal
import numpy as np
import tensorflow as tf

"""
SwinTransformer basic blocks:
PatchEmbedding, PatchMerging, MLPHead, Window-MultiHeadSelfAttention(W-MSA/SW-MSA), SwinTransformerBlock, 
BasicLayer(stage creator), SwinTransFormer(Backbone)
Extra Support Functions: window_partition, window_shift_back
"""

"""
This scripts implements SwinTransformer which accommodate any size of input images
Built-in input padding system engineering
"""

"""
NOTICE: In the paper (Liu et al), each model stage is combined with a SwinTransformerBlock and a PatchMerging 
sequentially, in open source however, the stage is on the other way around (PatchMerging -> SwinTransBlock)
Therefore, in the final stage (stage 4), there will be no PatchMerging Layer followed
"""


def window_partition(inputs, window_size: int) -> any:
    """
    This function implements shifted window method (sw)
    :return:
    """
    # assert that window_size is a square thus window_size can be an int rather than a tuple
    batches, height, width, channels = inputs.shape
    x = tf.reshape(inputs, shape=[batches, height // window_size, window_size,
                                  width // window_size, window_size, channels])
    # x.shape: [batches, n_windows_height, n_windows_width, window_size, window_size, channels]
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    # x.shape: [batches * n_windows, window_size, window_size, channels]
    x = tf.reshape(x, shape=[-1, window_size, window_size, channels])
    return x


def window_shift_back(inputs, window_size: int, height: int, width: int) -> any:
    """
    Shifts partitioned windows back
    :return:
    """
    batches_windows, window_size_height, window_size_width, channels = inputs.shape
    if window_size != window_size_height or window_size != window_size_width:
        print(formatwarning(f'argument window_size and retrieved window size does not match, '
                            f'expected [{window_size}, {window_size}], got[{window_size_height}, {window_size_width}]',
                            category=FutureWarning, filename='./SwinTransformer.py', lineno=52))
    batches = batches_windows / ((height / window_size) * (width / window_size))
    x = tf.reshape(inputs, shape=[batches, height / window_size, width / window_size,
                                  window_size, window_size, channels])
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=[batches, height, width, channels])

    return x


class PatchEmbedding(Layer):
    def __init__(self, patch_size=4, embed_dim=96, layer_norm=LayerNormalization):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embedding = Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='same',
                                      kernel_initializer=LecunNormal(), bias_initializer=Zeros())
        self.layer_normalization = layer_norm(epsilon=1e-6) if layer_norm else Activation('linear')

    def call(self, x, *args, **kwargs):
        _, height, width, _ = x.shape
        # padding is needed if height, width can not be fully divided by patch_size
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            # padding direction: [channel_front, channel_back], [top, bottom], [left, right]
            padder = tf.constant([[0, 0],
                                  [0, self.patch_size - height % self.patch_size],
                                  [0, self.patch_size - width % self.patch_size]])
            x = tf.pad(x, padder)

        # down sampling
        x = self.patch_embedding(x)
        batches, height, width, channels = x.shape  # [batches, 224 // patch_size, 224 // patch_size, embed_dim]
        # feature flatten, axis=1
        x = tf.reshape(x, shape=(batches, height * width, channels))
        x = self.layer_normalization(x)

        return x, height, width


class PatchMerging(Layer):
    def __init__(self, in_channels, layer_norm=LayerNormalization):
        super(PatchMerging, self).__init__()
        self.in_channels = in_channels
        # self.height = height
        # self.width = width

        self.layer_normalization = layer_norm(epsilon=1e-6) if layer_norm else Activation('linear')
        self.dim_reduction = Dense(in_channels * 2, use_bias=False,
                                   kernel_initializer=TruncatedNormal(stddev=.02))

    def call(self, x, height=None, width=None, *args, **kwargs):
        """
        x.shape -> [batches, height x width, channels]
        """
        assert height is not None and width is not None
        batches, length, channels = x.shape
        assert length == height * width, "length and feature size does not match: [{},{}]" \
            .format(length, height * width)
        x = tf.reshape(x, shape=(batches, height, width, channels))

        # padding is needed if height, width can not be fully divided by patch_size
        if height % 2 != 0 or width % 2 != 0:
            padder = tf.constant([0, 0], [0, 1], [0, 1], [0, 0])
            x = tf.pad(x, padder)

        # Interval sampling - patch merging
        x1, x2 = x[:, ::2, ::2, :], x[:, ::2, 1::2, :]
        x3, x4 = x[:, 1::2, ::2, :], x[:, 1::2, 1::2, :]
        x = tf.concat([x1, x2, x3, x4], axis=-1)  # shape: [batches, H/2, W/2, channels * 4]
        x = tf.reshape(x, shape=(batches, -1, channels * 4))
        x = self.layer_normalization(x)  # shape: [batches, H*W/4, channels * 4]
        x = self.dim_reduction(x)  # shape: [batches, H*W/4, channels * 2]

        return x


class FeedForwardNetwork(Layer):
    def __init__(self, in_channels, expansion_rate=4, drop_rate=0.):
        super(FeedForwardNetwork, self).__init__()
        self.drop_rate = drop_rate
        self.feedforward0 = Dense(int(in_channels * expansion_rate), kernel_initializer=TruncatedNormal())
        self.non_linearity = Activation('gelu')
        self.feedforward1 = Dense(in_channels, kernel_initializer=TruncatedNormal())
        self.dropout = Dropout(rate=drop_rate)

    def call(self, inputs, *args, **kwargs):
        x = self.feedforward0(inputs)
        x = self.non_linearity(x)
        if self.drop_rate > 0.:
            x = self.dropout(x)
        x = self.feedforward1(x)
        if self.drop_rate > 0.:
            x = self.dropout(x)

        return x


class WindowMultiHeadAttention(Layer):
    """
    Attention = Softmax((Query .* Key.T) / √d + B) .* Value
    build: relative position bias parameter building
    """

    def __init__(self, in_channels, n_heads, window_size=7, use_bias=False,
                 attention_drop_rate=0., projection_drop_rate=0.):
        super(WindowMultiHeadAttention, self).__init__()
        self.relative_pos_bias_index = None
        self.relative_pos_bias_table = None

        self.in_channels = in_channels
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self.n_heads = n_heads
        self.head_dimension = in_channels // n_heads
        self.scaler = self.head_dimension ** -.5

        self.query_key_value = Dense(3 * in_channels, use_bias=use_bias,
                                     kernel_initializer=TruncatedNormal(), bias_initializer=Zeros())
        self.attention_drop = Dropout(rate=attention_drop_rate)
        self.projection = Dense(in_channels, use_bias=use_bias,
                                kernel_initializer=TruncatedNormal(), bias_initializer=Zeros())
        self.projection_drop = Dropout(rate=projection_drop_rate)

    def build(self, input_shape):
        # relative position bias
        # shape: [(2 * height - 1) * (2 * width - 1), n_heads]
        self.relative_pos_bias_table = self.add_weight(
            shape=[(2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.n_heads],
            initializer=TruncatedNormal(stddev=.02),
            trainable=True, dtype=tf.float32
        )

        coor_height, coor_width = np.arange(self.window_size[0]), np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coor_height, coor_width, indexing='ij'))  # [2, height * width]
        coords_flatten = np.reshape(coords, newshape=(2, -1))
        # [2, height * width, 1] - [2, 1, height * width]
        relative_pos = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, h*w, h*w] broadcasting
        relative_pos = np.transpose(relative_pos, (1, 2, 0))  # [h*w, h*w, 2]
        # x = x + (window_size - 1), y = y + (window_size - 1)
        relative_pos[:, :, 0] += self.window_size[0] - 1
        relative_pos[:, :, 1] += self.window_size[1] - 1
        # x = x * (2 * window_size - 1)
        relative_pos[:, :, 0] *= 2 * self.window_size[0] - 1
        relative_pos_index = np.sum(relative_pos, axis=-1)

        # [height * width, height * width, 1]
        self.relative_pos_bias_index = tf.Variable(tf.convert_to_tensor(relative_pos_index),
                                                   trainable=False, dtype=tf.float64, )

    # FIXME <an argument of attention_mask/mask should be added>
    def call(self, inputs, mask=None, *args, **kwargs):
        """
        :param mask:
        :param inputs: shape: [batches * n_windows, height * width, embedding_dimension]
        :param args: //
        :param kwargs: //
        :return: attention matrix B
        """
        batches_nwindows, length, embed_dim = inputs.shape
        W_qkv = self.query_key_value(inputs)  # [batches * n_windows, height * width, 3 * embedding_dimension]
        W_qkv = tf.reshape(W_qkv, shape=[batches_nwindows, length, 3, self.n_heads, embed_dim // self.n_heads])
        # shape_transpose: [3, batches * n_windows, n_heads, height * width, dim_per_head]
        W_qkv = tf.transpose(W_qkv, perm=(2, 0, 3, 1, 4))
        W_query, W_key, W_value = W_qkv[0], W_qkv[1], W_qkv[2]

        # alpha.shape: [batches * n_windows, n_heads, height * width, height * width]
        alpha = tf.matmul(a=W_query, b=W_key, transpose_b=True) * self.scaler

        # relative_position_bias.shape: [(height * width) * (height * width), n_heads]
        relative_pos_bias = tf.gather(self.relative_pos_bias_table,
                                      tf.reshape(self.relative_pos_bias_index, [-1]))
        # relative_position_bias.shape: [height * width, height * width, n_heads]
        relative_pos_bias = tf.reshape(relative_pos_bias,
                                       shape=[self.window_size[0] * self.window_size[1],
                                              self.window_size[0] * self.window_size[1], -1])
        # relative_position_bias.shape: [n_heads, height * width, height * width]
        relative_pos_bias = tf.transpose(relative_pos_bias, perm=[2, 0, 1])
        # relative_position_bias.shape: [None, n_heads, height * width, height * width]
        relative_pos_bias = tf.expand_dims(relative_pos_bias, axis=0)
        # alpha.shape = [batches * windows, n_heads, height * width, height * width]
        alpha_bias = alpha + relative_pos_bias

        if mask is not None:
            # mask.shape: [n_windows, height * width, height * width]
            n_windows = mask.shape[0]
            # mask.shape = [1, n_windows, 1, height * width, height * width]
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0)
            alpha_bias = tf.reshape(alpha_bias, shape=[batches_nwindows // n_windows, n_windows,
                                                       self.n_heads, length, length])
            alpha_bias_mask = alpha_bias + mask
            alpha_bias = tf.reshape(alpha_bias_mask, shape=[-1, self.n_heads, length, length])

        alpha_prime = tf.nn.softmax(alpha_bias, axis=-1)
        # alpha_prime.shape = [batches * windows, n_heads, height * width, height * width]
        alpha_prime = self.attention_drop(alpha_prime)

        # x.shape: [batches * windows, n_heads, height * width, dimension_per_head]
        x = tf.matmul(alpha_prime, W_value)
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        x = tf.reshape(x, shape=(batches_nwindows, length, embed_dim))
        x = self.projection(x)
        x = self.projection_drop(x)

        return x


class SwinTransformerBlock(Layer):
    def __init__(self, in_channels, n_heads, window_size=7, shifted_size=0, use_bias=False,
                 attention_drop_rate=0., projection_drop_rate=0., mlp_drop_rate=0., drop_path_rate=0.):
        super(SwinTransformerBlock, self).__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.shifted_size = shifted_size
        self.window_size = window_size
        self.mlp_drop_rate = mlp_drop_rate
        assert self.shifted_size < self.window_size, "shifted size {} can not be greater than window size {}." \
            .format(self.shifted_size, self.window_size)

        self.layer_norm0 = LayerNormalization(epsilon=1e-5)
        self.window_attention = WindowMultiHeadAttention(in_channels, n_heads, window_size, use_bias=use_bias,
                                                         attention_drop_rate=attention_drop_rate,
                                                         projection_drop_rate=projection_drop_rate)
        self.drop_path = Dropout(rate=drop_path_rate, noise_shape=[None, 1, 1]) \
            if drop_path_rate > 0. else Activation('linear')
        self.layer_norm1 = LayerNormalization(epsilon=1e-5)
        self.feedforward = FeedForwardNetwork(in_channels, drop_rate=mlp_drop_rate)
        self.feature_add = Add()

    # FIXME <an argument of attention_mask should be added>
    def call(self, inputs, attention_mask=None, *args, **kwargs):
        # FIXME <the outside public declaration can be optimized>
        # self.height & self.width will be declared outside the class
        height, width = self.height, self.width
        batches, length, channels = inputs.shape
        assert length == height * width, "feature size and length does not match: {}/{}" \
            .format(height * width, length)

        x = self.layer_norm0(inputs)
        # shape: [batches, length, channels] -> [batches, height, width, channels]
        x = tf.reshape(x, shape=[batches, height, width, channels])

        # padding is needed if height and width can not be fully divided by window_size
        # FIXME <row_padder is decided by width or height? same as col_padder>
        row_padder = (self.window_size - width % self.window_size) % self.window_size  # height padding
        col_padder = (self.window_size - height % self.window_size) % self.window_size  # width padding
        if row_padder > 0 or col_padder > 0:
            padder = tf.constant([[0, 0], [0, row_padder], [0, col_padder], [0, 0]])
            x = tf.pad(x, padder)

        _, height_pad, width_pad, _ = x.shape

        if self.shifted_size > 0:
            shifted_x = tf.roll(x, shift=(-self.shifted_size, -self.shifted_size), axis=(1, 2))
        else:
            shifted_x, attention_mask = None, x

        # shifted_x.shape = [batches * windows, window_size, window_size, channels]
        window_x = window_partition(shifted_x, self.window_size)
        window_x = tf.reshape(window_x, shape=[-1, self.window_size ** 2, channels])
        # attention_x.shape: [batches * windows, window_size ** 2, channels]
        attention_x = self.window_attention(window_x, mask=attention_mask)
        # windows merging
        attention_x = tf.reshape(attention_x, shape=[-1, self.window_size, self.window_size, channels])
        shifted_x = window_shift_back(attention_x, self.window_size, height_pad, width_pad)
        if self.shifted_size > 0:
            x = tf.roll(shifted_x, shift=(self.shifted_size, self.shifted_size), axis=(1, 2))
        else:
            x = shifted_x

        x = tf.slice(x, begin=[0, 0, 0, 0], size=[batches, height, width, channels])
        x = tf.reshape(x, shape=[batches, -1, channels])
        x = self.drop_path(x)
        x = self.feature_add([x, inputs])
        x1 = self.layer_norm1(x)
        x1 = self.feedforward(x1)
        x1 = self.drop_path(x1)
        x1 = self.feature_add([x, x1])

        return x1


class BasicLayer(Layer):
    def __init__(self, in_channels, depth, n_heads, window_size=7, use_bias=True,
                 attention_drop_rate=0., projection_drop_rate=0., mlp_drop_rate=0.,
                 drop_path_rate: float or list = 0., down_sample=None):
        super(BasicLayer, self).__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.block_group = [
            SwinTransformerBlock(in_channels=in_channels, n_heads=n_heads, window_size=window_size,
                                 shifted_size=0 if (index % 2 == 0) else self.shift_size,
                                 use_bias=use_bias, attention_drop_rate=attention_drop_rate,
                                 projection_drop_rate=projection_drop_rate, mlp_drop_rate=mlp_drop_rate,
                                 drop_path_rate=drop_path_rate[index] if isinstance(drop_path_rate,
                                                                                    list) else drop_path_rate)
            for index in range(depth)
        ]

        if down_sample is not None:
            self.down_sample = down_sample(embed_dim=in_channels)
        else:
            self.down_sample = None

    def _create_mask(self, height, width, *kwargs):
        height_pad = int(np.ceil(height / self.window_size)) * self.window_size
        width_pad = int(np.ceil(width / self.window_size)) * self.window_size
        mask = np.zeros([1, height_pad, width_pad, 1])
        count = 0
        slices_height = [slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                         slice(-self.shift_size, None)]
        slices_width = [slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)]
        for h in slices_height:
            for w in slices_width:
                mask[:, h, w, :] = count
                count = count + 1

        mask = tf.convert_to_tensor(mask, dtype=tf.float32)  # [1, window_size, window_size, 1]
        mask_window = window_partition(mask, self.window_size)  # [n_windows, window_size, window_size, 1]
        mask_window = tf.reshape(mask_window, shape=[-1, self.window_size ** 2])  # [n_windows, window_size ** 2]
        # [n_windows, 1, window_size ** 2] - [n_windows, window_size ** 2, 1], broadcasting
        mask_window = tf.expand_dims(mask_window, 1) - tf.expand_dims(mask_window, 2)
        mask_window = tf.where(mask_window != 0, -100., mask_window)
        mask_window = tf.where(mask_window ==0, 0., mask_window)

        return mask_window

    # FIXME <arguments of height and width of feature maps should be added>
    def call(self, x, height=None, width=None, *args, **kwargs):
        assert height is not None and width is not None
        attention_mask = self._create_mask(height=height, width=width)
        for block in self.block_group:
            block.height, block.width = height, width
            x = block(x, attention_mask=attention_mask)

        if self.down_sample is not None:
            x = self.down_sample(x, height=height, width=width)
            height, width = (height + 1) // 2, (width + 1) // 2

        return x, height, width


class SwinTransformer(Model):
    def __init__(self, n_classes, patch_size, embed_dim=96, depth=(2, 2, 6, 2), n_heads=(3, 6, 12, 24),
                 window_size=7, use_bias=True, attention_drop_rate=0., projection_drop_rate=0.,
                 mlp_drop_rate=0., drop_path_rate=0., drop_rate=0., norm=LayerNormalization):
        super(SwinTransformer, self).__init__()
        self.n_classes = n_classes
        self.n_layers = len(depth)
        self.embed_dim = embed_dim
        self.mlp_drop_rate = mlp_drop_rate

        self.patch_embedding = PatchEmbedding(patch_size, embed_dim, layer_norm=norm)
        self.prior_dropout = Dropout(rate=drop_rate)

        drop_path_scheduler = [value for value in np.linspace(0, drop_path_rate, sum(depth))]

        self.stage_layers = []
        for ith_layer in range(self.n_layers):
            layer = BasicLayer(in_channels=int(embed_dim * (2 ** ith_layer)),
                               depth=depth[ith_layer], n_heads=n_heads[ith_layer],
                               window_size=window_size,
                               use_bias=use_bias,
                               attention_drop_rate=attention_drop_rate,
                               projection_drop_rate=projection_drop_rate,
                               mlp_drop_rate=mlp_drop_rate,
                               # FIXME <>
                               drop_path_rate=drop_path_scheduler[sum(depth[: ith_layer]): sum(depth[: ith_layer + 1])],
                               down_sample=PatchMerging if ith_layer < self.n_layers else None)
            self.stage_layers.append(layer)

        self.layer_normalization = norm(epsilon=1e-6)
        self.head = Dense(n_classes, kernel_initializer=TruncatedNormal(stddev=.02),
                          bias_initializer=Zeros())

    def call(self, inputs, training=None, mask=None):
        # sequential
        # patch embedding
        x, height, width = self.patch_embedding(inputs)
        x = self.prior_dropout(x)
        for layer in self.stage_layers:
            x, height, width = layer(x, height, width)

        x = self.layer_normalization(x)
        # x = GlobalAveragePooling2D(x)
        x = tf.reduce_min(x, axis=1)
        x = self.head(x)

        return x

