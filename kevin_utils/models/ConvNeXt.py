from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, Layer, Input
from keras.layers import LayerNormalization, DepthwiseConv2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.initializers.initializers_v2 import Constant
import numpy as np
import tensorflow as tf

KERNEL_INITIALIZER = {
    "class_name": "TruncatedNormal",
    "config": {
        "stddev": 0.2
    }
}

BIAS_INITIALIZER = "Zeros"


class PatchifyStemBlock(Layer):
    def __init__(self, n_filters):
        super(PatchifyStemBlock, self).__init__()
        self.down_sample = Conv2D(n_filters, kernel_size=(4, 4), strides=4,
                                  kernel_initializer=KERNEL_INITIALIZER, bias_initializer=BIAS_INITIALIZER)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, *args, **kwargs):
        x = self.down_sample(inputs)
        x = self.layer_norm(x)

        return x


class DownSamplingBlock(Layer):
    def __init__(self, n_filters):
        super(DownSamplingBlock, self).__init__()
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.down_sample = Conv2D(n_filters, kernel_size=(2, 2), strides=2, padding='valid',
                                  kernel_initializer=KERNEL_INITIALIZER, bias_initializer=BIAS_INITIALIZER)

    def call(self, inputs, *args, **kwargs):
        x = self.layer_norm(inputs)
        x = self.down_sample(x)

        return x


class ConvNeXtBlock(Layer):
    def __init__(self, out_channels, drop_path_rate=0., layer_scale_initializer=1e-6):
        super(ConvNeXtBlock, self).__init__()
        self.drop_path_rate = drop_path_rate
        self.layer_scale_initializer = layer_scale_initializer

        self.depth_wise = DepthwiseConv2D(kernel_size=(7, 7), strides=1, padding='same',
                                          kernel_initializer=KERNEL_INITIALIZER, bias_initializer=BIAS_INITIALIZER)
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.point_wise0 = Conv2D(out_channels * 4, kernel_size=(1, 1), strides=1, padding='same',
                                  kernel_initializer=KERNEL_INITIALIZER, bias_initializer=BIAS_INITIALIZER)
        self.non_linearity = Activation('gelu')
        self.point_wise1 = Conv2D(out_channels, kernel_size=(1, 1), strides=1, padding='same',
                                  kernel_initializer=KERNEL_INITIALIZER, bias_initializer=BIAS_INITIALIZER)
        self.drop_path = Dropout(rate=drop_path_rate, noise_shape=(None, 1, 1, 1))

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=input_shape[-1], trainable=True, dtype=tf.float32,
                                     initializer=Constant(value=self.layer_scale_initializer))

    def call(self, inputs, *args, **kwargs):
        x = self.depth_wise(inputs)
        x = self.layer_norm(x)
        x = self.point_wise0(x)
        x = self.non_linearity(x)
        x = self.point_wise1(x)
        if self.layer_scale_initializer > 0.:
            x = x * self.gamma
        if self.drop_path_rate > 0.:
            x = self.drop_path(x)

        return x


def ConvNeXt(input_shape, classes, version='tiny', drop_path_rate=0.,
             layer_scale_initializer=1e-6, include_top=True):
    """

    :param input_shape:
    :param classes:
    :param version:
    :param drop_path_rate:
    :param layer_scale_initializer:
    :param include_top:
    :return:
    """
    channels_dict = {'tiny': [96, 192, 284, 768], 'small': [96, 192, 284, 768],
                     'base': [128, 256, 512, 1024], 'large': [192, 384, 768, 1536],
                     'extra_large': [256, 512, 1024, 2048]}

    block_dict = {'tiny': [3, 3, 9, 3], 'small': [3, 3, 27, 3], 'base': [3, 3, 27, 3],
                  'large': [3, 3, 27, 3], 'extra_large': [3, 3, 27, 3]}

    if version not in channels_dict:
        raise KeyError(f'No key word named "{version}", available key words are: {channels_dict.keys()}')
    else:
        channels = channels_dict[version]
        blocks = block_dict[version]

    drop_path_rates = np.linspace(0, drop_path_rate, sum(blocks))

    # inputs
    ins = Input(input_shape)
    # patchify down sampling
    x = PatchifyStemBlock(96)(ins)

    # stage 1
    stage1_blocks = [ConvNeXtBlock(channels[0], drop_path_rates[index],
                                   layer_scale_initializer=layer_scale_initializer) for index in range(blocks[0])]
    for block in stage1_blocks:
        x = block(x)

    # stage 2
    x = DownSamplingBlock(channels[1])(x)
    stage2_blocks = [ConvNeXtBlock(channels[1], drop_path_rates[index],
                                   layer_scale_initializer=layer_scale_initializer) for index in range(blocks[1])]
    for block in stage2_blocks:
        x = block(x)

    # stage 3
    x = DownSamplingBlock(channels[2])(x)
    stage3_blocks = [ConvNeXtBlock(channels[2], drop_path_rates[index],
                                   layer_scale_initializer=layer_scale_initializer) for index in range(blocks[2])]
    for block in stage3_blocks:
        x = block(x)

    # stage 4
    x = DownSamplingBlock(channels[3])(x)
    stage4_blocks = [ConvNeXtBlock(channels[3], drop_path_rates[index],
                                   layer_scale_initializer=layer_scale_initializer) for index in range(blocks[3])]
    for block in stage4_blocks:
        x = block(x)

    if include_top is True:
        x = GlobalAveragePooling2D()(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dense(classes, kernel_initializer=KERNEL_INITIALIZER, bias_initializer=BIAS_INITIALIZER)(x)
        x = Activation('softmax')(x)

    model = Model(ins, x, name=f'ConvNeXt_{version}')

    assert 0. <= drop_path_rate <= 1.
    return model
