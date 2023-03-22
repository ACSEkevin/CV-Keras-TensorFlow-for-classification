from keras.backend import int_shape
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, AveragePooling2D, DepthwiseConv2D
from keras.layers import BatchNormalization, Add, Input, Flatten, GlobalAveragePooling2D, ReLU
from keras.layers import Reshape, Layer, Multiply, Conv2DTranspose, UpSampling2D
from keras.activations import hard_sigmoid, swish, relu, silu
from keras.models import Model


def make_divisible(n_channel, divisor=8, main_channel=None):
    """
    To make sure that the input channel can be entirely divided
    :param n_channel:
    :param divisor:
    :param main_channel:
    :return:
    """
    main_channel = divisor if main_channel is None else main_channel
    channel = max(main_channel, int(n_channel + divisor / 2) // divisor * divisor)
    if channel < 0.9 * n_channel:
        channel = channel + divisor

    return channel


class HardSigmoid(Layer):
    def __init__(self):
        super(HardSigmoid, self).__init__()
        self.hard_sigmoid = hard_sigmoid

    def call(self, inputs, *args, **kwargs):
        return self.hard_sigmoid(inputs)


class HardSwish(Layer):
    def __init__(self):
        super(HardSwish, self).__init__()
        self.hard_sigmoid = hard_sigmoid

    def call(self, inputs, *args, **kwargs):
        return inputs * self.hard_sigmoid(inputs)


def SqueezeExcite(inputs, n_filters, shrink_ratio=.25, **kwargs):
    """
    Squeeze and excite module
    :param shrink_ratio:
    :param inputs:
    :param n_filters:
    :param kwargs:
    :return:
    """
    x = GlobalAveragePooling2D(keepdims=True)(inputs)

    x = Conv2D(make_divisible(n_filters * shrink_ratio), kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(n_filters, kernel_size=(1, 1), padding='same')(x)
    x = HardSigmoid()(x)

    x = Multiply()([inputs, x])

    return x


def InvertedResBlock(inputs, kernel_size, in_channels, out_channels, strides, se_block: bool, activation=None, alpha=1.0):
    """
    Inverted residual block
    :param strides:
    :param inputs:
    :param kernel_size:
    :param in_channels:
    :param out_channels:
    :param se_block:
    :param activation:
    :param alpha:
    :return:
    """
    start_channels = int(inputs.shape[-1] * alpha)
    in_channels, out_channels = int(in_channels * alpha), int(out_channels * alpha)
    activations = {'hard_swish': HardSwish, 'relu': ReLU,
                   'hs': HardSwish}
    if activation not in activations:
        raise AttributeError(f'argument activation has no key called {activation},'
                             f'please refer to these available choices: {activations.keys()}')
    else:
        NonLinearity = activations[activation]

    # stage - dimension adjustment
    x = Conv2D(in_channels, kernel_size=(1, 1), padding='same')(inputs)
    x = BatchNormalization(momentum=.99)(x)
    x = NonLinearity()(x)

    # stage - depth-wise conv2d
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                        padding='same' if strides == 1 else 'valid')(x)
    x = BatchNormalization(momentum=.99)(x)
    x = NonLinearity()(x)

    if se_block is True:
        x = SqueezeExcite(x, n_filters=in_channels)

    # stage - dimension adjustment
    x = Conv2D(out_channels, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization(momentum=.99)(x)

    # shortcut check
    x = Add()([inputs, x]) if strides == 1 and start_channels == out_channels else x

    return x


def MobileNet_v3_large(input_shape, classes, alpha=1., include_top=True):
    """
    MobileNet V3 Large version
    :param input_shape:
    :param classes:
    :param alpha:
    :param include_top:
    :return:
    """
    # stage - beginning conv
    ins = Input(shape=input_shape)
    x = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='same')(ins)
    x = BatchNormalization(momentum=.9)(x)
    x = HardSigmoid()(x)

    # stage - inverted block (b-neck)
    x = InvertedResBlock(x, (3, 3), 16, 16, 1, se_block=False, activation='relu', alpha=alpha)
    x = InvertedResBlock(x, (3, 3), 64, 24, 2, se_block=False, activation='relu', alpha=alpha)
    x = InvertedResBlock(x, (3, 3), 72, 24, 1, se_block=False, activation='relu', alpha=alpha)
    x = InvertedResBlock(x, (5, 5), 72, 40, 2, se_block=True, activation='relu', alpha=alpha)
    x = InvertedResBlock(x, (5, 5), 120, 40, 1, se_block=True, activation='relu', alpha=alpha)
    x = InvertedResBlock(x, (5, 5), 120, 40, 1, se_block=True, activation='relu', alpha=alpha)
    x = InvertedResBlock(x, (3, 3), 240, 80, 2, se_block=False, activation='hs', alpha=alpha)
    x = InvertedResBlock(x, (3, 3), 200, 80, 1, se_block=False, activation='hs', alpha=alpha)
    x = InvertedResBlock(x, (3, 3), 184, 80, 1, se_block=False, activation='hs', alpha=alpha)
    x = InvertedResBlock(x, (3, 3), 184, 80, 1, se_block=False, activation='hs', alpha=alpha)
    x = InvertedResBlock(x, (3, 3), 480, 112, 1, se_block=True, activation='hs', alpha=alpha)
    x = InvertedResBlock(x, (3, 3), 672, 112, 1, se_block=True, activation='hs', alpha=alpha)
    x = InvertedResBlock(x, (5, 5), 672, 160, 2, se_block=True, activation='hs', alpha=alpha)
    x = InvertedResBlock(x, (5, 5), 960, 160, 1, se_block=True, activation='hs', alpha=alpha)
    x = InvertedResBlock(x, (5, 5), 960, 160, 1, se_block=True, activation='hs', alpha=alpha)

    # stage - ending conv2d & global pool
    x = Conv2D(960, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization(momentum=.9)(x)
    x = HardSwish()(x)
    x = HardSigmoid()(x)
    x = GlobalAveragePooling2D(keepdims=True)(x)

    if include_top is True:
        x = Conv2D(1280, kernel_size=(1, 1), padding='same')(x)
        x = HardSwish()(x)

        x = Conv2D(classes, kernel_size=(1, 1), padding='same')(x)
        x = Flatten()(x)
        x = Activation('softmax')(x)

    model = Model(ins, x, name='MobileNetV3_large')

    return model
