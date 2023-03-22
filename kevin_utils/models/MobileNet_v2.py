from keras.backend import int_shape
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, AveragePooling2D, DepthwiseConv2D
from keras.layers import BatchNormalization, Add, Input, Flatten, GlobalAveragePooling2D, ReLU, Reshape
from keras.activations import hard_sigmoid, swish, relu
from keras.models import Model
from tensorflow.python.ops.init_ops_v2 import glorot_uniform


def ReLU6(x):
    return relu(x, max_value=6)


def ConvBlock(input, n_filters, kernel_size, strides):
    x = Conv2D(n_filters, kernel_size=kernel_size, strides=strides, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation(ReLU6)(x)

    return x


def LinearBottleneck(input, n_filters, kernel_size, strides, expansion_rate=6, alpha=1., residual=False):
    # define input & output channels
    in_channel = int_shape(input)[-1] * expansion_rate
    out_channel = int(n_filters * alpha)

    # dimension ascension
    x = ConvBlock(input, in_channel, kernel_size=(1, 1), strides=(1, 1))

    # Depth-wise Conv block
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(ReLU6)(x)

    # Dimension reduction
    x = Conv2D(out_channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    if residual is True:
        x = Add()([x, input])

    return x


def InvertedResBlock(input, n_filters, kernel_size, strides, repeat, expansion_rate=6, alpha=1.):
    x = LinearBottleneck(input, n_filters, kernel_size, strides, expansion_rate, alpha)

    for _ in range(1, repeat):
        x = LinearBottleneck(x, n_filters, kernel_size, 1, expansion_rate, alpha, residual=True)

    return x


def MobileNet_v2(input_shape, classes, alpha=1.0, include_top=True):
    ins = Input(input_shape)

    # fundamental convolution
    x = ConvBlock(ins, n_filters=32, kernel_size=(3, 3), strides=(2, 2))

    # stage - Inverted block
    x = InvertedResBlock(x, 16, kernel_size=(3, 3), strides=(1, 1), repeat=1, expansion_rate=1)
    x = InvertedResBlock(x, 24, kernel_size=(3, 3), strides=(2, 2), repeat=2, alpha=alpha)
    x = InvertedResBlock(x, 32, kernel_size=(3, 3), strides=(2, 2), repeat=3, alpha=alpha)
    x = InvertedResBlock(x, 64, kernel_size=(3, 3), strides=(2, 2), repeat=4, alpha=alpha)
    x = InvertedResBlock(x, 96, kernel_size=(3, 3), strides=(1, 1), repeat=3, alpha=alpha)
    x = InvertedResBlock(x, 160, kernel_size=(3, 3), strides=(2, 2), repeat=3, alpha=alpha)
    x = InvertedResBlock(x, 320, kernel_size=(3, 3), strides=(1, 1), repeat=1, alpha=alpha)

    # stage - last conv block
    x = ConvBlock(x, n_filters=1280, kernel_size=(1, 1), strides=(1, 1))

    # stage - AvgPool and fully connected
    if include_top is True:
        x = GlobalAveragePooling2D()(x)
        x = Reshape(target_shape=(1, 1, 1280))(x)
        x = Dropout(rate=0.)(x)
        x = Conv2D(classes, (1, 1), padding='same')(x)
        x = Activation('softmax')(x)
    outs = Reshape(target_shape=(classes, ))(x)

    model = Model(ins, outs, name='MobileNetV2')

    return model

