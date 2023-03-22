from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, DepthwiseConv2D, Layer
from keras.layers import BatchNormalization, Input, Flatten, GlobalAveragePooling2D, Reshape
from keras.layers import Concatenate, Multiply, Add
from keras.activations import swish
from keras.models import Model, Sequential

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'},
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3,
        'mode': 'fan_out',
        'distribution': 'uniform'},
}


class SiLU(Layer):
    def __init__(self):
        super(SiLU, self).__init__()
        self.silu = swish

    def call(self, inputs, *args, **kwargs):
        return self.silu(inputs)


class _SqueezeAndExcite_(Layer):
    def __init__(self, n_filters, scaler=.25):
        super(_SqueezeAndExcite_, self).__init__()
        self.scale_channels = max(1, int(n_filters * scaler))
        self.out_channels = n_filters

        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.fully_connected0 = Conv2D(self.scale_channels, kernel_size=(1, 1), padding='same',
                                       activation='swish', kernel_initializer=CONV_KERNEL_INITIALIZER)
        self.fully_connected1 = Conv2D(self.out_channels, kernel_size=(1, 1), padding='same',
                                       activation='sigmoid', kernel_initializer=CONV_KERNEL_INITIALIZER)
        self.feature_mul = Multiply()

    def call(self, inputs, *args, **kwargs):
        x = self.global_avg_pool(inputs)
        x = self.fully_connected0(x)
        x = self.fully_connected1(x)
        x = self.feature_mul([inputs, x])

        return x


def SqueezeAndExcite(inputs, scaler=.25):
    """

    :param inputs:
    :param scaler:
    :return:
    """
    in_channels = max(1, int(inputs.shape[-1] * scaler))

    x = GlobalAveragePooling2D(keepdims=True)(inputs)

    x = Conv2D(in_channels, kernel_size=(1, 1), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER, activation='swish')(x)
    x = Conv2D(inputs.shape[-1], kernel_size=(1, 1), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER, activation='sigmoid')(x)

    x = Multiply()([inputs, x])

    return x


class _MBConvBlock_(Layer):
    def __init__(self, in_channels, out_channels, strides, expansion_rate=4., drop_connect_rate=0.):
        super(_MBConvBlock_, self).__init__()
        self.in_channels = int(in_channels * expansion_rate)
        self.out_channels = out_channels
        self.short_cut = (in_channels == out_channels and strides == 1)
        self.drop_connect_rate = drop_connect_rate

        self.dim_expansion = Conv2D(self.in_channels, kernel_size=(1, 1), padding='same',
                                    kernel_initializer=CONV_KERNEL_INITIALIZER)
        self.batch_norm0 = BatchNormalization(momentum=.9, epsilon=1e-3)
        self.activation0 = Activation('swish')

        self.depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same',
                                              kernel_initializer=CONV_KERNEL_INITIALIZER)
        self.batch_norm1 = BatchNormalization(momentum=.9, epsilon=1e-3)
        self.activation1 = Activation('swish')

        self.squeeze_excite = SqueezeAndExcite

        self.dim_reduction = Conv2D(self.out_channels, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                    kernel_initializer=CONV_KERNEL_INITIALIZER)
        self.batch_norm2 = BatchNormalization(momentum=.9, epsilon=1e-3)
        self.drop_out = Dropout(rate=self.drop_connect_rate, noise_shape=(None, 1, 1, 1))
        self.feature_add = Add()

        assert expansion_rate != 1
        assert strides <= 2

    def call(self, inputs, *args, **kwargs):
        # stage - dimension adjustment
        x = self.dim_expansion(inputs)
        x = self.batch_norm0(x)
        x = self.activation0(x)

        # stage - depth-wise conv2d
        x = self.depthwise_conv(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)

        # stage - attention module
        x = self.squeeze_excite(x)

        # stage - dimension adjustment
        x = self.dim_reduction(x)
        x = self.batch_norm2(x)

        if self.short_cut is True:
            if self.drop_connect_rate > 0.:
                x = self.drop_out(x)

            x = self.feature_add([inputs, x])

        return x


def MBConvBlock(inputs, in_channels, out_channels, strides, expansion_rate=4., drop_connect_rate=0.):
    """

    :param inputs:
    :param in_channels:
    :param out_channels:
    :param strides:
    :param expansion_rate:
    :param drop_connect_rate:
    :return:
    """
    short_cut = (in_channels == out_channels and strides == 1)
    in_channels = int(in_channels * expansion_rate)

    # stage - dimension adjustment
    x = Conv2D(in_channels, kernel_size=(1, 1), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER)(inputs)
    x = BatchNormalization(momentum=.9, epsilon=1e-3)(x)
    x = Activation('swish')(x)

    # stage - depth-wise conv2d
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same',
                        kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(momentum=.9, epsilon=1e-3)(x)
    x = Activation('swish')(x)

    # stage - squeeze and excite
    x = SqueezeAndExcite(x)

    # stage - dimension adjustment
    x = Conv2D(out_channels, kernel_size=(1, 1), strides=(1, 1), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(momentum=.9, epsilon=1e-3)(x)

    if short_cut is True:
        if drop_connect_rate > 0.:
            x = Dropout(rate=drop_connect_rate, noise_shape=(None, 1, 1, 1))(x)

        x = Add()([inputs, x])

    return x


class _FusedMBConvBlock_(Layer):
    def __init__(self, in_channels, out_channels, strides, expansion_rate=4., drop_connect_rate=0.):
        super(_FusedMBConvBlock_, self).__init__()
        self.in_channels = int(in_channels * expansion_rate)
        self.out_channels = out_channels
        self.short_cut = (in_channels == out_channels and strides == 1)
        self.drop_connect_rate = drop_connect_rate

        self.expand_conv = Conv2D(self.in_channels, kernel_size=(3, 3), strides=strides, padding='same',
                                  kernel_initializer=CONV_KERNEL_INITIALIZER)
        self.batch_norm0 = BatchNormalization(momentum=.9, epsilon=1e-3)
        self.activation = Activation('swish')

        self.project_conv = Conv2D(self.out_channels, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                   kernel_initializer=CONV_KERNEL_INITIALIZER)
        self.batch_norm1 = BatchNormalization(momentum=.9, epsilon=1e-3)

        self.drop_out = Dropout(rate=self.drop_connect_rate, noise_shape=(None, 1, 1, 1))
        self.feature_add = Add()

    def call(self, inputs, *args, **kwargs):
        # stage - expand conv2d
        x = self.expand_conv(inputs)
        x = self.batch_norm0(x)
        x = self.activation(x)

        # stage - project conv2d
        x = self.project_conv(x)
        x = self.batch_norm1(x)

        if self.short_cut is True:
            if self.drop_connect_rate > 0.:
                x = self.drop_out(x)

            x = self.feature_add([inputs, x])

        return x


def FusedMBConvBlock(inputs, in_channels, out_channels, strides, expansion_rate=4., drop_connect_rate=0.):
    """

    :param inputs:
    :param in_channels:
    :param out_channels:
    :param strides:
    :param expansion_rate:
    :param drop_connect_rate:
    :return:
    """
    short_cut = (in_channels == out_channels and strides == 1)
    in_channels = int(in_channels * expansion_rate)

    # stage - fused conv block
    x = Conv2D(in_channels, kernel_size=(3, 3), strides=strides, padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER)(inputs)
    x = BatchNormalization(momentum=.9, epsilon=1e-3)(x)
    x = Activation('swish')(x)

    # stage - dimension adjustment
    x = Conv2D(out_channels, kernel_size=(1, 1), strides=(1, 1), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(momentum=.9, epsilon=1e-3)(x)

    if short_cut is True:
        if drop_connect_rate > 0.:
            x = Dropout(rate=drop_connect_rate, noise_shape=(None, 1, 1, 1))(x)
        x = Add()([inputs, x])

    return x


def MBRepeater(inputs, repeat, in_channels, out_channels, strides, expansion_rate=4., drop_connect_rate=0.):
    """

    :param repeat:
    :param inputs:
    :param in_channels:
    :param out_channels:
    :param strides:
    :param expansion_rate:
    :param drop_connect_rate:
    :return:
    """
    x = MBConvBlock(inputs, in_channels, out_channels, strides, expansion_rate, drop_connect_rate)
    for _ in range(1, repeat):
        x = MBConvBlock(x, out_channels, out_channels, 1, expansion_rate, drop_connect_rate)

    assert repeat == int(repeat)
    assert expansion_rate != 1
    return x


def FusedMBRepeater(inputs, repeat, in_channels, out_channels, strides, expansion_rate=4., drop_connect_rate=0.):
    """

    :param inputs:
    :param repeat:
    :param in_channels:
    :param out_channels:
    :param strides:
    :param expansion_rate:
    :param drop_connect_rate:
    :return:
    """
    x = FusedMBConvBlock(inputs, in_channels, out_channels, strides, expansion_rate, drop_connect_rate)
    for _ in range(1, repeat):
        x = FusedMBConvBlock(x, out_channels, out_channels, 1, expansion_rate, drop_connect_rate)

    assert repeat == int(repeat)
    assert expansion_rate != 1
    return x


def EfficientNet_v2(input_shape, classes, drop_connect_rate=0., drop_rate=0., include_top=True):
    """

    :param input_shape:
    :param classes:
    :param drop_connect_rate:
    :param drop_rate:
    :param include_top:
    :return:
    """
    ins = Input(input_shape)

    # stage - beginning conv2d
    x = Conv2D(24, kernel_size=(3, 3), strides=(2, 2), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER)(ins)
    x = BatchNormalization(momentum=.9, epsilon=1e-3)(x)
    x = Activation('swish')(x)

    # stage - Fused mobile block
    x = FusedMBRepeater(x, 2, 24, 24, 1, expansion_rate=4., drop_connect_rate=drop_connect_rate)
    x = FusedMBRepeater(x, 4, 24, 48, 2, expansion_rate=4., drop_connect_rate=drop_connect_rate)
    x = FusedMBRepeater(x, 4, 48, 64, 2, expansion_rate=4., drop_connect_rate=drop_connect_rate)

    # stage - mobile block
    x = MBRepeater(x, 6, 64, 128, 2, expansion_rate=4., drop_connect_rate=drop_connect_rate)
    x = MBRepeater(x, 9, 128, 160, 1, expansion_rate=6., drop_connect_rate=drop_connect_rate)
    x = MBRepeater(x, 15, 160, 272, 2, expansion_rate=6., drop_connect_rate=drop_connect_rate)

    # stage - ending conv2d
    x = Conv2D(1792, kernel_size=(1, 1), strides=(1, 1), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)

    if include_top is True:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, activation='softmax', kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
        x = Dropout(rate=drop_rate)(x)

    model = Model(ins, x, name='EfficientNetV2S')

    assert classes == int(classes)
    return model
