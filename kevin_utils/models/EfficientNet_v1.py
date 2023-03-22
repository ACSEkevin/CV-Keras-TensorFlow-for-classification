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


class Swish(Layer):
    def __init__(self):
        super(Swish, self).__init__()
        self.swish = swish

    def call(self, inputs, *args, **kwargs):
        return self.swish(inputs)


class SqueeseAndExcite(Layer):
    def __init__(self, inputs, scale_rate=.25):
        super(SqueeseAndExcite, self).__init__()
        self.in_channels = inputs.shape[-1] * scale_rate
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.fully_connected0 = Conv2D(int(inputs.shape[-1] * scale_rate), kernel_size=(1, 1), padding='same',
                                       kernel_initializer=CONV_KERNEL_INITIALIZER, activation='swish')
        # self.activation0 = Activation('swish')
        self.fully_connected1 = Conv2D(inputs.shape[-1], kernel_size=(1, 1), padding='same',
                                       kernel_initializer=CONV_KERNEL_INITIALIZER, activation='sigmoid')

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


def MBConv(inputs, in_channels, out_channels, kernel_size, strides=1, drop_connect_rate=.2, alpha=1.):
    """

    :param alpha:
    :param inputs:
    :param in_channels:
    :param out_channels:
    :param strides:
    :param expansion_rate:
    :param drop_connect_rate:
    :return:
    """
    # start_channels = int(inputs.shape[-1] * alpha)
    in_channels, out_channels = int(in_channels * alpha), int(out_channels * alpha)

    # stage - dimension adjustment
    x = Conv2D(in_channels, kernel_size=(1, 1), padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER)(inputs)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    # stage - depth-wise conv2d
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same',
                        depthwise_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    # squeeze and excite module
    x = SqueeseAndExcite(x)(x)
    # x = SqueezeAndExcite(x)

    # stage - dimension adjustment
    x = Conv2D(out_channels, kernel_size=(1, 1), padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization()(x)

    # whole block dropout & residual
    if strides == 1 and in_channels == out_channels:
        x = Dropout(rate=drop_connect_rate, noise_shape=(None, 1, 1, 1))(x) if drop_connect_rate > 0 else x
        x = Add()([inputs, x])

    return x


def BlockRepeater(n_repeat, inputs, in_channels, out_channels, kernel_size, strides=1, drop_connect_rate=.2, alpha=1.):
    x = MBConv(inputs, in_channels, out_channels, kernel_size, strides, drop_connect_rate=drop_connect_rate, alpha=alpha)
    for _ in range(1, n_repeat):
        x = MBConv(x, out_channels, out_channels, kernel_size, strides=1, drop_connect_rate=drop_connect_rate, alpha=alpha)

    return x


def EfficientNet(input_shape, classes, drop_connect_rate=.2, drop_date=0., include_top=True):
    """

    :param drop_connect_rate:
    :param input_shape:
    :param classes:
    :param drop_date:
    :param alpha:
    :param include_top:
    :return:
    """
    # input stage
    ins = Input(input_shape)

    # stage - beginning conv2d
    x = Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER)(ins)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    # stage 2 - 8
    x = BlockRepeater(1, x, in_channels=32, out_channels=16, strides=1, kernel_size=(3, 3),
                      drop_connect_rate=drop_connect_rate, alpha=1)
    x = BlockRepeater(2, x, in_channels=16, out_channels=24, strides=2, kernel_size=(3, 3),
                      drop_connect_rate=drop_connect_rate, alpha=6)
    x = BlockRepeater(2, x, in_channels=24, out_channels=40, strides=2, kernel_size=(5, 5),
                      drop_connect_rate=drop_connect_rate, alpha=6)
    x = BlockRepeater(3, x, in_channels=40, out_channels=80, strides=2, kernel_size=(3, 3),
                      drop_connect_rate=drop_connect_rate, alpha=6)
    x = BlockRepeater(3, x, in_channels=80, out_channels=112, strides=1, kernel_size=(5, 5),
                      drop_connect_rate=drop_connect_rate, alpha=6)
    x = BlockRepeater(4, x, in_channels=112, out_channels=192, strides=2, kernel_size=(5, 5),
                      drop_connect_rate=drop_connect_rate, alpha=6)
    x = BlockRepeater(1, x, in_channels=192, out_channels=320, strides=1, kernel_size=(3, 3),
                      drop_connect_rate=drop_connect_rate, alpha=6)

    # stage - ending conv
    x = Conv2D(1280, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    if include_top is True:
        x = GlobalAveragePooling2D()(x)
        x = Dropout(rate=drop_date)(x)
        x = Dense(classes, activation='softmax', kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)

    model = Model(ins, x, name='EfficientB0')
    return model
