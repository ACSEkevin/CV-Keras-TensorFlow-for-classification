from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, DepthwiseConv2D, Layer
from keras.layers import BatchNormalization, Input, Flatten, GlobalAveragePooling2D, SeparableConv2D
from keras.models import Model, Sequential


def MobileConvBlock(input, n_filters, strides, depth_multiplier=1):
    # depth-wise conv
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same', depth_multiplier=depth_multiplier)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Point-wise conv
    x = Conv2D(n_filters, kernel_size=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def MobileNet_v1(input_shape, classes, drop_rate=.3, include_top=True):
    ins = Input(input_shape)
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(ins)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # stage 1
    x = MobileConvBlock(x, n_filters=64, strides=(1, 1))

    # stage 2
    x = MobileConvBlock(x, n_filters=128, strides=(2, 2))
    x = MobileConvBlock(x, n_filters=128, strides=(1, 1))

    # stage 3
    x = MobileConvBlock(x, n_filters=256, strides=(2, 2))
    x = MobileConvBlock(x, n_filters=256, strides=(1, 1))

    # stage 4
    x = MobileConvBlock(x, n_filters=512, strides=(2, 2))
    for _ in range(5):
        x = MobileConvBlock(x, n_filters=512, strides=(1, 1))

    # stage 5
    x = MobileConvBlock(x, n_filters=1024, strides=(2, 2))
    x = MobileConvBlock(x, n_filters=1024, strides=(1, 1))

    # top
    x = GlobalAveragePooling2D()(x)

    if include_top is True:
        x = Dense(1024, activation='relu')(x)
        x = Dropout(rate=drop_rate)(x)
        x = Dense(classes, activation='softmax')(x)

    model = Model(ins, x, name='MobileNetV1')

    return model


class MobileNet_test(Model):
    def __init__(self, input_shape, classes, depth_multiplier=1, drop_rate=0., include_top=False):
        super(MobileNet_test, self).__init__()
        self.input_layer = Input(input_shape)
        self.depth_multiplier = depth_multiplier
        self.include_top= include_top

        # self.mobile_conv = _MobileConvBlock
        self.beginning_block = Sequential([
            Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same'),
            BatchNormalization(),
            Activation('relu')])
        self.global_avgpool = GlobalAveragePooling2D()
        self.fully_connected = Dense(1024, activation='relu')
        self.drop_out = Dropout(rate=drop_rate)
        self.out_layer = Dense(classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.beginning_block(inputs)

        # stage 1
        x = _MobileConvBlock(n_filters=64, strides=(1, 1))(x)

        # stage 2
        x = _MobileConvBlock(n_filters=128, strides=(2, 2))(x)
        x = _MobileConvBlock(n_filters=128, strides=(1, 1))(x)

        # stage 3
        x = _MobileConvBlock(n_filters=256, strides=(2, 2))(x)
        x = _MobileConvBlock(n_filters=256, strides=(1, 1))(x)

        # stage 4
        x = _MobileConvBlock(n_filters=512, strides=(2, 2))(x)
        for _ in range(5):
            x = _MobileConvBlock(n_filters=512, strides=(1, 1))(x)

        # stage 5
        x = _MobileConvBlock(n_filters=1024, strides=(2, 2))(x)
        x = _MobileConvBlock(n_filters=1024, strides=(1, 1))(x)

        # stage - average pooling
        x = self.global_avgpool(x)

        if self.include_top is True:
            x = self.fully_connected(x)
            x = self.drop_out(x)
            x = self.out_layer(x)

        return x


class _MobileConvBlock(Layer):
    def __init__(self, n_filters, strides, depth_multiplier=1):
        super(_MobileConvBlock, self).__init__()
        # self.input_layer = inputs
        # depth-wise conv
        self.depthwise_conv = Sequential([
            DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same', depth_multiplier=depth_multiplier),
            BatchNormalization(),
            Activation('relu')
        ])

        # Point-wise conv
        self.pointwise_conv = Sequential([
            Conv2D(n_filters, kernel_size=(1, 1), padding='same'),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, inputs, *args, **kwargs):
        x = self.depthwise_conv(inputs)
        x = self.pointwise_conv(x)

        return x

