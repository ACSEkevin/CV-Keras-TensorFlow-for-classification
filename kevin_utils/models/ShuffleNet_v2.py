from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, DepthwiseConv2D, Layer
from keras.layers import BatchNormalization, Input, Flatten, GlobalAveragePooling2D, Reshape
from keras.layers import Concatenate
from keras.models import Model, Sequential
import tensorflow as tf


def ShuffleNet_v2(input_shape, classes, version='1.0x', include_top=True):
    channels_list = list()
    version_list = {'0.5x': [24, 48, 96, 192], '1.0x': [24, 116, 232, 464], '2.0x': [24, 244, 488, 976]}

    if version not in version_list:
        raise ValueError(f'no argument dict named {version}, please refer to these available '
                         f'keys: {version_list.keys()}')
    else:
        channels_list = version_list[version]

    input_layer = Input(input_shape)
    x = ConvBlock(channels_list[0], kernel_size=(3, 3), strides=(2, 2))(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = ShuffleStages(x, n_outchannel=channels_list[1], n_blocks=4)
    x = ShuffleStages(x, n_outchannel=channels_list[2], n_blocks=8)
    x = ShuffleStages(x, n_outchannel=channels_list[3], n_blocks=4)
    x = ConvBlock(n_filters=1024 if version != '2.0x' else 2048,
                  kernel_size=(1, 1), strides=(1, 1))(x)
    x = GlobalAveragePooling2D()(x)

    if include_top is True:
        x = Dense(classes, activation='softmax')(x)

    model = Model(input_layer, x, name='ShuffleNetV2')
    return model


class ShuffleNet_v2_test(Model):
    """
    ShuffleNet v2
    :param input_shape: tensor, list or tuple, shape of input data
    :param classes: int, number of categorical class
    :param include_top: bool, whether remains the fully connected layers
    :return: keras.models.Model
    """

    def __init__(self, input_shape, classes, version='1.0x', include_top=True):
        super(ShuffleNet_v2_test, self).__init__()
        self.classes = classes
        self.include_top = include_top
        self.channels_list = list()

        if version == '1.0x':
            self.channels_list = [24, 48, 96, 192]
        elif version == '0.5x':
            self.channels_list = [24, 116, 232, 464]
        elif version == '2.0x':
            self.channels_list = [24, 244, 488, 976]

        self.input_layer = Input(input_shape)
        self.beginning_conv = ConvBlock(self.channels_list[0], kernel_size=(3, 3), strides=(2, 2))
        self.beginning_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        self.stage2 = ShuffleStage(n_outchannel=self.channels_list[1], n_blocks=4)
        self.stage3 = ShuffleStage(n_outchannel=self.channels_list[2], n_blocks=8)
        self.stage4 = ShuffleStage(n_outchannel=self.channels_list[3], n_blocks=4)
        self.ending_conv = ConvBlock(n_filters=1024 if version != '2.0x' else 2048,
                                     kernel_size=(1, 1), strides=(1, 1))
        self.global_avgpool = GlobalAveragePooling2D()
        self.out = Dense(classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.beginning_conv(inputs)
        x = self.beginning_pool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.ending_conv(x)
        x = self.global_avgpool(x)

        if self.include_top is True:
            x = self.out(x)

        return x


class ConvBlock(Layer):
    def __init__(self, n_filters, kernel_size, strides, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv_layer = Conv2D(n_filters, kernel_size, strides, padding, use_bias=False)
        self.batch_norm = BatchNormalization(momentum=.9)
        self.activation = Activation('relu')

    def call(self, inputs, *args, **kwargs):
        x = self.conv_layer(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x


class ChannelShuffle(Layer):
    def __init__(self, shape, groups: int = 2):
        super(ChannelShuffle, self).__init__()
        data_size, height, width, n_channels = shape
        assert n_channels % 2 == 0
        sub_channels = n_channels // groups

        self.reshape1 = Reshape(target_shape=(height, width, groups, sub_channels))
        self.reshape2 = Reshape(target_shape=(height, width, n_channels))

    def call(self, inputs, *args, **kwargs):
        x = self.reshape1(inputs)
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        x = self.reshape2(x)

        return x


class DepthWiseConvBlock(Layer):
    def __init__(self, kernel_size, strides, padding='same'):
        super(DepthWiseConvBlock, self).__init__()
        self.depth_wise_conv = DepthwiseConv2D(kernel_size=kernel_size,
                                               strides=strides,
                                               padding=padding)
        self.batch_norm = BatchNormalization(momentum=.9)

    def call(self, inputs, *args, **kwargs):
        x = self.depth_wise_conv(inputs)
        x = self.batch_norm(x)

        return x


class ChannelSplit(Layer):
    def __init__(self, split_batch: int = 2):
        super(ChannelSplit, self).__init__()
        self.split_batch = split_batch
        self.split = tf.split

    def call(self, inputs, *args, **kwargs):
        batch1, batch2 = tf.split(value=inputs, num_or_size_splits=self.split_batch, axis=-1)
        return batch1, batch2


def ShuffleBlock_1(inputs, n_outchannel, strides: int = 1, groups: int = 2):
    """

    :param inputs:
    :param n_outchannel:
    :param strides:
    :param groups:
    :return:
    """
    dim_reduce_channel = n_outchannel // 2

    branch1, branch2 = ChannelSplit()(inputs)
    branch1 = ConvBlock(n_filters=dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(branch1)
    branch1 = DepthWiseConvBlock(kernel_size=(3, 3), strides=strides)(branch1)
    branch1 = ConvBlock(n_filters=dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(branch1)

    x = Concatenate()([branch1, branch2])
    x = ChannelShuffle(x.shape, groups=groups)(x)

    assert n_outchannel % 2 == 0
    return x


def ShuffleBlock_2(inputs, n_outchannel, strides: int = 2, groups: int = 2):
    """
    ShuffleNet block 2
    :param inputs:
    :param n_outchannel:
    :param strides:
    :param groups:
    :return:
    """
    dim_reduce_channel = n_outchannel // 2

    branch1 = ConvBlock(n_filters=dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(inputs)
    branch1 = DepthWiseConvBlock(kernel_size=(3, 3), strides=strides)(branch1)
    branch1 = ConvBlock(n_filters=dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(branch1)

    # branch 2
    branch2 = DepthWiseConvBlock(kernel_size=(3, 3), strides=strides)(inputs)
    branch2 = ConvBlock(n_filters=dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(branch2)

    # concat
    x = Concatenate()([branch1, branch2])
    x = ChannelShuffle(x.shape, groups=groups)(x)

    assert strides == 2
    assert n_outchannel % 2 == 0
    return x


def ShuffleStages(inputs, n_outchannel: int, n_blocks: int):
    x = ShuffleBlock_2(inputs, n_outchannel=n_outchannel)
    for index in range(1, n_blocks):
        x = ShuffleBlock_1(x, n_outchannel=n_outchannel, strides=1)

    return x


class ShuffleBlock1(Layer):
    def __init__(self, n_outchannel, strides, groups: int = 3):
        super(ShuffleBlock1, self).__init__()
        self.n_out_channel = n_outchannel
        self.dim_reduce_channel = n_outchannel // 2
        self.strides = strides
        self.groups = groups

        self.channel_split = ChannelSplit()
        self.conv_block = ConvBlock
        self.depth_wise_conv = DepthWiseConvBlock
        self.concat = Concatenate()
        self.channel_shuffle = ChannelShuffle

        assert n_outchannel % 2 == 0

    def call(self, inputs, *args, **kwargs):
        branch1, branch2 = self.channel_split(inputs)
        # branch1 = self.conv_block(n_filters=self.dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(branch1)
        branch1 = ConvBlock(n_filters=self.dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(branch1)
        # branch1 = self.depth_wise_conv(kernel_size=(3, 3), strides=self.strides)(branch1)
        branch1 = DepthWiseConvBlock(kernel_size=(3, 3), strides=self.strides)(branch1)
        # branch1 = self.conv_block(n_filters=self.dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(branch1)
        branch1 = ConvBlock(n_filters=self.dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(branch1)

        x = self.concat([branch1, branch2])
        # x = self.channel_shuffle(x.shape, groups=self.groups)(x)
        x = ChannelShuffle(x.shape, groups=self.groups)(x)
        return x


class ShuffleBlock2(Layer):
    def __init__(self, n_outchannel, strides: int = 2, groups: int = 3):
        super(ShuffleBlock2, self).__init__()
        self.n_out_channel = n_outchannel
        self.dim_reduce_channel = n_outchannel // 2
        self.strides = strides
        self.groups = groups

        self.conv_block = ConvBlock
        self.depth_wise_conv = DepthWiseConvBlock
        self.concat = Concatenate()
        self.channel_shuffle = ChannelShuffle

        assert strides == 2
        assert n_outchannel % 2 == 0

    def call(self, inputs, *args, **kwargs):
        # branch 1
        # branch1 = self.conv_block(n_filters=self.dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(inputs)
        branch1 = ConvBlock(n_filters=self.dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(inputs)
        # branch1 = self.depth_wise_conv(kernel_size=(3, 3), strides=self.strides)(branch1)
        branch1 = DepthWiseConvBlock(kernel_size=(3, 3), strides=self.strides)(branch1)
        # branch1 = self.conv_block(n_filters=self.dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(branch1)
        branch1 = ConvBlock(n_filters=self.dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(branch1)

        # branch 2
        # branch2 = self.depth_wise_conv(kernel_size=(3, 3), strides=self.strides)(inputs)
        branch2 = DepthWiseConvBlock(kernel_size=(3, 3), strides=self.strides)(inputs)
        # branch2 = self.conv_block(n_filters=self.dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(branch2)
        branch2 = ConvBlock(n_filters=self.dim_reduce_channel, kernel_size=(1, 1), strides=(1, 1))(branch2)

        # concat
        x = self.concat([branch1, branch2])
        x = self.channel_shuffle(x.shape, groups=self.groups)(x)

        return x


class ShuffleStage(Layer):
    def __init__(self, n_outchannel: int, n_blocks: int):
        super(ShuffleStage, self).__init__()
        self.n_out_channel = n_outchannel
        self.n_block = n_blocks

    def call(self, inputs, *args, **kwargs):
        x = ShuffleBlock2(n_outchannel=self.n_out_channel)(inputs)
        for index in range(1, self.n_block):
            x = ShuffleBlock1(n_outchannel=self.n_out_channel, strides=1)(x)

        return x
