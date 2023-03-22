from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, AveragePooling2D
from keras.layers import BatchNormalization, ZeroPadding2D, Input, Flatten, Concatenate
from keras.models import Model, load_model, Sequential
from keras.layers import Layer, GlobalAveragePooling2D


def GoogLeNet_v1(input_shape, classes, drop_rate=.5, batch_norm=False, include_top=False, aux_output=False):
    """
    GoogLeNet (inception_v1) with 3 auxiliary softmax outputs
    :param input_shape: shape of input img, can be a tensor, tuple or a list
    :param classes: number of output categories
    :param drop_rate: drop out rate in fully connected layers of auxiliary output
    :param batch_norm: adding batch normalization layer at the beginning stage, default: false
    :param include_top: remain fully connected later or not, default: False
    :param aux_output: network with auxiliary outputs or not, output three models predictions if True, default: False
    :return: Keras.models.Model
    """
    # stage - input

    global aux_out1, aux_out2
    ins = Input(input_shape, dtype='float32')

    # stage - beginning conv
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(ins)
    if batch_norm is True:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # stage - beginning conv2
    x = Conv2D(192, kernel_size=(3, 3), padding='same')(x)
    if batch_norm is True:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # stage - inception 3a 3b
    x = InceptionBlock(64, 96, 128, 16, 32, 32)(x)
    x = InceptionBlock(128, 128, 192, 32, 96, 64)(x)

    # stage - inception3 max pool
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # stage - 4a 4b 4c 4d 4e
    x = InceptionBlock(192, 96, 208, 16, 48, 64)(x)
    if aux_output is True:
        aux_out1 = InceptionAuxiliary(classes, drop_rate=drop_rate)(x)
    x = InceptionBlock(160, 112, 224, 24, 64, 64)(x)
    x = InceptionBlock(128, 128, 256, 24, 64, 64)(x)
    x = InceptionBlock(112, 144, 288, 32, 64, 64)(x)
    if aux_output is True:
        aux_out2 = InceptionAuxiliary(classes, drop_rate=drop_rate)(x)
    x = InceptionBlock(256, 160, 320, 32, 128, 128)(x)

    # stage - inception4 max pool
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # stage - inception 5a 5b
    x = InceptionBlock(256, 160, 320, 32, 128, 128)(x)
    x = InceptionBlock(384, 192, 184, 48, 128, 128)(x)

    # stage - ending
    x = GlobalAveragePooling2D(keepdims=True)(x)

    if include_top is True:
        x = Flatten()(x)
        x = Dropout(rate=0.3)(x)
        x = Dense(classes, activation='softmax')(x)

    if aux_output is True:
        model = Model(ins, [aux_out1, aux_out2, x], name='GoogLeNet_auxiliary')
    else:
        model = Model(ins, x, name='GoogLeNet')

    return model


class InceptionBlock(Layer):
    """
    Inception block with dimension reduction
    Include: dimension adjust conv1x1, conv3x3, conv5x5 and max pool 3x3
    each branch haas 1x1 conv to implement dimension adjustment
    Inception block is established by concatenation of four branches together
    """
    def __init__(self,ch_dim_rdc, ch_3x3_reduce, ch_3x3, ch_5x5_reduce, ch_5x5, pool_proj, **kwargs):
        super(InceptionBlock, self).__init__(**kwargs)
        self.dim_reduction = Conv2D(ch_dim_rdc, kernel_size=(1, 1))

        self.branch_3x3 = Sequential([
            Conv2D(ch_3x3_reduce, kernel_size=(1, 1), activation='relu'),
            Conv2D(ch_3x3, kernel_size=(3, 3), padding='same', activation='relu')
        ])
        self.branch_5x5 = Sequential([
            Conv2D(ch_5x5_reduce, kernel_size=(1, 1), activation='relu'),
            Conv2D(ch_5x5, kernel_size=(5, 5),padding='same', activation='relu')
        ])

        self.branch_max_pool = Sequential([
            MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'),
            Conv2D(pool_proj, kernel_size=(1, 1), activation='relu')
        ])

        self.batch_norm = BatchNormalization()
        self.concat = Concatenate()

    def call(self, inputs, *args, **kwargs):
        outs = self.concat([self.dim_reduction(inputs), self.branch_3x3(inputs),
                            self.branch_5x5(inputs), self.branch_max_pool(inputs)])

        return outs


class InceptionAuxiliary(Layer):
    """
    Inception Auxiliary outputs are introduced by Google AI team with outputs from inception block '4b' and '4e'
    Architecture: max pool 3x3, conv1x1, to fully connected layers finally softmax activation
    """
    def __init__(self, classes, drop_rate=0.5, **kwargs):
        super(InceptionAuxiliary, self).__init__(**kwargs)
        self.avgpool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))
        self.conv1x1 = Conv2D(128, kernel_size=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.fullyconnected = Dense(1024, activation='relu')
        self.drop_out_flatten = Dropout(rate=drop_rate)
        self.drop_out = Dropout(rate=drop_rate)
        self.softmax_out = Dense(classes, activation='softmax')

    def call(self, inputs, *args, **kwargs):
        x = self.avgpool(inputs)
        x = self.flatten(x)
        x = self.drop_out_flatten(x)
        x = self.fullyconnected(x)
        x = self.drop_out(x)
        x = self.softmax_out(x)

        return x
