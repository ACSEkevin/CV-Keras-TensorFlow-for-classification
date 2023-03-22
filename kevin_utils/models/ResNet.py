from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, AveragePooling2D
from keras.layers import BatchNormalization, Add, ZeroPadding2D, Input, Flatten
from keras.models import Model
from tensorflow.python.ops.init_ops_v2 import glorot_uniform


def ConvBnAc(input, n_filters, kernel_size, strides, padding, activtion=True):
    x = Conv2D(n_filters, kernel_size=kernel_size, strides=strides, padding=padding,
               kernel_initializer=glorot_uniform(seed=0))(input)
    x = BatchNormalization()(x)
    if activtion is True:
        x = Activation('relu')(x)
        return x
    else:
        return x


def ResBlock(input, n_filters, kernel_size):
    """
    Residual block combined with 3 conv2D layers with a shortcut
    """
    f1, f2, f3 = n_filters
    shortcut = input

    x = ConvBnAc(input, f1, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    x = ConvBnAc(x, f2, kernel_size, strides=(1, 1), padding='same')
    x = ConvBnAc(x, f3, kernel_size=(1, 1), strides=(1, 1), padding='valid')

    x = Add()([x, shortcut])

    return x


def ConvBlock(input, n_filters, kernel_size, stride):
    """
    Convolutional block combined with 3 conv2D layers
    adding a shortcut with one 1x1 conv2D so that input dimension matches outputs'
    """
    f1, f2, f3 = n_filters
    shortcut = input

    x = ConvBnAc(input, f1, kernel_size=(1, 1), strides=stride, padding='valid')
    x = ConvBnAc(x, f2, kernel_size=kernel_size, strides=(1, 1), padding='same')
    x = ConvBnAc(x, f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', activtion=False)

    shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=stride, padding='valid')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def ResNet_50(input_shape, classes):

    ins = Input(input_shape)
    # stage1 conv
    x = ZeroPadding2D(padding=(3, 3))(ins)

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='valid', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(strides=(2, 2))(x)

    # stage 2
    x = ConvBlock(x, n_filters=(64, 64, 256), kernel_size=(3, 3), stride=(1, 1))
    x = ResBlock(x, n_filters=(64, 64, 256), kernel_size=(3, 3))
    x = ResBlock(x, n_filters=(64, 64, 256), kernel_size=(3, 3))

    # stage 3
    x = ConvBlock(x, n_filters=(128, 128, 512), kernel_size=(3, 3), stride=(2, 2))
    x = ResBlock(x, n_filters=(128, 128, 512), kernel_size=(3, 3))
    x = ResBlock(x, n_filters=(128, 128, 512), kernel_size=(3, 3))
    x = ResBlock(x, n_filters=(128, 128, 512), kernel_size=(3, 3))

    # stage 4
    x = ConvBlock(x, n_filters=(256, 256, 1024), kernel_size=(3, 3), stride=(2, 2))
    x = ResBlock(x, n_filters=(256, 256, 1024), kernel_size=(3, 3))
    x = ResBlock(x, n_filters=(256, 256, 1024), kernel_size=(3, 3))
    x = ResBlock(x, n_filters=(256, 256, 1024), kernel_size=(3, 3))
    x = ResBlock(x, n_filters=(256, 256, 1024), kernel_size=(3, 3))
    x = ResBlock(x, n_filters=(256, 256, 1024), kernel_size=(3, 3))

    # stage 5
    x = ConvBlock(x, n_filters=(512, 512, 2048), kernel_size=(3, 3), stride=(1, 1))
    x = ResBlock(x, n_filters=(512, 512, 2048), kernel_size=(3, 3))
    x = ResBlock(x, n_filters=(512, 512, 2048), kernel_size=(3, 3))

    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)

    # fc
    x = Flatten()(x)
    outs = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(x)

    model = Model(ins, outs, name='ResNet50')
    return model


def ResNet_101(input_shape, classes, include_top=True):
    pass


def ResNet_152(input_shape, classes, include_top=True):
    pass

