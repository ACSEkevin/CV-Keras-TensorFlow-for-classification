from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, Layer, ZeroPadding2D
from keras.layers import BatchNormalization, Input, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Concatenate, Multiply
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


def DenseLayerBottleneck(inputs, growth_rate: int, drop_rate=0.):
    """

    :param inputs:
    :param growth_rate:
    :param drop_rate:
    :return:
    """
    # layer 1
    x = BatchNormalization(momentum=.99)(inputs)
    x = Activation('relu')(x)
    x = Conv2D(4 * growth_rate, kernel_size=(1, 1), strides=1, padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)

    # layer 2
    x = BatchNormalization(momentum=.99)(x)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, kernel_size=(3, 3), strides=1, padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)

    if drop_rate > 0.:
        x = Dropout(rate=drop_rate)(x)

    x = Concatenate()([inputs, x])

    return x


def DenseBlock(inputs, growth_rate, repeat: int, drop_rate=0.):
    """

    :param growth_rate:
    :param inputs:
    :param repeat:
    :param drop_rate:
    :return:
    """
    x = DenseLayerBottleneck(inputs, growth_rate, drop_rate)
    for count in range(1, repeat):
        # in_channels = x.shape[-1]
        x = DenseLayerBottleneck(x, growth_rate, drop_rate)

    return x


def TransitionBlock(inputs, reduction_rate):
    """

    :param inputs:
    :param reduction_rate:
    :return:
    """
    out_channels = int(inputs.shape[-1] * reduction_rate)
    x = BatchNormalization(momentum=.99, epsilon=1e-5)(inputs)
    x = Activation('relu')(x)
    x = Conv2D(out_channels, kernel_size=(1, 1), strides=1, padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)

    x = AveragePooling2D(pool_size=(2, 2))(x)

    return x


def DenseNet121(inputs_shape, classes, growth_rate=32, reduction_rate=.5, drop_rate=0., include_top=True):
    """

    :param reduction_rate:
    :param growth_rate:
    :param inputs_shape:
    :param classes:
    :param drop_rate:
    :param include_top:
    :return:
    """
    ins = Input(inputs_shape)

    # stage - beginning conv2d
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(ins)
    x = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding='same',
               kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = BatchNormalization(momentum=.99, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # stage - dense block x transition
    x = DenseBlock(x, growth_rate=growth_rate, repeat=6, drop_rate=drop_rate)
    x = TransitionBlock(x, reduction_rate=reduction_rate)
    x = DenseBlock(x, growth_rate=growth_rate, repeat=12, drop_rate=drop_rate)
    x = TransitionBlock(x, reduction_rate=reduction_rate)
    x = DenseBlock(x, growth_rate=growth_rate, repeat=24, drop_rate=drop_rate)
    x = TransitionBlock(x, reduction_rate=reduction_rate)
    x = DenseBlock(x, growth_rate=growth_rate, repeat=16, drop_rate=drop_rate)

    # stage - model head
    x = BatchNormalization(momentum=.99, epsilon=1e-5)(x)

    if include_top is True:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, activation='softmax')(x)

    model = Model(ins, x, name='DenseNet121')

    assert classes == int(classes)
    assert 0. <= reduction_rate <= 1.
    assert 0. <= drop_rate <= 1.
    return model


