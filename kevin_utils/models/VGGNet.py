from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, BatchNormalization


def VGG16(input_shape, classes=1000, include_top=True):
    ins = Input(shape=input_shape, name='input')
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv_0')(ins)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='maxpool_0')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='conv_2')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='conv_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='maxpool_1')(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv_4')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv_5')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv_6')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='maxpool_2')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv_7')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv_8')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv_9')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='maxpool_3')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv_10')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv_11')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv_12')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='maxpool_4')(x)

    if include_top is True:
        x = Flatten(name='flatten')(x)
        x = Dense(units=256, activation='relu', name='fc_0')(x)
        # x = Dropout(rate=0.4, name='fc0_dropout')(x)
        x = Dense(units=128, activation='relu', name='fc_1')(x)
        # Dropout(rate=0.5, name='fc1_dropout'),
        x = Dense(classes, activation='softmax', name='fc_2')(x)

    model = Model(ins, x, name='VGG-16')

    return model


def VGG19(input_shape, classes=1000, include_top=True):
    """
    :param input_shape: shape of input images
    :param classes: number of category in the label
    :param include_top: fully connected layer included or not
    :return: vgg-19 model
    """
    pass

