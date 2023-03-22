from keras.models import Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, Activation, AvgPool2D, Input, AveragePooling2D
from keras.layers import BatchNormalization, MaxPooling2D


def LeNet_5(input_shape, classes, include_top=True):
    ins = Input(shape=input_shape)
    x = Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', name='conv_0')(ins)
    x = Activation('sigmoid', name='activation_0')(x)
    x = AvgPool2D(pool_size=(2, 2), name='avgpool_0')(x)
    x = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', name='conv_1')(x)
    x = Activation('sigmoid', name='activation_1')(x)
    x = AvgPool2D(pool_size=(2, 2), name='avgpool_1')(x)

    if include_top is True:
        x = Flatten(name='flt_0')(x)
        x = Dense(units=120, activation='sigmoid', name='fc_0')(x)
        x = Dense(units=84, activation='sigmoid', name='fc_1')(x)
        x = Dense(units=classes, activation='softmax', name='output')(x)

    model = Model(ins, x, name='LeNet-5')

    return model


class LeNet_5_(Model):
    def __init__(self, input_shape, classes, drop_rate=0., include_top=True):
        super(LeNet_5_, self).__init__()
        self.classes = classes
        self.include_top = include_top

        self.input_layer = Input(input_shape)
        self.conv_layer0 = Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid')
        self.activation0 = Activation('sigmoid')
        self.avg_pool0 = AveragePooling2D()

        self.conv_layer1 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid')
        self.activation1 = Activation('sigmoid')
        self.avg_pool1 = AveragePooling2D()

        self.flatten = Flatten()
        self.fully_connected0 = Dense(units=120, activation='sigmoid')
        self.dropout0 = Dropout(rate=drop_rate)
        self.fully_connected1 = Dense(units=84, activation='sigmoid')
        self.dropout1 = Dropout(rate=drop_rate)
        self.out_layer = Dense(units=self.classes, activation='softmax')

    def call(self, input, training=None, mask=None):
        x = self.conv_layer0(input)
        x = self.activation0(x)
        x = self.avg_pool0(x)

        x = self.conv_layer1(x)
        x = self.activation1(x)
        x = self.avg_pool1(x)

        if self.include_top is True:
            x = self.flatten(x)
            x = self.fully_connected0(x)
            x = self.dropout0(x)
            x = self.fully_connected1(x)
            x = self.dropout1(x)
            x = self.out_layer(x)

        return x


class LeNet_v2(Model):
    def __init__(self,input_shape, classes, drop_rate=0., include_top=True):
        super(LeNet_v2, self).__init__()
        self.classes = classes
        self.include_top = include_top
        self.conv_block = self._conv_block

        self.input_layer = Input(input_shape)
        self.conv_layer0 = Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid')
        self.activation0 = Activation('relu')
        self.max_pool0 = MaxPooling2D()

        self.conv_layer1 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid')
        self.activation1 = Activation('relu')
        self.max_pool1 = MaxPooling2D()

        self.flatten = Flatten()
        self.fully_connected0 = Dense(units=120, activation='sigmoid')
        self.dropout0 = Dropout(rate=drop_rate)
        self.fully_connected1 = Dense(units=84, activation='sigmoid')
        self.dropout1 = Dropout(rate=drop_rate)
        self.out_layer = Dense(units=self.classes, activation='softmax')

    def call(self, input, training=None, mask=None):
        x = self.conv_layer0(input)
        x = self.activation0(x)
        x = self.max_pool0(x)

        x = self.conv_layer1(x)
        x = self.activation1(x)
        x = self.max_pool1(x)

        if self.include_top is True:
            x = self.flatten(x)
            x = self.fully_connected0(x)
            x = self.dropout0(x)
            x = self.fully_connected1(x)
            x = self.dropout1(x)
            x = self.out_layer(x)

        return x

    def _conv_block(self, input, n_filters, kernel_size, strides, padding='same', activation='relu'):
        x = Conv2D(n_filters, kernel_size, strides=strides, padding=padding)(input)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)

        return x



