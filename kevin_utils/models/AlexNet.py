from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Activation, Input, BatchNormalization


class AlexNet(Model):
    def __init__(self, input_shape, classes, drop_rate=0.5, include_top=True):
        super(AlexNet, self).__init__()
        self.classes = classes
        self.include_top = include_top

        self.input_layer = Input(input_shape)
        self.conv_layer0 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4))
        self.batch_norm0 = BatchNormalization()
        self.activation0 = Activation('relu')

        self.max_pool0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.conv_layer1 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same')
        self.batch_norm1 = BatchNormalization()
        self.activation1 = Activation('relu')

        self.max_pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.conv_layer2 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv_layer3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv_layer4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')

        self.max_pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.flatten = Flatten()
        self.fully_connected0 = Dense(units=4096, activation='relu')
        self.dropout0 = Dropout(rate=drop_rate)
        self.fully_connected1 = Dense(units=2048, activation='relu')
        self.dropout1 = Dropout(rate=drop_rate)
        self.softmax = Dense(units=self.classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        # stage - ordinal conv
        x = self.conv_layer0(inputs)
        x = self.batch_norm0(x)
        x = self.activation0(x)
        x = self.max_pool0(x)

        x = self.conv_layer1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)
        x = self.max_pool1(x)

        # stage - conv without batch norm
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.max_pool2(x)

        # stage - flatten & fully connected
        if self.include_top is True:
            x = self.flatten(x)
            x = self.fully_connected0(x)
            x = self.dropout0(x)
            x = self.fully_connected1(x)
            x = self.dropout1(x)

            # output predictions
            x = self.softmax(x)

        return x
