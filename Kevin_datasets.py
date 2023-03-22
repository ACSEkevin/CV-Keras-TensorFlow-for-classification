import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt


def flower_dataset(shuffle=True, normalization=True, test_size=0.1, random_state=1):
    # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    data = h5py.File('flower_photos/flower_dataset_227.h5', "r")
    x_train, x_test, y_train, y_test = train_test_split(np.array(data['dataset']), np.array(data['label']),
                                            test_size=test_size, shuffle=shuffle, random_state=random_state)
    if normalization is True:
        return x_train / 255, y_train, x_test / 255, y_test
    elif normalization is False:
        return x_train, y_train, x_test, y_test


# x_train, y_train, x_test, y_test = flower_dataset()
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# index = 2250
# plt.imshow(x_train[index])
# plt.title(y_train[index])
# plt.show()


def mnist_dataset(normalization=True):
    # loading dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])
    # Normalization
    if normalization is True:
        x_train, x_test = x_train / 255, x_test / 255
        return x_train, y_train, x_test, y_test
    elif normalization is False:
        return x_train, y_train, x_test, y_test


# x_train, y_train, x_test, y_test = mnist_dataset()
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

def cifar10_dataset(normalization=True):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])
    if normalization is True:
        x_train, x_test = x_train / 255, x_test / 255
        return x_train, y_train, x_test, y_test
    elif normalization is False:
        return x_train, y_train, x_test, y_test


# x_train, y_train, x_test, y_test = cifar10_dataset()
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

def cat_dog_dataset(shuffle=True, normalization=True, test_size=0.2, random_state=1):
    data = h5py.File('cat_dog_dataset/cats_dogs.h5', "r")
    x_train, x_test, y_train, y_test = train_test_split(np.array(data['dataset']), np.array(data['label']),
                                                        test_size=test_size, shuffle=shuffle, random_state=random_state)
    if normalization is True:
        return x_train / 255, y_train, x_test / 255, y_test
    elif normalization is False:
      return x_train, y_train, x_test, y_test

# x_train, y_train, x_test, y_test = cat_dog_dataset()
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# print(y_train[: 10])
# print(y_test[: 10])

