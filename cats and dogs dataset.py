import keras
import tensorflow as tf
import numpy as np
import PIL.Image as Pimg
import matplotlib.pyplot as plt


def get_file(resize=(224, 224)):
    images_cat, images_dog = [], []
    label_cat, label_dog = [], []
    fnames = ['cat.{}.jpg'.format(i) for i in range(2000)]
    for fname in fnames:
        cats = Pimg.open('./cat_dog_dataset/cat' + f'/{fname}')
        cats = cats.resize(resize)
        cats_np = np.array(cats)
        images_cat.append(cats_np)
        label_cat.append(0)
        print(f'processing {fname}')

    fnames = ['dog.{}.jpg'.format(i) for i in range(2000)]
    for fname in fnames:
        dogs = Pimg.open('./cat_dog_dataset/dog' + f'/{fname}')
        dogs = dogs.resize(resize)
        dogs_np = np.array(dogs)
        images_dog.append(dogs_np)
        label_dog.append(1)
        print(f'processing {fname}')

    images_dataset = np.vstack((images_dog, images_cat))
    images_label = np.hstack((label_dog, label_cat))

    # datasets_row = np.array([images_dataset, images_label], dtype='int32').T
    # np.random.shuffle(datasets_row)
    # dataset, label = list(datasets_row[:, 0]), list(datasets_row[:, 1])
    # return np.array(dataset), np.array(label)
    return images_dataset, images_label


dataset, label = get_file()
print(dataset.shape, label.shape)

import h5py
file = h5py.File('cats_dogs.h5', 'w')
file['dataset'], file['label'] = dataset, label
file.close()

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(dataset[i])
    plt.title(label[i])
    plt.axis("off")

plt.show()




# cats = Pimg.open('./cat_dog_dataset/cat' + f'/cat.2390.jpg').convert('RGB')
# img_np = np.array(cats, 'uint8')
# plt.imshow(cats)
# plt.axis('off')
# plt.show()
