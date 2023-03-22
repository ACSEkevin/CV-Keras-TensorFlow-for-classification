import tensorflow as tf
import numpy as np
AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)
print(data_root)
for item in data_root.iterdir():
    print(item)


label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)

# labeling
label_to_index = dict((name, index) for index, name in enumerate(label_names))
print(label_to_index)


import random
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
print(image_count)

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]for path in all_image_paths]
print("First 10 labels indices: ", all_image_labels[:10])


img_path = all_image_paths[0]
img_raw = tf.io.read_file(img_path)
# convert to tensor
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)


def preprocess_image(image, shape=None):
    if shape is None:
        shape = [227, 227]
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, shape)
    # image /= 255.0  # normalize to [0,1] range  return image
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# construct labels, converting dtype to int64
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
# get images and preloading
image_ds = path_ds.map(load_and_preprocess_image)
print(image_ds)

# zip both and put them into a new dataset
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
print(image_label_ds)
#
# features = np.array(list(image_ds))
# label = np.array(all_image_labels)
# print(features.shape)
# print(np.array(all_image_labels).shape)
#
# import h5py
# file = h5py.File('flower_dataset_227.h5', 'w')
# file['dataset'], file['label'] = features, label
# file.close()







