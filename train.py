from Kevin_datasets import flower_dataset, mnist_dataset, cifar10_dataset, cat_dog_dataset
from keras.optimizers import Adam, Adadelta, SGD
from keras.metrics import SparseCategoricalAccuracy, Accuracy
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
from keras.applications import NASNetMobile, EfficientNetV2M

# dataset preprocessing
x_train, y_train, x_test, y_test = cat_dog_dataset(normalization=True)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

# model importing
from kevin_utils.models import ShuffleNet_v2, AlexNet, ConvNeXt, VisionTransformer, VisionTransformerTest

# model = ShuffleNet_v2(input_shape, classes=2, version='2.0x')
model = VisionTransformer(classes=2, n_encoders=4, num_heads=6)

model.compile(loss=SparseCategoricalCrossentropy(),
              optimizer=Adam(learning_rate=0.0007), metrics=SparseCategoricalAccuracy())

# print(model.summary())

history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"model train loss: {train_loss}, train accuracy: {train_acc}")
print(f"model test loss: {test_loss}, test accuracy: {test_acc}")

train_acc, train_loss = history.history['sparse_categorical_accuracy'], history.history['loss']
val_acc, val_loss = history.history['val_sparse_categorical_accuracy'], history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc, color='purple')
plt.plot(val_acc, color='red')
plt.xlabel('$epochs$')
plt.ylabel('$accuracy$')
plt.legend(['train_accuracy', 'val_accuracy'])

plt.subplot(1, 2, 2)
plt.plot(train_loss, color='deeppink')
plt.plot(val_loss, color='deepskyblue')
plt.xlabel('$epochs$')
plt.ylabel('$loss$')
plt.legend(['train_loss', 'val_loss'])
plt.show()
