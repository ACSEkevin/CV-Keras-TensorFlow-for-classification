# CV-Keras-TensorFlow-for-classification
## Author: ACSEKevin <hzhang205@sheffield.ac.uk>
This is a project including most of the CNN backbones which can be found in package `kevin_utils.models`<p>
#### Notice: 1. SwinTransformer in this version accepts inputs with different size of images instead of the original version (224 x 224 x 3),  has not been tested yet, however. If there is any problem when using models, please refer to <https://github.com/microsoft/Swin-Transformer><br> 2. The datasets have been removed from the dataset directories, please add own dataset before run the code. The origial datasets: ImageNet21k, Pascal VOC 2012, cat_and_dog_dataset, flowers_dataset, gesture_dataset.<p>

In `main`, `train.py` for training the model, `Kevin_datasets.py` for wrapping the dataset processing procedures.<br>
In package `kevin_utils.models`, the models are listed:
1. Classic networks: Lenet-5, Lenet-5(Modified), AlexNet, VGG (Visual Geometry Group from Oxford University)
2. GoogLeNet (Inception V1), ResNet (18, 34, 50, 101, 152), ResNeXt
3. MobileNet V2/V3, ShuffleNet V2
4. EfficientNet V1/V2, DenseNet
5. VisionTransformer, SwinTransformer, ConvNeXt<p>

#### The models are provided for future convenient, please make a citations when cloning and using this code.
## Reference:<p>
Yann, L. (1998) LeNet <http://yann.lecun.com/exdb/lenet/index.html><br>
Krizhevsky, A. (2012) AlexNet <http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf><br>
Visual Geometry Group VGG <https://arxiv.org/abs/1409.1556><br>
Google AI Research GoogLeNet(Inceptionv1) <https://arxiv.org/abs/1409.4842><br>
He, K. Microsoft AI Research Team, ResNet <https://arxiv.org/abs/1512.03385><br>
Xie, S. ResNeXt <https://arxiv.org/abs/1611.05431><br>
Huang, G. DenseNet <https://arxiv.org/abs/1608.06993><br>
Hu, J. (2017) SENet(Squeeze-and-Excitation Networks) <https://arxiv.org/abs/1709.01507><br>
Sandler, M. (2018) MobileNet V2 <https://arxiv.org/abs/1801.04381><br>
Howard, A. (2019) MobileNet V3 <https://arxiv.org/abs/1905.02244><br>
Zhang, X. (2017) ShuffleNet V1 <https://arxiv.org/abs/1707.01083><br>
Ma, N. (2018) ShuffleNet V2 <https://arxiv.org/abs/1807.11164><br>
