from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, Layer
from keras.layers import Concatenate, Multiply, Add, LayerNormalization
from keras.models import Model, Sequential
from keras.initializers.initializers_v2 import LecunNormal, Zeros, GlorotUniform
import numpy as np
import tensorflow as tf


class PatchEmbedding(Layer):
    def __init__(self, img_size=224, n_embedding=768, patch_size=16, strides=None):
        super(PatchEmbedding, self).__init__()
        if strides is None:
            strides = patch_size
        self.img_size = img_size
        self.n_embedding = n_embedding
        assert self.n_embedding == (patch_size ** 2) * 3
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_embedding = Conv2D(n_embedding, kernel_size=patch_size, strides=strides, padding='same',
                                      kernel_initializer=LecunNormal(), bias_initializer=Zeros())

    def call(self, inputs, *args, **kwargs):
        batches, height, width, channels = inputs.shape
        x = self.patch_embedding(inputs)
        # x.shape: [batches, grid_height(14), grid_width(14), channels]
        x = tf.reshape(x, [batches, self.n_patches, self.n_embedding])

        assert height * width == self.img_size ** 2, 'input image size: {}*{} does not match img_size: {}' \
            .format(height, width, self.img_size)
        return x


class PositionalEncoding(Layer):
    def __init__(self, n_embedding=768, n_patches=14 ** 2):
        super(PositionalEncoding, self).__init__()
        self.position_embedding = None
        self.class_token = None
        self.n_embedding = n_embedding
        self.n_patches = n_patches

    def build(self, input_shape):
        self.class_token = self.add_weight(shape=[1, 1, self.n_embedding], initializer=Zeros(),
                                           trainable=True, dtype=tf.float32)
        self.position_embedding = self.add_weight(shape=[1, self.n_patches + 1, self.n_embedding],
                                                  initializer=Zeros(), trainable=True, dtype=tf.float32)

    def call(self, inputs, *args, **kwargs):
        class_token = tf.broadcast_to(self.class_token, shape=[inputs.shape[0], 1, self.n_embedding])
        # x = tf.concat([inputs, class_token], axis=1)
        x = Concatenate(axis=1)([inputs, class_token])
        x = x + self.position_embedding

        return x


class MultiHeadSelfAttention(Layer):
    def __init__(self, n_embedding, n_head=8, scaler=None, use_bias=False,
                 attention_drop_rate=0., projection_drop_rate=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_embedding = n_embedding
        self.n_head = n_head
        self.head_dim = n_embedding // n_head
        self.scaler = self.head_dim if scaler is None else scaler

        self.query_key_value = Dense(n_embedding * 3, use_bias=use_bias)
        self.attention_drop = Dropout(rate=attention_drop_rate)

        self.projection = Dense(n_embedding, use_bias=use_bias)
        self.projection_drop = Dropout(rate=projection_drop_rate)

    def call(self, inputs, *args, **kwargs):
        batches, patches_token, n_embedding = inputs.shape
        W_qkv = self.query_key_value(inputs)
        W_qkv = tf.reshape(W_qkv, shape=[batches, patches_token, 3, self.n_head, n_embedding // self.n_head])
        W_qkv = tf.transpose(W_qkv, perm=[2, 0, 3, 1, 4])
        # q_k_v.shape: [batches, num_head, patches_token, head_dimension]
        W_q, W_k, W_v = W_qkv[0], W_qkv[1], W_qkv[2]

        # alpha.shape: [batches, num_head, patches_token, patches_token]
        alpha = tf.matmul(a=W_q, b=W_k, transpose_b=True) * self.scaler
        alpha_prime = tf.nn.softmax(alpha, axis=-1)
        alpha_prime = self.attention_drop(alpha_prime)

        # b_row.shape: [batches, num_head, patches_token, head_dimension]
        b_row = tf.matmul(alpha_prime, W_v)
        b_row = tf.transpose(b_row, perm=[0, 2, 1, 3])
        W_b = tf.reshape(b_row, shape=[batches, patches_token, n_embedding])

        x = self.projection(W_b)
        x = self.projection_drop(x)
        return x


class FeedForwardLayer(Layer):
    def __init__(self, in_channels, expansion_rate=4, drop_rate=0.):
        super(FeedForwardLayer, self).__init__()
        self.in_channels = int(in_channels * expansion_rate)
        self.feedforward0 = Dense(self.in_channels)
        self.non_linearity = Activation('gelu')
        self.dropout0 = Dropout(drop_rate)

        self.feedforward1 = Dense(in_channels)
        self.dropout1 = Dropout(drop_rate)

    def call(self, inputs, *args, **kwargs):
        x = self.feedforward0(inputs)
        x = self.non_linearity(x)
        x = self.dropout0(x)

        x = self.feedforward1(x)
        x = self.dropout1(x)

        return x


class TransformerEncoder(Layer):
    def __init__(self, n_embedding, n_head=8, scaler=None, use_bias=True, expansion_rate=4,
                 attention_drop_rate=0., projection_drop_rate=0., ffn_drop_rate=0., drop_path_rate=0.):
        super(TransformerEncoder, self).__init__()
        self.attention_drop_rate = attention_drop_rate
        self.projection_drop_rate = projection_drop_rate
        self.drop_path_rate = drop_path_rate
        self.layer_norm0 = LayerNormalization(epsilon=1e-6)
        self.multihead_attention = MultiHeadSelfAttention(n_embedding, n_head=n_head, scaler=scaler,
                                                          use_bias=use_bias, attention_drop_rate=attention_drop_rate,
                                                          projection_drop_rate=projection_drop_rate)
        self.drop_path = Dropout(rate=drop_path_rate, noise_shape=(None, 1, 1))
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.feedforward = FeedForwardLayer(n_embedding, expansion_rate=expansion_rate, drop_rate=ffn_drop_rate)
        self.feature_add = Add()

    def call(self, inputs, *args, **kwargs):
        x = self.layer_norm0(inputs)
        x = self.multihead_attention(x)
        x = self.drop_path(x) if self.drop_path_rate > 0. else x
        x = self.feature_add([inputs, x])

        x1 = self.layer_norm1(x)
        x1 = self.feedforward(x1)
        x1 = self.drop_path(x1) if self.drop_path_rate > 0. else x1
        x1 = self.feature_add([x, x1])

        return x1


class VisionTransformerTest(Model):
    def __init__(self, classes, img_size=224,
                 n_encoders=12, n_embedding=768, patch_size=16, use_bias=True,
                 n_head=8, scaler=None, drop_rate_scheduler: np.linspace = None,
                 expansion_rate=4,
                 attention_drop_rate=0., projection_drop_rate=0., ffn_drop_rate=0., drop_rate=0.,
                 drop_path_rate=0., representation_size=None):
        super(VisionTransformerTest, self).__init__()
        # self.input_shape = input_shape
        self.classes = classes
        self.n_encoders = n_encoders
        self.n_embedding = n_embedding
        self.drop_rate_scheduler = drop_rate_scheduler
        self.patch_embedding = PatchEmbedding(img_size, n_embedding, patch_size)
        self.n_patches = self.patch_embedding.n_patches
        self.positional_encoding = PositionalEncoding(n_embedding, self.n_patches)
        self.prior_dropout = Dropout(rate=drop_rate)
        self.attention_blocks = [TransformerEncoder(n_embedding, n_head, scaler, use_bias=use_bias,
                                                    expansion_rate=expansion_rate,
                                                    attention_drop_rate=attention_drop_rate,
                                                    projection_drop_rate=projection_drop_rate,
                                                    ffn_drop_rate=ffn_drop_rate,
                                                    drop_path_rate=drop_path_rate if not drop_rate_scheduler else
                                                   drop_rate_scheduler[index])
                                 for index in range(n_encoders)]

        self.layer_normalization = LayerNormalization(epsilon=1e-6)
        if representation_size is not None:
            self.has_logits = True
            self.pre_logits = Dense(representation_size, activation='tanh')
        else:
            self.has_logits = False
            self.pre_logits = Activation('linear')
        self.linear = Dense(classes, kernel_initializer=Zeros())

    def call(self, inputs, training=None, mask=None):
        x = self.patch_embedding(inputs)
        x = self.positional_encoding(x)
        x = self.prior_dropout(x)

        for encoder_block in self.attention_blocks:
            x = encoder_block(x)

        x = self.layer_normalization(x)
        # extract class token
        x = self.pre_logits(x[:, 0])
        x = self.linear(x)

        return x


# def ViTBase16(classes, n_encoders=12, n_head=8, has_logits=False):
#     model = VisionTransformer(classes=classes,
#                               img_size=224,
#                               n_encoders=n_encoders,
#                               n_embedding=768,
#                               n_head=n_head)


