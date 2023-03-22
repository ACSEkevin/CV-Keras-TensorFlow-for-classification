from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, Layer
from keras.layers import Concatenate, Multiply, Add, LayerNormalization, MultiHeadAttention
from keras.models import Model, Sequential
from keras.initializers.initializers_v2 import LecunNormal, Zeros, GlorotUniform
import numpy as np
import tensorflow as tf


class ImgPatchEmbedding(Layer):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super(ImgPatchEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.img_size = (img_size, img_size)
        self.grids = (img_size // patch_size, img_size // patch_size)
        self.n_patches = self.grids[0] * self.grids[1]
        self.projection = Conv2D(self.embed_dim, kernel_size=(16, 16), strides=16, padding='same',
                                 kernel_initializer=LecunNormal(), bias_initializer=Zeros())

    def call(self, inputs, *args, **kwargs):
        batch, height, width, channels = inputs.shape
        x = self.projection(inputs)
        x = tf.reshape(x, [batch, self.n_patches, self.embed_dim])

        assert (height, width) == self.img_size, \
            f"Input image size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        return x


class ConcatTokenAddPositionEncoding(Layer):
    def __init__(self, embed_dim=768, n_patches=14 ** 2):
        super(ConcatTokenAddPositionEncoding, self).__init__()
        self.position_embedding = None
        self.class_token = None
        self.embed_dim = embed_dim
        self.n_patches = n_patches

    def build(self, input_shape):
        self.class_token = self.add_weight(shape=[1, 1, self.embed_dim], initializer=Zeros(),
                                           trainable=True, dtype=tf.float32)
        self.position_embedding = self.add_weight(shape=[1, self.n_patches + 1, self.embed_dim],
                                                  initializer=Zeros(), trainable=True, dtype=tf.float32)

    def call(self, inputs, *args, **kwargs):
        class_token = tf.broadcast_to(self.class_token, [inputs.shape[0], 1, self.embed_dim])

        # class_token.shape -> [batch, 1, dim_embedding] x inputs.shape -> [batch, n_patches, dim_embedding]
        # x = tf.concat([class_token, inputs], axis=1)
        x = Concatenate(axis=1)([class_token, inputs])
        # x = Add()([x, self.position_embedding])  # not feasible as Add can not broadcast tensors
        x = x + self.position_embedding  # broadcasting methodology for residual connection

        return x


class VisionMHAttention(Layer):
    def __init__(self, embed_dim, num_heads=8, scaler=None,
                 use_bias=True, attention_drop_rate=0., transition_drop_rate=0.):
        super(VisionMHAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_drop_rate = attention_drop_rate
        self.transition_drop_rate = transition_drop_rate

        self.head_dimension = self.embed_dim // self.num_heads
        self.scaler = scaler if scaler else self.head_dimension ** -0.5

        self.query_key_value = Dense(self.embed_dim * 3, kernel_initializer=GlorotUniform(),
                                     use_bias=use_bias, bias_initializer=Zeros())
        self.attention_drop = Dropout(rate=self.attention_drop_rate)

        self.transition = Dense(self.embed_dim, kernel_initializer=GlorotUniform(),
                                use_bias=use_bias, bias_initializer=Zeros())
        self.transition_drop = Dropout(rate=self.transition_drop_rate)

    def call(self, inputs, *args, **kwargs):
        batch, patches_ctoken, channels = inputs.shape

        # stage - instantiate Wq, Qk, Wv
        # W_qkv.shape: [batch, patches_class_token, 3 * channels]
        W_qkv = self.query_key_value(inputs)
        # W_qkv.shape -> [batch, patches_class_token, 3, num_heads, dimension_per_head]
        W_qkv = tf.reshape(W_qkv, [batch, patches_ctoken, 3, self.num_heads, channels // self.num_heads])
        # W_qkv.shape -> [3, batch, num_heads, patches_class_token, dimension_per_head]
        W_qkv = tf.transpose(W_qkv, [2, 0, 3, 1, 4])
        # retrieve matrices query, key and value
        W_query, W_key, W_value = W_qkv[0], W_qkv[1], W_qkv[2]

        # stage - compute alpha and alpha^prime
        # b.transpose.shape -> [batch, num_heads, dimension_per_head, patches_class_token]
        # W_alpha.shape -> [batch, num_heads, patches_class_token, patches_class_token]
        W_alpha = tf.matmul(a=W_query, b=W_key, transpose_b=True) * self.scaler
        W_alpha_prime = tf.nn.softmax(W_alpha, axis=-1)
        W_alpha_prime = self.attention_drop(W_alpha_prime)

        # stage - compute output
        # x.shape -> [batch, num_heads, patches_class_token, dimension_per_head]
        x = tf.matmul(W_alpha_prime, W_value)
        # x.shape -> [batch, patches_class_token, num_heads, dimension_per_head]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        # x.shape -> [batch, patches_class_token, num_heads * dimension_per_head]
        x = tf.reshape(x, shape=[batch, patches_ctoken, channels])

        x = self.transition(x)
        x = self.transition_drop(x)

        return x


class FeedForwardNetwork(Layer):
    def __init__(self, in_features, expansion_rate=4, drop_rate=0.):
        super(FeedForwardNetwork, self).__init__()
        self.in_units = int(in_features * expansion_rate)
        self.fully_connected0 = Dense(self.in_units, kernel_initializer=GlorotUniform(),
                                      bias_initializer=Zeros())
        self.non_linearity = Activation('gelu')
        self.fully_connected1 = Dense(in_features, kernel_initializer=GlorotUniform(),
                                      bias_initializer=Zeros())
        self.drop_out = Dropout(rate=drop_rate)

    def call(self, inputs, *args, **kwargs):
        x = self.fully_connected0(inputs)
        x = self.non_linearity(x)
        x = self.drop_out(x)
        x = self.fully_connected1(x)
        x = self.drop_out(x)

        return x


class EncoderBlock(Layer):
    def __init__(self, embed_dim, num_heads=8, scaler=None, use_bias=True,
                 drop_path_rate=0., attention_drop_rate=0., transition_drop_rate=0.):
        super(EncoderBlock, self).__init__()
        self.attention_drop_rate = attention_drop_rate
        self.transition_drop_rate = transition_drop_rate

        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.multihead_attention = VisionMHAttention(embed_dim, num_heads=num_heads, scaler=scaler, use_bias=use_bias,
                                                     attention_drop_rate=self.attention_drop_rate,
                                                     transition_drop_rate=self.transition_drop_rate)
        self.drop_path = Dropout(rate=drop_path_rate, noise_shape=(None, 1, 1))
        self.feedforward = FeedForwardNetwork(embed_dim, drop_rate=drop_path_rate)
        self.feature_add = Add()

    def call(self, inputs, *args, **kwargs):
        x = self.layer_norm(inputs)
        x = self.multihead_attention(x)
        if self.attention_drop_rate > 0.:
            x = self.drop_path(x)

        x1 = self.feature_add([inputs, x])

        x1 = self.layer_norm(x1)
        x1 = self.feedforward(x1)
        if self.attention_drop_rate > 0.:
            x1 = self.drop_path(x1)

        out = self.feature_add([x, x1])

        return out


class VisionTransformer(Model):
    def __init__(self, classes, n_encoders=12, img_size=224, patch_size=16,
                 embed_dim=768, num_heads=8, scaler=None, use_bias=True,
                 drop_rate=0., attention_drop_rate=0., transition_drop_rate=0., representation_size=None):
        super(VisionTransformer, self).__init__()
        self.classes = classes
        self.n_encoders = n_encoders
        self.drop_rate = drop_rate
        self.use_bias = use_bias

        self.patch_embedding = ImgPatchEmbedding(img_size, patch_size, embed_dim)
        n_patches = self.patch_embedding.n_patches
        self.concat_token_add_pe = ConcatTokenAddPositionEncoding(embed_dim, n_patches)

        self.drop_out = Dropout(rate=drop_rate)

        self.attention_drop_rate_list = np.linspace(0., attention_drop_rate, self.n_encoders)
        self.encoder_list = [EncoderBlock(embed_dim, num_heads, scaler, self.use_bias,
                                          attention_drop_rate=attention_drop_rate,
                                          transition_drop_rate=transition_drop_rate,
                                          drop_path_rate=self.attention_drop_rate_list[index]) for index in
                             range(self.n_encoders)]

        self.layer_norm = LayerNormalization(epsilon=1e-5)

        if representation_size is not None:
            self.has_logits = True
            self.pre_logits = Dense(representation_size, activation='tanh')
        else:
            self.has_logits = False
            self.pre_logits = Activation('linear')

        self.linear = Dense(self.classes, kernel_initializer=Zeros())

    def call(self, inputs, training=None, mask=None):
        x = self.patch_embedding(inputs)
        x = self.concat_token_add_pe(x)
        x = self.drop_out(x) if self.drop_rate > 0. else x

        for encoder in self.encoder_list:
            x = encoder(x)

        x = self.layer_norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.linear(x)

        return x

