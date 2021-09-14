import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa


class downsample(tf.keras.Model):
    def __init__(self, num_channels, size, apply_instancenorm=True):
        super(downsample, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.conv = layers.Conv2D(num_channels, size, strides=2, padding='same',
                                  kernel_initializer=initializer, use_bias=False)
        self.apply_instancenorm = apply_instancenorm
        if apply_instancenorm:
            self.instance_norm = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)

        self.leaky_relu = layers.LeakyReLU()

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.conv(x)
        if self.apply_instancenorm:
            x = self.instance_norm(x)
        out = self.leaky_relu(x)
        return out

class upsample(tf.keras.Model):
    def __init__(self, num_channels, size, apply_dropout=False):
        super(upsample, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.dconv = layers.Conv2DTranspose(num_channels, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        self.instance_norm = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)
        self.apply_dropout = apply_dropout
        if apply_dropout:
            self.dropout = layers.Dropout(0.5)
        self.relu = layers.ReLU()

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.dconv(x)
        x = self.instance_norm(x)
        if self.apply_dropout:
            x = self.dropout(x)
        out = self.relu(x)
        return out
