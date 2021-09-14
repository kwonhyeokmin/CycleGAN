import tensorflow as tf
import tensorflow.keras.layers as layers
from layers import downsample, upsample
import tensorflow_addons as tfa

class Generator(tf.keras.Model):
    def __init__(self, num_channels):
        super(Generator, self).__init__()
        self.down_stack = [
            downsample(64, 4, apply_instancenorm=False),
            downsample(128, 4),
            downsample(256, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
            downsample(512, 4),
        ]
        self.up_stack = [
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4),
            upsample(256, 4),
            upsample(128, 4),
            upsample(64, 4),
        ]
        initializer = tf.random_normal_initializer(0., 0.02)
        self.dconv = layers.Conv2DTranspose(num_channels, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        # Down sampling
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Up sampling
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = layers.Concatenate()([x, skip])

        out = self.dconv(x)
        return out


class Discriminator(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.down_1 = downsample(64, 4, False)
        self.down_2 = downsample(128, 4)
        self.down_3 = downsample(256, 4)

        self.zero_pad_1 = layers.ZeroPadding2D()

        initializer = tf.random_normal_initializer(0., 0.02)
        gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        self.conv_1 = layers.Conv2D(512, 4, strides=2,
                               kernel_initializer=initializer,
                               use_bias=False)
        self.norm_1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)
        self.leaky_relu = layers.LeakyReLU()
        self.zero_pad_2 = layers.ZeroPadding2D()
        self.conv_2 = layers.Conv2D(1, 4, strides=2,
                               kernel_initializer=initializer)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.zero_pad_1(x)
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.leaky_relu(x)
        x = self.zero_pad_2(x)
        out = self.conv_2(x)
        return out

class CycleGan(tf.keras.Model):
    def __init__(self, monet_generator, photo_generator,
                 monet_discriminator, photo_discriminator,
                 lambda_cycle=10,):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle

    def compile(
            self,
            m_gen_optimizer,
            p_gen_optimizer,
            m_disc_optimizer,
            p_disc_optimizer,
            gen_loss_fn,
            disc_loss_fn,
            cycle_loss_fn,
            identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, data):
        real_monet, real_photo = data
        batch_size = tf.shape(data)[0]

        with tf.GradientTape(persistent=True) as tape:
            # photo to monet and back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet to photo and back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # discriminator used for check, inputing real images
            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator used for check, inputing fake images
            disc_fake_monet = self.m_disc(fake_monet, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluate generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # evaluate total cycle consistancy loss
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + \
                               self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            # evaluate total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + \
                                   self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + \
                                   self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        # Calculate gradients
        monet_generator_gradients = tape.gradient(total_monet_gen_loss, self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss, self.p_gen.trainable_variables)
        monet_discriminator_gradients = tape.gradient(monet_disc_loss, self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss, self.p_disc.trainable_variables)

        # Apply gradients
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients, self.m_gen.trainable_variables))
        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients, self.p_gen.trainable_variables))
        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients, self.m_disc.trainable_variables))
        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients, self.p_disc.trainable_variables))

        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }

    def output_models(self):
        self.m_gen.save_weights('monet_generator.h5', save_format='h5')
        self.p_gen.save_weights('photo_generator.h5', save_format='h5')
        self.m_disc.save_weights('monet_discriminator.h5', save_format='h5')
        self.p_disc.save_weights('photo_discriminator.h5', save_format='h5')
