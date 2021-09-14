import tensorflow as tf
import logging
import argparse
import time
from config import cfg
from data.dataset import Dataset
from model import Generator, Discriminator, CycleGan
from losses import *

# 로그 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if args.gpu_ids != '-1' and '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    return args


if __name__ == '__main__':
    args = parse_args()
    devices = ['/gpu:{}'.format(x) for x in args.gpu_ids.split(',')]
    strategy = tf.distribute.MirroredStrategy(devices=devices)

    # Load dataset
    logger.info('Load datasets')
    monet_ds = Dataset('monet_tfrec', cfg).load_dataset(labeled=True).batch(cfg.batch_size)
    photo_ds = Dataset('photo_tfrec', cfg).load_dataset(labeled=True).batch(cfg.batch_size)

    with strategy.scope():
        # Define generator and discriminator for cyclegan
        monet_generator = Generator(cfg.output_channels)
        photo_generator = Generator(cfg.output_channels)
        monet_discriminator = Discriminator(cfg.output_channels)
        photo_discriminator = Discriminator(cfg.output_channels)

        # Define optimaizer
        monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # Define loss function
        gen_loss_fn = generator_loss
        disc_loss_fn = discriminator_loss
        cycle_loss_fn = calc_cycle_loss
        identity_loss_fn = identity_loss

        logger.info('Define CycleGan model')
        cycle_gan_model = CycleGan(
            monet_generator, photo_generator, monet_discriminator, photo_discriminator, lambda_cycle=5
        )

        cycle_gan_model.compile(
            m_gen_optimizer = monet_generator_optimizer,
            p_gen_optimizer = photo_generator_optimizer,
            m_disc_optimizer = monet_discriminator_optimizer,
            p_disc_optimizer = photo_discriminator_optimizer,
            gen_loss_fn = generator_loss,
            disc_loss_fn = discriminator_loss,
            cycle_loss_fn = calc_cycle_loss,
            identity_loss_fn = identity_loss
        )

        logger.info('Model training start')
        cycle_gan_model.fit(
            tf.data.Dataset.zip((monet_ds, photo_ds)),
            epochs=cfg.n_epochs
        )

    logger.info('Model training end. Save models')
    cycle_gan_model.output_models()
