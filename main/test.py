import cv2
import numpy as np
from model import Generator
import argparse
from data.dataset import Dataset
import tensorflow as tf
from config import cfg
import matplotlib.pyplot as plt


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

    photo_ds = Dataset('photo_tfrec', cfg).load_dataset(labeled=True).batch(cfg.batch_size)

    with strategy.scope():
        monet_generator = Generator(cfg.output_channels)
        monet_generator(tf.zeros(([1, 256, 256, 3])))
        monet_generator.load_weights('../monet_generator.h5')
        _, ax = plt.subplots(5, 2, figsize=(12, 12))
        for i, img in enumerate(photo_ds.take(5)):
            prediction = monet_generator(img, training=False)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input Photo")
            ax[i, 1].set_title("Monet-esque")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
        plt.show()