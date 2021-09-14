import tensorflow as tf
from utils.common_utils import read_tfrecord
from functools import partial

AUTOTUNE = tf.data.experimental.AUTOTUNE

class Dataset(object):
    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg
        self.data_paths = tf.io.gfile.glob(str(cfg.DATA_PATH + '/{}/*.tfrec'.format(name)))

    def load_dataset(self, labeled=True, ordered=False):
        dataset = tf.data.TFRecordDataset(self.data_paths)
        dataset = dataset.map(partial(read_tfrecord, image_size=self.cfg.input_shape), num_parallel_calls=AUTOTUNE)
        return dataset

