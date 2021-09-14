import os
from pathlib import Path


class Config:
    n_epochs = 75
    batch_size = 1
    lr = 2e-4
    beta = 0.5
    output_channels = 3
    input_shape = [256, 256]

    # optimizer
    monet_generator_optimizer = 'adam'
    photo_generator_optimizer = 'adam'
    monet_discriminator_optimizer = 'adam'
    photo_discriminator_optimizer = 'adam'

    # directories
    ROOT_PATH = Path(__file__).resolve().parent.parent
    DATA_PATH = os.path.join(ROOT_PATH, 'data')


cfg = Config()