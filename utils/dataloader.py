import os

import torch
from torch.utils.data import DataLoader, Subset

from utils.load_ljspeech import LJSPEECH


def loaders(dataset_name='lj_speech', path=os.getcwd(), seq_len=8192, batch_size=64, shuffle=True, num_workers=0):
    """
    Create dataloaders for given dataset.
    :param dataset_name: Name of the dataset to be loaded (str, default='lj_speech')
    :param path: Full path of data
    :param seq_len: Maximum sequence length - to enforce same size for all training samples (1, seq_len)
    :param batch_size: Batch size (for training dataloader)
    :param shuffle: whether to shuffle training data or not (bool, default=True)
    :param num_workers: How many sub-processes to use for data loading (only for training set).
                        Set it to zero to load the data inside the main process (int, default=0)
    :return: Train and test dataloaders for the given dataset
    """

    if dataset_name == 'lj_speech':
        # one dataset for training, one for testing (remove segmentation, normalization and augmentation)
        ds = LJSPEECH(root=path, segment=True, seq_len=seq_len, normalize=True, augment=True)
        ds2 = LJSPEECH(root=path)
    else:
        raise ValueError('Dataset not available. Current version supports only "lj_speech".')

    # TODO: extend to support VCTK dataset as well

    ''' train/test split: as in the official implementation of Melgan, we keep 
        the first 10 audio clips for testing and the rest for training. '''

    train_ds, test_ds = Subset(ds, torch.arange(10, len(ds))), Subset(ds2, torch.arange(10))

    # create dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_dl, test_dl
