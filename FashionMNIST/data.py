import torch
from torch.utils.data import random_split, DataLoader

import torchvision.transforms as T
from torchvision.datasets import FashionMNIST

import numpy as np

from pathlib import Path


# TODO (optional): Dynamically compute DATA_MEAN, DATA_STD
DATA_MEAN = 0.2861
DATA_STD = 0.3530

LABEL_NAMES = {
    0: "t_shirt_or_top",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle_boot"
}

# NOTE: We duplicate the grayscale channel
IM_DIMS = (2, 28, 28)
NDIMS_TOTAL = np.prod(IM_DIMS)


def unnormalize(x):
    return x * DATA_STD + DATA_MEAN


def get_trainval_datasets(config):

    n_eval = int(config["val_percentage"] * 60000)
    n_train = 60000 - n_eval

    trainval_ds = FashionMNIST(Path(__file__).parent / "data",
                               train=True,
                               download=True,
                               transform=T.Compose([T.ToTensor(),
                                                    T.Normalize(DATA_MEAN, DATA_STD),
                                                    T.Lambda(lambda x: x.repeat(2, 1, 1))]))

    train_ds, val_ds = random_split(trainval_ds,
                                    [n_train, n_eval],
                                    generator=torch.Generator().manual_seed(config["seed"]))

    # Additionally add noise to train transform to stabilize training
    train_ds.dataset.transform = T.Compose([train_ds.dataset.transform,
                                            T.Lambda(lambda x: x + config["augm_sigma"] * torch.randn_like(x))])

    return train_ds, val_ds


def get_test_dataset():
    test_ds = FashionMNIST(Path(__file__).parent / "data",
                           train=False,
                           download=True,
                           transform=T.Compose([
                               T.ToTensor(),
                               T.Normalize(DATA_MEAN, DATA_STD),
                               T.Lambda(lambda x: x.repeat(2, 1, 1))
                           ]))

    return test_ds


def get_dataloader(ds, config):
    pin_memory = True if config["device"] == "gpu" else False
    return DataLoader(ds,
                      batch_size=config["batch_size"],
                      shuffle=True,
                      num_workers=2,
                      pin_memory=pin_memory,
                      drop_last=True)
