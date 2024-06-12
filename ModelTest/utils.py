from torchvision import models
from torch import nn
import time
import logging
import torch
import numpy as np



def load_vgg16():
    model = models.vgg16(weights=None)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=num_ftrs, out_features=2)
    return model


def load_resnet():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model


def load_inception():
    model = models.inception_v3(weights=None, init_weights=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

def create_logger(logs_dir, level=logging.DEBUG, log_format=None):
    if log_format is None:
        log_format = logging.Formatter("%(asctime)s: %(message)s")

    logger = logging.getLogger("train-logger")
    logger.setLevel(level)
    f_handler = logging.FileHandler(
        logs_dir / f'train_{time.strftime("%d%m%Y-%H%M%S")}.log'
    )
    f_handler.setFormatter(log_format)

    p_handler = logging.StreamHandler()
    p_handler.setFormatter(log_format)

    logger.addHandler(f_handler)
    logger.addHandler(p_handler)

    return logger


def create_loaders(
    batch_size, dataset, train_idx, test_idx
):
    """Create necessary loaders to train the model, based on the actual fold

    Args:
        batch_size (int): Number of images to load in a single training step
        dataset (torchvision.datasets.ImageFolder): Dataset with loaded images
        train_idx (np.ndarray): Indexes of the train set images
        test_idx (np.ndarray): Indexes of the test set images

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: train and test dataloader
    """
    np.random.shuffle(train_idx)
    train_idx, val_idx = np.split(train_idx, [int(len(train_idx) * 0.8)])
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_subsampler,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_subsampler,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_subsampler,
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    m = load_inception()