import numpy as np
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

NUM_WORKERS = os.cpu_count()


def load_split_train_val(datadir, valid_size=.2, batch_size=64, num_workers=0):
    transformations = transforms.Compose([
        transforms.Resize([224, 244]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # TO-DO use different transform for validation set

    train_data = datasets.ImageFolder(datadir, transform=transformations)

    val_data = datasets.ImageFolder(datadir,
                                    transform=transformations)
    image_datasets = {'train': train_data, 'val': val_data}

    num_train = len(image_datasets['train'])

    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_data, sampler=val_sampler, batch_size=batch_size,
                                             num_workers=num_workers)

    # dataloaders = {'train': trainloader, 'val': valloader}
    return train_loader, val_loader, image_datasets

# def create_dataloaders(
#     train_dir: str,
#     test_dir: str,
#     transform: transforms.Compose,
#     batch_size: int,
#     num_workers: int=NUM_WORKERS
# ):
#   """Creates training and testing DataLoaders.
#   Takes in a training directory and testing directory path and turns
#   them into PyTorch Datasets and then into PyTorch DataLoaders.
#   Args:
#     train_dir: Path to training directory.
#     test_dir: Path to testing directory.
#     transform: torchvision transforms to perform on training and testing data.
#     batch_size: Number of samples per batch in each of the DataLoaders.
#     num_workers: An integer for number of workers per DataLoader.
#   Returns:
#     A tuple of (train_dataloader, test_dataloader, class_names).
#     Where class_names is a list of the target classes.
#     Example usage:
#       train_dataloader, test_dataloader, class_names = \
#         = create_dataloaders(train_dir=path/to/train_dir,
#                              test_dir=path/to/test_dir,
#                              transform=some_transform,
#                              batch_size=32,
#                              num_workers=4)
#   """
#   # Use ImageFolder to create dataset(s)
#   train_data = datasets.ImageFolder(train_dir, transform=transform)
#   test_data = datasets.ImageFolder(test_dir, transform=transform)
#
#   # Get class names
#   class_names = train_data.classes
#
#   # Turn images into data loaders
#   train_dataloader = DataLoader(
#       train_data,
#       batch_size=batch_size,
#       shuffle=True,
#       num_workers=num_workers,
#       pin_memory=True,
#   )
#   test_dataloader = DataLoader(
#       test_data,
#       batch_size=batch_size,
#       shuffle=False,
#       num_workers=num_workers,
#       pin_memory=True,
#   )
#
#   return train_dataloader, test_dataloader, class_names