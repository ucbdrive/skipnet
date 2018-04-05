"""prepare CIFAR and SVHN
"""

from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


crop_size = 32
padding = 4


def prepare_train_data(dataset='cifar10', batch_size=128,
                       shuffle=True, num_workers=4):

    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
    elif 'svhn' in dataset:
        transform_train =transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4377, 0.4438, 0.4728),
                                         (0.1980, 0.2010, 0.1970)),
                ])
        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data',
            split='train',
            download=True,
            transform=transform_train
        )

        transform_extra = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4300,  0.4284, 0.4427),
                                 (0.1963,  0.1979, 0.1995))

        ])

        extraset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data',
            split='extra',
            download=True,
            transform = transform_extra
        )

        total_data =  torch.utils.data.ConcatDataset([trainset, extraset])

        train_loader = torch.utils.data.DataLoader(total_data,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
    else:
        train_loader = None
    return train_loader


def prepare_test_data(dataset='cifar10', batch_size=128,
                      shuffle=False, num_workers=4):

    if 'cifar' in dataset:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.__dict__[dataset.upper()](root='/tmp/data',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
    elif 'svhn' in dataset:
        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4524,  0.4525,  0.4690),
                                         (0.2194,  0.2266,  0.2285)),
                ])
        testset = torchvision.datasets.__dict__[dataset.upper()](
                                               root='/tmp/data',
                                               split='test',
                                               download=True,
                                               transform=transform_test)
        np.place(testset.labels, testset.labels == 10, 0)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
    else:
        test_loader = None
    return test_loader
