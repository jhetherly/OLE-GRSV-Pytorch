import numpy as np
import torch
import torchvision
from torch.utils.data.sampler import Sampler


class ClassBalancedBatchSampler(Sampler):
    r"""Samples elements such that the classes in each batch are balanced.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, class_indices, batch_size, imbalance_protocol='longest', np_rng=None):
        self.data_source = data_source
        self.class_indices = class_indices
        self.batch_size = batch_size
        self.np_rng = np_rng
        self.class_sizes = {key: len(class_indices[key]) for key in class_indices.keys()}
        self.n_classes = len(class_indices.keys())

        if imbalance_protocol.lower() == 'longest':
            self.size = np.max(list(self.class_sizes.values()))
        if imbalance_protocol.lower() == 'shortest':
            self.size = np.min(list(self.class_sizes.values()))
        if imbalance_protocol.lower() == 'mid':
            self.size = np.mean(list(self.class_sizes.values()))
        self.n_batches = int(self.size/batch_size + 0.5)

    def __iter__(self):
        self.n = 0
        if self.np_rng is not None:
            for key in self.class_indices.keys():
                self.np_rng.shuffle(self.class_indices[key])
        return self

    def __next__(self):
        if self.n < self.n_batches:
            result = []
            for key in sorted(self.class_indices.keys()):
                for i in range(self.n*self.batch_size, (self.n + 1)*self.batch_size):
                    result += [self.class_indices[key][i%self.class_sizes[key]]]
            self.n += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return self.n_batches


def create_CIFAR10_dataloaders(base_transform, random_transform, val_frac,
                               training_batch_size,
                               validation_batch_size,
                               testing_batch_size,
                               np_rng=np.random):
    train_dataset = torchvision.datasets.CIFAR10(root='data', train=True,
                                                 download=True, transform=random_transform)
    val_dataset = torchvision.datasets.CIFAR10(root='data', train=True,
                                               download=True, transform=base_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='data', train=False,
                                                download=True, transform=base_transform)
    initial_train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                                           shuffle=False, num_workers=0)
    initial_test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                          shuffle=False, num_workers=0)
    
    train_class_indices = {}
    for i, data in enumerate(initial_train_dataloader, 0):
        _, labels = data
        label = labels[0]

        train_class_indices.setdefault(np.asscalar(label.data.numpy()), []).append(i)
    
    val_class_indices = {}
    for c in train_class_indices.keys():
        indices = train_class_indices[c]
        np_rng.shuffle(indices)
        n_val_samples = int(len(indices)*val_frac)
        train_class_indices[c] = indices[:-n_val_samples]
        val_class_indices[c] = indices[-n_val_samples:]

    test_class_indices = {}
    for i, data in enumerate(initial_test_dataloader, 0):
        _, labels = data
        label = labels[0]

        test_class_indices.setdefault(np.asscalar(label.data.numpy()), []).append(i)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_sampler=ClassBalancedBatchSampler(train_dataset,
            train_class_indices, batch_size=training_batch_size,
            np_rng=np_rng),
        num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
        batch_sampler=ClassBalancedBatchSampler(val_dataset,
            val_class_indices, batch_size=validation_batch_size,
            np_rng=np_rng),
        num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
        batch_sampler=ClassBalancedBatchSampler(test_dataset,
            test_class_indices, batch_size=testing_batch_size,
            np_rng=np_rng),
        num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader
