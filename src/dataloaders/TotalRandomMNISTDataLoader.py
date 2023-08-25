import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from torchvision.transforms import transforms


class TotalRandomMNISTDataset(Dataset):
    def __init__(self, mnist_path, p=0.0, train=True, transform=None):
        super().__init__()
        self.train_ds = MNIST(
            mnist_path, train=train, download=True, transform=transform
        )
        self.p = p
        self.transform = None
        self.images = self.train_ds.data
        self.targets = self.train_ds.targets.numpy()
        self.distribution_dict = {}
        for idx, v in enumerate(self.targets):
            if v not in self.distribution_dict:
                self.distribution_dict[v] = [idx]
            else:
                self.distribution_dict[v].append(idx)

    def __len__(self):
        return self.targets.size

    def __getitem__(self, item):
        image, label = self.train_ds.__getitem__(
            item
        )  # self.images[item] #torch.unsqueeze(self.images[item], dim=0).double()
        assert label == self.targets[item]
        # label = self.targets[item]
        from scipy.stats import bernoulli

        res = bernoulli.rvs(p=self.p)
        out = None
        if res == 1:
            # idx = np.random.Generator.choice(self.distribution_dict[label])
            idx = np.random.choice(list(range(self.targets.size)))
            out, a_label = self.train_ds.__getitem__(
                idx
            )  # torch.unsqueeze(self.images[idx],dim=0).double()
            # assert label == a_label
        else:
            out = image

        return image, out, label


class TotalRandomMNISTDataloader(pl.LightningDataModule):
    def __init__(self, batch_size, path, p=0.0, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.p = float(p)
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            ]
        )

        self.train_ds = TotalRandomMNISTDataset(path, p=self.p, transform=transform)

        self.val_ds = TotalRandomMNISTDataset(
            path, train=False, p=0.0, transform=transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            num_workers=os.cpu_count(),
        )


def main():
    train_ds = MNIST("../../Datasets/mnist", train=True, download=True)
    target = train_ds.targets.numpy()

    labels = np.unique(target)
    target = list(target)
    distribution = [target.count(v) for v in labels]
    plt.bar(x=labels, height=distribution)
    plt.show()


if __name__ == "__main__":
    main()
