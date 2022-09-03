import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import resize
from torchvision.io import read_image, ImageReadMode


def get_metadata(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
    classes = []
    dict_of_imgs = dict()
    for line in lines:
        folder, _, cls, path = line.strip().split('/')
        if cls not in classes:
            classes.append(cls)
            dict_of_imgs[cls] = []
        dict_of_imgs[cls].append(os.path.join(folder, 'images', path))
    return dict_of_imgs, classes


class MiniImagenetDataset(Dataset):
    def __init__(self, imgs, labels, device, transform=True):
        self.img_name = imgs
        self.labels = tensor(labels, dtype=torch.long)
        self.transform = transform
        self.device = device
        if transform:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(84),
                transforms.RandomHorizontalFlip()
            ])

    def __len__(self) -> int:
        return len(self.img_name)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_name[index]
        label = self.labels[index]
        img = self._load_and_preprocess_image(img_path)
        img_t = img
        if self.transform:
            img_t = self.transform(img_t)
        return img_t, label

    def _load_and_preprocess_image(self, img_path) -> tensor:
        img = read_image(img_path, mode=ImageReadMode.RGB)
        img = resize(img, (84, 84))
        return img.float().div(255)


def get_pretrain_dataloader(
        path: str, batch_size: int, device: torch.device, transform=True, shuffle=True) -> DataLoader:
    dict_of_imgs, classes = get_metadata(path)
    imgs = []
    labels = []
    for i, cls in enumerate(classes):
        for img in dict_of_imgs[cls]:
            imgs.append(img)
            labels.append(i)
    dataset = MiniImagenetDataset(imgs, labels, device, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=6, drop_last=True)
    return dataloader


class SequentialTaskSampler:
    pass
