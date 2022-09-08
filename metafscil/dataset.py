import os
import random
from typing import Tuple

import torch
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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
    def __init__(self, imgs, labels, transform=True):
        self.img_name = imgs
        self.labels = tensor(labels, dtype=torch.long)
        self.transform = transform
        if transform:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(84),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(84),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.img_name)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_name[index]
        label = self.labels[index]
        img = self._load_and_preprocess_image(img_path)
        img_t = img
        img_t = self.transform(img_t)
        return img_t, label

    def _load_and_preprocess_image(self, img_path) -> tensor:
        img = read_image(img_path, mode=ImageReadMode.RGB)
        return img.float().div(255)


def get_pretrain_dataloader(
        path: str, batch_size: int, transform=True, shuffle=True) -> DataLoader:
    dict_of_imgs, classes = get_metadata(path)
    imgs = []
    labels = []
    for i, cls in enumerate(classes):
        for img in dict_of_imgs[cls]:
            imgs.append(img)
            labels.append(i)
    dataset = MiniImagenetDataset(imgs, labels, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=12, pin_memory=True)
    return dataloader


class SequentialTaskSampler:
    def __init__(
        self,
        path: str,
        n_way: int = 5,
        n_shot: int = 5,
        n_query: int = 250,
        n_support: int = 250,
        n_base_task: int = 20,
        n_sample_base_per_class: int = 50,
        n_sample_query: int = 50
    ):
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_support = n_support
        self.n_base_task = n_base_task
        self.n_sample_base_per_class = n_sample_base_per_class
        self.n_sample_query = n_sample_query
        self.dict_of_imgs, self.classes = get_metadata(path)
        self.n_class = len(self.classes)
        self.shuffle_classes = self.classes[:]

        # self._separate_images()
        self.support_pools = dict()
        self.query_pools = dict()

    # def _separate_images(self):
    #     self.support_pools = dict()
    #     self.query_pools = dict()
    #     for cls in self.classes:
    #         imgs = self.dict_of_imgs[cls]
    #         random.shuffle(imgs)
    #         self.support_pools[cls] = imgs[:self.n_support]
    #         self.query_pools[cls] = imgs[self.n_support:]

    def new_sequence(self):
        # self.query_imgs = []
        # self.query_labels = []
        random.shuffle(self.shuffle_classes)
        for cls in self.shuffle_classes:
            imgs = self.dict_of_imgs[cls]
            random.shuffle(imgs)
            self.support_pools[cls] = imgs[:self.n_support]
            self.query_pools[cls] = imgs[self.n_support:]
            # random.shuffle(self.support_pools[cls])
            # random.shuffle(self.query_pools[cls])
        self.session = -1

    def new_session(self):
        self.session += 1
        self.support_imgs = []
        self.support_labels = []
        self.query_imgs = []
        self.query_labels = []
        if self.session == 0:
            start_cls = 0
            end_cls = self.n_base_task
            self.session_n_class = end_cls
            n_support_imgs = self.n_sample_base_per_class
        else:
            start_cls = self.n_base_task + (self.session-1) * self.n_way
            end_cls = start_cls + self.n_way
            self.session_n_class += self.n_way
            n_support_imgs = self.n_shot
        classes = self.shuffle_classes[start_cls:end_cls]
        for cls in classes:
            self.support_imgs += self.support_pools[cls][:n_support_imgs]
            self.support_labels += [self.shuffle_classes.index(cls)] * n_support_imgs
        for cls in self.shuffle_classes[:end_cls]:
            random.shuffle(self.query_pools[cls])
            self.query_imgs += self.query_pools[cls][:self.n_sample_query]
            self.query_labels += [self.shuffle_classes.index(cls)] * self.n_sample_query

        self.support_loader = self._get_loader(self.support_imgs, self.support_labels,
                                               min(len(self.support_imgs), 128), shuffle=True)
        self.query_loader = self._get_loader(self.query_imgs, self.query_labels, self.n_sample_query, shuffle=False)

        self.support_iter = iter(self.support_loader)
        self.query_iter = iter(self.query_loader)

    def _get_loader(self, imgs, labels, batch_size, shuffle=True):
        dataset = MiniImagenetDataset(imgs, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=2)
        return dataloader

    def sample_query(self):
        try:
            return next(self.query_iter)
        except StopIteration:
            self.query_iter = iter(self.query_loader)
            return next(self.query_iter)

    def sample_support(self):
        try:
            return next(self.support_iter)
        except StopIteration:
            self.support_iter = iter(self.support_loader)
            return next(self.support_iter)


class EpisodeSampler:
    def __init__(
        self,
        n_way: int = 5,
        n_shot: int = 5,
        n_base_task: int = 60,
    ):
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_base_task = n_base_task
        self.new_sequence()

    def new_sequence(self):
        self.query_imgs = []
        self.query_labels = []
        self.classes = []
        self.session = -1

    def new_session(self):
        self.session += 1

        # training set
        self.support_imgs = []
        self.support_labels = []
        imgs, clss = get_metadata(f"metadata/mini_imagenet/session_{self.session + 1}.txt")
        self.classes += clss
        for cls in clss:
            self.support_imgs += imgs[cls][:]
            self.support_labels += [self.classes.index(cls)] * len(imgs[cls])

        # testing set
        imgs, _ = get_metadata(f"metadata/mini_imagenet/test_{self.session + 1}.txt")
        for cls in clss:
            self.query_imgs += imgs[cls][:]
            self.query_labels += [self.classes.index(cls)] * len(imgs[cls])

        self.support_loader = self._get_loader(self.support_imgs, self.support_labels, min(len(self.support_imgs), 256))
        self.query_loader = self._get_loader(self.query_imgs, self.query_labels, 100, transform=False)

        self.support_iter = iter(self.support_loader)

    def _get_loader(self, imgs, labels, batch_size, transform=True):
        dataset = MiniImagenetDataset(imgs, labels, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
        return dataloader

    def sample_support(self):
        try:
            return next(self.support_iter)
        except StopIteration:
            self.support_iter = iter(self.support_loader)
            return next(self.support_iter)
