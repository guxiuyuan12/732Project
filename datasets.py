import os
import pandas as pd
import torch
import torch.utils.data as data
import torchvision
import sys
from PIL import Image


class CatDogDataset(data.Dataset):
    def __init__(self, transform=None):
        super(CatDogDataset, self).__init__()
        self.root_dir = './data/train'

        self.transform = transform
        self.label = {"cat": 1, "dog": 0}

    def __len__(self):
        return 25000

    def __getitem__(self, idx):
        if idx % 2 == 0:
            label = self.label["cat"]
            img_path = os.path.join(self.root_dir, 'cat.' + str(idx//2) + '.jpg')
        else:
            label = self.label["dog"]
            img_path = os.path.join(self.root_dir, 'dog.' + str(idx//2)+'.jpg')

        image = Image.open(img_path, mode='r')
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class CatDogTrainset(data.Dataset):
    def __init__(self, transform=None):
        super(CatDogTrainset, self).__init__()
        self.root_dir = './data/train'

        self.transform = transform
        self.label = {"cat": 1, "dog": 0}

    def __len__(self):
        return 22000

    def __getitem__(self, idx):
        if idx % 2 == 0:
            label = self.label["cat"]
            img_path = os.path.join(self.root_dir, 'cat.' + str(idx//2) + '.jpg')
        else:
            label = self.label["dog"]
            img_path = os.path.join(self.root_dir, 'dog.' + str(idx//2)+'.jpg')

        image = Image.open(img_path, mode='r')
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class CatDogValset(data.Dataset):
    def __init__(self, transform=None):
        super(CatDogValset, self).__init__()
        self.root_dir = './data/train'

        self.transform = transform
        self.label = {"cat": 1, "dog": 0}

    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        idx += 22000
        if idx % 2 == 0:
            label = self.label["cat"]
            img_path = os.path.join(self.root_dir, 'cat.' + str(idx//2) + '.jpg')
        else:
            label = self.label["dog"]
            img_path = os.path.join(self.root_dir, 'dog.' + str(idx//2)+'.jpg')

        image = Image.open(img_path, mode='r')
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
