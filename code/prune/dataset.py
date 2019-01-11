import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

def loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.RandomSizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

def test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)

class DataSet:
    def __init__(self, torch_v=0.4):
        self.torch_v = torch_v

    def loader(self, path, batch_size=32, num_workers=4, pin_memory=True):
        '''normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'''
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if self.torch_v == 0.3:
            resize = transforms.RandomSizedCrop(224)
        else:
            resize = transforms.RandomResizedCrop(224)

        traindata_transforms = transforms.Compose([
            resize,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        return data.DataLoader(
            dataset=datasets.MNIST(root=path, train=True, download=True, transform=traindata_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory)

    def test_loader(self, path, batch_size=32, num_workers=4, pin_memory=True):
        '''normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'''
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if self.torch_v == 0.3:
            resize = transforms.Scale(256)
        else:
            resize = transforms.Resize(256)
        testdata_transforms = transforms.Compose([
            resize,
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
        return data.DataLoader(
            dataset=datasets.MNIST(root=path, train=False, download=True, transform=testdata_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory)
