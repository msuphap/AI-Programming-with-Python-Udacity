import time
import torch
import json
import numpy as np
import argparse, sys
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'valid': transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                   }

# TODO: Load the datasets with ImageFolder
directories = {'train': train_dir, 
               'valid': valid_dir, 
               'test' : test_dir}

image_datasets = {x: datasets.ImageFolder(directories[x], transform=data_transforms[x])
                  for x in ['train', 'valid', 'test']}
