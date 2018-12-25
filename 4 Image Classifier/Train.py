
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import tensor
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
import customize_flore


parser = argparse.ArgumentParser()

# 设置保存检查点的目录：
parser.add_argument('data_dir', nargs='*', action='store', default='./flowers/', help='train on the data directory')
parser.add_argument('--save_dir', dest='save_dir', action='store', default='./checkpoint.pth', help='the path where to save the checkpoint')

# 选择架构
parser.add_argument('--arch', dest='arch', action='store', default='vgg13', type = str, help='choose a pytorch model')

#设置超参数
parser.add_argument('--learning_rate', dest='learning_rate', action='store', default=0.01)
parser.add_argument('--hidden_units', type=int, dest='hidden_units', action='store', default=512)
parser.add_argument('--epochs', dest='epochs', action="store", type=int, default=20)

#使用 GPU 进行训练
parser.add_argument('--gpu', dest='gpu', action='store', default='gpu')

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs

trainloader, valiloader, testloader = customize_flore.load_data(where)

model, optimizer, criterion = customize_flore.preparation(structure,dropout,hidden_layer1,lr,power)

customize_flore.train_network(model, optimizer, criterion, epochs, 20, trainloader, power)

customize_flore.save_checkpoint(path,structure,hidden_layer1,dropout,lr)


print("Done!") 

