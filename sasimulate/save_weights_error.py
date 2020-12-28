import argparse
import collections
import cProfile
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pdb
import random
import shutil
import time
import warnings
from datetime import datetime
from pprint import pprint
from pyinstrument import Profiler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import SAsimulate2
import weight_mapping as wmp
from binary_converter import bit2float, float2bit
from models import *
from utils import progress_bar


now = datetime.now().date()
ran = random.randint(1, 231242)

def test(net, testloader, device, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save best accuracy.
    acc = 100.*correct/total
    return acc

def main():
   

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda")

    # Initialize Tensorboard
   
    # writer = SummaryWriter('runs/{}-{}'.format(now, "method0 (50 points) 1e-10 -- 1e-02, 5e-08"))
    # Create model
    model = ResNet18().to(device)
    #Load model state dict
    state_dict = torch.load("./checkpoint/resnet.pt")['net']
    simulate = SAsimulate(test_loader, model, state_dict, method="method0", mapped_gen = [])  
    # error_range = np.linspace(1e-10, 1e-02, 100)
    error_range = np.arange(1e-10, 1e-02, 5e-07)
    simulate.run(error_range)

class SAsimulate:
    def __init__(self, test_loader, model, state_dict, method, mapped_gen):
        self.test_loader = test_loader
        self.model = model
        self.state_dict = state_dict
        self.method = method
        self.mapped_gen = mapped_gen
        self.device = torch.device("cuda")
        model.load_state_dict(state_dict)
        self.criterion = nn.CrossEntropyLoss()
        model.to(self.device)

    def run(self, error_range):
      count = 0
      avr_error = 0.
      orig_weights_list = []

      orig_model = self.model
      total_param = 11169152
      if self.method == "method0":
          for error_total in error_range:
              count += 1
              print("Error rate: ", error_total)
              # model = method0(orig_model, total_param, error_total)
              state_dict = method0(self.state_dict, total_param, error_total)
              torch.save(state_dict, "./save_weights_error/error_rate_"+str(error_total)+".pt")
# Inject error without doing anything to the weight
def method0(state_dict, total_param, error_total):
    device = torch.device("cuda")
    with torch.no_grad():
      for name, param in state_dict.items():
          if "weight" not in name:
              continue
          else:
            shape = param.data.shape
            error_layer = (param.numel()/total_param)*error_total
            param_binary = float2bit(param.data, num_e_bits=8, num_m_bits=23, bias=127.)
            mask, mask1 = SAsimulate2.create_mask(param_binary, error_layer)
            output = SAsimulate2.make_SA2(param.data.view(-1), mask, mask1)
            param.data = output.view(shape)
    return state_dict

if __name__ == '__main__':
    main()

