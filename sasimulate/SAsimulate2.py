from __future__ import print_function

import argparse
import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from sasimulate.binary_converter import bit2float, float2bit
from pyinstrument import Profiler

## Example Convol Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

## Inject stuck at fault into weights using 2 masks
def make_SA2(weights, mask, mask1):
    assert weights.shape == weights.view(-1).shape
    assert mask.shape == mask.view(-1).shape
    assert mask1.shape == mask1.view(-1).shape
    conv_binary = float2bit(weights, num_e_bits=8, num_m_bits=23, bias=127.)
    shape = conv_binary.shape
    conv_binary = conv_binary.view(-1)
  ## Inject errors
    output = ((conv_binary + mask) > 0.).float() # inject stuck at 0
    output = ((output - mask1)> 0.).float()       # inject stuck at 1
    output = output.view(shape)
    float_tensor = bit2float(output, num_e_bits=8, num_m_bits=23, bias=127.)
    return float_tensor

## Inject stuck at fault into weights using 2 masks, for ECC
def make_SA_ECC(weights, mask, mask1):
    assert weights.shape == weights.view(-1).shape
    assert mask.shape == mask.view(-1).shape
    assert mask1.shape == mask1.view(-1).shape
    conv_binary = float2bit(weights, num_e_bits=8, num_m_bits=23, bias=127.)
    shape = conv_binary.shape
    conv_binary = conv_binary.view(-1)
  ## Inject errors
    output = ((conv_binary + mask) > 0.).float() # inject stuck at 0
    output = ((output - mask1)> 0.).float()       # inject stuck at 1
    output = output.view(shape)
    # float_tensor = bit2float(output, num_e_bits=8, num_m_bits=23, bias=127.)
    return output

## Inject stuck at fault into weights using 2 masks
def make_SA(weights, mask, mask1):
    assert weights.shape == weights.view(-1).shape
    assert mask.shape == mask.view(-1).shape
    assert mask1.shape == mask1.view(-1).shape
    weights = weights.view(-1)
    ## Inject errors
    output = ((weights + mask) > 0.).float() # inject stuck at 0
    output = ((output - mask1)> 0.).float()       # inject stuck at 1
    output = output.view(int(output.numel()/32), 32)
    float_tensor = bit2float(output, num_e_bits=8, num_m_bits=23, bias=127.)
    return float_tensor

#Calculate total numbers of error bits,
# input: Flatten binary weights and mask
# Output: Number of stuck bits 
def calculate_stuck(conv_binary, mask, mask1):
    stuck0 = torch.sum(mask*conv_binary, dim=1)
    stuck1 = torch.sum(mask1, dim=1) - torch.sum(mask1*conv_binary, dim=1)
    stuck_total = stuck0 + stuck1
    return stuck_total

def create_mask(conv_binary, error_rate):
    shape = conv_binary.shape
    mask = torch.zeros(conv_binary.shape)
    mask1 = torch.zeros(conv_binary.shape)
    mask, mask1 = mask.to("cuda"), mask1.to("cuda")
    num_SA = (error_rate * conv_binary.numel())
    mask = mask.view(-1)                 # mask flatten
    mask1= mask1.view(-1)
    if int(num_SA) != 0:
        error_list = torch.randint(high=mask.numel(), size=(int(num_SA/2), 2), device="cuda", dtype=torch.int32)
        SA0_idx, SA1_idx = error_list[:, 0], error_list[:, 1]
        mask = mask.scatter_(-1, SA0_idx.type(torch.long), torch.ones(mask.shape, device="cuda"))
        mask1 = mask1.scatter_(-1, SA1_idx.type(torch.long), torch.ones(mask1.shape, device="cuda"))
    # index = torch.randperm(mask.numel(), device = torch.device("cuda"))
    # # index = np.random.permutation(mask.numel())
    # mask = mask.view(-1)                 # mask flatten
    # mask1= mask1.view(-1)
    # num_SA0 = index[:int(num_SA)]
    # num_SA1 = index[(index.numel()-int(num_SA)):]
    # mask[num_SA0] = 1.
    # mask1[num_SA1] = 1.
    return mask, mask1



