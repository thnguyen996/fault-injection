from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sasimulate.binary_converter import float2bit, bit2float
import torchvision.models as models
import pdb
import numpy as np
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

## Inject stuck at fault into weights using 2 masks
def make_SA(weights, mask, mask1):
    ## Inject errors
    output = ((weights + mask) > 0.).float() # inject stuck at 0
    output = ((output - mask1)> 0.).float()       # inject stuck at 1
    # output = output.view(16, 16, 32)
    # float_tensor = bit2float(output, num_e_bits=8, num_m_bits=23, bias=127.)
    return output

#Calculate total numbers of error bits,
# input: Flatten binary weights and mask
# Output: Number of stuck bits 
def calculate_stuck(conv_binary, mask, mask1):
    stuck0 = torch.sum(mask*conv_binary, dim=1)
    stuck1 = torch.sum(mask1, dim=1) - torch.sum(mask1*conv_binary, dim=1)
    stuck_total = stuck0 + stuck1
    return stuck_total

##Calculate total numbers of error bits,
## input: Flatten binary weights and mask
## Output: Number of stuck bits 
#def calculate_stuck(conv_binary, mask, mask1):
#    stuck0 = torch.sum(mask*conv_binary)
#    stuck1 = torch.sum(mask1) - torch.sum(mask1*conv_binary)
#    stuck_total = stuck0 + stuck1
#    return stuck_total

## Create 2 mask: Stuck at 0 and stuck at 1
def create_mask(weight_shape, error_rate):
    shape_list = [*weight_shape, 32]
    mask = torch.zeros(shape_list, dtype=torch.int8)
    mask1 = torch.zeros(shape_list, dtype=torch.int8)
    num_SA = (error_rate * np.prod(shape_list))
    mask = mask.view(-1).to("cuda")                 # mask flatten
    mask1= mask1.view(-1).to("cuda")
    if int(num_SA) != 0:
        error_list = torch.randint(high=mask.numel(), size=(int(num_SA/2), 2), device="cuda", dtype=torch.int32)
        SA0_idx, SA1_idx = error_list[:, 0], error_list[:, 1]
        mask = mask.scatter_(-1, SA0_idx.type(torch.long), torch.ones(mask.shape, device="cuda", dtype=torch.int8))
        mask1 = mask1.scatter_(-1, SA1_idx.type(torch.long), torch.ones(mask1.shape, device="cuda", dtype=torch.int8))
    return mask, mask1

## Create 2 mask: Stuck at 0 and stuck at 1
def create_mask_bool(weight_shape, error_rate):
    shape_list = [*weight_shape, 32]
    mask = torch.zeros(shape_list, dtype=torch.bool)
    mask1 = torch.zeros(shape_list, dtype=torch.bool)
    num_SA = (error_rate * np.prod(shape_list))
    mask = mask.view(-1).to("cuda")                 # mask flatten
    mask1= mask1.view(-1).to("cuda")
    if int(num_SA) != 0:
        error_list = torch.randint(high=mask.numel(), size=(int(num_SA/2), 2), device="cuda", dtype=torch.int32)
        SA0_idx, SA1_idx = error_list[:, 0], error_list[:, 1]
        mask = mask.scatter_(-1, SA0_idx.type(torch.long), torch.ones(mask.shape, device="cuda", dtype=torch.bool))
        mask1 = mask1.scatter_(-1, SA1_idx.type(torch.long), torch.ones(mask1.shape, device="cuda", dtype=torch.bool))
    return mask, mask1

## Inject stuck at fault into weights using 2 masks
def make_SA_bool(weights, mask, mask1):
    ## Inject errors
    # output = ((weights + mask) > 0.)  # inject stuck at 0
    # output = ((output - mask1)> 0.)   # inject stuck at 1
    not_mask0 = torch.bitwise_not(mask) 
    output = torch.bitwise_and(weights, not_mask0) # inject stuck at 0
    output = torch.bitwise_or(output, mask1) # inject stuck at 1
    return output
# x = torch.randn(32, 1, 3, 3)

# output, stuck_total = make_SA(x, 0.00001)
# print(stuck_total)



