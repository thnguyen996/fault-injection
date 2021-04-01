import argparse
import collections
import cProfile

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
from torchvision import datasets, transforms
from pytorch_memlab import profile_every

import sasimulate.SAsimulate2
import sasimulate.SAsimulate3
from sasimulate import *
import sasimulate.weight_mapping as wmp
from sasimulate.binary_converter import bit2float, float2bit
torch.set_printoptions(profile='full')

# Inject error without doing anything to the weight
def method0(state_dict, total_param, error_total, device):
    device = device
    with torch.no_grad():
        for name, param in state_dict.items():
            if "weight" not in name:
                continue
            else:
                shape = param.data.shape
                print(name, shape)
                error_layer = (param.numel() / total_param) * error_total
                param_binary = float2bit(
                    param, num_e_bits=8, num_m_bits=23, bias=127.0
                ) > 0.
                mask, mask1 = SAsimulate3.create_mask_bool(shape, error_layer)
                output = SAsimulate3.make_SA_bool(param_binary.view(-1), mask, mask1)
                output = bit2float(output.view(param_binary.shape).type(torch.int8))
                param.data = output.view(shape)
                torch.cuda.empty_cache()
    return state_dict


# Perform weight mapping
def method1(state_dict, total_param, mapped_float, binary_path, error_total, device):
    index = torch.arange(16).to(device)
    index_map = wmp.mapallweights2(index).squeeze()
    _, indicies = torch.sort(index_map, dim=0)
    with torch.no_grad():
        for name, param in state_dict.items():
            if "weight" not in name:
                continue
            else:
                error_layer = (param.numel() / total_param) * error_total
                mapped_binary_dict = torch.load(
                    binary_path + str(name) + "_binary.pt",
                    map_location=device
                )
                mapped_binary_val = mapped_binary_dict[name]
                mapped_binary_val = mapped_binary_val.type(torch.int8)
                output = weight_map(
                    param.data, mapped_float[name], mapped_binary_val, error_layer, indicies.to(device), device
                )
                param.data = output
    return state_dict


def weight_map(weights, mapped_float, mapped_binary, error_rate, indicies, device):
    shape = weights.shape
    weights_flat = weights.view(-1)
    if weights_flat.numel() > 16:
        weight_binary = mapped_binary
    else:
        return weights
    # Creating masks for all weights in one layer
    mask0_binary, mask1_binary = SAsimulate3.create_mask(shape, error_rate=error_rate)
    # reporter = MemReporter()
    # reporter.report()
    mask0_binary, mask1_binary = (
        mask0_binary.view(int(mask0_binary.numel() / 32 / 16), 16, 32),
        mask1_binary.view(int(mask1_binary.numel() / 32 / 16), 16, 32),
    )
    new_weight_binary = torch.empty(
        [*mapped_binary.shape], device=device, dtype=torch.int8
    )
    for i in range(16):
        new_weight_binary[:, :, i, :] = SAsimulate3.make_SA(
            mapped_binary[:, :, i, :], mask0_binary, mask1_binary
        )
    half_shape = int(new_weight_binary.shape[0] / 2)
    new_weight = torch.empty(half_shape * 2, 16, 16, device=device)
    new_weight[0:half_shape, ...] = bit2float(
        new_weight_binary[0:half_shape, ...], num_e_bits=8, num_m_bits=23, bias=127.0
    )
    new_weight[half_shape : new_weight.shape[0], ...] = bit2float(
        new_weight_binary[half_shape : new_weight.shape[0], ...],
        num_e_bits=8,
        num_m_bits=23,
        bias=127.0,
    )
    binary_index = 0
    weight_index = 0

    dev_map = abs(mapped_float - new_weight) # Calculate deviation
    dev_sum_map = torch.sum(dev_map, dim=1)
    min_dev, best_map = torch.min(dev_sum_map, dim=1) # calculate best mapping
    best_map3d = best_map.unsqueeze(1).repeat(1, 16).unsqueeze(1)
    best_map_16 = torch.gather(new_weight, dim=1, index=best_map3d).squeeze(1) 
    idx_map = torch.index_select(indicies, dim=0, index=best_map)
    weight_remap = torch.gather(best_map_16, dim=1, index=idx_map)
    new_weights = weight_remap.view(shape)

    return new_weights


# Perform weight mapping and encoding
def method2(state_dict, total_param, mapped_float, binary_path, error_total, device):
    index = torch.arange(16).to(device)
    index_map = wmp.mapallweights2(index).squeeze()
    index_map = torch.cat((index_map, index_map), dim=0)
    _, indicies = torch.sort(index_map, dim=1)
    with torch.no_grad():
        for name, param in state_dict.items():
            if "weight" not in name:
                continue
            else:
                error_layer = (param.numel() / total_param) * error_total
                mapped_binary_dict = torch.load(
                    binary_path + str(name) + "_binary.pt",
                    map_location=device
                )
                mapped_binary_val = mapped_binary_dict[name]
                # mapped_binary_val = mapped_binary_val.type(torch.int8)
                output = weight_map2(
                    param.data, mapped_float[name], mapped_binary_val, error_layer, indicies.to(device), device
                )
                param.data = output
    return state_dict

# @profile_every(1)
def weight_map2(weights, mapped_float, mapped_binary, error_rate, indicies, device):
    shape = weights.shape
    weights_flat = weights.view(-1)
    if weights_flat.numel() > 16:
        weight_binary = mapped_binary
    else:
        return weights

    # Creating masks for all weights in one layer
    mask0_binary, mask1_binary = SAsimulate3.create_mask_bool(shape, error_rate=error_rate)
    # reporter = MemReporter()
    # reporter.report()
    mask0_binary, mask1_binary = (
        mask0_binary.view(int(mask0_binary.numel() / 32 / 16), 16, 32),
        mask1_binary.view(int(mask1_binary.numel() / 32 / 16), 16, 32),
    )

    flip_mapped = ~mapped_binary
    mapped_binary = torch.cat((mapped_binary, flip_mapped), dim=1)
    new_weight_binary = torch.empty(
        [*mapped_binary.shape], device=device, dtype=torch.bool
    )

    for i in range(32):
        new_weight_binary[:, i, :, :] = SAsimulate3.make_SA_bool(
                mapped_binary[:, i, :, :], mask0_binary, mask1_binary
        )
    new_weight_binary[:, 16:32, :, :] = ~new_weight_binary[:, 16:32, :, :]

    new_weight = torch.empty(new_weight_binary.shape[0], 32, 16, device=device)
    for idx in range(32):
        new_binary = new_weight_binary[:, idx, ...]
        new_weight[:, idx, :] = bit2float(new_binary.type(torch.int8))

    # half_shape = int(new_weight_binary.shape[0] / 4)
    # new_weight = torch.empty(half_shape * 4, 32, 16, device=device)
    # def part_weight_binary(new_weight_binary):
    #     for idx in range(0, new_weight_binary.shape[0], half_shape):
    #         yield new_weight_binary[idx:idx+half_shape, ...].type(torch.int8)

    # part_binary = part_weight_binary(new_weight_binary)
    # for idx, binary in zip(range(0, new_weight_binary.shape[0], half_shape), part_binary):
    #     new_weight[idx:idx+half_shape] = bit2float(binary)
    #     torch.cuda.empty_cache()

    # new_weight[0:half_shape, ...] = bit2float(
    #     next(part_binary), num_e_bits=8, num_m_bits=23, bias=127.0
    # )
    # new_weight[half_shape : new_weight.shape[0], ...] = bit2float(
    #     next(part_binary),
    #     num_e_bits=8,
    #     num_m_bits=23,
    #     bias=127.0,
    # )

    binary_index = 0
    weight_index = 0
    mapped_float = torch.cat((mapped_float, mapped_float), dim=1)

    dev_map = abs(mapped_float - new_weight) # Calculate deviation
    dev_sum_map = torch.sum(dev_map, dim=2)
    min_dev, best_map = torch.min(dev_sum_map, dim=1) # Calculate best mapping
    best_map3d = best_map.unsqueeze(1).repeat(1, 16).unsqueeze(1)
    best_map_16 = torch.gather(new_weight, dim=1, index=best_map3d).squeeze(1) # Choose best case in 32 cases
    idx_map = torch.index_select(indicies, dim=0, index=best_map)
    weight_remap = torch.gather(best_map_16, dim=1, index=idx_map)  # remap best mapping
    new_weights = weight_remap.view(shape)

    return new_weights

# ECC 
def ECC_method(state_dict, total_param, error_total, device):
    device = device
    with torch.no_grad():
        for name, param in state_dict.items():
            if "weight" not in name:
                continue
            else:
                shape = param.data.shape
                error_layer = (param.numel() / total_param) * error_total
                param_binary = float2bit(
                    param.data, num_e_bits=8, num_m_bits=23, bias=127.0
                )
                mask, mask1 = SAsimulate2.create_mask(param_binary, error_layer)
                output = SAsimulate2.make_SA_ECC(param.data.view(-1), mask, mask1)
                correct_binary = ECC(output, param_binary)
                float_tensor = bit2float(correct_binary, num_e_bits=8, num_m_bits=23, bias=127.0)
                param.data = float_tensor.view(shape)
    return state_dict

# ECC
# Input: Error tensor, original tensor
# Output: Corrected tensor using ecc 64 bits

def ECC(error_tensor, original_tensor):
    shape = error_tensor.shape
    error_flatten = error_tensor.view(-1)
    original_flatten = original_tensor.view(-1)
    error_bool = error_flatten > 0.0
    original_bool = original_flatten > 0.0
    stuck_bits = (error_bool ^ original_bool).float()
    stuck_bits_64 = stuck_bits.view(int(stuck_bits.numel() / 64), 64)
    error_64 = error_flatten.view(stuck_bits_64.shape)
    original_64 = original_flatten.view(stuck_bits_64.shape)
    sum_64 = torch.sum(stuck_bits_64, dim=1)
    index = torch.where(sum_64 == 1.)
    if index[0].shape[0] == 0:
        return error_tensor
    else:
        for i in index:
            error_64[i, :] = original_64[i, :]
    correct_tensor = error_64.view(shape)
    return correct_tensor


if __name__ == "__main__":
    main()
