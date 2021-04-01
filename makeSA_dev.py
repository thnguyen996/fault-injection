import torch
import math
from tqdm import tqdm
import pdb
import numpy as np
import cupy as cp
from collections import OrderedDict
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import random
from sasimulate import weight_mapping as wmp
from pyinstrument import Profiler

np.set_printoptions(threshold=sys.maxsize)

class sa_config:
    def __init__(self, test_loader, model, state_dict, method, writer=False, device=torch.device("cuda"), mapped_float=None, binary_path=None):
        self.test_loader = test_loader
        self.model = model
        self.state_dict = state_dict
        self.method = method
        self.device = device
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.total_param = 0
        self.writer_op = writer
        if method == "method1" or method == "method2":
            assert mapped_float != None, "Insert map weight path"
            self.map_weight_path = mapped_float
        if self.writer_op:
            now = datetime.now().date()
            ran = random.randint(1, 100)
            self.writer = SummaryWriter(
                "runs/{}-{}-{}-deviation".format(
                    now, ran, method
                )
            )
            print("Run ID: {}-{}".format(now, ran))

    def np_to_cp(self):

        for name, param in self.state_dict.items():
            if "weight" in name:
                param_np = param.cpu().numpy()
                param_cp = cp.asarray(param_np)
                cp.save("./save_cp/" + str(name) + ".npy", param_cp)
        print("Converted weights to cupy")

    def run(self, error_range, avr_point, validate, arg, state_dict, weight_path):
        count = 0
        avr_error = 0.0
        if self.method == "method0":
            for error_rate in error_range:
                running_error = []
                running_dev = []
                count += 1
                print("Error rate: ", error_rate)
                for i in range(avr_point):
                    state_dict =  method0(state_dict, weight_path, error_rate, num_bits=32)
                    self.model.load_state_dict(state_dict)
                    torch.cuda.empty_cache()
                    # acc1 = validate(arg)
                    dev = check_dev(weight_path, state_dict)
                    print("Running dev: " , dev)
                    running_dev.append(dev)
                    # running_error.append(100.0 - acc1)

                if self.writer_op:
                    avr_dev = sum(running_dev) / len(running_dev)
                    print("Avarage deviation: ", avr_dev)
                    self.writer.add_scalar("Average Dev", avr_dev, count)
                    self.writer.close()

        if self.method == "method1":
            for error_rate in error_range:
                running_error = []
                count += 1
                print("Error rate: ", error_rate)
                for i in range(avr_point):
                    state_dict =  method1(state_dict, self.map_weight_path, error_rate, num_bits=32)
                    self.model.load_state_dict(state_dict)
                    torch.cuda.empty_cache()
                    acc1 = validate(arg)
                    running_error.append(100.0 - acc1)

                if self.writer_op:
                    avr_error = sum(running_error) / len(running_error)
                    print("Avarage classification Error: ", avr_error)
                    self.writer.add_scalar("Average Error", avr_error, count)
                    self.writer.close()

        if self.method == "method2":
            for error_rate in error_range:
                running_error = []
                running_dev = []
                count += 1
                print("Error rate: ", error_rate)
                for i in range(avr_point):
                    state_dict =  method2(state_dict, self.map_weight_path, error_rate, num_bits=32)
                    self.model.load_state_dict(state_dict)
                    torch.cuda.empty_cache()
                    dev = check_dev(weight_path, state_dict)
                    print("Running dev: " , dev)
                    running_dev.append(dev)
                    # acc1 = validate(arg)

                if self.writer_op:
                    avr_dev = sum(running_dev) / len(running_dev)
                    print("Avarage deviation: ", avr_dev)
                    self.writer.add_scalar("Average Dev", avr_dev, count)
                    self.writer.close()

        if self.method == "ECC":
            set_map = cp.empty(32, dtype=cp.uint32)
            for i in range(32):
                set_map[i] = 2**i
            for error_rate in error_range:
                running_error = []
                count += 1
                print("Error rate: ", error_rate)
                for i in range(avr_point):
                    state_dict =  ECC_method(state_dict, weight_path, error_rate, set_map, num_bits=32)
                    self.model.load_state_dict(state_dict)
                    torch.cuda.empty_cache()
                    acc1 = validate(arg)
                    running_error.append(100.0 - acc1)

                if self.writer_op:
                    avr_error = sum(running_error) / len(running_error)
                    print("Avarage classification Error: ", avr_error)
                    self.writer.add_scalar("Average Error", avr_error, count)
                    self.writer.close()

        if self.method == "ECP":
            set_map = cp.empty(32, dtype=cp.uint32)
            for i in range(32):
                set_map[i] = 2**i
            for error_rate in error_range:
                running_error = []
                running_dev = []
                count += 1
                print("Error rate: ", error_rate)
                for i in range(avr_point):
                    state_dict =  ECP_method(state_dict, weight_path, error_rate, set_map, num_bits=32)
                    self.model.load_state_dict(state_dict)
                    torch.cuda.empty_cache()
                    dev = check_dev(weight_path, state_dict)
                    print("Running dev: " , dev)
                    running_dev.append(dev)
                    # acc1 = validate(arg)

                if self.writer_op:
                    avr_dev = sum(running_dev) / len(running_dev)
                    print("Avarage deviation: ", avr_dev)
                    self.writer.add_scalar("Average Dev", avr_dev, count)
                    self.writer.close()

def create_mask(param, error_rate, num_bits=32):
    # mempool = cp.get_default_memory_pool()
    # if param.numel() > 10000000:
    #     pdb.set_trace()
    if num_bits == 32:
        dtype = cp.uint32
        ftype = cp.float32
    numel = param.numel()
    num_SA = numel * num_bits * error_rate
    total_bits = numel * num_bits

#Generate mask
    mask = cp.random.randint(numel, size=(1, int(num_SA)), dtype=dtype)
    mask_bit = cp.random.randint(num_bits, size=(1, int(num_SA)), dtype=cp.int8)

    if int(num_SA) % 2:
        mask0 = mask[0, 0:int(num_SA/2)]
        mask1 = mask[0, int(num_SA/2)+1:int(num_SA)]
        mask0_bit = mask_bit[0, 0 : int(num_SA / 2)]
        mask1_bit = mask_bit[0, int(num_SA / 2)+1 : int(num_SA)]
    else:
        mask0 = mask[0, 0:int(num_SA/2)]
        mask1 = mask[0, int(num_SA/2):int(num_SA)]
        mask0_bit = mask_bit[0, 0 : int(num_SA / 2)]
        mask1_bit = mask_bit[0, int(num_SA / 2) : int(num_SA)]
    mask0 = (mask0, mask0_bit)
    mask1 = (mask1, mask1_bit)
    return mask0, mask1


def inject_error(weight, mask0, mask1, num_bits=32):
    if num_bits == 32:
        dtype = cp.uint32
        ftype = cp.float32
    shape = weight.shape
    weight_flatten = cp.ravel(weight).view(dtype)
    mask0, mask0_bit = mask0
    mask1, mask1_bit = mask1
    zero = cp.zeros(1, dtype=dtype)

    if (mask0.__len__() is not 0) or (mask1.__len__() is not 0):
        for b in range(num_bits):
            fault = cp.full(weight_flatten.size, 2**b, dtype=dtype)
            bit_loc0 = cp.where(mask0_bit == b, mask0, zero).nonzero()[0]
            bit_loc1 = cp.where(mask1_bit == b, mask1, zero).nonzero()[0]
            uniform0 = cp.zeros(weight_flatten.size, dtype=dtype)
            uniform1 = cp.zeros(weight_flatten.size, dtype=dtype)
# Inject bit error
            if bit_loc0.__len__() > 0:
                cp.put(uniform0, mask0[bit_loc0], fault)
                cp.put(uniform1, mask1[bit_loc1], fault)
# Stuck at 0
                not_mask0 = cp.invert(uniform0)
                weight_flatten = cp.bitwise_and(weight_flatten, not_mask0)
# Stuck at 1 
                weight_flatten = cp.bitwise_or(weight_flatten, uniform1)
        weight_float = weight_flatten.view(ftype)
        return  cp.reshape(weight_float, shape)
    else:
        return weight




def count_total_param(state_dict):
    total = 0
    for name, param in state_dict.items():
        if "weight" not in name:
            continue
        else:
            total += param.numel()
    return total


def method0(state_dict, weight_path, error_rate, num_bits=32):
    for name, param in state_dict.items():
        if "weight" not in name:
            continue
        else:
            weight = cp.load(weight_path + str(name) + ".npy")
            mask0, mask1 = create_mask(param, error_rate)
            param_error = inject_error(weight, mask0, mask1, num_bits)
            param_error_np = cp.asnumpy(param_error)
            param_error_torch = torch.from_numpy(param_error_np)
            param.copy_(param_error_torch)
    return state_dict

#Perform weight mapping

def method1(state_dict, map_weight_path, error_rate, num_bits=32):
    index = torch.arange(16).to("cuda")
    index_map = wmp.mapallweights2(index).squeeze()
    _, indicies = torch.sort(index_map, dim=0)
    for name, param in state_dict.items():
        if "weight" not in name:
            continue
        else:
            shape = param.shape
            weight = np.load(map_weight_path + str(name) + ".npy")
            mask0, mask1 = create_mask(param, error_rate, num_bits)
            output = weight_mapping(weight, mask0, mask1, indicies, num_bits)
            param.copy_(output.view(shape))
    return state_dict


def weight_mapping(weight, mask0, mask1, indicies, num_bits=32):
    new_weight = np.copy(weight)
    for i in range(16):
        weight_case_i = cp.asarray(weight[:, i, :])
        weight_error_i = inject_error(weight_case_i, mask0, mask1, num_bits)
        new_weight[:, i, :] = cp.asnumpy(weight_error_i)

    mapped_float = torch.from_numpy(weight)
    new_weight = torch.from_numpy(new_weight)

    dev_map = abs(mapped_float - new_weight) # Calculate deviation
    dev_sum_map = torch.sum(dev_map, dim=1)
    min_dev, best_map = torch.min(dev_sum_map, dim=1) # calculate best mapping
    best_map3d = best_map.unsqueeze(1).repeat(1, 16).unsqueeze(1)
    best_map_16 = torch.gather(new_weight, dim=1, index=best_map3d).squeeze(1) 
    idx_map = torch.index_select(indicies, dim=0, index=best_map)
    weight_remap = torch.gather(best_map_16, dim=1, index=idx_map)

    return weight_remap

# Weight mapping + encoding

def method2(state_dict, map_weight_path, error_rate, num_bits=32):
    index = torch.arange(16).to("cuda")
    index_map = wmp.mapallweights2(index).squeeze()
    index_map = torch.cat((index_map, index_map), dim=0)
    _, indicies = torch.sort(index_map, dim=1)
    for name, param in tqdm(state_dict.items(), desc="Executing method2: ", leave=False):
        if "weight" in name:
            shape = param.shape
            weight = np.load(map_weight_path + str(name) + ".npy")
            mask0, mask1 = create_mask(param, error_rate, num_bits)
            output = weight_mapping_encode(weight, mask0, mask1, indicies, num_bits)
            param.copy_(output.view(shape))
    return state_dict


def weight_mapping_encode(weight, mask0, mask1, indicies, num_bits=32):
# Add flip mapping
    weight_flip = np.invert(weight.view(np.uint32)).view(np.float32)
    flip_map = np.concatenate((weight, weight_flip), axis=1)
    old_map = np.concatenate((weight, weight), axis=1)

    for i in range(32):
        weight_case_i = cp.asarray(flip_map[:, i, :])
        weight_error_i = inject_error(weight_case_i, mask0, mask1, num_bits)
        flip_map[:, i, :] = cp.asnumpy(weight_error_i)

    flip_map[:, 16:32, :] = np.invert(flip_map[:, 16:32, :].view(np.uint32)).view(np.float32)
    new_weight = torch.from_numpy(flip_map)
    mapped_float = torch.from_numpy(old_map)

    dev_map = abs(mapped_float - new_weight) # Calculate deviation
    dev_sum_map = torch.sum(dev_map, dim=2)
    min_dev, best_map = torch.min(dev_sum_map, dim=1) # calculate best mapping
    best_map3d = best_map.unsqueeze(1).repeat(1, 16).unsqueeze(1)
    best_map_16 = torch.gather(new_weight, dim=1, index=best_map3d).squeeze(1) 
    idx_map = torch.index_select(indicies, dim=0, index=best_map)
    weight_remap = torch.gather(best_map_16, dim=1, index=idx_map)

    return weight_remap

# Perform ECC(72, 64)
def ECC_method(state_dict, weight_path, error_rate, set_map, num_bits=32):
    for name, param in tqdm(state_dict.items(), desc="Executing ECC: ", leave=False):
        if "weight" in name:
            weight = cp.load(weight_path + str(name) + ".npy")
            orig_weight = cp.copy(weight)
            mask0, mask1 = create_mask(param, error_rate)
            param_error = inject_error(weight, mask0, mask1, num_bits)
            correct_param = ECC(param_error, orig_weight, set_map)
            param_error_np = cp.asnumpy(correct_param)
            param_error_torch = torch.from_numpy(param_error_np)
            param.copy_(param_error_torch)
    return state_dict

def ECC(error_weight, orig_weight, set_map):
    orig_shape = error_weight.shape
    error_weight, orig_weight  = cp.ravel(error_weight.view(cp.uint32)), cp.ravel(orig_weight.view(cp.uint32))
    shape = ( int( error_weight.__len__()/2 ), 2 )
# Reshape 64 bit in one row 
    error_weight, orig_weight = cp.reshape(error_weight, shape), cp.reshape(orig_weight, shape)
# Calculate stuck bits
    stuck_bits = cp.bitwise_xor(error_weight, orig_weight)
    stuck_bits_sum = cp.sum(stuck_bits, axis=1)
    error = cp.concatenate(cp.in1d(stuck_bits_sum, set_map).nonzero())

    if error.__len__() == 0:
        return cp.reshape(error_weight, orig_shape).view(cp.float32)
    else:
        error_weight[error, :] = orig_weight[error, :]
        return cp.reshape(error_weight, orig_shape).view(cp.float32)

# Perform ECP
def ECP_method(state_dict, weight_path, error_rate, set_map, num_bits=32):
    for name, param in tqdm(state_dict.items(), desc="Executing ECP: ", leave=False):
        if "weight" in name:
            weight = cp.load(weight_path + str(name) + ".npy")
            orig_weight = cp.copy(weight)
            mask0, mask1 = create_mask(param, error_rate)
            param_error = inject_error(weight, mask0, mask1, num_bits)
            correct_param = ECP(param_error, orig_weight, set_map)
            param_error_np = cp.asnumpy(correct_param)
            param_error_torch = torch.from_numpy(param_error_np)
            param.copy_(param_error_torch)
    return state_dict

def ECP(error_weight, orig_weight, set_map):
    orig_shape = error_weight.shape
    error_weight, orig_weight  = cp.ravel(error_weight.view(cp.uint32)), cp.ravel(orig_weight.view(cp.uint32))
    shape = ( int( error_weight.__len__()/16), 16)
# Reshape 64 bit in one row 
    error_weight, orig_weight = cp.reshape(error_weight, shape), cp.reshape(orig_weight, shape)
# Calculate stuck bits
    stuck_bits = cp.bitwise_xor(error_weight, orig_weight)
    stuck_bits_sum = cp.sum(stuck_bits, axis=1)
    error = cp.concatenate(cp.in1d(stuck_bits_sum, set_map).nonzero())

    if error.__len__() == 0:
        return cp.reshape(error_weight, orig_shape).view(cp.float32)
    else:
        error_weight[error, :] = orig_weight[error, :]
        return cp.reshape(error_weight, orig_shape).view(cp.float32)

def check_dev(weight_path, state_dict):
    dev_sum = 0
    for name, param in state_dict.items():
        if "weight" in name:
            weight = cp.load(weight_path + str(name) + ".npy")
            weight_np = cp.asnumpy(weight)
            weight_tensor = torch.from_numpy(weight_np).to("cuda")
            dev = abs(weight_tensor - param)
            dev_flat = dev.view(-1)
            dev_sum += torch.sum(dev_flat, 0).item()
    return dev_sum


