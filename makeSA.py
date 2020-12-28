import torch
from tqdm import tqdm
import pdb
import numpy as np
import cupy as cp
from collections import OrderedDict
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import random
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

        if self.writer_op:
            now = datetime.now().date()
            ran = random.randint(1, 100)
            self.writer = SummaryWriter(
                "runs/{}-{}-{}".format(
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
                count += 1
                print("Error rate: ", error_rate)
                for i in range(avr_point):
                    state_dict =  method0(state_dict, weight_path, error_rate, num_bits=32)
                    self.model.load_state_dict(state_dict)
                    torch.cuda.empty_cache()
                    acc1 = validate(arg)
                    running_error.append(100.0 - acc1)

                if self.writer_op:
                    avr_error = sum(running_error) / len(running_error)
                    print("Avarage classification Error: ", avr_error)
                    self.writer.add_scalar("Average Error", avr_error, count)
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

# Stuck-at-faults injection

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

