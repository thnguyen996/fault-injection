import numpy as np
from pprint import pprint
import torch
import pdb
import collections
def switch1(w):
    weight = w.clone()
    temp = weight.clone()
    weight[0::2] = temp[1::2]
    weight[1::2] = temp[0::2]
    return weight

def switch2(w):
    weight = w.clone()
    weight = weight.view(int(w.numel()/2), 2)
    temp = weight.clone()
    weight[0::2, :] = temp[1::2, :]
    weight[1::2, :] = temp[0::2, :]
    return weight.view(-1)

def switch4(w):
    weight = w.clone()
    weight = weight.view(int(w.numel()/4), 4)
    temp = weight.clone()
    weight[0::2, :] = temp[1::2, :]
    weight[1::2, :] = temp[0::2, :]
    return weight.view(-1)


def switch8(w):
    weight = w.clone()
    weight = weight.view(int(w.numel()/8), 8)
    temp = weight.clone()
    weight[0::2, :] = temp[1::2, :]
    weight[1::2, :] = temp[0::2, :]
    return weight.view(-1)


def remap(weight_tensor, index):
    stack_index = torch.stack((weight_tensor, index))
    new_weights = stack_index.sort(1).values[0]
    return new_weights

# Given a weight tensor, calculate all the possible mapping case

def map_cases2(weight_tensor):
    w = weight_tensor.view(-1)
    w0 = w.clone()
    w1 = switch1(w)
    w2 = switch2(w)
    w3 = switch1(w2)
    w4 = switch4(w)
    w5 = switch1(w4)
    w6 = switch2(w4)
    w7 = switch1(w6)

    w8 = switch8(w)
    w9 = switch1(w8)
    w10 = switch2(w8)
    w11 = switch1(w10)
    w12 = switch4(w8)
    w13 = switch1(w12)
    w14 = switch2(w12)
    w15 = switch1(w14)

    w16 = torch.roll(w, 16)
    w17 = torch.roll(w1, 16)
    w18 = torch.roll(w2, 16)
    w19 = torch.roll(w3, 16)
    w20 = torch.roll(w4, 16)
    w21 = torch.roll(w5, 16)
    w22 = torch.roll(w6, 16)
    w23 = torch.roll(w7, 16)

    w24 = torch.roll(w8, 16)
    w25 = torch.roll(w9, 16)
    w26 = torch.roll(w10, 16)
    w27 = torch.roll(w11, 16)
    w28 = torch.roll(w12, 16)
    w29 = torch.roll(w13, 16)
    w30 = torch.roll(w14, 16)
    w31 = torch.roll(w15, 16)
    weight_cases = torch.stack((w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, 
        w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27, w28, w29, w30, w31))
    return weight_cases

# Return list of mapped weights 
def mapallweights(weights):
    assert weights.shape == weights.view(-1).shape
    map_save = []
    if (weights.numel() % 16) == 0 and weights.numel() >= 16:
        for i in range(0, weights.numel(), 16): 
            map_dict = collections.OrderedDict({})
            weight16 = weights[i:i+16]
            map_dict = map_cases(weight16)
            map_save.append(map_dict)
        return map_save
    else:
        remainder = weights.numel() % 16
        print("Weights are not divisible by 16, skipping: ", remainder, "weights")
        if weights.numel() > 16:
            weights_map_length = weights.numel() - remainder
            for i in range(0, weights_map_length, 16): 
                map_dict = {}
                weight16 = weights[i:i+16]
                map_dict = map_cases(weight16)
                map_save.append(map_dict)
            map_save.append(weights[weights_map_length:weights.numel()])
            return map_save
        else:
            map_save.append(weights)
            return map_save

def mapallweights2(weights):
    assert weights.shape == weights.view(-1).shape
    num_groups = int(weights.numel()/32)
    map_tensor = torch.empty(num_groups, 32, 32)
    if (weights.numel() % 32) == 0 and weights.numel() >= 32:
        for i, j in zip(range(0, weights.numel(), 32), range(map_tensor.shape[0])): 
            weight32 = weights[i:i+32]
            map_dict = map_cases2(weight32)
            map_tensor[j, ...] = map_dict
        return map_tensor
    else:
        remainder = weights.numel() % 32
        print("Weights are not divisible by 32, skipping: ", remainder, "weights")
        if weights.numel() > 32:
            weights_map_length = weights.numel() - remainder
            for i, j in zip(range(0, weights_map_length, 32), range(map_tensor.shape[0])): 
                weight32 = weights[i:i+32]
                map_dict = map_cases2(weight32)
                map_tensor[j, ...] = map_dict
            return map_tensor, weights[weights_map_length : weights.numel()]
        else:
            return weights

## For debugging

def main():

    torch.set_printoptions(profile='full')

    x = torch.arange(32)
    x = x.view(-1)
    map_weight = mapallweights2(x)
    pprint(map_weight.squeeze(), width=200)

if __name__ == "__main__":
    main()
