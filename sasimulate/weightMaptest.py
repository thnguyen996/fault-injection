import numpy as np
import torch
import cProfile

def way(tensor, index):
    newtensor = torch.randn(16)
    for i in range(16):
        newtensor[i] = tensor[torch.where(index == i)]
    return newtensor

x = torch.randn(16)

index = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14], dtype=float)

print(way(x, index))
def switch1(w):
    weight = w.clone()
    for i in np.arange(0, weight.numel(), 2):
            a = weight[i].clone()
            weight[i] = weight[i+1]
            weight[i+1] = a
    return weight

print(switch1(x))
cProfile.run("way(x, index)")
cProfile.run("switch1(x)")
