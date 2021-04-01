import argparse
import collections
import json
import os
import pdb

import numpy as np
import cupy as cp
import torch
from tqdm import tqdm

import sasimulate.weight_mapping as wmp


def save_map(state_dict, save_map_path, device):
    state_dict = torch.load(args.weights, map_location=device)
    save_weights = collections.OrderedDict({})
    save_binary = collections.OrderedDict({})
# Save mapped float weights
    for name, param in tqdm(state_dict.items(), desc="Saving weight maps: "):
        if "weight" in name:
            pdb.set_trace()
            weights = param.view(-1)
            map_cases = wmp.mapallweights2(weights)
            map_cases_np = map_cases.cpu().numpy()
            map_cases_cp = cp.asarray(map_cases_np)
            cp.save("../" + save_map_path + str(name) + ".npy", map_cases_cp)

