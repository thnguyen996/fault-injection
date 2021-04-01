import argparse
import collections
import cProfile
import os
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

now = datetime.now().date()
ran = random.randint(1, 231242)

torch.set_printoptions(profile='full')

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
            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
    # Save best accuracy.
    acc = 100.0 * correct / total
    return acc


def main():

    # print('Loading mapped weights:')
    # mapped_gen = load_mapped_weights()
    # print("Weights loaded")

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    args = parser.parse_args()

    print("==> Preparing data..")

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    device = torch.device("cuda")

    # Initialize Tensorboard

    writer = SummaryWriter(
        "runs/{}-{}".format(now, "ECC (100 points) 1e-10 -- 1e-03, 10")
    )
    # Create model
    model = ResNet18().to(device)
    # Load model state dict
    state_dict = torch.load("./checkpoint/resnet.pt")["net"]
    simulate = SAsimulate(
        test_loader, model, state_dict, method="method0", mapped_gen=[], writer=writer
    )
    error_range = np.linspace(1e-10, 1e-03, 10)
    simulate.run(error_range)


class SAsimulate:
    def __init__(self, test_loader, model, state_dict, method, mapped_gen, writer):
        self.test_loader = test_loader
        self.model = model
        self.state_dict = state_dict
        self.method = method
        self.mapped_gen = mapped_gen
        self.device = torch.device("cuda")
        self.writer = writer
        model.load_state_dict(state_dict)
        self.criterion = nn.CrossEntropyLoss()
        model.to(self.device)

    def run(self, error_range):
        count = 0
        avr_error = 0.0
        des = np.arange(7.071e-05, 0.0001, 9.999e-07)
        orig_weights_list = []

        orig_model = self.model
        total_param = 11173962
        if self.method == "method0":
            for error_total in error_range:
                running_error = []
                running_deviation = []
                count += 1
                print("Error rate: ", error_total)
                for i in range(100):
                    model = method0(orig_model, total_param, error_total)
                    acc1 = test(model, self.test_loader, self.device, self.criterion)
                    running_error.append(100.0 - acc1)
                    orig_model.load_state_dict(self.state_dict)

                avr_error = sum(running_error) / len(running_error)
                print("Avarage classification Error: ", avr_error)
                self.writer.add_scalar("Average Error", avr_error, count)
                self.writer.close()


# Inject error without doing anything to the weight
def method0(model, total_param, error_total):
    device = torch.device("cuda")
    with torch.no_grad():
        for name, param in model.named_parameters():
            shape = param.data.shape
            error_layer = (param.numel() / total_param) * error_total
            param_binary = float2bit(
                param.data, num_e_bits=8, num_m_bits=23, bias=127.0
            )
            mask, mask1 = SAsimulate2.create_mask(param_binary, error_layer)
            # mask, mask1 = mask.to(device), mask1.to(device)
            output = SAsimulate2.make_SA2(param.data.view(-1), mask, mask1)
            correct_binary = ECC(output, param_binary)
            float_tensor = bit2float(correct_binary, num_e_bits=8, num_m_bits=23, bias=127.0)
            param.data = float_tensor.view(shape)
    return model

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
