"""Train CIFAR10 with PyTorch."""
import argparse
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pdb

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import save_weight
from torchsummary import summary

from models import *
from utils import progress_bar
import makeSA_dev


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--method", default="method0", type=str, help="Running method")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=False, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=500, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="../data", train=False, download=False, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
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

# Model
print("==> Building model..")
# net = vgg19_bn()
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()

state_dict = torch.load("./checkpoint/resnet.pt")['net']
net = net.to(device)
net.load_state_dict(state_dict)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     net.load_state_dict(state_dict['net'])
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )


def test(epoch):
    global best_acc
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
    acc = 100.0 * correct / total
    return acc
    # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/resnet.pt')
    #     best_acc = acc

# state_error = makeSA.method0(state_dict, "./save_cp/", error_rate=1e-08)
# net.load_state_dict(state_error)
# test(0)

if args.method == "method0":
    SAsimulate = makeSA_dev.sa_config(
        testloader,
        net,
        state_dict,
        args.method,
        writer=True
    )
    error_range = np.logspace(-8, -1, 80)
    if not os.path.isdir("./save_cp"):
        os.mkdir("./save_cp")
        SAsimulate.np_to_cp()
    SAsimulate.run(error_range, 100, test, 0, state_dict, "./save_cp/")

if args.method == "method2":
    if not os.path.isdir("./save_map"):
        os.mkdir("./save_map")
        save_weight.save_map(state_dict, "./save_map/", device)

    SAsimulate = makeSA_dev.sa_config(
        testloader,
        net,
        state_dict,
        args.method,
        writer=True,
        mapped_float="./save_map/"
    )

    error_range = np.logspace(-6, -1, 60)
    SAsimulate.run(error_range, 100, test, 0, state_dict, "./save_cp/")

if args.method == "ECC":
    SAsimulate = makeSA.sa_config(
        testloader,
        net,
        state_dict,
        args.method,
        writer=True
    )
    error_range = np.logspace(-10, -1, 100)
    SAsimulate.run(error_range, 40, test, 0, state_dict, "./save_cp/")

if args.method == "ECP":
    SAsimulate = makeSA_dev.sa_config(
        testloader,
        net,
        state_dict,
        args.method,
        writer=True
    )
    error_range = np.logspace(-8, -1, 80)
    SAsimulate.run(error_range, 100, test, 0, state_dict, "./save_cp/")
