import torch
import random
from datetime import datetime
from sasimulate.method import *
import pdb
from torch.utils.tensorboard import SummaryWriter

class config:
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
            if mapped_float == None:
                raise Exception("Input weight mappped float path")
            if binary_path == None:
                raise Exception("Input binary weight path")
            self.mapped_float = torch.load(mapped_float, map_location=self.device)
            self.binary_path = binary_path
        if self.writer_op:
            now = datetime.now().date()
            ran = random.randint(1, 1000)
            self.writer = SummaryWriter(
                "runs/{}-{}-{}".format(
                    now, ran, method 
                )
            )
            print("Run ID: {}-{}".format(now, ran))

# Calculate total_param 
        with torch.no_grad():
            for name, value in model.named_parameters():
                if "weight" not in name:
                    continue
                else:
                    self.total_param += value.numel()
        print("Total numbers of weights: {}".format(self.total_param))

    def run(self, error_range, avr_point, validate, arg, state_dict_path):
        count = 0
        avr_error = 0.0
        if self.method == "method0":
            for error_total in error_range:
                running_error = []
                count += 1
                print("Error rate: ", error_total)
                for i in range(avr_point):
                    orig_state_dict = torch.load(state_dict_path, map_location=self.device)
                    state_dict =  method0(orig_state_dict, self.total_param, error_total, self.device)
                    self.model.load_state_dict(state_dict)
                    torch.cuda.empty_cache()
                    acc1 = validate(arg)
                    running_error.append(100.0 - acc1)

                if self.writer_op:
                    avr_error = sum(running_error) / len(running_error)
                    print("Avarage classification Error: ", avr_error)
                    self.writer.add_scalar("Average Error", avr_error, count)
                    self.writer.close()

        if self.method == "method1":
            for error_total in error_range:
                running_error = []
                count += 1
                print("Error rate: ", error_total)
                for i in range(avr_point):
                    orig_state_dict = torch.load(state_dict_path)
                    state_dict =  method1(orig_state_dict, self.total_param, self.mapped_float, self.binary_path, error_total, self.device)
                    self.model.load_state_dict(state_dict)
                    acc1 = validate(arg)
                    running_error.append(100.0 - acc1)

                if self.writer_op:
                    avr_error = sum(running_error) / len(running_error)
                    print("Avarage classification Error: ", avr_error)
                    self.writer.add_scalar("Average Error", avr_error, count)
                    self.writer.close()

        if self.method == "method2":
            for error_total in error_range:
                running_error = []
                count += 1
                print("Error rate: ", error_total)
                for i in range(avr_point):
                    orig_state_dict = torch.load(state_dict_path)
                    state_dict =  method2(orig_state_dict, self.total_param, self.mapped_float, self.binary_path, error_total, self.device)
                    self.model.load_state_dict(state_dict)
                    acc1 = validate(arg)
                    running_error.append(100.0 - acc1)

                if self.writer_op:
                    avr_error = sum(running_error) / len(running_error)
                    print("Avarage classification Error: ", avr_error)
                    self.writer.add_scalar("Average Error", avr_error, count)
                    self.writer.close()

        if self.method == "ECC_method":
            for error_total in error_range:
                running_error = []
                count += 1
                print("Error rate: ", error_total)
                for i in range(avr_point):
                    orig_state_dict = torch.load(state_dict_path)
                    state_dict =  ECC_method(orig_state_dict, self.total_param, error_total, self.device)
                    self.model.load_state_dict(state_dict)
                    torch.cuda.empty_cache()
                    acc1 = validate(arg)
                    running_error.append(100.0 - acc1)

                if self.writer_op:
                    avr_error = sum(running_error) / len(running_error)
                    print("Avarage classification Error: ", avr_error)
                    self.writer.add_scalar("Average Error", avr_error, count)
                    self.writer.close()
