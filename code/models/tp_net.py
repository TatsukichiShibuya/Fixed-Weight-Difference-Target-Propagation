from models.tp_layer import tp_layer
from models.net import net
from models.my_functions import parameterized_function
from utils import calc_angle
from copy import deepcopy

import sys
import time
import wandb
import numpy as np
import torch
from torch import nn
from torch.autograd.functional import jacobian


class tp_net(net):
    def __init__(self, depth, direct_depth, in_dim, hid_dim, out_dim, loss_function, device, params=None):
        self.depth = depth
        self.direct_depth = direct_depth
        self.loss_function = loss_function
        self.device = device
        self.MSELoss = nn.MSELoss(reduction="sum")
        self.layers = self.init_layers(in_dim, hid_dim, out_dim, params)
        self.back_trainable = (params["bf1"]["type"] == "parameterized")

    def init_layers(self, in_dim, hid_dim, out_dim, params):
        layers = [None] * self.depth
        dims = [in_dim] + [hid_dim] * (self.depth - 1) + [out_dim]
        for d in range(self.depth - 1):
            layers[d] = tp_layer(dims[d], dims[d + 1], self.device, params)
        params_last = deepcopy(params)
        params_last["ff2"]["act"] = params["last"]
        layers[-1] = tp_layer(dims[-2], dims[-1], self.device, params_last)
        return layers

    def forward(self, x, update=True):
        y = x
        for d in range(self.depth):
            y = self.layers[d].forward(y, update=update)
        return y

    def train(self, train_loader, valid_loader, epochs, lr, lrb, std, stepsize, log, params=None):
        # Pre-train the feedback weights
        for e in range(params["epochs_backward"]):
            torch.cuda.empty_cache()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.train_back_weights(x, y, lrb, std, loss_type=params["loss_feedback"])

        # Train the feedforward and feedback weights
        for e in range(epochs + 1):
            print(f"Epoch: {e}")
            torch.cuda.empty_cache()
            start_time = time.time()
            if e > 0:
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    # Train feedback weights
                    for i in range(params["epochs_backward"]):
                        self.train_back_weights(x, y, lrb, std, loss_type=params["loss_feedback"])
                    # Train forward weights
                    self.compute_target(x, y, stepsize)
                    self.update_weights(x, lr)
            end_time = time.time()

            # Compute Positive semi-definiteness (the strict condition) and Trace (the weak condition)
            eigenvalues_ratio = [torch.zeros(1, device=self.device) for d in range(self.depth)]
            eigenvalues_trace = [torch.zeros(1, device=self.device) for d in range(self.depth)]
            for x, y in valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    self.forward(x)
                    for d in range(1, self.depth - self.direct_depth + 1):
                        h1 = self.layers[d].input[0]
                        gradf = jacobian(self.layers[d].forward, h1)
                        h2 = self.layers[d].forward(h1)
                        gradg = jacobian(self.layers[d].backward_function_1.forward, h2)
                        eig, _ = torch.linalg.eig(gradf @ gradg)
                        eigenvalues_ratio[d] += (eig.real > 0).sum() / len(eig.real)
                        eigenvalues_trace[d] += torch.trace(gradf @ gradg)
            for d in range(self.depth):
                eigenvalues_ratio[d] /= len(valid_loader)
                eigenvalues_trace[d] /= len(valid_loader)

            # Predict
            with torch.no_grad():
                train_loss, train_acc = self.test(train_loader)
                valid_loss, valid_acc = self.test(valid_loader)
            # Logging
            if log:
                log_dict = {}
                log_dict["train loss"] = train_loss
                log_dict["valid loss"] = valid_loss
                if train_acc is not None:
                    log_dict["train accuracy"] = train_acc
                if valid_acc is not None:
                    log_dict["valid accuracy"] = valid_acc
                log_dict["time"] = end_time - start_time
                for d in range(1, self.depth - self.direct_depth + 1):
                    log_dict[f"eigenvalue ratio {d}"] = eigenvalues_ratio[d].item()
                    log_dict[f"eigenvalue trace {d}"] = eigenvalues_trace[d].item()

                wandb.log(log_dict)
            else:
                print(f"\tTrain Loss       : {train_loss}")
                print(f"\tValid Loss       : {valid_loss}")
                if train_acc is not None:
                    print(f"\tTrain Acc        : {train_acc}")
                if valid_acc is not None:
                    print(f"\tValid Acc        : {valid_acc}")

                for d in range(1, self.depth - self.direct_depth + 1):
                    print(f"\teigenvalue ratio-{d}: {eigenvalues_ratio[d].item()}")
                for d in range(1, self.depth - self.direct_depth + 1):
                    print(f"\teigenvalue trace-{d}: {eigenvalues_trace[d].item()}")

    def train_back_weights(self, x, y, lrb, std, loss_type="L-DRL"):
        if not self.back_trainable:
            return

        self.forward(x)
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            if loss_type == "DTP":
                q = self.layers[d - 1].output.detach().clone()
                q = q + torch.normal(0, std, size=q.shape, device=self.device)
                q_upper = self.layers[d].forward(q)
                h = self.layers[d].backward_function_1.forward(q_upper)
                loss = self.MSELoss(h, q)
            elif loss_type == "DRL":
                h = self.layers[d - 1].output.detach().clone()
                q = h + torch.normal(0, std, size=h.shape, device=self.device)
                for _d in range(d, self.depth - self.direct_depth + 1):
                    q = self.layers[_d].forward(q)
                for _d in range(d, self.depth - self.direct_depth + 1):
                    h = self.layers[_d].forward(h)
                for _d in reversed(range(d, self.depth - self.direct_depth + 1)):
                    q = self.layers[_d].backward_function_1.forward(q)
                    q = self.layers[_d].backward_function_2.forward(q, self.layers[_d - 1].output)
                loss = self.MSELoss(self.layers[d].input.clone(), q)
            elif loss_type == "L-DRL":
                h = self.layers[d - 1].output.detach().clone()
                q = h + torch.normal(0, std, size=h.shape, device=self.device)
                q_up = self.layers[d].forward(q)
                _q_up = self.layers[d].backward_function_1.forward(q_up)
                q_rec = self.layers[d].backward_function_2.forward(_q_up, h)
                h_up = self.layers[d].forward(h)
                r_up = h_up + torch.normal(0, std, size=h_up.shape, device=self.device)
                _r_up = self.layers[d].backward_function_1.forward(r_up)
                r_rec = self.layers[d].backward_function_2.forward(_r_up, h)
                loss = -((q - h) * (q_rec - h)).sum() + self.MSELoss(r_rec, h) / 2
            else:
                raise NotImplementedError()
            self.layers[d].zero_grad()
            loss.backward(retain_graph=True)
            self.layers[d].update_backward(lrb / len(x))

    def compute_target(self, x, y, stepsize):
        y_pred = self.forward(x)
        loss = self.loss_function(y_pred, y)
        for d in range(self.depth):
            self.layers[d].zero_grad()
        loss.backward(retain_graph=True)

        with torch.no_grad():
            for d in range(self.depth - self.direct_depth, self.depth):
                self.layers[d].target = self.layers[d].output - \
                    stepsize * self.layers[d].output.grad

            for d in reversed(range(self.depth - self.direct_depth)):
                plane = self.layers[d + 1].backward_function_1.forward(self.layers[d + 1].target)
                diff = self.layers[d + 1].backward_function_2.forward(plane, self.layers[d].output)
                self.layers[d].target = diff

    def update_weights(self, x, lr):
        self.forward(x)
        for d in range(self.depth):
            loss = self.MSELoss(self.layers[d].target, self.layers[d].output)
            self.layers[d].zero_grad()
            loss.backward(retain_graph=True)
            self.layers[d].update_forward(lr / len(x))
