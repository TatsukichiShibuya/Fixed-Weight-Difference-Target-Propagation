from models.bp_layer import bp_layer
from models.net import net

import time
import wandb
import numpy as np
import torch
from torch import nn


class bp_net(net):
    def __init__(self, depth, in_dim, hid_dim, out_dim, activation_function, loss_function, algorithm, device):
        self.device = device
        self.depth = depth
        self.layers = self.init_layers(in_dim, hid_dim, out_dim, activation_function)
        self.loss_function = loss_function
        self.MSELoss = nn.MSELoss(reduction="sum")
        self.algorithm = algorithm

    def init_layers(self, in_dim, hid_dim, out_dim, activation_function):
        layers = [None] * self.depth
        dims = [in_dim] + [hid_dim] * (self.depth - 1) + [out_dim]
        for d in range(self.depth - 1):
            layers[d] = bp_layer(dims[d], dims[d + 1], activation_function, self.device)
        layers[-1] = bp_layer(dims[-2], dims[-1], "linear", self.device)
        return layers

    def train(self, train_loader, valid_loader, epochs, lr, log):
        for e in range(epochs):
            start_time = time.time()

            # train forward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.update_weights(x, y, lr)

            end_time = time.time()
            print(f"epochs {e}: {end_time - start_time:.2f}")

            # predict
            with torch.no_grad():
                train_loss, train_acc = self.test(train_loader)
                valid_loss, valid_acc = self.test(valid_loader)
                if log:
                    log_dict = {"train loss": train_loss,
                                "valid loss": valid_loss}
                    if train_acc is not None:
                        log_dict["train accuracy"] = train_acc
                    if valid_acc is not None:
                        log_dict["valid accuracy"] = valid_acc
                    log_dict["time"] = end_time - start_time

                    wandb.log(log_dict)
                else:
                    print(f"\ttrain loss     : {train_loss}")
                    print(f"\tvalid loss     : {valid_loss}")
                    if train_acc is not None:
                        print(f"\ttrain acc      : {train_acc}")
                    if valid_acc is not None:
                        print(f"\tvalid acc      : {valid_acc}")

    def update_weights(self, x, y, lr):
        y_pred = self.forward(x).requires_grad_()
        y_pred.retain_grad()
        loss = self.loss_function(y_pred, y)
        batch_size = len(y)
        self.zero_grad()
        loss.backward()
        g = y_pred.grad
        grad = [None] * self.depth
        with torch.no_grad():
            for d in reversed(range(self.depth)):
                g = g * self.layers[d].activation_derivative(self.layers[d].linear_activation)
                grad[d] = g.T @ self.layers[d - 1].linear_activation if d >= 1 else g.T @ x
                if self.algorithm == "BP":
                    g = g @ self.layers[d].weight
                elif self.algorithm == "FA":
                    g = g @ self.layers[d].fixed_weight
        for d in range(self.depth):
            self.layers[d].weight = (self.layers[d].weight - (lr / batch_size)
                                     * grad[d]).detach().requires_grad_()

    def zero_grad(self):
        for d in range(self.depth):
            if self.layers[d].weight.grad is not None:
                self.layers[d].weight.grad.zero_()

    def forward(self, x, update=True):
        y = x
        for d in range(self.depth):
            y = self.layers[d].forward(y, update=update)
        return y
