import torch
from torch import nn
from utils import batch_normalization

import sys


class bp_layer:
    def __init__(self, in_dim, out_dim, activation_function, device):
        # weights
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        nn.init.orthogonal_(self.weight)

        self.fixed_weight = torch.empty(out_dim, in_dim, device=device)
        nn.init.normal_(self.fixed_weight, self.weight.mean().item(), self.weight.std().item())

        # functions
        if activation_function == "linear":
            self.activation_function = (lambda x: x)
            self.activation_derivative = (lambda x: 1)
        elif activation_function == "tanh":
            self.activation_function = nn.Tanh()
            self.activation_derivative = (lambda x: 1 - torch.tanh(x)**2)
        else:
            sys.tracebacklimit = 0
            raise NotImplementedError(f"activation_function : {activation_function} ?")

        # activation
        self.linear_activation = None
        self.activation = None

    def forward(self, x, update=True):
        if update:
            self.linear_activation = x @ self.weight.T
            self.activation = self.activation_function(self.linear_activation)
            return self.activation
        else:
            a = x @ self.weight.T
            h = self.activation_function(a)
            return h
