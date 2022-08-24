import torch
from torch import nn
from abc import ABCMeta, abstractmethod
from utils import batch_normalization


class abstract_function(metaclass=ABCMeta):
    def __init__(self, in_dim, out_dim, layer, device):
        self.layer = layer
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

    @abstractmethod
    def forward(self, input, original=None):
        raise NotImplementedError()

    def update(self, lr):
        # Nothing to do
        return

    def zero_grad(self):
        # Nothing to do
        return

    def get_grad(self):
        # Nothing to do
        return None


class identity_function(abstract_function):
    def __init__(self, in_dim, out_dim, layer, device, params):
        super().__init__(in_dim, out_dim, layer, device)
        self.weight = torch.eye(out_dim, in_dim, device=device)
        if (params["act"] is None) or (params["act"] == "linear"):
            self.activation_function = (lambda x: x)
        elif params["act"] == "linear-BN":
            self.activation_function = (lambda x: batch_normalization(x))
        else:
            raise NotImplementedError()

    def forward(self, input, original=None):
        return self.activation_function(input @ self.weight.T)


class parameterized_function(abstract_function):
    def __init__(self, in_dim, out_dim, layer, device, params):
        super().__init__(in_dim, out_dim, layer, device)
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        if params["init"] == "uniform":
            nn.init.uniform_(self.weight, -1e-2, 1e-2)
        elif params["init"] == "gaussian":
            nn.init.normal_(self.weight, 0, 1e-3)
        elif params["init"] == "orthogonal":
            nn.init.orthogonal_(self.weight)
        else:
            raise NotImplementedError()
        if params["act"] == "tanh":
            self.activation_function = nn.Tanh()
        elif params["act"] == "linear":
            self.activation_function = (lambda x: x)
        elif params["act"] == "tanh-BN":
            tanh = nn.Tanh()
            self.activation_function = (lambda x: batch_normalization(tanh(x)))
        elif params["act"] == "linear-BN":
            self.activation_function = (lambda x: batch_normalization(x))

    def forward(self, input, original=None):
        return self.activation_function(input @ self.weight.T)

    def update(self, lr):
        self.weight = (self.weight - lr * self.weight.grad).detach().requires_grad_()

    def zero_grad(self):
        if self.weight.grad is not None:
            self.weight.grad.zero_()

    def get_grad(self):
        return self.weight.grad


class random_function(abstract_function):
    def __init__(self, in_dim, out_dim, layer, device, params):
        super().__init__(in_dim, out_dim, layer, device)
        self.weight = torch.empty(out_dim, in_dim, device=device)
        if params["init"] == "uniform":
            nn.init.uniform_(self.weight, -1e-3, 1e-3)
        elif params["init"] == "gaussian":
            nn.init.normal_(self.weight, 0, 1e-3)
        elif params["init"] == "orthogonal":
            nn.init.orthogonal_(self.weight)
        elif "orthogonal" in params["init"]:
            scale = 10**(-float(params["init"].split("-")[1]))
            nn.init.orthogonal_(self.weight)
            self.weight *= scale
        elif "gaussian" in params["init"]:
            std = 10**(-float(params["init"].split("-")[1]))
            nn.init.normal_(self.weight, 0, std)
        elif "uniform" in params["init"]:
            range_val = 10**(-float(params["init"].split("-")[1]))
            nn.init.uniform_(self.weight, -range_val, range_val)
        elif "eye" in params["init"]:
            scale = 10**(-float(params["init"].split("-")[1]))
            nn.init.eye_(self.weight)
            self.weight *= scale
        elif "constant" in params["init"]:
            scale = 10**(-float(params["init"].split("-")[1]))
            nn.init.constant_(self.weight, scale)
        elif "same" in params["init"]:
            nn.init.normal_(self.weight, 0, 5e-2)
        elif "rank" in params["init"]:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        if "sparse" in params["init"]:
            sparse_matrix = torch.ones((out_dim, in_dim), device=device)
            sparse_ratio = float(params["init"].split("-")[-1])
            sparse_dim = int(in_dim * sparse_ratio)
            zero_mask = (torch.randperm(in_dim) < sparse_dim)
            for i in range(out_dim):
                sparse_matrix[i][zero_mask] = 0
                zero_mask = torch.cat([zero_mask[-1].reshape(1), zero_mask[:-1]])
            self.weight = (self.weight * sparse_matrix).detach().requires_grad_()

        if params["act"] == "tanh":
            self.activation_function = nn.Tanh()
        elif params["act"] == "linear":
            self.activation_function = (lambda x: x)
        elif params["act"] == "tanh-BN":
            tanh = nn.Tanh()
            self.activation_function = (lambda x: batch_normalization(tanh(x)))
        elif params["act"] == "linear-BN":
            self.activation_function = (lambda x: batch_normalization(x))
        else:
            raise NotImplementedError()

    def forward(self, input, original=None):
        return self.activation_function(input @ self.weight.T)


class difference_function(abstract_function):
    def __init__(self, in_dim, out_dim, layer, device, params):
        super().__init__(in_dim, out_dim, layer, device)
        if (params["act"] is None) or (params["act"] == "linear"):
            self.activation_function = (lambda x: x)
        elif params["act"] == "linear-BN":
            self.activation_function = (lambda x: batch_normalization(x))
        else:
            raise NotImplementedError()

    def forward(self, input, original=None):
        with torch.no_grad():
            upper = self.layer.forward(original, update=False)
            rec = self.layer.backward_function_1.forward(upper)
            difference = original - rec
        return self.activation_function(input + difference)
