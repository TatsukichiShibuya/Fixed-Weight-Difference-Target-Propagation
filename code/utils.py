import os
import numpy as np
import torch
from torch import nn
import math
import wandb


def combined_loss(pred, label, device="cpu", num_classes=10):
    batch_size = pred.shape[0]
    dim = pred.shape[1]
    E = torch.eye(dim, device=device)
    E1 = E[:, :num_classes]
    E2 = E[:, num_classes:]
    ce = nn.CrossEntropyLoss(reduction="sum")
    mse = nn.MSELoss(reduction="sum")
    return ce(pred @ E1, (label @ E1).max(axis=1).indices) + 1e-3 * mse(pred @ E2, label @ E2)


def calc_accuracy(pred, label):
    max_index = pred.max(axis=1).indices
    return (max_index == label).sum().item() / label.shape[0]


def calc_accuracy_combined(pred, label, num_classes=10):
    data_size = pred.shape[0]
    pred_max = pred[:, :num_classes].max(axis=1).indices
    label_max = label[:, :num_classes].max(axis=1).indices
    return (pred_max == label_max).sum().item() / data_size


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def calc_angle(v1, v2):
    cos = (v1 * v2).sum(axis=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-12)
    cos = torch.clamp(cos, min=-1, max=1)
    acos = torch.acos(cos) * 180 / math.pi
    angle = 180 - torch.abs(acos - 180)
    return angle


def batch_normalization(x, mean=None, std=None):
    if mean is None:
        mean = torch.mean(x, dim=0)
    if std is None:
        std = torch.std(x, dim=0)
    return (x - mean) / (std + 1e-12)


def batch_normalization_inverse(y, mean, std):
    return y * std + mean


def set_wandb(args, params):
    config = args.copy()
    name = {"ff1": "forward_function_1",
            "ff2": "forward_function_2",
            "bf1": "backward_function_1",
            "bf2": "backward_function_2"}
    for n in name.keys():
        config[name[n]] = params[n]["type"]
        config[name[n] + "_init"] = params[n]["init"]
        config[name[n] + "_activation"] = params[n]["act"]
    config["last_activation"] = params["last"]
    wandb.init(config=config)


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        os.environ['OMP_NUM_THREADS'] = '1'
    return device
