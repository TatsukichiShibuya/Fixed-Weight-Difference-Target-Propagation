from utils import worker_init_fn, set_seed, combined_loss, set_wandb, set_device
from dataset import make_MNIST, make_FashionMNIST, make_CIFAR10, make_CIFAR100

from models.bp_net import bp_net
from models.tp_net import tp_net

import os
import sys
import wandb
import torch
import argparse
import numpy as np
from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
BP_LIST = ["BP", "FA", "sFA"]
TP_LIST = ["TP", "DTP", "DTP-BN", "FWDTP", "FWDTP-BN", "ITP", "ITP-BN"]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="MNIST",
                        choices=["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"])
    parser.add_argument("--algorithm", type=str, default="FWDTP-BN", choices=BP_LIST + TP_LIST)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test", action="store_true")

    # model architecture
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--direct_depth", type=int, default=1)
    parser.add_argument("--in_dim", type=int, default=784)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=10)

    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--learning_rate_backward", "-lrb", type=float, default=1e-3)
    parser.add_argument("--std_backward", "-sb", type=float, default=1e-2)
    parser.add_argument("--stepsize", type=float, default=1e-2)

    parser.add_argument("--label_augmentation", action="store_true")

    # setting of tp_layer
    parser.add_argument("--forward_function_1", "-ff1", type=str, default="identity",
                        choices=["identity", "random", "parameterized"])
    parser.add_argument("--forward_function_2", "-ff2", type=str, default="identity",
                        choices=["identity", "random", "parameterized"])
    parser.add_argument("--backward_function_1", "-bf1", type=str, default="identity",
                        choices=["identity", "random", "parameterized"])
    parser.add_argument("--backward_function_2", "-bf2", type=str, default="identity",
                        choices=["identity", "random", "difference"])

    # neccesary if {parameterized, random} was choosed
    parser.add_argument("--forward_function_1_init", "-ff1_init", type=str, default="orthogonal",
                        choices=["orthogonal", "gaussian", "uniform"])
    parser.add_argument("--forward_function_2_init", "-ff2_init", type=str, default="orthogonal",
                        choices=["orthogonal", "gaussian", "uniform"])
    parser.add_argument("--backward_function_1_init", "-bf1_init", type=str, default="uniform",
                        choices=["orthogonal", "gaussian", "uniform",
                                 "orthogonal-0", "orthogonal-1", "orthogonal-2", "orthogonal-3", "orthogonal-4",
                                 "gaussian-0", "gaussian-1", "gaussian-1", "gaussian-2", "gaussian-3", "gaussian-4",
                                 "uniform-0", "uniform-1", "uniform-2", "uniform-3", "uniform-4",
                                 "eye-0", "eye-1", "eye-2", "eye-3", "eye-4",
                                 "constant-0", "constant-1", "constant-2", "constant-3", "constant-4",
                                 "rank-1", "rank-2", "rank-4", "rank-8", "same"])
    parser.add_argument("--backward_function_2_init", "-bf2_init", type=str, default="orthogonal",
                        choices=["orthogonal", "gaussian", "uniform"])
    parser.add_argument("--sparse_ratio", "-sr", type=float, default=-1)

    parser.add_argument("--forward_function_1_activation", "-ff1_act", type=str, default="linear",
                        choices=["tanh", "linear", "tanh-BN", "linear-BN"])
    parser.add_argument("--forward_function_2_activation", "-ff2_act", type=str, default="tanh",
                        choices=["tanh", "linear", "tanh-BN", "linear-BN"])
    parser.add_argument("--backward_function_1_activation", "-bf1_act", type=str, default="tanh",
                        choices=["tanh", "linear", "tanh-BN", "linear-BN"])
    parser.add_argument("--backward_function_2_activation", "-bf2_act", type=str, default="linear",
                        choices=["tanh", "linear", "tanh-BN", "linear-BN"])
    parser.add_argument("--forward_last_activation", type=str, default="linear",
                        choices=["tanh", "linear", "tanh-BN", "linear-BN"])

    parser.add_argument("--loss_feedback", type=str, default="DTP",
                        choices=["DTP", "DRL", "L-DRL"])
    parser.add_argument("--epochs_backward", type=int, default=5)

    # wandb
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--agent", action="store_true")

    args = parser.parse_args()
    return args


def main(**kwargs):
    set_seed(kwargs["seed"])
    device = set_device()
    print(f"DEVICE: {device}")
    if kwargs["algorithm"] in TP_LIST:
        params = set_params(kwargs)
        print("Forward  : ", end="")
        print(f"{params['ff1']['type']}({params['ff1']['act']},{params['ff1']['init']})", end="")
        print(f" -> {params['ff2']['type']}({params['ff2']['act']},{params['ff2']['init']})")
        print("Backward : ", end="")
        print(f"{params['bf1']['type']}({params['bf1']['act']},{params['bf1']['init']})", end="")
        print(f" -> {params['bf2']['type']}({params['bf2']['act']},{params['bf2']['init']})")
        if kwargs["log"]:
            set_wandb(kwargs, params)
    elif kwargs["algorithm"] in BP_LIST:
        print("Forward  : ", end="")
        print(f"{kwargs['forward_function_2_activation']}, orthogonal")
        if kwargs["log"]:
            config = kwargs.copy()
            config["activation_function"] = kwargs['forward_function_2_activation']
            wandb.init(config=config)

    if kwargs["dataset"] == "MNIST":
        num_classes = 10
        trainset, validset, testset = make_MNIST(kwargs["label_augmentation"],
                                                 kwargs["out_dim"], kwargs["test"])
    elif kwargs["dataset"] == "FashionMNIST":
        num_classes = 10
        trainset, validset, testset = make_FashionMNIST(kwargs["label_augmentation"],
                                                        kwargs["out_dim"], kwargs["test"])
    elif kwargs["dataset"] == "CIFAR10":
        num_classes = 10
        trainset, validset, testset = make_CIFAR10(kwargs["label_augmentation"],
                                                   kwargs["out_dim"], kwargs["test"])
    elif kwargs["dataset"] == "CIFAR100":
        num_classes = 100
        trainset, validset, testset = make_CIFAR100(kwargs["label_augmentation"],
                                                    kwargs["out_dim"], kwargs["test"])
    else:
        raise NotImplementedError()

    if kwargs["label_augmentation"]:
        loss_function = (lambda pred, label: combined_loss(pred, label, device, num_classes))
    else:
        loss_function = nn.CrossEntropyLoss(reduction="sum")

    # make dataloader
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=kwargs["batch_size"],
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=kwargs["batch_size"],
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=kwargs["batch_size"],
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True,
                                              worker_init_fn=worker_init_fn)

    # initialize model
    if kwargs["algorithm"] in BP_LIST:
        model = bp_net(kwargs["depth"], kwargs["in_dim"], kwargs["hid_dim"],
                       kwargs["out_dim"], kwargs["forward_function_2_activation"],
                       loss_function, kwargs["algorithm"], device)
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["learning_rate"],
                    kwargs["log"])
    elif kwargs["algorithm"] in TP_LIST:
        model = tp_net(kwargs["depth"], kwargs["direct_depth"], kwargs["in_dim"],
                       kwargs["hid_dim"], kwargs["out_dim"], loss_function, device, params=params)
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["learning_rate"],
                    kwargs["learning_rate_backward"], kwargs["std_backward"], kwargs["stepsize"],
                    kwargs["log"], {"loss_feedback": kwargs["loss_feedback"], "epochs_backward": kwargs["epochs_backward"]})

    # test
    loss, acc = model.test(test_loader)
    print(f"Test Loss      : {loss}")
    if acc is not None:
        print(f"Test Acc       : {acc}")


def set_params(kwargs):
    name = {"ff1": "forward_function_1",
            "ff2": "forward_function_2",
            "bf1": "backward_function_1",
            "bf2": "backward_function_2"}
    params = {}
    sparse_ratio = ("-sparse-" + str(kwargs["sparse_ratio"])
                    ) if 1 >= kwargs["sparse_ratio"] >= 0 else ""

    params["last"] = "linear"
    if kwargs["algorithm"] == "TP":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh"}
        params["bf1"] = {"type": "parameterized",
                         "init": kwargs[name["bf1"] + "_init"],
                         "act": "tanh"}
        params["bf2"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
    elif kwargs["algorithm"] == "DTP":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh"}
        params["bf1"] = {"type": "parameterized",
                         "init": kwargs[name["bf1"] + "_init"],
                         "act": "tanh"}
        params["bf2"] = {"type": "difference",
                         "init": None,
                         "act": "linear"}
    elif kwargs["algorithm"] == "DTP-BN":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh-BN"}
        params["bf1"] = {"type": "parameterized",
                         "init": kwargs[name["bf1"] + "_init"],
                         "act": "tanh-BN"}
        params["bf2"] = {"type": "difference",
                         "init": None,
                         "act": "linear-BN"}
    elif kwargs["algorithm"] == "FWDTP":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh"}
        params["bf1"] = {"type": "random",
                         "init": kwargs[name["bf1"] + "_init"] + sparse_ratio,
                         "act": "tanh"}
        params["bf2"] = {"type": "difference",
                         "init": None,
                         "act": "linear"}
    elif kwargs["algorithm"] == "FWDTP-BN":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh-BN"}
        params["bf1"] = {"type": "random",
                         "init": kwargs[name["bf1"] + "_init"] + sparse_ratio,
                         "act": "tanh-BN"}
        params["bf2"] = {"type": "difference",
                         "init": None,
                         "act": "linear-BN"}
        params["last"] = kwargs["forward_last_activation"]
    elif kwargs["algorithm"] == "ITP":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh"}
        params["bf1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["bf2"] = {"type": "difference",
                         "init": None,
                         "act": "linear"}
    elif kwargs["algorithm"] == "ITP-BN":
        params["ff1"] = {"type": "identity",
                         "init": None,
                         "act": "linear"}
        params["ff2"] = {"type": "parameterized",
                         "init": kwargs[name["ff2"] + "_init"],
                         "act": "tanh-BN"}
        params["bf1"] = {"type": "identity",
                         "init": None,
                         "act": "linear-BN"}
        params["bf2"] = {"type": "difference",
                         "init": None,
                         "act": "linear-BN"}
    return params


if __name__ == '__main__':
    FLAGS = vars(get_args())
    main(**FLAGS)
