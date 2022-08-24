# Fixed-Weight Difference Target Propagation
---
The attached codes were used for the experiments conducted in the submitted paper "Fixed-Weight Difference Target Propagation".
## How to use
You can run each method on each dataset by calling the `Main.py` script and specifying the needed command-line arguments. For example, the script bellow is the case on FW-DTP with batch normalization on MNIST:
```
python Main.py --algorithm=FWDTP-BN --dataset=MNIST --forward_last_activation=linear-BN --learning_rate=0.1 --stepsize=0.04 --test
```
Due to the difference of device, the values reported in the paper is not necessarily obtained.
### Method
You can use each method used in the paper by setting the arguments `--algorithm` and `--loss_feedback` as shown below:
| Method            | `--algorithm` | `--loss_feedback` |
| ----------------- | ------------- | ----------------- |
| FW-DTP with BN    | FWDTP-BN      | any string        |
| FW-DTP without BN | FWDTP         | any string        |
| DTP with BN       | DTP-BN        | DTP               |
| DTP without BN    | DTP           | DTP               |
| DRL without BN    | DTP           | DRL               |
| L-DRL without BN  | DTP           | LDRL              |
| BP                | BP            | any string        |
Also more detailed setting are available by setting `--forward_function_1`,`--forward_function_2`,`--backward_function_1` and `--backward_function_2` (they are corresponding to f_mu, f_nu, g_mu and g_nu with the notations in Section 3.2).
### Architecture
Default architecture is set to the architecture used for MNIST in the paper. You can change the architecture by setting the argumnts `--in_dim`, `--hid_dim`, `--out_dim` and `--depth` (they are corresponding to the input dimension, the number of hidden units, the output dimension and the number of layers).
### Dataset
You can use all datasets used in the experiments (MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100) by setting the argument `--dataset` as bellow:
| Dataset       | `--dataset`  |
| ------------- | ------------ |
| MNIST         | MNIST        |
| Fashion-MNIST | FashionMNIST |
| CIFAR-10      | CIFAR10      |
| CIFAR-100     | CIFAR100     |
When you change the dataset, you also should change the architecture since the input dimension `--in_dim` and the output dimension `--out_dim` are not automatically changed.
### Hyperparameter
You can change the values of hyperparameters by setting the arguments as shown below:
| Parameter                             | Argument                   |
| ------------------------------------- | -------------------------- |
| Learning rate for feedforward network | `--learning_rate`          |
| Stepsize                              | `--stepsize`               |
| Learning rate for feedback network    | `--learning_rate_backward` |
| Standard deviation                    | `--std_backward`           |
| The feedback update frequency         | `--epochs_backward`        |
