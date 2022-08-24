from utils import calc_accuracy, calc_angle, calc_accuracy_combined

import torch
from torch import nn
from abc import ABCMeta, abstractmethod


class net(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x, update=True):
        raise NotImplementedError()

    def predict(self, x):
        return self.forward(x, update=False)

    def test(self, data_loader):
        pred, label = None, None
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.predict(x)
            pred = y_pred if pred is None else torch.concat([pred, y_pred])
            label = y if label is None else torch.concat([label, y])

        if isinstance(self.loss_function, nn.CrossEntropyLoss):  # classification
            return self.loss_function(pred, label) / len(data_loader.dataset), calc_accuracy(pred, label)
        elif isinstance(self.loss_function, nn.MSELoss):  # regression
            return self.loss_function(pred, label) / len(data_loader.dataset), None
        else:
            return self.loss_function(pred, label) / len(data_loader.dataset), calc_accuracy_combined(pred, label,
                                                                                                      data_loader.dataset.num_classes)
