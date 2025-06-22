import torch
import torch.nn as nn

# Complex NMSE Metric
def nmse_loss(pred, target):
    power = torch.sum(target ** 2, dim=[1, 2, 3, 4])
    mse = torch.sum((pred - target) ** 2, dim=[1, 2, 3, 4])
    return torch.mean(mse / (power + 1e-10))

# Standard MSE Loss
def mse_loss():
    return nn.MSELoss()
