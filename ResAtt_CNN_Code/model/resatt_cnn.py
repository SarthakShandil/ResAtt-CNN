import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch, C, T, W = x.size()
        squeeze = x.view(batch, C, -1).mean(dim=2)
        excitation = torch.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(batch, C, 1, 1)
        return x * excitation

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class ResAttCNN(nn.Module):
    def __init__(self, in_channels=2, base_channels=32, num_blocks=3):
        super(ResAttCNN, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.att = SEBlock(base_channels)
        self.output_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.input_conv(x))
        out = self.res_blocks(out)
        out = self.att(out)
        out = self.output_conv(out)
        return out

