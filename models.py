import torch
import torch.nn as nn
import numpy as np

class DQNSolver(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.squeeze(-1)  # FrameStack dimension
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)
        return x

class DQNSolverResNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.forward_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward_conv(self, x):
        x = self.initial_conv(x)
        x = self.maxpool(self.res_block1(x))
        x = self.maxpool(self.res_block2(x))
        return x

    def forward(self, x):
        x = x.squeeze(-1)
        conv_out = self.forward_conv(x).view(x.size(0), -1)
        return self.fc(conv_out)
