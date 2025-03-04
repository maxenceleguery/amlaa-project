import torch
import torch.nn as nn
import numpy as np

class DQNSolver(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        #input_shape = (4, 84, 84)
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
        o = self.conv(torch.zeros(*shape))
        return int(np.prod(o.size()))

    def forward(self, x: torch.Tensor):
        x = x.squeeze(-1)
        conv_out = self.conv(x).reshape(x.size()[0], -1)
        return self.fc(conv_out)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding='same', 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding='same', 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=(3, 3), 
                    padding='same', 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)

        return x

class DQNSolverResNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNSolverResNet, self).__init__()
        #input_shape = (4, 84, 84)  # Fix input shape for processing stacked frames

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
        """Calculate output size after conv layers."""
        o = self.forward_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward_conv(self, x):
        """Apply convolutional and residual layers."""
        x = self.initial_conv(x)
        x = self.maxpool(self.res_block1(x))
        x = self.maxpool(self.res_block2(x))
        return x

    def forward(self, x: torch.Tensor):
        """Compute forward pass."""
        x = x.squeeze(-1)
        conv_out = self.forward_conv(x).reshape(x.size()[0], -1)
        return self.fc(conv_out)
    
if __name__ == "__main__":
    block = ResidualBlock(in_channels=64, out_channels=64)
    x = torch.randn(4, 64, 32, 32)  # Batch size 4, 64 channels, 32x32 spatial size
    out = block(x)
    assert out.shape == x.shape, "Output shape should match input shape when stride=1"

    block = ResidualBlock(in_channels=32, out_channels=64)
    x = torch.randn(4, 32, 32, 32)
    out = block(x)
    assert out.shape == (4, 64, 32, 32), "Output shape mismatch when increasing channels"

    block = ResidualBlock(in_channels=64, out_channels=64)
    x = torch.randn(4, 64, 32, 32)
    out = block(x)
    assert torch.all(out != x)

    model = DQNSolver(input_shape=(4, 84, 84), n_actions=6)
    x = torch.randn(4, 4, 84, 84)  # Batch size 4
    out = model(x)
    assert out.shape == (4, 6), "DQNSolver output shape mismatch"

    model = DQNSolverResNet(input_shape=(4, 84, 84), n_actions=6)
    x = torch.randn(4, 4, 84, 84)  # Batch size 4
    out = model(x)
    assert out.shape == (4, 6), "DQNSolverResNet output shape mismatch"

    model = DQNSolverResNet(input_shape=(4, 84, 84), n_actions=6)
    x = torch.randn(4, 4, 84, 84)
    features = model.forward_conv(x)
    assert len(features.shape) == 4, "Feature map should have 4 dimensions (batch, channels, H, W)"