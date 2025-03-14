import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

#models
class DQNSolver(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
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
            nn.Conv2d(input_shape[0], 8, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.res_block1 = ResidualBlock(8, 16)
        self.res_block2 = ResidualBlock(16, 16)
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


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNetwork, self).__init__()
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
            nn.Linear(512, n_actions),
            nn.Softmax(dim=-1)  # Outputs action probabilities
        )

    def _get_conv_out(self, shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)


class NoisyLinear(nn.Module):
    """Noisy Linear layer for NoisyNet exploration"""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        """Initialize weights and bias"""
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.in_features**0.5)
        self.bias_sigma.data.fill_(self.std_init / self.in_features**0.5)

    def sample_noise(self):
        """Sample noise for NoisyNet exploration"""
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class RainbowDQN(nn.Module):
    """Rainbow DQN with Dueling Network, Noisy Layers, and Distributional C51"""
    def __init__(self, state_dim, action_dim, atom_size=51, v_min=-10, v_max=10):
        super(RainbowDQN, self).__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, atom_size)

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # Dueling network
        self.advantage = NoisyLinear(128, action_dim * atom_size)
        self.value = NoisyLinear(128, atom_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        advantage = self.advantage(x).view(-1, self.action_dim, self.atom_size)
        value = self.value(x).view(-1, 1, self.atom_size)

        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)  # Dueling
        q_dist = F.softmax(q_dist, dim=-1)  # Distributional output
        return q_dist

    def reset_noise(self):
        """Resample noise for NoisyNet"""
        self.advantage.sample_noise()
        self.value.sample_noise()

    
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
