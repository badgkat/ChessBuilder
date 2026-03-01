# training/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class ChessNet(nn.Module):
    def __init__(self, board_size=8, num_channels=13, policy_size=8513, num_res_blocks=4, hidden_channels=128):
        super().__init__()
        # Initial convolution
        self.conv_input = nn.Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(hidden_channels)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_dropout = nn.Dropout(0.3)
        self.policy_fc = nn.Linear(32 * board_size * board_size, policy_size)

        # Value head
        self.value_conv = nn.Conv2d(hidden_channels, 4, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * board_size * board_size, 64)
        self.value_dropout = nn.Dropout(0.3)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Shared trunk
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_dropout(p)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_dropout(v)
        v = torch.tanh(self.value_fc2(v))

        return p, v
