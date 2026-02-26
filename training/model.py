# training/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self, board_size=8, num_channels=13, policy_size=8513):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        # Update the policy head to output 8513 dimensions.
        self.fc_policy = nn.Linear(256, policy_size)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, x):
        # x shape: (batch_size, num_channels, board_size, board_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))
        return policy, value