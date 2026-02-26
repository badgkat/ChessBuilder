# training/dataset.py
import torch
from torch.utils.data import Dataset
import os
import numpy as np

class ChessDataset(Dataset):
    def __init__(self, data_file=None):
        if data_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(script_dir, '..', 'training_data.npz')
        if os.path.exists(data_file):
            data = np.load(data_file)
            self.states = data['states']           # shape: (N, num_channels, board_size, board_size)
            self.policy_targets = data['policy_targets']  # shape: (N, policy_size)
            self.value_targets = data['value_targets']    # shape: (N, 1)
        else:
            raise FileNotFoundError(f"Data file {data_file} not found. Please run training/selfplay.py first.")
            
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        policy_target = self.policy_targets[idx]
        value_target = self.value_targets[idx]
        return state, policy_target, value_target
