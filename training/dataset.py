# training/dataset.py
import torch
from torch.utils.data import Dataset
import os
import numpy as np


def _build_flip_index_map(board_size=8, policy_size=8513):
    """Precompute mapping from original policy index to horizontally-flipped index."""
    flip_map = np.arange(policy_size, dtype=np.int64)

    # Standard moves [0, 4095]: (sr*8+sc)*64 + (dr*8+dc)
    for i in range(board_size**4):
        src_idx = i // (board_size * board_size)
        dst_idx = i % (board_size * board_size)
        sr, sc = divmod(src_idx, board_size)
        dr, dc = divmod(dst_idx, board_size)
        fsc = board_size - 1 - sc
        fdc = board_size - 1 - dc
        flip_map[i] = (sr * board_size + fsc) * (board_size * board_size) + (dr * board_size + fdc)

    # collect_gold [4096]: no change
    # (index 4096 stays 4096)

    # Purchase [4097, 4416]: 4097 + type_idx*64 + dr*8+dc
    purchase_start = board_size**4 + 1
    for t in range(5):
        for sq in range(board_size * board_size):
            dr, dc = divmod(sq, board_size)
            fdc = board_size - 1 - dc
            orig = purchase_start + t * (board_size * board_size) + sq
            flipped = purchase_start + t * (board_size * board_size) + dr * board_size + fdc
            flip_map[orig] = flipped

    # Transfer gold [4417, 8512]: 4417 + (sr*8+sc)*64 + (dr*8+dc)
    transfer_start = purchase_start + 5 * (board_size * board_size)
    for i in range(board_size**4):
        src_idx = i // (board_size * board_size)
        dst_idx = i % (board_size * board_size)
        sr, sc = divmod(src_idx, board_size)
        dr, dc = divmod(dst_idx, board_size)
        fsc = board_size - 1 - sc
        fdc = board_size - 1 - dc
        orig = transfer_start + i
        flipped = transfer_start + (sr * board_size + fsc) * (board_size * board_size) + (dr * board_size + fdc)
        flip_map[orig] = flipped

    return flip_map


# Precompute once at module load
_FLIP_INDEX_MAP = _build_flip_index_map()


class ChessDataset(Dataset):
    def __init__(self, data_file=None, augment=False):
        if data_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(script_dir, '..', 'training_data.npz')
        if os.path.exists(data_file):
            data = np.load(data_file)
            self.states = data['states']
            self.policy_targets = data['policy_targets']
            self.value_targets = data['value_targets']
        else:
            raise FileNotFoundError(f"Data file {data_file} not found. Please run training/selfplay.py first.")

        self.augment = augment
        self.base_len = len(self.states)

    def __len__(self):
        return self.base_len * 2 if self.augment else self.base_len

    def __getitem__(self, idx):
        if self.augment and idx >= self.base_len:
            # Flipped example
            real_idx = idx - self.base_len
            state = np.flip(self.states[real_idx], axis=2).copy()  # flip columns
            policy = self.policy_targets[real_idx][_FLIP_INDEX_MAP]
            value = self.value_targets[real_idx]
            return state, policy, value
        else:
            real_idx = idx if not self.augment else idx
            return self.states[real_idx], self.policy_targets[real_idx], self.value_targets[real_idx]
