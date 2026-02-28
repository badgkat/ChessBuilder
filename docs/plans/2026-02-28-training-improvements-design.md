# Training Pipeline Improvements Design

**Date:** 2026-02-28
**Goal:** Make the training pipeline effective — fix data quality bugs, eliminate code duplication, add replay buffer, upgrade model architecture, add temperature decay and data augmentation.

---

## 1. Shared Legal Actions & Bug Fix

**Problem:** Legal action enumeration is duplicated 3 times (`get_model_move`, `get_training_example`, `get_random_move`), and `get_model_move` uses a different legality check (`is_move_legal` with full game copy) while the others use `simulate_move_is_safe` (lightweight in-place). This inconsistency means the model could select moves the training data considers illegal.

**Solution:** Extract a single `get_legal_actions()` method. Use `simulate_move_is_safe` for standard moves (fast, in-place) and direct board manipulation for purchases. All three callers become thin wrappers:

- `get_random_move()` → `random.choice(self.get_legal_actions())`
- `get_training_example()` → uniform distribution over `self.get_legal_actions()`
- `get_model_move()` → mask model output to `self.get_legal_actions()`

## 2. Replay Buffer

**Problem:** Each iteration overwrites `training_data.npz` with only the latest games. Model never sees older data → catastrophic forgetting.

**Solution:** Append new games to growing buffer with configurable max size (default 500k examples). Rolling window drops oldest examples when buffer exceeds max. Implemented in `selfplay.py` — load existing file, append, trim, save.

## 3. Model Architecture Upgrade (Moderate)

**Problem:** 2 conv layers → 256-dim FC → 8513 policy head. The 256→8513 bottleneck makes learning a useful policy nearly impossible.

**New architecture:**
```
Input: 13 × 8 × 8
Conv2d(13, 128, 3×3, pad=1) + BN + ReLU
4× Residual Block: Conv(128,128,3×3) + BN + ReLU + Conv(128,128,3×3) + BN + skip
Policy: Conv(128,32,1×1) + BN + ReLU → Flatten(2048) → Dropout(0.3) → FC(2048, 8513)
Value: Conv(128,4,1×1) + BN + ReLU → Flatten(256) → FC(256,64) + ReLU + Dropout(0.3) → FC(64,1) + tanh
```

- 4 residual blocks (10 conv layers total)
- BatchNorm for stable training
- Policy bottleneck: 2048→8513 instead of 256→8513
- Dropout(0.3) on FC layers
- Same `ChessNet(num_channels=13, policy_size=8513)` interface
- Old checkpoints incompatible (fresh start required)

## 4. Temperature Decay

**Problem:** Fixed `temperature=3` (very high exploration) throughout all training.

**Solution:** Schedule based on iteration:
- Iterations 0-2: temperature=2.0 (explore)
- Iterations 3-6: temperature=1.0 (balanced)
- Iterations 7+: temperature=0.5 (exploit)

Pass iteration number from `iterative_training.py` → `generate_selfplay_data()` → `get_model_move()`.

## 5. Data Augmentation (Horizontal Flip)

**Problem:** Limited training data from expensive self-play.

**Solution:** For each training example, also include its horizontal mirror (flip columns a↔h). Chess is symmetric across the vertical axis.

- Flip board state channels along column axis
- Remap policy target indices to reflect mirrored src/dst squares
- Doubles effective training data
- Implemented as transform in `dataset.py` at load time

## Files Affected

- `src/game.py` — new `get_legal_actions()`, refactor 3 callers
- `training/model.py` — new residual architecture
- `training/selfplay.py` — replay buffer, temperature parameter
- `training/dataset.py` — horizontal flip augmentation
- `training/interative_training.py` — pass iteration to selfplay, temperature schedule
- `training/main.py` — same temperature/iteration changes for GUI
