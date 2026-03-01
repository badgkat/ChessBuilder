# Fix Training Rewards & Move Limit

**Date:** 2026-03-01
**Goal:** Make training produce useful signal by fixing the reward scale bug and removing the punitive move limit.

---

## Problem

Training is producing no useful learning signal. After 9 iterations and 98k examples:

- **95% of games hit the 200-move limit** and receive a value target of -5.
- **The value head uses tanh** (output range [-1, +1]), but targets are -5, -3, -0.5, +3, +5. The model cannot represent these values. Every value-head gradient step is wasted trying to push tanh toward an unreachable target.
- **One label dominates**: with 95% of examples at -5, the value head learns "every position is equally terrible" instead of distinguishing good from bad positions.
- **Asymmetric rewards** (white win=+5, black win=-3) add unnecessary complexity with no clear benefit.

The replay buffer contains 98k examples generated under these broken rewards. Training on them reinforces bad signal.

## Changes

### 1. Fix Reward Values (`src/game.py` — `get_outcome()`)

Standard AlphaZero-style symmetric rewards that fit within tanh range:

| Outcome | Current | New |
|---------|---------|-----|
| White wins | +5 | +1 |
| Black wins | -3 | -1 |
| Natural draw | -0.5 | 0 |
| Forced draw (max moves) | -5 | 0 |

No special-casing for max-moves draws. A draw is a draw.

### 2. Raise Move Limit (`training/selfplay.py`)

- Change `max_moves` default from 200 to 1000.
- Games should end via the built-in draw rules (threefold repetition, 50-move rule, insufficient material) or checkmate — not an artificial timeout.
- The 1000 cap is a safety valve only, not a training signal.

### 3. Wipe Replay Buffer

- Delete `training_data.npz` before retraining.
- All existing data was generated under broken reward values.

### 4. Fresh Checkpoint

- Delete `models/chess_model_checkpoint.pt` and start training from scratch.
- The value head learned to fit impossible targets — those weights are not salvageable.

## What Does NOT Change

- Model architecture (4 residual blocks, 128 hidden channels)
- Temperature decay schedule (2.0 → 1.0 → 0.5)
- Horizontal flip data augmentation
- Learning rate (0.001 Adam)
- Replay buffer mechanics (append + trim to 500k)
- Self-play game generation logic (beyond max_moves)

## Files Affected

- `src/game.py` — `get_outcome()`: replace reward values
- `training/selfplay.py` — change `max_moves` default from 200 to 1000
- `training_data.npz` — delete (manual step)
- `models/chess_model_checkpoint.pt` — delete (manual step)

## Success Criteria

After retraining with fixed rewards:

1. Value targets in training data are all within [-1, +1].
2. Fewer than 50% of games hit the 1000-move safety limit (games end naturally).
3. Value loss decreases over iterations (the value head is actually learning).
4. Win/loss/draw distribution becomes more balanced over time as the model improves.
