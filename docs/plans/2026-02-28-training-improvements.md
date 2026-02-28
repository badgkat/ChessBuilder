# Training Pipeline Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the training pipeline effective by fixing data quality bugs, eliminating code duplication, adding a replay buffer, upgrading the model architecture, adding temperature decay, and adding data augmentation.

**Architecture:** Extract a shared `get_legal_actions()` method in `game.py` to replace 3 duplicated implementations. Upgrade `ChessNet` to use residual blocks with batch norm. Add a replay buffer in `selfplay.py` that appends to existing data. Add temperature decay and horizontal-flip augmentation in the training loop.

**Tech Stack:** Python, PyTorch, NumPy, Pygame, pytest

---

### Task 1: Extract shared `get_legal_actions()` method

**Files:**
- Modify: `src/game.py:1540-1606` (where `get_random_move` lives)
- Test: `tests/test_game.py`

**Context:** Legal action enumeration is duplicated in `get_random_move()` (line 1540), `get_training_example()` (line 388), and `get_model_move()` (line 112). The first two use `simulate_move_is_safe` (fast, in-place) while `get_model_move` uses `is_move_legal` (slow, full copy). We unify on the fast approach.

**Step 1: Write the failing test**

Add to `tests/test_game.py`:

```python
def test_get_legal_actions_returns_list(game_instance):
    """get_legal_actions returns a non-empty list of 4-tuples."""
    g = game_instance
    g.new_game()
    actions = g.get_legal_actions()
    assert isinstance(actions, list)
    assert len(actions) > 0
    for a in actions:
        assert len(a) == 4
        assert a[0] in ("move", "collect_gold", "purchase", "transfer_gold")


def test_get_legal_actions_matches_random_move(game_instance):
    """get_legal_actions should produce the same set as get_random_move draws from."""
    g = game_instance
    g.new_game()
    actions = g.get_legal_actions()
    # get_random_move should only return actions that are in get_legal_actions
    for _ in range(50):
        g.new_game()
        move = g.get_random_move()
        if move:
            assert move in g.get_legal_actions()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_game.py::test_get_legal_actions_returns_list tests/test_game.py::test_get_legal_actions_matches_random_move -v`
Expected: FAIL with `AttributeError: 'Game' object has no attribute 'get_legal_actions'`

**Step 3: Implement `get_legal_actions()` — insert ABOVE `get_random_move()` (before line 1540)**

```python
    def get_legal_actions(self):
        """Return list of all legal 4-tuple actions for the current player."""
        legal_actions = []
        board_size = board.BOARD_SIZE

        # 1. Standard moves (including promotions)
        for r in range(board_size):
            for c in range(board_size):
                piece = self.board[r][c]
                if piece and piece.color == self.turn:
                    moves = board.get_valid_moves(piece, (r, c), self.board, self.en_passant)
                    for move in moves:
                        if self.simulate_move_is_safe((r, c), move):
                            if piece.type == 'P' and self.move_leads_to_promotion(piece, (r, c), move):
                                for promo in ['Q', 'R', 'B', 'N']:
                                    legal_actions.append(("move", (r, c), move, promo))
                            else:
                                legal_actions.append(("move", (r, c), move, None))

        # 2. Gold collection (pawns, not in check)
        if not self.is_in_check(self.turn):
            for r in range(board_size):
                for c in range(board_size):
                    piece = self.board[r][c]
                    if piece and piece.color == self.turn and piece.type == 'P':
                        legal_actions.append(("collect_gold", (r, c), None, None))

        # 3. Purchase actions (king gold, adjacent empty squares)
        king, king_pos = None, None
        for r in range(board_size):
            for c in range(board_size):
                piece = self.board[r][c]
                if piece and piece.color == self.turn and piece.type == 'K':
                    king, king_pos = piece, (r, c)
                    break
            if king_pos:
                break
        if king and king.gold > 0:
            kr, kc = king_pos
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = kr + dr, kc + dc
                    if board.in_bounds(nr, nc) and self.board[nr][nc] is None:
                        for p_type in ['P', 'N', 'B', 'R', 'Q']:
                            cost = board.PIECE_COST.get(p_type)
                            if cost and king.gold >= cost:
                                if p_type == 'P' and (nr == 0 or nr == board.BOARD_SIZE - 1):
                                    continue
                                self.board[nr][nc] = board.Piece(p_type, king.color)
                                if not self.is_in_check(king.color):
                                    legal_actions.append(("purchase", king_pos, (nr, nc), p_type))
                                self.board[nr][nc] = None

        # 4. Gold transfers (not in check, pieces with gold)
        if not self.is_in_check(self.turn):
            for r in range(board_size):
                for c in range(board_size):
                    piece = self.board[r][c]
                    if piece and piece.color == self.turn and piece.gold > 0:
                        visible = board.get_visible_squares(piece, (r, c), self.board)
                        for target in visible:
                            legal_actions.append(("transfer_gold", (r, c), target, None))

        return legal_actions
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_game.py::test_get_legal_actions_returns_list tests/test_game.py::test_get_legal_actions_matches_random_move -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/game.py tests/test_game.py
git commit -m "feat: extract shared get_legal_actions() method"
```

---

### Task 2: Refactor callers to use `get_legal_actions()`

**Files:**
- Modify: `src/game.py:1540-1606` (`get_random_move`)
- Modify: `src/game.py:388-468` (`get_training_example`)
- Modify: `src/game.py:137-211` (legal action block inside `get_model_move`)
- Test: `tests/test_game.py`

**Step 1: Refactor `get_random_move()`**

Replace the entire body (lines 1541-1606) with:

```python
    def get_random_move(self):
        """Return a random legal action as a 4-tuple (action_type, src, dst, extra)."""
        legal_actions = self.get_legal_actions()
        if not legal_actions:
            return None
        return random.choice(legal_actions)
```

**Step 2: Run existing tests to verify they still pass**

Run: `python -m pytest tests/test_game.py::test_get_random_move_returns_4_tuple tests/test_game.py::test_get_random_move_includes_gold_collection -v`
Expected: PASS

**Step 3: Refactor `get_training_example()`**

Replace lines 396-466 (the legal action enumeration block) with:

```python
    def get_training_example(self):
        """
        Returns (board_state, policy_target, player) where policy_target is
        a uniform distribution over all legal actions in the 8513-dim action space.
        """
        board_state = self.encode_board_state()
        total_actions = 8513
        policy_target = np.zeros(total_actions, dtype=np.float32)
        legal_actions = self.get_legal_actions()

        # Uniform distribution over legal actions
        if legal_actions:
            probability = 1.0 / len(legal_actions)
            for action in legal_actions:
                action_type, src, dst, pt = action
                index = self.move_to_index(action_type, src, dst, pt)
                policy_target[index] = probability

        return board_state, policy_target, self.turn
```

**Step 4: Run existing test to verify it still passes**

Run: `python -m pytest tests/test_game.py::test_training_example_only_legal_actions -v`
Expected: PASS

**Step 5: Refactor `get_model_move()`**

Replace lines 137-211 (the legal action enumeration block, from `# 2. Build the list of legal actions.` through `legal_actions.append(candidate)` for transfer_gold) with:

```python
        # 2. Build the list of legal actions (shared method).
        legal_actions = self.get_legal_actions()
```

Keep everything else in `get_model_move` unchanged (lines 112-136 for model inference, and lines 213-250 for index mapping and sampling).

**Step 6: Run all game tests to verify nothing broke**

Run: `python -m pytest tests/test_game.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/game.py
git commit -m "refactor: use shared get_legal_actions() in all 3 callers"
```

---

### Task 3: Upgrade model architecture

**Files:**
- Modify: `training/model.py` (full rewrite)
- Test: `tests/test_training.py`

**Step 1: Write the failing test**

Add to `tests/test_training.py`:

```python
def test_model_residual_architecture():
    """New model should have residual blocks and correct output shapes."""
    from training.model import ChessNet
    import torch
    model = ChessNet(num_channels=13, policy_size=8513)
    x = torch.randn(2, 13, 8, 8)
    policy, value = model(x)
    assert policy.shape == (2, 8513)
    assert value.shape == (2, 1)
    # Model should have more parameters than old 2-layer version (~50k)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 500_000, f"Model too small: {num_params} params"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training.py::test_model_residual_architecture -v`
Expected: FAIL (old model has ~50k params, assertion requires >500k)

**Step 3: Rewrite `training/model.py`**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training.py::test_model_residual_architecture -v`
Expected: PASS

**Step 5: Delete old checkpoint (incompatible architecture)**

```bash
rm -f models/chess_model_checkpoint.pt
```

**Step 6: Run all training tests**

Run: `python -m pytest tests/test_training.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add training/model.py tests/test_training.py
git commit -m "feat: upgrade ChessNet to residual architecture with batch norm"
```

---

### Task 4: Implement replay buffer in selfplay

**Files:**
- Modify: `training/selfplay.py`
- Test: `tests/test_training.py`

**Step 1: Write the failing test**

Add to `tests/test_training.py`:

```python
import tempfile
import numpy as np

def test_replay_buffer_appends():
    """Selfplay should append to existing data, not overwrite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, 'training_data.npz')
        # Create fake existing data
        existing = {
            'states': np.random.randn(10, 13, 8, 8).astype(np.float32),
            'policy_targets': np.random.randn(10, 8513).astype(np.float32),
            'value_targets': np.random.randn(10, 1).astype(np.float32),
        }
        np.savez(data_path, **existing)

        from training.selfplay import generate_selfplay_data
        generate_selfplay_data(num_games=2, model=None, device=None, data_path=data_path)
        result = np.load(data_path)
        # Should have MORE than the original 10 examples
        assert result['states'].shape[0] > 10
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training.py::test_replay_buffer_appends -v`
Expected: FAIL (`generate_selfplay_data() got an unexpected keyword argument 'data_path'`)

**Step 3: Modify `training/selfplay.py`**

Add `data_path` and `max_buffer_size` parameters to `generate_selfplay_data`. After generating new data, load existing data from `data_path` (if it exists), concatenate, trim to `max_buffer_size`, and save.

Replace the function signature (line 13) with:

```python
def generate_selfplay_data(num_games=10, model=None, device=None, check_interruption=None, max_moves=200, data_path=None, max_buffer_size=500_000, iteration=0):
```

Add at line 8 (after `_DATA_PATH`):

```python
def _get_temperature(iteration):
    """Temperature decay schedule."""
    if iteration <= 2:
        return 2.0
    elif iteration <= 6:
        return 1.0
    else:
        return 0.5
```

Replace the data path default at the start of the function body (after `global global_game_counter`):

```python
    if data_path is None:
        data_path = _DATA_PATH
```

Replace the temperature in the model move call (line 51):

```python
            if model is not None:
                temp = _get_temperature(iteration)
                move = game_instance.get_model_move(model, device, temperature=temp, use_dirichlet=True, epsilon=0.25, alpha=0.3, sample=True)
```

Replace the save block at the end (lines 94-98) with:

```python
    new_states = np.array(states, dtype=np.float32)
    new_policy = np.array(policy_targets, dtype=np.float32)
    new_values = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

    # Replay buffer: load existing data and append
    if os.path.exists(data_path):
        try:
            existing = np.load(data_path)
            new_states = np.concatenate([existing['states'], new_states], axis=0)
            new_policy = np.concatenate([existing['policy_targets'], new_policy], axis=0)
            new_values = np.concatenate([existing['value_targets'], new_values], axis=0)
        except Exception:
            pass  # If file is corrupt, start fresh

    # Trim to max buffer size (keep most recent)
    if new_states.shape[0] > max_buffer_size:
        new_states = new_states[-max_buffer_size:]
        new_policy = new_policy[-max_buffer_size:]
        new_values = new_values[-max_buffer_size:]

    np.savez(data_path, states=new_states, policy_targets=new_policy, value_targets=new_values)
    print(f"Generated self-play data from {num_games} games. Buffer size: {new_states.shape[0]}. Total games so far: {global_game_counter}.")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training.py::test_replay_buffer_appends -v`
Expected: PASS

**Step 5: Commit**

```bash
git add training/selfplay.py tests/test_training.py
git commit -m "feat: add replay buffer and temperature decay to selfplay"
```

---

### Task 5: Plumb iteration through training loops

**Files:**
- Modify: `training/interative_training.py:34` (pass iteration to selfplay)
- Modify: `training/main.py:59` (pass iteration to selfplay)

**Step 1: Update `training/interative_training.py`**

Change line 34 from:

```python
        generate_selfplay_data(num_games=games_per_iter, model=model, device=device)
```

to:

```python
        generate_selfplay_data(num_games=games_per_iter, model=model, device=device, iteration=iteration)
```

**Step 2: Update `training/main.py` TrainingWorker.run()**

Change line 59 (inside the per-game loop) from:

```python
                generate_selfplay_data(num_games=1, model=model, device=device)
```

to:

```python
                generate_selfplay_data(num_games=1, model=model, device=device, iteration=iteration)
```

**Step 3: Run training tests to verify nothing broke**

Run: `python -m pytest tests/test_training.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add training/interative_training.py training/main.py
git commit -m "feat: pass iteration to selfplay for temperature decay"
```

---

### Task 6: Add horizontal flip data augmentation

**Files:**
- Modify: `training/dataset.py`
- Test: `tests/test_training.py`

**Context:** The `move_to_index` encoding maps (src_row, src_col) → (dst_row, dst_col) to flat indices. Horizontal flip mirrors columns: col → 7-col. The index scheme is:
- `[0, 4095]`: standard moves — `(sr*8+sc)*64 + (dr*8+dc)`
- `4096`: collect_gold (unchanged by flip)
- `[4097, 4416]`: purchase — `4097 + type_idx*64 + dr*8+dc`
- `[4417, 8512]`: transfer_gold — `4417 + (sr*8+sc)*64 + (dr*8+dc)`

**Step 1: Write the failing test**

Add to `tests/test_training.py`:

```python
def test_dataset_augmentation_doubles_data():
    """Dataset with augmentation should have 2x the examples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, 'training_data.npz')
        states = np.random.randn(5, 13, 8, 8).astype(np.float32)
        policy = np.zeros((5, 8513), dtype=np.float32)
        policy[:, 0] = 1.0  # dummy
        values = np.zeros((5, 1), dtype=np.float32)
        np.savez(data_path, states=states, policy_targets=policy, value_targets=values)

        from training.dataset import ChessDataset
        ds = ChessDataset(data_file=data_path, augment=True)
        assert len(ds) == 10  # 5 original + 5 flipped
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training.py::test_dataset_augmentation_doubles_data -v`
Expected: FAIL (`__init__() got an unexpected keyword argument 'augment'`)

**Step 3: Implement augmentation in `training/dataset.py`**

Replace the entire file with:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training.py::test_dataset_augmentation_doubles_data -v`
Expected: PASS

**Step 5: Enable augmentation in training loops**

In `training/interative_training.py`, change line 38 from:

```python
        dataset = ChessDataset()
```

to:

```python
        dataset = ChessDataset(augment=True)
```

In `training/main.py`, change line 65 from:

```python
            dataset = ChessDataset()
```

to:

```python
            dataset = ChessDataset(augment=True)
```

In `training/train.py`, change line 69 from:

```python
    dataset = ChessDataset()
```

to:

```python
    dataset = ChessDataset(augment=True)
```

**Step 6: Run all training tests**

Run: `python -m pytest tests/test_training.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add training/dataset.py training/interative_training.py training/main.py training/train.py tests/test_training.py
git commit -m "feat: add horizontal flip data augmentation"
```

---

### Task 7: Integration smoke test

**Files:**
- Test: `tests/test_training.py`

**Step 1: Write integration test**

Add to `tests/test_training.py`:

```python
def test_full_pipeline_with_improvements():
    """Smoke test: selfplay with replay buffer -> augmented dataset -> residual model training."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, 'training_data.npz')

        # Run 3 selfplay games (random, no model)
        from training.selfplay import generate_selfplay_data
        generate_selfplay_data(num_games=3, model=None, device=None, data_path=data_path, iteration=0)
        assert os.path.exists(data_path)

        # Run 2 more games — replay buffer should append
        generate_selfplay_data(num_games=2, model=None, device=None, data_path=data_path, iteration=1)
        data = np.load(data_path)
        total_examples = data['states'].shape[0]
        assert total_examples > 0

        # Load with augmentation
        from training.dataset import ChessDataset
        ds = ChessDataset(data_file=data_path, augment=True)
        assert len(ds) == total_examples * 2

        # Train 1 batch with new model
        from training.model import ChessNet
        import torch
        device = torch.device('cpu')
        model = ChessNet(num_channels=13, policy_size=8513).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loader = torch.utils.data.DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True)
        batch = next(iter(loader))
        states, policy_targets, value_targets = [b.to(device) for b in batch]
        policy_pred, value_pred = model(states)
        log_probs = torch.nn.functional.log_softmax(policy_pred, dim=1)
        loss_policy = -torch.sum(policy_targets * log_probs) / policy_targets.shape[0]
        loss_value = torch.nn.functional.mse_loss(value_pred, value_targets)
        loss = loss_policy + loss_value
        loss.backward()
        optimizer.step()
        assert loss.item() > 0
```

**Step 2: Run the integration test**

Run: `python -m pytest tests/test_training.py::test_full_pipeline_with_improvements -v`
Expected: PASS

**Step 3: Run ALL tests**

Run: `python -m pytest tests/test_game.py tests/test_training.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/test_training.py
git commit -m "test: add full pipeline integration test with all improvements"
```
