# Fix Training Rewards Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the broken reward scale and move limit so training produces useful learning signal.

**Architecture:** Two code changes — fix `get_outcome()` to return values within tanh range [-1, +1], and raise `max_moves` from 200 to 1000. Then wipe stale data/checkpoint files generated under the broken rewards.

**Tech Stack:** Python, PyTorch, Pygame, pytest

---

### Task 1: Write failing test for correct reward values

**Files:**
- Modify: `tests/test_training.py`

**Step 1: Write the failing test**

Add this test at the end of `tests/test_training.py`:

```python
def test_get_outcome_values_within_tanh_range(game_instance):
    """get_outcome() must return values in [-1, +1] to match tanh output."""
    g = game_instance
    g.new_game()

    # White wins
    g.game_over = True
    g.winner = "white"
    assert g.get_outcome() == 1

    # Black wins
    g.winner = "black"
    assert g.get_outcome() == -1

    # Natural draw
    g.winner = "draw"
    assert g.get_outcome() == 0

    # Forced draw (max moves)
    g.max_moves_reached = True
    assert g.get_outcome() == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py::test_get_outcome_values_within_tanh_range -v`
Expected: FAIL — current values are +5, -3, -0.5, -5

---

### Task 2: Fix get_outcome() to return correct values

**Files:**
- Modify: `src/game.py:1482-1493`

**Step 1: Replace `get_outcome()` implementation**

Change from:
```python
    def get_outcome(self):
        if not self.game_over:
            return None  # Game not finished yet
        if self.winner == "draw":
            # Check if the draw resulted from reaching max moves
            if hasattr(self, "max_moves_reached") and self.max_moves_reached:
                return -5    # Strongly discourage a forced draw
            else:
                return -0.5  # Only slightly discourage a natural draw
        # For wins, reward white wins strongly and punish losses moderately.
        return 5 if self.winner == "white" else -3
```

To:
```python
    def get_outcome(self):
        if not self.game_over:
            return None
        if self.winner == "draw":
            return 0
        return 1 if self.winner == "white" else -1
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_training.py::test_get_outcome_values_within_tanh_range -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `pytest -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/game.py tests/test_training.py
git commit -m "fix: use symmetric rewards within tanh range (+1/0/-1)"
```

---

### Task 3: Raise max_moves default from 200 to 1000

**Files:**
- Modify: `training/selfplay.py:22`

**Step 1: Change the default parameter**

On line 22, change:
```python
def generate_selfplay_data(num_games=10, model=None, device=None, check_interruption=None, max_moves=200, data_path=None, max_buffer_size=500_000, iteration=0):
```

To:
```python
def generate_selfplay_data(num_games=10, model=None, device=None, check_interruption=None, max_moves=1000, data_path=None, max_buffer_size=500_000, iteration=0):
```

**Step 2: Run full test suite**

Run: `pytest -v`
Expected: All tests pass (no test depends on max_moves=200)

**Step 3: Commit**

```bash
git add training/selfplay.py
git commit -m "fix: raise selfplay max_moves from 200 to 1000"
```

---

### Task 4: Delete stale training data and checkpoint

**Files:**
- Delete: `training_data.npz`
- Delete: `models/chess_model_checkpoint.pt`

**Step 1: Delete the files**

```bash
rm training_data.npz models/chess_model_checkpoint.pt
```

**Step 2: Verify files are gone**

```bash
ls training_data.npz models/chess_model_checkpoint.pt
```
Expected: "No such file or directory" for both

**Step 3: Commit**

```bash
git add -u training_data.npz models/chess_model_checkpoint.pt
git commit -m "chore: wipe stale training data and checkpoint from broken rewards"
```

Note: Only commit if these files are tracked by git. If they're in `.gitignore`, just delete them without committing.

---

### Task 5: Verify everything works end-to-end

**Step 1: Run full test suite**

Run: `pytest -v`
Expected: All tests pass

**Step 2: Quick selfplay smoke test**

Run: `python -c "from training.selfplay import generate_selfplay_data; generate_selfplay_data(num_games=2, data_path='test_smoke.npz')"`

Verify output shows `Buffer size: <N>` and no errors.

**Step 3: Verify new reward values in generated data**

```python
python -c "
import numpy as np
d = np.load('test_smoke.npz')
v = d['value_targets'].flatten()
print('Unique values:', np.unique(v))
print('All within [-1,1]:', all(-1 <= x <= 1 for x in v))
d.close()
"
```
Expected: Unique values are subset of {-1, 0, 1} and all within [-1, 1].

**Step 4: Clean up smoke test file**

```bash
rm test_smoke.npz
```
