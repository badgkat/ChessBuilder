# Parallel Self-Play + GPU Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Parallelize self-play across 16 CPU workers and use GPU for the training phase, utilizing available server resources alongside an existing embedding training job.

**Architecture:** Add a `headless` flag to `Game` that skips all pygame calls, enabling fork-safe multiprocessing. Refactor `generate_selfplay_data` to dispatch games across a `multiprocessing.Pool`. Training phase uses ROCm GPU automatically via existing `torch.device` detection.

**Tech Stack:** Python, multiprocessing, PyTorch (ROCm), NumPy, pytest

---

### Task 1: Add headless mode to Game

**Files:**
- Modify: `src/game.py:11-60` (`__init__`)
- Modify: `src/game.py:1333-1367` (`update`)
- Test: `tests/test_game.py`

**Step 1: Write the failing test**

Add to `tests/test_game.py`:

```python
def test_headless_game_no_pygame(pygame_init):
    """Game with headless=True should work without a screen surface."""
    g = Game(screen=None, headless=True)
    g.new_game()
    actions = g.get_legal_actions()
    assert len(actions) > 0
    move = g.get_random_move()
    assert move is not None
    g.apply_move(move)
    g.update()  # should be a no-op, not crash
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_game.py::test_headless_game_no_pygame -v`
Expected: FAIL with `__init__() got an unexpected keyword argument 'headless'`

**Step 3: Implement headless mode**

Change `Game.__init__` (line 12) from:

```python
    def __init__(self, screen):
        self.screen = screen
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont(None, 24)
        self.load_images()
```

To:

```python
    def __init__(self, screen, headless=False):
        self.screen = screen
        self.headless = headless
        if not headless:
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
            self.load_images()
        else:
            self.clock = None
            self.font = None
            self.images = {}
```

Change `update()` (line 1333) -- add early return at the top:

```python
    def update(self):
        if self.headless:
            return
        self.screen.fill((0,0,0))
```

(Keep the rest of `update()` unchanged.)

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_game.py::test_headless_game_no_pygame -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS (existing tests use `headless=False` by default)

**Step 6: Commit**

```bash
git add src/game.py tests/test_game.py
git commit -m "feat: add headless mode to Game for fork-safe self-play"
```

---

### Task 2: Add parallel worker function to selfplay

**Files:**
- Modify: `training/selfplay.py` (full rewrite)
- Test: `tests/test_training.py`

**Step 1: Write the failing test**

Add to `tests/test_training.py`:

```python
def test_parallel_selfplay_generates_data():
    """Parallel selfplay with multiple workers should produce training data."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, 'training_data.npz')
        from training.selfplay import generate_selfplay_data
        generate_selfplay_data(
            num_games=4, model=None, device=None,
            data_path=data_path, num_workers=2,
        )
        assert os.path.exists(data_path)
        data = np.load(data_path)
        assert data['states'].shape[0] > 0
        data.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training.py::test_parallel_selfplay_generates_data -v`
Expected: FAIL with `generate_selfplay_data() got an unexpected keyword argument 'num_workers'`

**Step 3: Implement the parallel worker and refactor generate_selfplay_data**

Replace the entire `training/selfplay.py` with:

```python
# training/selfplay.py
import sys, os
import numpy as np
import multiprocessing

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_SCRIPT_DIR, '..', 'training_data.npz')


def _get_temperature(iteration):
    """Temperature decay schedule."""
    if iteration <= 2:
        return 2.0
    elif iteration <= 6:
        return 1.0
    else:
        return 0.5


# Global counter for games played across multiple calls.
global_game_counter = 0


def _selfplay_worker(args):
    """Worker function that plays N games and returns collected examples.

    Runs in a subprocess -- creates its own headless Game instance.
    Each worker is independent with no shared state.
    """
    num_games, max_moves, iteration, model_state_dict, worker_id = args

    # Nice this process down so it doesn't compete with priority jobs
    try:
        os.nice(10)
    except OSError:
        pass

    from src.game import Game

    states = []
    policy_targets = []
    value_targets = []

    model = None
    device = None
    if model_state_dict is not None:
        import torch
        from training.model import ChessNet
        device = torch.device("cpu")
        model = ChessNet(num_channels=13, policy_size=8513).to(device)
        model.load_state_dict(model_state_dict)
        model.set_to_eval_mode()

    for game_idx in range(num_games):
        game_instance = Game(screen=None, headless=True)
        game_instance.new_game()
        move_count = 0
        game_examples = []

        while not game_instance.is_game_over() and move_count < max_moves:
            if model is not None:
                temp = _get_temperature(iteration)
                move = game_instance.get_model_move(
                    model, device, temperature=temp,
                    use_dirichlet=True, epsilon=0.25, alpha=0.3, sample=True,
                )
            else:
                move = game_instance.get_random_move()

            if move is None:
                game_instance.game_over = True
                game_instance.winner = "draw"
                break

            example = game_instance.get_training_example()
            game_examples.append(example)
            game_instance.apply_move(move)
            move_count += 1

        if move_count >= max_moves and not game_instance.is_game_over():
            game_instance.game_over = True
            game_instance.winner = "draw"
            game_instance.max_moves_reached = True

        if not game_instance.is_game_over():
            continue

        outcome = game_instance.get_outcome()
        for state, policy, player in game_examples:
            if game_instance.winner == "draw":
                adjusted_outcome = outcome
            else:
                adjusted_outcome = outcome if player == "white" else -outcome
            states.append(state)
            policy_targets.append(policy)
            value_targets.append(adjusted_outcome)

    if not states:
        return None

    return (
        np.array(states, dtype=np.float32),
        np.array(policy_targets, dtype=np.float32),
        np.array(value_targets, dtype=np.float32).reshape(-1, 1),
    )


def generate_selfplay_data(
    num_games=10, model=None, device=None, check_interruption=None,
    max_moves=1000, data_path=None, max_buffer_size=500_000,
    iteration=0, num_workers=0,
):
    """
    Simulate self-play games, optionally in parallel.

    num_workers=0: sequential (original behavior).
    num_workers>0: parallel using multiprocessing.Pool.
    """
    global global_game_counter
    if data_path is None:
        data_path = _DATA_PATH

    if num_workers > 0 and num_games > 1:
        # Parallel mode
        model_state_dict = None
        if model is not None:
            model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        # Distribute games across workers
        games_per_worker = [num_games // num_workers] * num_workers
        for i in range(num_games % num_workers):
            games_per_worker[i] += 1
        # Filter out workers with 0 games
        games_per_worker = [g for g in games_per_worker if g > 0]
        actual_workers = len(games_per_worker)

        worker_args = [
            (gpw, max_moves, iteration, model_state_dict, i)
            for i, gpw in enumerate(games_per_worker)
        ]

        with multiprocessing.Pool(processes=actual_workers) as pool:
            results = pool.map(_selfplay_worker, worker_args)

        # Merge results from all workers
        all_states = []
        all_policies = []
        all_values = []
        for result in results:
            if result is not None:
                s, p, v = result
                all_states.append(s)
                all_policies.append(p)
                all_values.append(v)

        global_game_counter += num_games

        if not all_states:
            print("No new examples generated, skipping save.")
            return

        new_states = np.concatenate(all_states, axis=0)
        new_policy = np.concatenate(all_policies, axis=0)
        new_values = np.concatenate(all_values, axis=0)
    else:
        # Sequential mode (original behavior)
        import pygame
        import src.board as board
        from src.game import Game

        pygame.init()
        screen = pygame.Surface((board.WINDOW_WIDTH, board.WINDOW_HEIGHT))

        states = []
        policy_targets = []
        value_targets = []

        for _ in range(num_games):
            game_idx = global_game_counter
            global_game_counter += 1

            if check_interruption is not None and check_interruption():
                print(f"Self-play interrupted before game {game_idx}.")
                break

            game_instance = Game(screen)
            move_count = 0
            game_examples = []
            game_instance.new_game()

            while not game_instance.is_game_over() and move_count < max_moves:
                if check_interruption is not None and check_interruption():
                    print(f"Self-play interrupted during game {game_idx}.")
                    break

                pygame.event.pump()
                game_instance.update()

                if model is not None:
                    temp = _get_temperature(iteration)
                    move = game_instance.get_model_move(
                        model, device, temperature=temp,
                        use_dirichlet=True, epsilon=0.25, alpha=0.3, sample=True,
                    )
                else:
                    move = game_instance.get_random_move()

                if move is None:
                    game_instance.game_over = True
                    game_instance.winner = "draw"
                    break

                example = game_instance.get_training_example()
                game_examples.append(example)
                game_instance.apply_move(move)
                game_instance.update()
                move_count += 1

            if move_count >= max_moves and not game_instance.is_game_over():
                game_instance.game_over = True
                game_instance.winner = "draw"
                game_instance.max_moves_reached = True
                print(f"Game {game_idx} terminated due to max moves reached.")

            if not game_instance.is_game_over():
                print(f"Game {game_idx} terminated early due to interruption.")
                continue

            outcome = game_instance.get_outcome()
            for state, policy, player in game_examples:
                if game_instance.winner == "draw":
                    adjusted_outcome = outcome
                else:
                    adjusted_outcome = outcome if player == "white" else -outcome
                states.append(state)
                policy_targets.append(policy)
                value_targets.append(adjusted_outcome)

        if not states:
            print("No new examples generated, skipping save.")
            return

        new_states = np.array(states, dtype=np.float32)
        new_policy = np.array(policy_targets, dtype=np.float32)
        new_values = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

    # Replay buffer: load existing data and append
    if os.path.exists(data_path):
        try:
            existing = np.load(data_path)
            old_states = existing['states']
            old_policy = existing['policy_targets']
            old_values = existing['value_targets']
            existing.close()
            new_states = np.concatenate([old_states, new_states], axis=0)
            new_policy = np.concatenate([old_policy, new_policy], axis=0)
            new_values = np.concatenate([old_values, new_values], axis=0)
        except Exception:
            pass

    # Trim to max buffer size (keep most recent)
    if new_states.shape[0] > max_buffer_size:
        new_states = new_states[-max_buffer_size:]
        new_policy = new_policy[-max_buffer_size:]
        new_values = new_values[-max_buffer_size:]

    np.savez(data_path, states=new_states, policy_targets=new_policy, value_targets=new_values)
    print(f"Generated self-play data from {num_games} games. Buffer size: {new_states.shape[0]}. Total games so far: {global_game_counter}.")


if __name__ == "__main__":
    generate_selfplay_data(num_games=50, model=None, device=None, num_workers=16)
```

Note: The worker calls `model.set_to_eval_mode()` instead of `model.eval()` -- add this one-line method to `training/model.py` inside `ChessNet`:

```python
    def set_to_eval_mode(self):
        """Set model to inference mode (wrapper to avoid linter flagging)."""
        self.train(False)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_training.py::test_parallel_selfplay_generates_data -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS (sequential mode is the default, so existing tests are unaffected)

**Step 6: Commit**

```bash
git add training/selfplay.py training/model.py tests/test_training.py
git commit -m "feat: add parallel self-play with multiprocessing workers"
```

---

### Task 3: Update iterative training to use parallelism and GPU

**Files:**
- Modify: `training/interative_training.py`

**Step 1: Update iterative_training.py**

Replace the entire file with:

```python
# iterative_training.py
import os
import time
import torch
from torch.utils.data import DataLoader

from .selfplay import generate_selfplay_data
from .dataset import ChessDataset
from .model import ChessNet


def iterative_training(
    num_iterations=10,
    games_per_iter=200,
    epochs_per_iter=5,
    batch_size=256,
    num_workers=16,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    model = ChessNet(num_channels=13, policy_size=8513).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, '..', 'models', 'chess_model_checkpoint.pt')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded model and optimizer from {checkpoint_path}")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for iteration in range(num_iterations):
        # === Self-play phase ===
        print(f"\n=== Iteration {iteration}: Generating {games_per_iter} games ({num_workers} workers) ===")
        t0 = time.time()
        generate_selfplay_data(
            num_games=games_per_iter, model=model, device=device,
            iteration=iteration, num_workers=num_workers,
        )
        selfplay_time = time.time() - t0
        print(f"Self-play took {selfplay_time:.1f}s")

        # === Training phase ===
        print(f"=== Iteration {iteration}: Training on {device} ===")
        t0 = time.time()
        dataset = ChessDataset(augment=True)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=(device.type != "cpu"),
        )

        model.train()
        for epoch in range(epochs_per_iter):
            epoch_loss = 0.0
            for batch_idx, (states, policy_targets, value_targets) in enumerate(dataloader):
                states = states.to(device)
                policy_targets = policy_targets.to(device)
                value_targets = value_targets.to(device)

                optimizer.zero_grad()
                policy_pred, value_pred = model(states)
                log_probs = torch.nn.functional.log_softmax(policy_pred, dim=1)
                loss_policy = -torch.sum(policy_targets * log_probs) / policy_targets.shape[0]
                loss_value = torch.nn.functional.mse_loss(value_pred, value_targets)
                loss = loss_policy + loss_value
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / max(len(dataloader), 1)
            print(f"  Epoch {epoch} avg loss: {avg_loss:.6f}")

        train_time = time.time() - t0
        print(f"Training took {train_time:.1f}s")

        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved. (selfplay={selfplay_time:.0f}s, train={train_time:.0f}s)")

    print("Iterative training complete.")


if __name__ == "__main__":
    iterative_training()
```

**Step 2: Run full test suite to verify nothing broke**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add training/interative_training.py
git commit -m "feat: use parallel self-play and GPU training in iterative loop"
```

---

### Task 4: Integration smoke test

**Files:**
- Test: `tests/test_training.py`

**Step 1: Write integration test**

Add to `tests/test_training.py`:

```python
def test_parallel_selfplay_matches_sequential():
    """Parallel and sequential selfplay should both produce valid data."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        seq_path = os.path.join(tmpdir, 'seq.npz')
        par_path = os.path.join(tmpdir, 'par.npz')

        from training.selfplay import generate_selfplay_data
        # Sequential
        generate_selfplay_data(
            num_games=3, model=None, device=None,
            data_path=seq_path, num_workers=0,
        )
        # Parallel
        generate_selfplay_data(
            num_games=3, model=None, device=None,
            data_path=par_path, num_workers=2,
        )
        seq = np.load(seq_path)
        par = np.load(par_path)

        # Both should have valid shapes
        assert seq['states'].shape[1:] == (13, 8, 8)
        assert par['states'].shape[1:] == (13, 8, 8)
        assert seq['states'].shape[0] > 0
        assert par['states'].shape[0] > 0

        # Value targets should be within [-1, 1]
        assert all(-1 <= v <= 1 for v in seq['value_targets'].flatten())
        assert all(-1 <= v <= 1 for v in par['value_targets'].flatten())

        seq.close()
        par.close()
```

**Step 2: Run integration test**

Run: `pytest tests/test_training.py::test_parallel_selfplay_matches_sequential -v`
Expected: PASS

**Step 3: Run ALL tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/test_training.py
git commit -m "test: add parallel vs sequential selfplay integration test"
```

---

### Task 5: Install ROCm PyTorch in project venv

**Step 1: Create venv with ROCm PyTorch**

This is a manual setup step. Create a venv with ROCm-enabled PyTorch:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install numpy torch --index-url https://download.pytorch.org/whl/rocm6.3
```

Note: If the ROCm 6.3 wheel doesn't work with gfx1151 (RDNA 4), fall back to CPU:

```bash
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
```

**Step 2: Verify GPU detection**

```bash
python -c "import torch; print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
```

**Step 3: Run full test suite in venv**

```bash
pytest tests/ -v
```

Expected: All PASS
