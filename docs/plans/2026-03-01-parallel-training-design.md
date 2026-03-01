# Parallel Self-Play + GPU Training

**Date:** 2026-03-01
**Goal:** Utilize available server resources (32 threads, 128GB RAM, Radeon 8060S GPU via ROCm) to run training alongside an existing embedding model job.

---

## Context

The server is an AMD Ryzen AI MAX+ 395 (16C/32T) with 128GB unified memory and a Radeon 8060S iGPU (RDNA 4, 20 CUs, 133GB visible via shared memory). An embedding model training job runs at nice 5, using ~1 CPU core, 4 workers, and 61% GPU. Plenty of headroom exists for ChessBuilder training.

Current self-play is single-threaded and depends on pygame for headless game simulation — both are unnecessary bottlenecks.

## Changes

### 1. Headless Game Mode (`src/game.py`)

**Problem:** `Game.__init__` requires a pygame screen surface. Self-play creates a `pygame.Surface` and calls `pygame.event.pump()` every move. Pygame is not fork-safe, blocking multiprocessing.

**Solution:** Add `headless=True` parameter to `Game.__init__`:
- Skip all surface/font/image loading when headless
- Make `update()` and `draw()` no-ops
- No pygame dependency at all in headless mode

This is a prerequisite for multiprocessing — without it, forked workers crash on pygame state.

### 2. Parallel Self-Play Workers (`training/selfplay.py`)

**Problem:** Self-play generates games sequentially. A single game at 1000 max moves takes 5-30s of CPU time. With 50 games per iteration, that's 4-25 minutes of wall time using one core while 30+ threads sit idle.

**Solution:** Use `multiprocessing.Pool` with 16 workers (configurable via `num_workers` parameter):

```
Main Process                    Worker Pool (16 workers)
    |                               |
    |-- dispatch games -----------> | worker_0: play ceil(N/16) games
    |                               | worker_1: play ceil(N/16) games
    |                               | ...
    |<-- collect numpy arrays ----  | worker_15: play ceil(N/16) games
    |                               |
    |-- merge + append to buffer    |
```

- Each worker creates its own headless `Game` instance (no shared state)
- Workers return `(states, policies, values)` numpy arrays
- For random self-play (no model): workers are fully independent
- For model self-play: pass `state_dict` bytes to workers, each loads its own CPU copy
- Workers run at nice 10 (lower priority than embedding job at nice 5)
- Fall back to sequential mode with `num_workers=0`

### 3. GPU Training Phase

Already works via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. With ROCm PyTorch installed in the venv, the training loop will use the Radeon 8060S automatically. The model is ~1.2M params — GPU memory usage will be <100MB.

Increase default batch size from 32 to 256 to better utilize GPU throughput. Add `num_workers=4` to the DataLoader for faster data loading.

### 4. Updated Iterative Training Loop

- Pass `num_workers=16` to `generate_selfplay_data()`
- Increase default `games_per_iter` from 50 to 200 (parallelism makes this feasible)
- DataLoader gets `batch_size=256`, `num_workers=4`
- Log wall-clock time per phase (self-play vs training) for monitoring

## What Does NOT Change

- Game logic, legal action generation, reward values
- Model architecture (4 residual blocks, 128 hidden channels)
- Replay buffer mechanics (append + trim to 500k)
- Temperature decay schedule
- Data augmentation (horizontal flip)

## Files Affected

- `src/game.py` — Add `headless` parameter to `Game.__init__`, guard pygame calls
- `training/selfplay.py` — Add `_selfplay_worker()`, refactor `generate_selfplay_data()` to use `multiprocessing.Pool`
- `training/interative_training.py` — Pass `num_workers`, increase batch size, add timing

## Resource Budget

| Resource | Embedding Job | ChessBuilder | Remaining |
|----------|--------------|-------------|-----------|
| CPU threads | ~5 (1 main + 4 workers) | 16 workers + 1 main | 10 idle |
| GPU compute | ~61% | <5% (training phase only) | ~34% |
| GPU memory | ~40GB | <1GB | ~92GB |
| System RAM | ~11GB | ~20GB (16 workers × ~1GB) | ~97GB |

## Success Criteria

1. Self-play phase completes 200 games in under 5 minutes (vs current ~25 min sequential estimate).
2. Full iterative training loop runs without impacting embedding training throughput.
3. All existing tests pass unchanged.
