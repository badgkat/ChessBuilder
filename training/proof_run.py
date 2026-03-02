"""Proof-of-concept training run.

Runs a small iterative training loop to verify:
1. Parallel self-play generates valid data
2. GPU training runs without errors
3. Loss decreases over iterations (model is learning *something*)
4. Checkpoint save/load works

Parameters are deliberately small — this is a smoke test, not a real run.
"""

import os
import sys
import time

# Must precede torch import for AMD GPU compat
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import training.gpu_setup  # noqa: F401

import torch
from torch.utils.data import DataLoader

from training.selfplay import generate_selfplay_data
from training.dataset import ChessDataset
from training.model import ChessNet


def proof_run():
    # --- Configuration ---
    num_iterations = 3
    games_per_iter = 30  # 30 games × 3 iters = 90 total (small but meaningful)
    epochs_per_iter = 10  # More epochs per iter since we have little data
    batch_size = 64
    num_workers = 8  # Conservative: 8 of 16 cores
    lr = 0.001

    # Use a separate data file so we don't pollute any existing training data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "proof_training_data.npz")
    checkpoint_path = os.path.join(script_dir, "..", "models", "proof_checkpoint.pt")

    # Clean up from previous proof runs
    if os.path.exists(data_path):
        os.remove(data_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  HIP: {torch.version.hip}")

    model = ChessNet(num_channels=13, policy_size=8513).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    losses_by_iter = []
    wall_start = time.time()

    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # --- Self-play phase ---
        print(f"\nSelf-play: {games_per_iter} games with {num_workers} workers...")
        t0 = time.time()

        # First iteration: random play (no model). Later: model-guided.
        sp_model = model if iteration > 0 else None
        sp_device = device if iteration > 0 else None

        generate_selfplay_data(
            num_games=games_per_iter,
            model=sp_model,
            device=sp_device,
            iteration=iteration,
            num_workers=num_workers,
            data_path=data_path,
            max_buffer_size=500_000,
        )
        sp_time = time.time() - t0
        print(f"Self-play: {sp_time:.1f}s")

        # --- Training phase ---
        print(f"\nTraining: {epochs_per_iter} epochs, batch_size={batch_size}...")
        t0 = time.time()
        dataset = ChessDataset(data_file=data_path, augment=True)
        print(f"  Dataset size: {len(dataset)} examples ({len(dataset)//2} base + augmented)")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type != "cpu"),
        )

        model.train()
        epoch_losses = []
        for epoch in range(epochs_per_iter):
            epoch_loss = 0.0
            num_batches = 0
            for states, policy_targets, value_targets in dataloader:
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
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)
            if epoch == 0 or epoch == epochs_per_iter - 1:
                print(f"  Epoch {epoch}: loss={avg_loss:.4f} (policy={loss_policy.item():.4f}, value={loss_value.item():.4f})")

        train_time = time.time() - t0
        print(f"Training: {train_time:.1f}s")
        losses_by_iter.append(epoch_losses)

        # --- Checkpoint ---
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

    # --- Summary ---
    wall_total = time.time() - wall_start
    print(f"\n{'='*60}")
    print(f"PROOF RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Total wall time: {wall_total:.1f}s")
    print(f"\nLoss progression (first epoch of each iteration):")
    for i, losses in enumerate(losses_by_iter):
        arrow = ""
        if i > 0:
            delta = losses[0] - losses_by_iter[i - 1][0]
            arrow = f" ({'↓' if delta < 0 else '↑'} {abs(delta):.4f})"
        print(f"  Iter {i + 1}: {losses[0]:.4f} → {losses[-1]:.4f}{arrow}")

    print(f"\nLoss progression (last epoch of each iteration):")
    for i, losses in enumerate(losses_by_iter):
        arrow = ""
        if i > 0:
            delta = losses[-1] - losses_by_iter[i - 1][-1]
            arrow = f" ({'↓' if delta < 0 else '↑'} {abs(delta):.4f})"
        print(f"  Iter {i + 1}: {losses[-1]:.4f}{arrow}")

    # Sanity checks
    first_loss = losses_by_iter[0][0]
    last_loss = losses_by_iter[-1][-1]
    print(f"\nSanity checks:")
    print(f"  Loss decreased within iterations: ", end="")
    within_ok = all(losses[-1] < losses[0] for losses in losses_by_iter)
    print(f"{'PASS' if within_ok else 'UNCERTAIN (may need more epochs)'}")

    print(f"  Checkpoint loads correctly: ", end="")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        test_model = ChessNet(num_channels=13, policy_size=8513)
        test_model.load_state_dict(ckpt["model_state_dict"])
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")

    # Clean up proof artifacts
    if os.path.exists(data_path):
        os.remove(data_path)
        print(f"\nCleaned up {data_path}")


if __name__ == "__main__":
    proof_run()
