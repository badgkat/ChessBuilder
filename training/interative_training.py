# iterative_training.py
import training.gpu_setup  # noqa: F401 — must precede torch import
import os
import time
import torch
from torch.utils.data import DataLoader

from .selfplay import generate_selfplay_data
from .dataset import ChessDataset
from .model import ChessNet


def _stockfish_depth(iteration):
    """Curriculum schedule: increase Stockfish depth as training progresses."""
    if iteration < 20:
        return 1
    elif iteration < 50:
        return 2
    elif iteration < 100:
        return 3
    else:
        return 5


def iterative_training(
    num_iterations=10,
    games_per_iter=200,
    epochs_per_iter=5,
    batch_size=256,
    num_workers=16,
    stockfish_ratio=0.5,
    stockfish_depth_schedule=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    model = ChessNet(num_channels=13, policy_size=8513).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, '..', 'models', 'chess_model_checkpoint.pt')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    start_iteration = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = checkpoint.get("iteration", -1) + 1
        print(f"Resuming from iteration {start_iteration} (loaded {checkpoint_path})")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if stockfish_depth_schedule is None:
        stockfish_depth_schedule = _stockfish_depth

    end_iteration = start_iteration + num_iterations
    for iteration in range(start_iteration, end_iteration):
        # === Self-play phase ===
        sf_depth = stockfish_depth_schedule(iteration)
        sf_info = f", stockfish={stockfish_ratio:.0%} depth={sf_depth}" if stockfish_ratio > 0 else ""
        print(f"\n=== Iteration {iteration}: Generating {games_per_iter} games ({num_workers} workers{sf_info}) ===")
        t0 = time.time()
        generate_selfplay_data(
            num_games=games_per_iter, model=model, device=device,
            iteration=iteration, num_workers=num_workers,
            stockfish_ratio=stockfish_ratio, stockfish_depth=sf_depth,
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

    print(f"Iterative training complete. Ran iterations {start_iteration}-{end_iteration - 1}.")


if __name__ == "__main__":
    iterative_training()
