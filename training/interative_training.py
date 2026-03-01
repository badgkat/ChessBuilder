# iterative_training.py
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

# Import our components from training.
from .selfplay import generate_selfplay_data
from .dataset import ChessDataset
from .model import ChessNet

def iterative_training(num_iterations=10, games_per_iter=50, epochs_per_iter=5, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(num_channels=13, policy_size=8513).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  # The directory this script is in
    checkpoint_path = os.path.join(script_dir, '..', 'models', 'chess_model_checkpoint.pt')
    # Ensure checkpoint directory exists.
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded model and optimizer from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration}: Generating self-play data ===")
        # Use the current model for self-play.
        generate_selfplay_data(num_games=games_per_iter, model=model, device=device, iteration=iteration)
        
        print(f"=== Iteration {iteration}: Training model ===")
        # Load the newly generated data.
        dataset = ChessDataset(augment=True)  # This loads "training_data.npz"
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train for a few epochs.
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
            avg_loss = epoch_loss / len(dataloader)
            print(f"Iteration {iteration} Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint after each iteration.
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Iteration {iteration} checkpoint saved.")
    
    print("Iterative training complete.")

if __name__ == "__main__":
    iterative_training()
