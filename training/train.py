# training/train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import ChessNet
from .dataset import ChessDataset


def train(model, optimizer, dataloader, device, start_epoch, num_epochs, checkpoint_path):
    model.train()
    loss_fn_value = torch.nn.MSELoss()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        for batch_idx, (states, policy_targets, value_targets) in enumerate(dataloader):
            states = states.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            policy_pred, value_pred = model(states)
            
            # Cross-entropy loss for policy (soft targets)
            log_probs = torch.nn.functional.log_softmax(policy_pred, dim=1)
            loss_policy = -torch.sum(policy_targets * log_probs) / policy_targets.shape[0]
            loss_value = loss_fn_value(value_pred, value_targets)
            loss = loss_policy + loss_value
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}: Loss {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint at the end of each epoch
        torch.save({
            'iteration': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")
        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, '..', 'models', 'chess_model_checkpoint.pt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    start_epoch = 0
    num_epochs = 10  # Set this to the total number of epochs you want to train in this run
    
    # Resume training if a checkpoint exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('iteration', checkpoint.get('epoch', 0))
        print(f"Resuming training from epoch {start_epoch}")
    
    dataset = ChessDataset(augment=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    train(model, optimizer, dataloader, device, start_epoch, num_epochs, checkpoint_path)
    
if __name__ == "__main__":
    main()
