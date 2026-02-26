import random
import os

try:
    import torch
    from training.model import ChessNet
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class AI:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model = ChessNet(num_channels=13, policy_size=8513).to(device)
        self.load_checkpoint(checkpoint_path)
        self.ai_color = None  # Will be set later: 'white' or 'black'

    def load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("AI model loaded from checkpoint.")
        except Exception as e:
            print("No checkpoint loaded; using a new model.", e)

    def set_color(self, color_choice):
        """
        Sets the AI color. The input can be 'white', 'black', or 'random'.
        If 'random', then randomly choose one of the two.
        """
        if color_choice.lower() == 'random':
            self.ai_color = random.choice(['white', 'black'])
        elif color_choice.lower() in ['white', 'black']:
            self.ai_color = color_choice.lower()
        else:
            raise ValueError("Invalid color choice. Please choose 'white', 'black', or 'random'.")
        print(f"AI will play as: {self.ai_color}")

    def get_move(self, game):
        """
        Given a game instance, if it is the AI's turn, return the AI move by calling the game's
        get_model_move method. Otherwise, return None.
        """
        #print("AI.get_move called. Game turn:", game.turn, "AI color:", game.ai_color)
        if game.turn == game.ai_color and not game.game_over:
            move = game.get_model_move(self.model, self.device, temperature=1.0, use_dirichlet=False, sample=False)
            #print("AI.get_move returning move:", move, flush=True)
            return move
        #print("AI.get_move: condition not met (either wrong turn or game over), returning None", flush=True)
        return None
