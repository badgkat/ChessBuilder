# training/selfplay.py
import sys, os, pygame
import numpy as np
import src.board as board
from src.game import Game

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_SCRIPT_DIR, '..', 'training_data.npz')

# Global counter for games played across multiple calls.
global_game_counter = 0

def generate_selfplay_data(num_games=10, model=None, device=None, check_interruption=None, max_moves=200):
    """
    Simulate self-play games.
    If a model is provided, it selects moves; otherwise, moves are random.
    Training examples are saved in "training_data.npz".
    If max_moves is exceeded in a game, it is forced to end (draw).
    The optional check_interruption callback should return True when an interruption is requested.
    """
    global global_game_counter
    pygame.init()
    screen = pygame.Surface((board.WINDOW_WIDTH, board.WINDOW_HEIGHT))

    states = []
    policy_targets = []
    value_targets = []
    
    for _ in range(num_games):
        game_idx = global_game_counter  # use the global counter
        global_game_counter += 1         # increment for next game

        if check_interruption is not None and check_interruption():
            print(f"Self-play interrupted before game {game_idx}.")
            break

        game_instance = Game(screen)
        move_count = 0
        game_examples = []
        game_instance.new_game()  # Reset game state

        while not game_instance.is_game_over() and move_count < max_moves:
            if check_interruption is not None and check_interruption():
                print(f"Self-play interrupted during game {game_idx}.")
                break

            pygame.event.pump()
            game_instance.update()

            if model is not None:
                move = game_instance.get_model_move(model, device, temperature=3, use_dirichlet=True, epsilon=0.25, alpha=0.3, sample=True)
            else:
                move = game_instance.get_random_move()
            
            if move is None:
                # No legal moves — game should end
                game_instance.game_over = True
                game_instance.winner = "draw"
                break

            # Record a training example before applying the move.
            example = game_instance.get_training_example()
            game_examples.append(example)

            game_instance.apply_move(move)
            game_instance.update()
            move_count += 1

        # Force termination if max_moves exceeded
        if move_count >= max_moves and not game_instance.is_game_over():
            game_instance.game_over = True
            game_instance.winner = "draw"
            game_instance.max_moves_reached = True  # Mark forced termination
            print(f"Game {game_idx} terminated due to max moves reached.")

        if not game_instance.is_game_over():
            print(f"Game {game_idx} terminated early due to interruption.")
            continue

        outcome = game_instance.get_outcome()
        for state, policy, player in game_examples:
            if game_instance.winner == "draw":
                # For draws (normal or forced) both players get the same outcome value.
                adjusted_outcome = outcome
            else:
                # For wins/losses, give the outcome from the white perspective,
                # then flip it for non-white moves.
                adjusted_outcome = outcome if player == "white" else -outcome
            states.append(state)
            policy_targets.append(policy)
            value_targets.append(adjusted_outcome)


    states = np.array(states, dtype=np.float32)
    policy_targets = np.array(policy_targets, dtype=np.float32)
    value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)
    
    np.savez(_DATA_PATH, states=states, policy_targets=policy_targets, value_targets=value_targets)
    print(f"Generated self-play data from {num_games} games. Total games so far: {global_game_counter}.")

if __name__ == "__main__":
    # For example, running 50 games in one batch:
    generate_selfplay_data(num_games=50, model=None, device=None, check_interruption=lambda: False)
