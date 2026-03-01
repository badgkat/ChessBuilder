# training/selfplay.py
import sys, os, pygame
import numpy as np
import src.board as board
from src.game import Game

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

def generate_selfplay_data(num_games=10, model=None, device=None, check_interruption=None, max_moves=1000, data_path=None, max_buffer_size=500_000, iteration=0):
    """
    Simulate self-play games.
    If a model is provided, it selects moves; otherwise, moves are random.
    Training examples are saved in "training_data.npz".
    If max_moves is exceeded in a game, it is forced to end (draw).
    The optional check_interruption callback should return True when an interruption is requested.
    """
    global global_game_counter
    if data_path is None:
        data_path = _DATA_PATH
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
                temp = _get_temperature(iteration)
                move = game_instance.get_model_move(model, device, temperature=temp, use_dirichlet=True, epsilon=0.25, alpha=0.3, sample=True)
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
            pass  # If file is corrupt, start fresh

    # Trim to max buffer size (keep most recent)
    if new_states.shape[0] > max_buffer_size:
        new_states = new_states[-max_buffer_size:]
        new_policy = new_policy[-max_buffer_size:]
        new_values = new_values[-max_buffer_size:]

    np.savez(data_path, states=new_states, policy_targets=new_policy, value_targets=new_values)
    print(f"Generated self-play data from {num_games} games. Buffer size: {new_states.shape[0]}. Total games so far: {global_game_counter}.")

if __name__ == "__main__":
    # For example, running 50 games in one batch:
    generate_selfplay_data(num_games=50, model=None, device=None, check_interruption=lambda: False)
