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
        model.train(False)

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
    max_moves=50_000, data_path=None, max_buffer_size=500_000,
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
