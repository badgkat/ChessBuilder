import pytest
import pygame
import os
import sys
import numpy as np

from src.game import Game
from src import board


@pytest.fixture(scope="module")
def pygame_init():
    pygame.init()
    screen = pygame.display.set_mode((1, 1))
    yield
    pygame.display.quit()
    pygame.quit()


@pytest.fixture
def game_instance(pygame_init):
    screen = pygame.display.set_mode((board.WINDOW_WIDTH, board.WINDOW_HEIGHT))
    g = Game(screen)
    return g


def test_selfplay_random_game_completes(game_instance):
    """A game played with random moves terminates."""
    g = game_instance
    g.new_game()
    max_moves = 300
    moves_made = 0
    for i in range(max_moves):
        if g.is_game_over():
            break
        move = g.get_random_move()
        if move is None:
            break
        g.apply_move(move)
        moves_made += 1
    # At least one move was made
    assert moves_made > 0


def test_training_example_shapes(game_instance):
    """Training examples have correct shapes throughout a game."""
    g = game_instance
    g.new_game()
    for _ in range(10):
        if g.is_game_over():
            break
        state, policy, player = g.get_training_example()
        assert state.shape == (13, 8, 8)
        assert policy.shape == (8513,)
        assert player in ('white', 'black')
        assert policy.sum() > 0  # Should have at least one legal action

        move = g.get_random_move()
        if move is None:
            break
        g.apply_move(move)


def test_move_to_index_round_trip(game_instance):
    """move_to_index produces valid indices for all action types."""
    g = game_instance
    g.new_game()

    # Standard move
    idx = g.move_to_index("move", (6, 4), (5, 4), None)
    assert 0 <= idx < 4096

    # Collect gold
    idx = g.move_to_index("collect_gold", (6, 4), None, None)
    assert idx == 4096

    # Purchase
    idx = g.move_to_index("purchase", (7, 4), (6, 3), 'N')
    assert 4097 <= idx < 4097 + 320

    # Transfer gold
    idx = g.move_to_index("transfer_gold", (6, 4), (5, 3), None)
    assert 4097 + 320 <= idx < 8513


def test_replay_buffer_appends():
    """Selfplay should append to existing data, not overwrite."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, 'training_data.npz')
        # Create fake existing data
        existing = {
            'states': np.random.randn(10, 13, 8, 8).astype(np.float32),
            'policy_targets': np.random.randn(10, 8513).astype(np.float32),
            'value_targets': np.random.randn(10, 1).astype(np.float32),
        }
        np.savez(data_path, **existing)

        from training.selfplay import generate_selfplay_data
        generate_selfplay_data(num_games=2, model=None, device=None, data_path=data_path)
        result = np.load(data_path)
        num_states = result['states'].shape[0]
        result.close()  # Close file handle to avoid Windows PermissionError
        # Should have MORE than the original 10 examples
        assert num_states > 10


def test_model_residual_architecture():
    """New model should have residual blocks and correct output shapes."""
    from training.model import ChessNet
    import torch
    model = ChessNet(num_channels=13, policy_size=8513)
    x = torch.randn(2, 13, 8, 8)
    policy, value = model(x)
    assert policy.shape == (2, 8513)
    assert value.shape == (2, 1)
    # Model should have more parameters than old 2-layer version (~50k)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 500_000, f"Model too small: {num_params} params"
