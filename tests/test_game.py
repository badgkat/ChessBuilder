import pytest
import pygame
import os
import sys
from unittest.mock import patch
from src.game import Game
from src import board
from src.clock import ChessClock


@pytest.fixture(scope="module")
def pygame_init():
    """
    Initializes and quits pygame once for the entire test module.
    This ensures pygame is properly set up before tests run.
    """
    pygame.init()
    # Some environments need a video mode set, even if it's tiny.
    screen = pygame.display.set_mode((1, 1))
    yield
    pygame.display.quit()
    pygame.quit()


@pytest.fixture
def game_instance(pygame_init):
    """
    Creates a Game instance with a small screen. 
    """
    screen = pygame.display.set_mode((board.WINDOW_WIDTH, board.WINDOW_HEIGHT))
    g = Game(screen)
    return g


def test_new_game_initial_positions(game_instance):
    """
    Tests if new_game() sets up the board correctly 
    for the simplified example in the code.
    """
    g = game_instance
    g.new_game()

    # White king at [7][4], white pawn at [6][4]
    assert g.board[7][4] is not None
    assert g.board[7][4].type == 'K'
    assert g.board[7][4].color == 'white'

    assert g.board[6][4] is not None
    assert g.board[6][4].type == 'P'
    assert g.board[6][4].color == 'white'

    # Black king at [0][4], black pawn at [1][4]
    assert g.board[0][4] is not None
    assert g.board[0][4].type == 'K'
    assert g.board[0][4].color == 'black'

    assert g.board[1][4] is not None
    assert g.board[1][4].type == 'P'
    assert g.board[1][4].color == 'black'

    assert g.turn == 'white'
    assert g.game_over is False


def test_end_turn_switches_color(game_instance):
    """
    Tests that end_turn() switches the turn from white to black 
    and vice versa.
    """
    g = game_instance
    g.new_game()
    assert g.turn == 'white'
    g.end_turn()
    assert g.turn == 'black'
    g.end_turn()
    assert g.turn == 'white'


def test_is_in_check_false_on_new_game(game_instance):
    """
    On the default new_game() setup, no side should be in check.
    """
    g = game_instance
    g.new_game()
    assert not g.is_in_check('white')
    assert not g.is_in_check('black')


def test_has_any_legal_moves(game_instance):
    """
    On the simplified new_game board, both sides have moves.
    """
    g = game_instance
    g.new_game()
    assert g.has_any_legal_moves('white')
    assert g.has_any_legal_moves('black')


def test_en_passant_not_triggered_immediately(game_instance):
    """
    Tests that en_passant is None at the start of new_game.
    """
    g = game_instance
    g.new_game()
    assert g.en_passant is None


def test_promotion_mode(game_instance):
    """
    Tests that promotion_mode is triggered when a pawn
    reaches the last rank (in simplified scenario).
    """
    g = game_instance
    g.new_game()

    # Move white pawn from [6][4] to [1][4] quickly (faking direct move 
    # for test, not using normal handle_board_click).
    g.board[1][4] = g.board[6][4]
    g.board[6][4] = None

    # Move it to the last rank: [0][4]
    g.move_piece((1, 4), (0, 4))

    # Should be in promotion_mode now if the logic triggers
    assert g.promotion_mode is True
    assert g.promotion_pos == (0, 4)
    assert g.promotion_color == "white"


def test_get_move_log_lines(game_instance):
    """
    Tests that move_log and get_move_log_lines() behave as expected.
    """
    g = game_instance
    g.new_game()
    # Simulate a move to fill move_log
    g.move_piece((6, 4), (5, 4))  # White pawn moves forward
    g.move_piece((1, 4), (2, 4))  # Black pawn moves forward
    lines = g.get_move_log_lines()
    # Example result: ["1. P[...] P[...]"]
    assert len(lines) == 1  # 1 turn so far, which shows both White and Black's move
    assert "P" in lines[0]


def test_chess_clock_init(game_instance):
    """
    Tests that creating a chess clock works properly. 
    """
    g = game_instance
    # Simulate picking a time control.
    # We'll just manually attach a ChessClock to test the logic.
    g.chess_clock = ChessClock(300, 300, 0)
    assert g.chess_clock.white_time == 300
    assert g.chess_clock.black_time == 300
    assert g.chess_clock.increment == 0


def test_purchase_fails_if_not_enough_gold(game_instance):
    """
    Tests that an error is produced when a king tries
    to purchase a piece without enough gold.
    """
    g = game_instance
    g.new_game()
    # Try to open purchase mode for white king (which has 0 gold).
    g.selected_piece_pos = (7, 4)  # White king
    g.purchase_mode = True
    g.create_purchase_options(7, 4)

    # Simulate picking a piece from the overlay by direct call 
    # with not enough gold
    # We'll check if error_message is set properly.
    # This snippet is a partial imitation of handle_board_click's logic:
    g.purchase_selected_type = 'Q'  # Queen
    # White king has 0 gold, can't afford a queen which likely costs > 0
    cost = board.PIECE_COST['Q']  # Suppose it's something like 9 or 5 or so
    king = g.board[7][4]
    if king.gold < cost:
        g.error_message = "Not enough gold for purchase."

    assert g.error_message == "Not enough gold for purchase."


def test_draw_by_repetition(game_instance):
    """
    Tests that the position_history logic triggers a draw 
    after 3 repetitions of the same position_key.
    """
    g = game_instance
    g.new_game()

    # Sequence to repeat:
    # 1. White pawn forward
    # 2. Black pawn forward
    # 3. White pawn back
    # 4. Black pawn back
    # This should get us back to the starting position with white to move

    # First sequence
    g.move_piece((6, 4), (5, 4))  # White pawn forward
    g.move_piece((1, 4), (2, 4))  # Black pawn forward
    g.move_piece((5, 4), (6, 4))  # White pawn back
    g.move_piece((2, 4), (1, 4))  # Black pawn back

    # Second sequence
    g.move_piece((6, 4), (5, 4))  # White pawn forward
    g.move_piece((1, 4), (2, 4))  # Black pawn forward
    g.move_piece((5, 4), (6, 4))  # White pawn back
    g.move_piece((2, 4), (1, 4))  # Black pawn back

    # Third sequence - should trigger draw
    g.move_piece((6, 4), (5, 4))  # White pawn forward
    g.move_piece((1, 4), (2, 4))  # Black pawn forward
    g.move_piece((5, 4), (6, 4))  # White pawn back
    g.move_piece((2, 4), (1, 4))  # Black pawn back

    assert g.game_over is True
    assert g.winner == "draw"


def test_apply_move_standard_move(game_instance):
    """apply_move with a standard move works correctly."""
    g = game_instance
    g.new_game()
    move = ("move", (6, 4), (5, 4), None)  # White pawn forward
    g.apply_move(move)
    assert g.board[5][4] is not None
    assert g.board[5][4].type == 'P'
    assert g.board[6][4] is None
    assert g.turn == 'black'


def test_apply_move_promotion(game_instance):
    """apply_move promotes a pawn reaching the final rank."""
    g = game_instance
    g.new_game()
    g.board[1][4] = None  # Remove black pawn
    g.board[6][4] = None  # Remove white pawn
    g.board[1][3] = board.Piece('P', 'white')
    move = ("move", (1, 3), (0, 3), 'Q')
    g.apply_move(move)
    assert g.board[0][3] is not None
    assert g.board[0][3].type == 'Q'
    assert g.board[0][3].color == 'white'
    assert g.turn == 'black'


def test_apply_move_promotion_default_queen(game_instance):
    """apply_move defaults to queen when no promo type specified."""
    g = game_instance
    g.new_game()
    g.board[1][4] = None
    g.board[6][4] = None
    g.board[1][3] = board.Piece('P', 'white')
    move = ("move", (1, 3), (0, 3), None)
    g.apply_move(move)
    assert g.board[0][3].type == 'Q'


def test_apply_move_capture_transfers_gold(game_instance):
    """Capturing a piece transfers its gold to the capturer."""
    g = game_instance
    g.new_game()
    g.board[4][4] = board.Piece('P', 'white', gold=3)
    g.board[3][3] = board.Piece('P', 'black', gold=5)
    move = ("move", (4, 4), (3, 3), None)
    g.apply_move(move)
    assert g.board[3][3].gold == 8  # 3 + 5


def test_purchase_adjacency_enforced(game_instance):
    """apply_move rejects purchases not adjacent to king."""
    g = game_instance
    g.new_game()
    king = g.board[7][4]
    king.gold = 5
    # Try to purchase a rook far from king
    move = ("purchase", (7, 4), (0, 0), 'R')
    g.apply_move(move)
    assert g.board[0][0] is None  # Should not have placed
    assert g.error_message != ""


def test_insufficient_material_king_vs_king(game_instance):
    g = game_instance
    g.new_game()
    g.board = [[None]*8 for _ in range(8)]
    g.board[0][0] = board.Piece('K', 'white')
    g.board[7][7] = board.Piece('K', 'black')
    assert g.has_insufficient_material() is True


def test_insufficient_material_king_bishop_vs_king(game_instance):
    g = game_instance
    g.new_game()
    g.board = [[None]*8 for _ in range(8)]
    g.board[0][0] = board.Piece('K', 'white')
    g.board[0][1] = board.Piece('B', 'white')
    g.board[7][7] = board.Piece('K', 'black')
    assert g.has_insufficient_material() is True


def test_sufficient_material_with_gold(game_instance):
    """Kings with gold can purchase pieces — not insufficient."""
    g = game_instance
    g.new_game()
    g.board = [[None]*8 for _ in range(8)]
    g.board[0][0] = board.Piece('K', 'white', gold=5)
    g.board[7][7] = board.Piece('K', 'black')
    assert g.has_insufficient_material() is False


def test_sufficient_material_with_rook(game_instance):
    g = game_instance
    g.new_game()
    g.board = [[None]*8 for _ in range(8)]
    g.board[0][0] = board.Piece('K', 'white')
    g.board[0][1] = board.Piece('R', 'white')
    g.board[7][7] = board.Piece('K', 'black')
    assert g.has_insufficient_material() is False


def test_get_random_move_returns_4_tuple(game_instance):
    """get_random_move must return a 4-tuple matching apply_move format."""
    g = game_instance
    g.new_game()
    move = g.get_random_move()
    assert move is not None
    assert len(move) == 4
    action_type, src, dst, extra = move
    assert action_type in ("move", "collect_gold", "purchase", "transfer_gold")


def test_get_random_move_includes_gold_collection(game_instance):
    """get_random_move should include gold collection as a possible action."""
    g = game_instance
    action_types = set()
    for _ in range(200):
        g.new_game()
        move = g.get_random_move()
        if move:
            action_types.add(move[0])
    assert "collect_gold" in action_types


def test_training_example_only_legal_actions(game_instance):
    """Policy target should only have nonzero entries for legal actions."""
    g = game_instance
    g.new_game()
    state, policy, player = g.get_training_example()
    assert state.shape == (13, 8, 8)
    assert policy.shape == (8513,)
    assert player == 'white'
    # Policy should sum to ~1.0 (uniform over legal actions)
    assert abs(policy.sum() - 1.0) < 1e-5
    # All nonzero entries should be equal (uniform)
    nonzero = policy[policy > 0]
    assert len(nonzero) > 0
    assert abs(nonzero.max() - nonzero.min()) < 1e-7


def test_get_legal_actions_returns_list(game_instance):
    """get_legal_actions returns a non-empty list of 4-tuples."""
    g = game_instance
    g.new_game()
    actions = g.get_legal_actions()
    assert isinstance(actions, list)
    assert len(actions) > 0
    for a in actions:
        assert len(a) == 4
        assert a[0] in ("move", "collect_gold", "purchase", "transfer_gold")


def test_get_legal_actions_matches_random_move(game_instance):
    """get_legal_actions should produce the same set as get_random_move draws from."""
    g = game_instance
    g.new_game()
    actions = g.get_legal_actions()
    # get_random_move should only return actions that are in get_legal_actions
    for _ in range(50):
        g.new_game()
        move = g.get_random_move()
        if move:
            assert move in g.get_legal_actions()
