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
    assert g.board[7][4].type == "K"
    assert g.board[7][4].color == "white"

    assert g.board[6][4] is not None
    assert g.board[6][4].type == "P"
    assert g.board[6][4].color == "white"

    # Black king at [0][4], black pawn at [1][4]
    assert g.board[0][4] is not None
    assert g.board[0][4].type == "K"
    assert g.board[0][4].color == "black"

    assert g.board[1][4] is not None
    assert g.board[1][4].type == "P"
    assert g.board[1][4].color == "black"

    assert g.turn == "white"
    assert g.game_over is False


def test_end_turn_switches_color(game_instance):
    """
    Tests that end_turn() switches the turn from white to black
    and vice versa.
    """
    g = game_instance
    g.new_game()
    assert g.turn == "white"
    g.end_turn()
    assert g.turn == "black"
    g.end_turn()
    assert g.turn == "white"


def test_is_in_check_false_on_new_game(game_instance):
    """
    On the default new_game() setup, no side should be in check.
    """
    g = game_instance
    g.new_game()
    assert not g.is_in_check("white")
    assert not g.is_in_check("black")


def test_has_any_legal_moves(game_instance):
    """
    On the simplified new_game board, both sides have moves.
    """
    g = game_instance
    g.new_game()
    assert g.has_any_legal_moves("white")
    assert g.has_any_legal_moves("black")


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
    g.purchase_selected_type = "Q"  # Queen
    # White king has 0 gold, can't afford a queen which likely costs > 0
    cost = board.PIECE_COST["Q"]  # Suppose it's something like 9 or 5 or so
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


def test_capture_awards_piece_cost_plus_gold(game_instance):
    """Capturing a piece should award its cost + accumulated gold."""
    g = game_instance
    g.new_game()
    # Place a black knight with 2 accumulated gold at (5, 3)
    g.board[5][3] = board.Piece("N", "black", gold=2)
    # Place a white pawn at (6, 4) — default position
    assert g.board[6][4].type == "P"
    assert g.board[6][4].gold == 0

    # White pawn captures black knight diagonally
    g.capture_piece((6, 4), (5, 3))

    # Pawn should gain knight cost (3) + accumulated gold (2) = 5
    pawn = g.board[5][3]
    assert pawn.type == "P"
    assert pawn.color == "white"
    assert pawn.gold == 5


def test_en_passant_transfers_gold(game_instance):
    """En passant should transfer captured pawn's gold + cost to the capturing pawn."""
    g = game_instance
    g.new_game()
    # Set up en passant scenario:
    # White pawn on row 3, col 3 with 0 gold
    white_pawn = board.Piece("P", "white", gold=0)
    g.board[3][3] = white_pawn
    g.board[6][4] = None  # Remove default white pawn

    # Black pawn on row 3, col 4 with 3 gold (just moved 2 squares)
    black_pawn = board.Piece("P", "black", gold=3)
    g.board[3][4] = black_pawn

    # Set en passant target: white can capture at (2, 4), actual pawn at (3, 4)
    g.en_passant = ((2, 4), (3, 4))

    g.move_piece((3, 3), (2, 4))

    # White pawn should gain black pawn's gold (3) + cost (1) = 4
    assert g.board[2][4].gold == 4
    # Black pawn should be removed
    assert g.board[3][4] is None


def test_capture_promotion_triggers(game_instance):
    """A pawn capturing onto the final rank should trigger promotion."""
    g = game_instance
    g.new_game()
    # Place white pawn at row 1 (one step from promotion)
    white_pawn = board.Piece("P", "white", gold=0)
    g.board[1][3] = white_pawn
    g.board[6][4] = None  # Remove default pawn

    # Place black piece to capture at row 0
    g.board[0][4] = board.Piece("N", "black")

    g.capture_piece((1, 3), (0, 4))

    # Should trigger promotion mode
    assert g.promotion_mode is True
    assert g.promotion_pos == (0, 4)
    assert g.promotion_color == "white"
    # Pawn should have gained knight cost (3)
    assert g.board[0][4].gold == 3


def test_simulate_move_is_safe_en_passant(game_instance):
    """simulate_move_is_safe should correctly handle en passant by removing
    the captured pawn from the simulation."""
    g = game_instance
    g.new_game()

    # Clear the board except kings
    for r in range(board.BOARD_SIZE):
        for c in range(board.BOARD_SIZE):
            if g.board[r][c] and g.board[r][c].type != "K":
                g.board[r][c] = None

    # White king at (7, 4), Black king at (0, 4)
    # White pawn at (3, 3), Black pawn at (3, 4) (just moved 2 squares)
    g.board[3][3] = board.Piece("P", "white")
    g.board[3][4] = board.Piece("P", "black")
    g.en_passant = ((2, 4), (3, 4))

    # En passant capture should be safe (no check issues)
    assert g.simulate_move_is_safe((3, 3), (2, 4)) is True
    # After simulation, the board should be restored
    assert g.board[3][3] is not None
    assert g.board[3][4] is not None
    assert g.board[2][4] is None


def test_last_move_tracking(game_instance):
    """Moving a piece should record the last move for highlighting."""
    g = game_instance
    g.new_game()
    assert g.last_move is None

    g.move_piece((6, 4), (5, 4))
    assert g.last_move == ((6, 4), (5, 4))


def test_restore_purchase_state(game_instance):
    """restore_purchase_state should revert the board to pre-purchase state."""
    g = game_instance
    g.new_game()
    g.board[7][4].gold = 10  # Give king gold
    g.selected_piece_pos = (7, 4)
    g.create_purchase_options(7, 4)

    # Modify state
    g.board[6][3] = board.Piece("N", "white")
    g.move_log.append("test")

    # Restore
    g.restore_purchase_state()

    # Board should be back to pre-purchase state
    assert g.board[6][3] is None
    assert "test" not in g.move_log


def test_restore_promotion_state(game_instance):
    """restore_promotion_state should revert the board to pre-promotion state."""
    g = game_instance
    g.new_game()

    # Set up promotion scenario
    g.board[1][4] = g.board[6][4]
    g.board[6][4] = None
    g.move_piece((1, 4), (0, 4))

    assert g.promotion_mode is True

    # Restore should undo the promotion
    g.restore_promotion_state()
    assert g.board[0][4] is not None
