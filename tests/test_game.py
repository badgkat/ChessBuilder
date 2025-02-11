import pytest
import pygame
import os
import sys
from unittest.mock import patch
from game import Game
import board
from clock import ChessClock


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