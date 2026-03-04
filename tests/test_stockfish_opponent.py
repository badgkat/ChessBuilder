import pytest
import shutil

from src.game import Game
from training.stockfish_opponent import (
    board_to_fen,
    uci_to_chessbuilder_move,
    StockfishOpponent,
)


@pytest.fixture
def headless_game():
    g = Game(screen=None, headless=True)
    g.new_game()
    return g


def test_board_to_fen_initial(headless_game):
    """FEN for ChessBuilder initial position: K+P each side."""
    fen = board_to_fen(headless_game)
    parts = fen.split()
    assert len(parts) == 6
    # Piece placement: black king on e8, black pawn on e7, white pawn on e2, white king on e1
    assert parts[0] == "4k3/4p3/8/8/8/8/4P3/4K3"
    assert parts[1] == "w"  # white to move
    assert parts[2] == "-"  # no castling
    assert parts[3] == "-"  # no en passant
    assert parts[4] == "0"  # halfmove clock
    assert parts[5] == "1"  # fullmove number


def test_board_to_fen_after_move(headless_game):
    """FEN updates correctly after a move."""
    g = headless_game
    # Move white pawn e2->e4 (row 6,col 4 -> row 4,col 4)
    g.apply_move(("move", (6, 4), (4, 4), None))
    fen = board_to_fen(g)
    parts = fen.split()
    assert parts[0] == "4k3/4p3/8/8/4P3/8/8/4K3"
    assert parts[1] == "b"  # black to move
    assert parts[3] == "e3"  # en passant target


def test_uci_to_chessbuilder_move_basic():
    """Standard move conversion."""
    move = uci_to_chessbuilder_move("e2e4", None)
    assert move == ("move", (6, 4), (4, 4), None)


def test_uci_to_chessbuilder_move_promotion():
    """Promotion move includes piece type."""
    move = uci_to_chessbuilder_move("e7e8q", None)
    assert move == ("move", (1, 4), (0, 4), "Q")

    move_n = uci_to_chessbuilder_move("a2a1n", None)
    assert move_n == ("move", (6, 0), (7, 0), "N")


def test_uci_to_chessbuilder_move_corners():
    """Edge squares convert correctly."""
    move = uci_to_chessbuilder_move("a1h8", None)
    assert move == ("move", (7, 0), (0, 7), None)

    move2 = uci_to_chessbuilder_move("h1a8", None)
    assert move2 == ("move", (7, 7), (0, 0), None)


@pytest.mark.skipif(
    shutil.which("stockfish") is None,
    reason="Stockfish binary not installed",
)
class TestStockfishOpponent:
    def test_get_move_returns_legal(self, headless_game):
        """Stockfish returns a move that is in the legal actions list."""
        sf = StockfishOpponent(depth=1)
        try:
            move = sf.get_move(headless_game)
            assert move is not None
            legal = headless_game.get_legal_actions()
            assert move in legal
        finally:
            sf.close()

    def test_get_move_is_standard_move(self, headless_game):
        """Stockfish should return a standard 'move' action, not gold actions."""
        sf = StockfishOpponent(depth=1)
        try:
            move = sf.get_move(headless_game)
            assert move[0] == "move"
        finally:
            sf.close()

    def test_multiple_moves(self, headless_game):
        """Stockfish can play several consecutive moves."""
        sf = StockfishOpponent(depth=1)
        try:
            g = headless_game
            for _ in range(6):
                if g.is_game_over():
                    break
                move = sf.get_move(g)
                if move is None:
                    break
                assert move in g.get_legal_actions()
                g.apply_move(move)
        finally:
            sf.close()
