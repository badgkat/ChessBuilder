"""Stockfish opponent for training bootstrap.

Wraps the Stockfish engine via python-chess to provide a competent opponent
that only plays standard chess moves (blind to gold/building mechanics).
Forces the model to learn basic survival before discovering building strategies.
"""

import random
import shutil

import chess
import chess.engine


def board_to_fen(game):
    """Convert ChessBuilder board state to a FEN string.

    Ignores gold amounts. Castling is always "-" (ChessBuilder has no castling).
    """
    rows = []
    for r in range(8):
        empty = 0
        row_str = ""
        for c in range(8):
            piece = game.board[r][c]
            if piece is None:
                empty += 1
            else:
                if empty > 0:
                    row_str += str(empty)
                    empty = 0
                letter = piece.type  # K, Q, R, B, N, P
                if piece.color == "black":
                    letter = letter.lower()
                row_str += letter
        if empty > 0:
            row_str += str(empty)
        rows.append(row_str)

    piece_placement = "/".join(rows)
    active = "w" if game.turn == "white" else "b"
    castling = "-"

    # En passant: game.en_passant is ((target_row, target_col), (pawn_row, pawn_col)) or None
    if game.en_passant is not None:
        ep_row, ep_col = game.en_passant[0]
        ep_file = chr(ep_col + ord("a"))
        ep_rank = str(8 - ep_row)
        en_passant = ep_file + ep_rank
    else:
        en_passant = "-"

    halfmove = str(game.halfmove_clock)
    # ChessBuilder doesn't track fullmove number; derive from move log
    fullmove = str(len(game.move_log) // 2 + 1)

    return f"{piece_placement} {active} {castling} {en_passant} {halfmove} {fullmove}"


def uci_to_chessbuilder_move(uci_str, game):
    """Convert a UCI move string (e.g. 'e2e4') to a ChessBuilder move tuple.

    Returns ("move", (src_row, src_col), (dst_row, dst_col), promo_type_or_None).
    """
    src_col = ord(uci_str[0]) - ord("a")
    src_row = 8 - int(uci_str[1])
    dst_col = ord(uci_str[2]) - ord("a")
    dst_row = 8 - int(uci_str[3])

    promo = None
    if len(uci_str) == 5:
        promo_map = {"q": "Q", "r": "R", "b": "B", "n": "N"}
        promo = promo_map.get(uci_str[4])

    return ("move", (src_row, src_col), (dst_row, dst_col), promo)


class StockfishOpponent:
    """Wraps Stockfish engine for use as a training opponent.

    Only produces standard chess moves — cannot collect gold, purchase, or transfer.
    Falls back to a random legal ChessBuilder move if Stockfish's suggestion
    isn't legal under ChessBuilder rules (e.g. castling).
    """

    def __init__(self, depth=1, stockfish_path=None):
        if stockfish_path is None:
            stockfish_path = shutil.which("stockfish")
        if stockfish_path is None:
            raise FileNotFoundError(
                "Stockfish binary not found. Install with: sudo apt install stockfish"
            )
        self.depth = depth
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def get_move(self, game):
        """Get a move for the current position.

        Returns a ChessBuilder move tuple (action_type, src, dst, extra).
        Falls back to a random legal move if Stockfish's move isn't valid.
        """
        fen = board_to_fen(game)
        try:
            board = chess.Board(fen)
        except ValueError:
            # Invalid FEN — fall back to random
            return _random_legal_move(game)

        try:
            result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
            uci_str = result.move.uci()
            cb_move = uci_to_chessbuilder_move(uci_str, game)
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
            return _random_legal_move(game)

        # Validate against ChessBuilder's legal actions
        legal_actions = game.get_legal_actions()
        # Only consider standard moves for matching (Stockfish can't do gold actions)
        legal_moves = [a for a in legal_actions if a[0] == "move"]

        for legal in legal_moves:
            if legal[1] == cb_move[1] and legal[2] == cb_move[2]:
                # Match on src/dst; use ChessBuilder's promotion type if applicable
                if cb_move[3] is not None:
                    if legal[3] == cb_move[3]:
                        return legal
                else:
                    if legal[3] is None:
                        return legal

        # Stockfish move not legal in ChessBuilder — pick a random legal move
        # Prefer standard moves over gold actions to stay "chess-like"
        if legal_moves:
            return random.choice(legal_moves)
        return _random_legal_move(game)

    def close(self):
        """Shut down the Stockfish engine process."""
        try:
            self.engine.quit()
        except Exception:
            pass


def _random_legal_move(game):
    """Fallback: return any random legal action."""
    legal = game.get_legal_actions()
    if not legal:
        return None
    return random.choice(legal)
