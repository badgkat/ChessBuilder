import pygame, sys, os, copy

# --- Constants ---
BOARD_SIZE = 8
SQUARE_SIZE = 80
RIGHT_PANEL_WIDTH = 200
BOTTOM_PANEL_HEIGHT = 100
BOARD_WIDTH = BOARD_SIZE * SQUARE_SIZE  # 640
BOARD_HEIGHT = BOARD_SIZE * SQUARE_SIZE  # 640
MARGIN_LEFT = 30
MARGIN_BOTTOM = 30
WINDOW_WIDTH = MARGIN_LEFT + BOARD_WIDTH + RIGHT_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT + MARGIN_BOTTOM + BOTTOM_PANEL_HEIGHT

WHITE_SQ = (232, 235, 239)
DARK_SQ = (125, 135, 150)
BLUE   = (0, 0, 255)     # valid moves
RED    = (255, 0, 0)     # captures
GREEN  = (0, 255, 0)     # gold transfers

RIGHT_PANEL_COLOR  = (200, 200, 200)
BOTTOM_PANEL_COLOR = (180, 180, 180)
NOTATION_PANEL_COLOR = (190, 190, 190)

# Colors for highlights, gold, etc.
SELECT_HIGHLIGHT_COLOR = (255, 255, 0, 80)  # semi-transparent yellow
GOLD_CIRCLE_COLOR = (218, 165, 32)         # golden rod
GOLD_TEXT_COLOR   = (220, 220, 220)        # light gray

PIECE_COST = {
    'P': 1,
    'N': 3,
    'B': 3,
    'R': 5,
    'Q': 9
}

KING_MOVES   = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
KNIGHT_MOVES = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
DIAGONAL_DIRS   = [(-1,-1), (-1,1), (1,-1), (1,1)]
ORTHOGONAL_DIRS = [(1,0), (-1,0), (0,1), (0,-1)]

def square_to_notation(row, col):
    """Converts board coordinates (row, col) to algebraic notation (e.g. e4)."""
    file = chr(col + ord('a'))
    rank = str(BOARD_SIZE - row)
    return file + rank

class Piece:
    def __init__(self, piece_type, color, gold=0):
        self.type = piece_type
        self.color = color
        self.gold = gold

    def __repr__(self):
        return f"{self.color[0]}{self.type}"

def in_bounds(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

def get_valid_moves(piece, pos, board, en_passant=None):
    moves = []
    r, c = pos
    if piece.type == 'K':
        for dr, dc in KING_MOVES:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc):
                target = board[nr][nc]
                if target is None or target.color != piece.color:
                    moves.append((nr, nc))
    elif piece.type == 'P':
        direction = -1 if piece.color == 'white' else 1
        nr = r + direction
        # Single-square advance:
        if in_bounds(nr, c) and board[nr][c] is None:
            moves.append((nr, c))
            # Double-square advance if on starting rank:
            if (piece.color == 'white' and r == 6) or (piece.color == 'black' and r == 1):
                nr2 = r + 2 * direction
                if in_bounds(nr2, c) and board[nr2][c] is None:
                    moves.append((nr2, c))
        # Diagonal captures:
        for dc2 in [-1, 1]:
            nc = c + dc2
            if in_bounds(nr, nc):
                target = board[nr][nc]
                if target is not None and target.color != piece.color:
                    moves.append((nr, nc))
            # En passant capture:
            if en_passant is not None:
                ep_target, _ = en_passant
                if (nr, c + dc2) == ep_target:
                    moves.append(ep_target)
    elif piece.type == 'N':
        for dr, dc in KNIGHT_MOVES:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc):
                target = board[nr][nc]
                if target is None or target.color != piece.color:
                    moves.append((nr, nc))
    elif piece.type == 'B':
        for dr, dc in DIAGONAL_DIRS:
            nr, nc = r, c
            while True:
                nr += dr; nc += dc
                if not in_bounds(nr, nc):
                    break
                target = board[nr][nc]
                if target is None:
                    moves.append((nr, nc))
                else:
                    if target.color != piece.color:
                        moves.append((nr, nc))
                    break
    elif piece.type == 'R':
        for dr, dc in ORTHOGONAL_DIRS:
            nr, nc = r, c
            while True:
                nr += dr; nc += dc
                if not in_bounds(nr, nc):
                    break
                target = board[nr][nc]
                if target is None:
                    moves.append((nr, nc))
                else:
                    if target.color != piece.color:
                        moves.append((nr, nc))
                    break
    elif piece.type == 'Q':
        for dr, dc in DIAGONAL_DIRS + ORTHOGONAL_DIRS:
            nr, nc = r, c
            while True:
                nr += dr; nc += dc
                if not in_bounds(nr, nc):
                    break
                target = board[nr][nc]
                if target is None:
                    moves.append((nr, nc))
                else:
                    if target.color != piece.color:
                        moves.append((nr, nc))
                    break
    return moves

def get_visible_squares(piece, pos, board):
    visible = []
    r, c = pos
    if piece.type == 'K':
        for dr, dc in KING_MOVES:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and board[nr][nc] is not None and board[nr][nc].color == piece.color:
                visible.append((nr, nc))
    elif piece.type == 'P':
        direction = -1 if piece.color == 'white' else 1
        for dc2 in [-1, 1]:
            nr, nc = r + direction, c + dc2
            if in_bounds(nr, nc) and board[nr][nc] is not None and board[nr][nc].color == piece.color:
                visible.append((nr, nc))
    elif piece.type == 'N':
        for dr, dc in KNIGHT_MOVES:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and board[nr][nc] is not None and board[nr][nc].color == piece.color:
                visible.append((nr, nc))
    elif piece.type in ['B','R','Q']:
        dirs = []
        if piece.type == 'B':
            dirs = DIAGONAL_DIRS
        elif piece.type == 'R':
            dirs = ORTHOGONAL_DIRS
        elif piece.type == 'Q':
            dirs = DIAGONAL_DIRS + ORTHOGONAL_DIRS
        for dr, dc in dirs:
            nr, nc = r, c
            while True:
                nr += dr; nc += dc
                if not in_bounds(nr, nc):
                    break
                if board[nr][nc] is not None:
                    if board[nr][nc].color == piece.color:
                        visible.append((nr, nc))
                    break
    return visible
