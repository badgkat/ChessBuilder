import pytest
from src.board import Piece, get_valid_moves, get_visible_squares, in_bounds

BOARD_SIZE = 8

def create_empty_board():
    return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def test_in_bounds():
    assert in_bounds(0, 0) == True
    assert in_bounds(7, 7) == True
    assert in_bounds(-1, 0) == False
    assert in_bounds(0, -1) == False
    assert in_bounds(8, 8) == False

def test_pawn_moves():
    board = create_empty_board()
    pawn = Piece('P', 'white')
    moves = get_valid_moves(pawn, (6, 3), board)
    assert (5, 3) in moves  # Single move forward
    assert (4, 3) in moves  # Double move forward from start position

def test_knight_moves():
    board = create_empty_board()
    knight = Piece('N', 'white')
    moves = get_valid_moves(knight, (4, 4), board)
    expected_moves = [(6, 5), (6, 3), (2, 5), (2, 3), (5, 6), (5, 2), (3, 6), (3, 2)]
    assert sorted(moves) == sorted(expected_moves)

def test_bishop_moves():
    board = create_empty_board()
    bishop = Piece('B', 'white')
    moves = get_valid_moves(bishop, (3, 3), board)
    assert (0, 0) in moves
    assert (6, 6) in moves
    assert (0, 6) in moves
    assert (6, 0) in moves

def test_rook_moves():
    board = create_empty_board()
    rook = Piece('R', 'white')
    moves = get_valid_moves(rook, (3, 3), board)
    assert (3, 0) in moves
    assert (3, 7) in moves
    assert (0, 3) in moves
    assert (7, 3) in moves

def test_queen_moves():
    board = create_empty_board()
    queen = Piece('Q', 'white')
    moves = get_valid_moves(queen, (3, 3), board)
    assert (0, 0) in moves
    assert (7, 7) in moves
    assert (0, 6) in moves
    assert (6, 0) in moves
    assert (3, 0) in moves
    assert (3, 7) in moves

def test_king_moves():
    board = create_empty_board()
    king = Piece('K', 'white')
    moves = get_valid_moves(king, (3, 3), board)
    expected_moves = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3), (4, 4)]
    assert sorted(moves) == sorted(expected_moves)

def test_pawn_captures():
    board = create_empty_board()
    pawn = Piece('P', 'white')
    board[5][2] = Piece('P', 'black')  # Opponent piece
    board[5][4] = Piece('P', 'black')  # Opponent piece
    moves = get_valid_moves(pawn, (6, 3), board)
    assert (5, 2) in moves  # Capture left diagonal
    assert (5, 4) in moves  # Capture right diagonal

def test_visible_squares():
    board = create_empty_board()
    bishop = Piece('B', 'white')
    board[5][5] = bishop
    visible = get_visible_squares(bishop, (5, 5), board)
    assert len(visible) == 0  # No friendly pieces to be visible
    
    board[3][3] = Piece('P', 'white')  # Friendly pawn in diagonal
    board[7][7] = Piece('P', 'white')  # Another friendly piece
    visible = get_visible_squares(bishop, (5, 5), board)
    assert (3, 3) in visible
    assert (7, 7) in visible
