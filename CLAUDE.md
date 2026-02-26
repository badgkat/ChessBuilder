# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChessBuilder is a Python chess variant built with pygame. It extends standard chess with an economic system: pieces accumulate gold through captures, can collect gold (pawn double-click), transfer gold to friendly pieces, and the king can purchase new pieces. The game is local two-player only, with no AI or network play.

## Commands

```bash
# Install dependencies
pip install -e .

# Run the game
python src/main.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_board.py

# Run a specific test
pytest tests/test_board.py::test_pawn_moves -v

# Format code
black src/ tests/
isort src/ tests/
```

## Architecture

All source code lives in `src/`, tests in `tests/`.

- **main.py** — Entry point and pygame event loop (30 FPS). Handles QUIT, ESC (pause/cancel overlays), C (copy move log to clipboard), mouse wheel (scroll move log), and mouse clicks (delegates to `game.handle_board_click`).
- **game.py** — The `Game` class is the central controller. Manages board state (8x8 2D list of `Piece` objects), turn tracking, move validation, check/checkmate/stalemate/draw detection, the gold economy, and all UI rendering including overlays (purchase, promotion, pause menu, time control). Uses a state machine pattern with modes: purchase, promotion, placement, pause menu, time control. Pre-state saving allows cancellation of overlay actions.
- **board.py** — Defines the `Piece` class (type, color, gold), board constants (dimensions, colors, piece costs), and pure functions: `get_valid_moves()` computes legal squares per piece type, `get_visible_squares()` finds friendly pieces in transfer range, `square_to_notation()` converts coords to algebraic notation.
- **clock.py** — `ChessClock` class for optional timed games with per-player time and increment. `format_time()` helper for MM:SS display.
- **assets/** — 12 PNG sprites: `{w,b}_{king,queen,rook,bishop,knight,pawn}.png`.

## Key Mechanics

- **Gold system**: Pieces gain gold from captures (captured piece's accumulated gold + its cost: Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9). Pawns can collect 1 gold by clicking them twice. Gold can be transferred to friendly pieces within capture range.
- **Piece purchasing**: Double-click king to open purchase overlay. Pieces are placed on empty squares adjacent to the king.
- **Draw detection**: Threefold repetition (position history dict), 50-move rule (100 halfmove counter), and stalemate.
- **Board perspective**: Coordinates transform between internal (row, col) and display coords based on whose turn it is (white views from bottom, black from top).

## Testing

Tests use `SDL_VIDEODRIVER=dummy` for headless pygame. Fixtures in each test file handle pygame init/quit. `test_game.py` patches `load_images` to skip asset loading. When adding tests that need a `Game` instance, follow the existing fixture pattern in `tests/test_game.py`.

## Formatting

Black (line-length 88, target Python 3.8) and isort (black profile). Configured in `pyproject.toml`.
