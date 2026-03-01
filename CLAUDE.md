# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChessBuilder is a custom chess variant built with Python/Pygame. It adds gold mechanics, piece purchasing, and gold transfers on top of standard chess rules. Each side starts with only a king and a pawn — players must accumulate gold and purchase additional pieces. Published on PyPI as `chessbuilder`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install in editable mode (dev)
pip install -e .

# Run the game
python main.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_game.py

# Run a specific test
pytest tests/test_game.py::test_new_game_initial_positions

# Run training UI (requires torch + PyQt5)
python -m training.main
```

## Architecture

### Core Game (`src/`)

- **`main.py`** — Entry point. Pygame init, 30 FPS game loop, event dispatch (ESC/C/mouse), AI turn execution.
- **`board.py`** — Stateless movement logic layer. Defines `Piece` class, board constants (8x8, 80px squares), `get_valid_moves()`, `get_visible_squares()`. Does NOT manage game state.
- **`game.py`** — Central game state manager (~1500 lines). The `Game` class owns the board array, turn tracking, gold mechanics, purchase/promotion overlays, move legality (including check validation via `copy_for_simulation()`), drawing, and AI move generation.
- **`clock.py`** — `ChessClock` class for optional time controls with increment.
- **`ai.py`** — `AI` class wrapping a PyTorch `ChessNet` model. Loads checkpoint from `models/chess_model_checkpoint.pt`, delegates move selection to `Game.get_model_move()`.

### Training Pipeline (`training/`)

- **`model.py`** — `ChessNet`: 2-conv-layer CNN with policy head (8513 actions) and value head.
- **`selfplay.py`** — Generates training games, outputs `training_data.npz`.
- **`dataset.py`** — PyTorch `Dataset` loader for `.npz` training data.
- **`main.py`** — PyQt5 training UI with progress tracking.

### Key Design Patterns

**Move representation**: `(action_type, src, dst, purchase_type)` where action_type is one of `"move"`, `"collect_gold"`, `"purchase"`, `"transfer_gold"`.

**Action space (8513 total)**: 4096 standard moves (64×64) + 1 gold collection + 320 purchases (5 types × 64 squares) + 4096 gold transfers (64×64).

**Board encoding for AI**: 13 channels of 8×8 — channels 0–5 white pieces (K,Q,R,B,N,P), 6–11 black pieces, channel 12 gold amounts.

**Simulation copies**: `Game.copy_for_simulation()` creates a lightweight game copy (excluding pygame surfaces) to validate moves don't leave the king in check.

**Display coordinate flipping**: Board perspective rotates based on current turn. `to_display_coords()` / `from_display_coords()` handle the conversion.

## Game-Specific Rules

- Piece costs: Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9 gold
- Pawns collect gold by clicking them again when already selected (and not in check)
- Kings open a purchase overlay by clicking again when selected
- Purchased pieces must be placed adjacent to your king without leaving you in check
- Captured pieces transfer their gold to the capturing piece
- Gold transfer uses same targeting logic as captures (between friendly pieces)
- No castling (king must move to build rook, so castling is never possible)
- En passant covers standard pawn movement only, not placed pawns

## Formatting

Configured for Black (line-length 88) and isort (profile "black"). See `[tool.black]` and `[tool.isort]` in pyproject.toml.

## Window Dimensions

1100×930 pixels: 30px left margin + 640px board (8×80) + 200px right panel, 640px board + 30px bottom margin + 100px bottom panel.
