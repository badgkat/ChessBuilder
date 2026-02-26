# ChessBuilder Bugfix & ML Pipeline Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all critical game logic bugs and make the ML training pipeline work end-to-end.

**Architecture:** Layer-by-layer — fix core game logic first (apply_move, get_random_move, promotion, draw rules), then fix ML pipeline on top (training examples, loss function, checkpoints, selfplay). The ML pipeline depends on correct game state, so order matters.

**Tech Stack:** Python 3.8+, Pygame, PyTorch, pytest

---

### Task 1: Quick cleanup — duplicate code and missing init

**Files:**
- Modify: `src/game.py:1382-1388` (remove duplicate checkmate block)
- Modify: `src/game.py:12-53` (add placement_mode to __init__)

**Step 1: Read and verify the duplicate block**

Confirm lines 1369-1375 and 1382-1388 in `end_turn()` are identical checkmate/stalemate checks.

**Step 2: Remove the duplicate block**

Delete lines 1382-1388 (the second `if not self.has_any_legal_moves(self.turn):` block). Keep the first one at 1369-1375.

The `end_turn()` method after the first checkmate block should be:
```python
        # New insufficient material check.
        if self.has_insufficient_material():
            self.game_over = True
            self.winner = "draw"
            return
```

Followed by nothing else (the duplicate block is removed).

**Step 3: Add placement_mode to __init__**

In `__init__`, after line 28 (`self.purchase_mode = False`), the placement_mode is never initialized. Add it:

```python
        self.placement_mode = False
```

Add this after `self.purchase_king_color = None` and before `self.pre_purchase_state = None` (around line 34).

**Step 4: Run tests**

Run: `pytest tests/test_game.py -v`
Expected: All existing tests pass.

**Step 5: Commit**

```bash
git add src/game.py
git commit -m "fix: remove duplicate checkmate check, init placement_mode in __init__"
```

---

### Task 2: Rewrite apply_move "move" handler

**Files:**
- Modify: `src/game.py:456-469` (apply_move method)
- Test: `tests/test_game.py`

**Context:** The current apply_move delegates to `move_piece()`/`capture_piece()` which are UI methods with side effects (promotion overlay, pygame rects). For AI and simulation, apply_move needs to handle moves directly: en passant, captures with gold transfer, promotion, and en passant state management.

**Step 1: Write failing tests**

Add to `tests/test_game.py`:

```python
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
    # Place white pawn on row 1 (one step from promotion)
    g.board[1][4] = None  # Remove black pawn
    g.board[6][4] = None  # Remove white pawn from start
    g.board[1][3] = board.Piece('P', 'white')
    move = ("move", (1, 3), (0, 3), 'Q')
    g.apply_move(move)
    assert g.board[0][3] is not None
    assert g.board[0][3].type == 'Q'
    assert g.board[0][3].color == 'white'
    assert g.turn == 'black'


def test_apply_move_promotion_default_queen(game_instance):
    """apply_move defaults to queen promotion when no type specified."""
    g = game_instance
    g.new_game()
    g.board[1][4] = None
    g.board[6][4] = None
    g.board[1][3] = board.Piece('P', 'white')
    move = ("move", (1, 3), (0, 3), None)
    g.apply_move(move)
    assert g.board[0][3].type == 'Q'


def test_apply_move_capture_promotion(game_instance):
    """apply_move handles capture + promotion correctly."""
    g = game_instance
    g.new_game()
    g.board[6][4] = None  # Remove white pawn
    g.board[1][3] = board.Piece('P', 'white')
    # Black pawn at 0,4 is actually black king, so place a target
    g.board[0][2] = board.Piece('N', 'black')
    move = ("move", (1, 3), (0, 2), 'R')
    g.apply_move(move)
    assert g.board[0][2].type == 'R'
    assert g.board[0][2].color == 'white'


def test_apply_move_capture_transfers_gold(game_instance):
    """Capturing a piece transfers its gold to the capturer."""
    g = game_instance
    g.new_game()
    g.board[4][4] = board.Piece('P', 'white', gold=3)
    g.board[3][3] = board.Piece('P', 'black', gold=5)
    move = ("move", (4, 4), (3, 3), None)
    g.apply_move(move)
    assert g.board[3][3].gold == 8  # 3 + 5
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_game.py::test_apply_move_promotion -v`
Expected: FAIL (promotion not handled in apply_move)

**Step 3: Rewrite apply_move's "move" block**

Replace the `if action_type == "move":` block (lines 464-469) with:

```python
        if action_type == "move":
            sr, sc = src
            dr, dc = dst
            mover = self.board[sr][sc]
            target = self.board[dr][dc]

            # En passant capture
            if mover.type == 'P' and self.en_passant is not None and dst == self.en_passant[0]:
                captured_pos = self.en_passant[1]
                self.board[captured_pos[0]][captured_pos[1]] = None
                self.board[dr][dc] = mover
                self.board[sr][sc] = None
                self.move_log.append(f"{mover.type}x{board.square_to_notation(dr, dc)} (e.p.)")
                self.halfmove_clock = 0
            elif target is not None:
                # Capture — transfer gold from captured piece
                mover.gold += target.gold
                self.board[dr][dc] = mover
                self.board[sr][sc] = None
                self.move_log.append(f"{mover.type}x{board.square_to_notation(dr, dc)}")
                self.halfmove_clock = 0
            else:
                # Normal move
                self.board[dr][dc] = mover
                self.board[sr][sc] = None
                self.move_log.append(f"{mover.type}{board.square_to_notation(dr, dc)}")
                self.halfmove_clock = 0 if mover.type == 'P' else self.halfmove_clock + 1

            # Promotion check
            if mover.type == 'P':
                final_rank = (mover.color == 'white' and dr == 0) or \
                             (mover.color == 'black' and dr == board.BOARD_SIZE - 1)
                if final_rank:
                    promo_type = purchase_type if purchase_type else 'Q'
                    mover.type = promo_type
                    self.move_log[-1] += f"={promo_type}"
                elif abs(sr - dr) == 2:
                    # Set en passant target for next move
                    if mover.color == 'white':
                        self.en_passant = ((sr - 1, sc), (dr, dc))
                    else:
                        self.en_passant = ((sr + 1, sc), (dr, dc))
                else:
                    self.en_passant = None
            else:
                self.en_passant = None

            self.selected_piece_pos = None
            self.clear_valid_actions()
            if not simulate:
                self.end_turn()
```

**Step 4: Run all tests**

Run: `pytest tests/test_game.py -v`
Expected: All tests pass including the new ones.

**Step 5: Commit**

```bash
git add src/game.py tests/test_game.py
git commit -m "fix: rewrite apply_move move handler with promotion and en passant"
```

---

### Task 3: Add purchase adjacency validation in apply_move

**Files:**
- Modify: `src/game.py:485-513` (apply_move purchase block)
- Test: `tests/test_game.py`

**Step 1: Write failing test**

```python
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
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_game.py::test_purchase_adjacency_enforced -v`
Expected: FAIL (purchase placed at (0,0) without adjacency check)

**Step 3: Add adjacency and pawn-rank validation**

In the `elif action_type == "purchase":` block, after the `king.gold < cost` check and before calling `self.purchase_piece`, add:

```python
                    # Validate adjacency to king
                    kr, kc = king_pos
                    if abs(kr - dst[0]) > 1 or abs(kc - dst[1]) > 1:
                        self.error_message = "Purchase must be adjacent to king."
                    elif self.board[dst[0]][dst[1]] is not None:
                        self.error_message = "Purchase square must be empty."
                    elif purchase_type == 'P' and (dst[0] == 0 or dst[0] == board.BOARD_SIZE - 1):
                        self.error_message = "Cannot place pawn on first or last rank."
                    else:
                        self.purchase_piece(dst, purchase_type)
                        self.move_log.append(f"${purchase_type}{board.square_to_notation(dst[0], dst[1])}")
                        king.gold -= cost
                        self.halfmove_clock += 1
```

This replaces the existing else block that had no adjacency check.

**Step 4: Run tests**

Run: `pytest tests/test_game.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/game.py tests/test_game.py
git commit -m "fix: enforce purchase adjacency and pawn rank in apply_move"
```

---

### Task 4: Fix has_insufficient_material

**Files:**
- Modify: `src/game.py:1330-1345`
- Test: `tests/test_game.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_game.py::test_sufficient_material_with_gold -v`
Expected: FAIL (current code doesn't check gold)

**Step 3: Rewrite has_insufficient_material**

Replace the method entirely:

```python
    def has_insufficient_material(self):
        pieces = []
        for row in self.board:
            for piece in row:
                if piece is not None:
                    pieces.append(piece)

        # Any piece with gold means purchases are possible
        if any(p.gold > 0 for p in pieces):
            return False

        # King vs King
        if len(pieces) == 2:
            return True

        # King + minor piece vs King
        if len(pieces) == 3:
            non_kings = [p for p in pieces if p.type != 'K']
            if len(non_kings) == 1 and non_kings[0].type in ['N', 'B']:
                return True

        # King + Bishop vs King + Bishop (same color square)
        if len(pieces) == 4:
            non_kings = [p for p in pieces if p.type != 'K']
            if len(non_kings) == 2 and all(p.type == 'B' for p in non_kings):
                bishop_squares = []
                for r in range(board.BOARD_SIZE):
                    for c in range(board.BOARD_SIZE):
                        if self.board[r][c] and self.board[r][c].type == 'B':
                            bishop_squares.append((r + c) % 2)
                if len(bishop_squares) == 2 and bishop_squares[0] == bishop_squares[1]:
                    return True

        return False
```

**Step 4: Run tests**

Run: `pytest tests/test_game.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/game.py tests/test_game.py
git commit -m "fix: improve insufficient material detection with gold check"
```

---

### Task 5: Fix get_random_move to return 4-tuple

**Files:**
- Modify: `src/game.py:1462-1474`
- Test: `tests/test_game.py`

**Step 1: Write failing test**

```python
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
    g.new_game()
    # Run many times to check gold collection appears
    action_types = set()
    for _ in range(200):
        g.new_game()
        move = g.get_random_move()
        if move:
            action_types.add(move[0])
    assert "collect_gold" in action_types
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_game.py::test_get_random_move_returns_4_tuple -v`
Expected: FAIL (returns 2-tuple)

**Step 3: Rewrite get_random_move**

Replace the entire method:

```python
    def get_random_move(self):
        """Return a random legal action as a 4-tuple (action_type, src, dst, extra)."""
        legal_actions = []

        # 1. Standard moves (including promotions)
        for r in range(board.BOARD_SIZE):
            for c in range(board.BOARD_SIZE):
                piece = self.board[r][c]
                if piece and piece.color == self.turn:
                    moves = board.get_valid_moves(piece, (r, c), self.board, self.en_passant)
                    for move in moves:
                        if self.simulate_move_is_safe((r, c), move):
                            if piece.type == 'P' and self.move_leads_to_promotion(piece, (r, c), move):
                                for promo in ['Q', 'R', 'B', 'N']:
                                    legal_actions.append(("move", (r, c), move, promo))
                            else:
                                legal_actions.append(("move", (r, c), move, None))

        # 2. Gold collection (pawns, not in check)
        if not self.is_in_check(self.turn):
            for r in range(board.BOARD_SIZE):
                for c in range(board.BOARD_SIZE):
                    piece = self.board[r][c]
                    if piece and piece.color == self.turn and piece.type == 'P':
                        legal_actions.append(("collect_gold", (r, c), None, None))

        # 3. Purchase actions
        king, king_pos = None, None
        for r in range(board.BOARD_SIZE):
            for c in range(board.BOARD_SIZE):
                piece = self.board[r][c]
                if piece and piece.color == self.turn and piece.type == 'K':
                    king, king_pos = piece, (r, c)
                    break
            if king_pos:
                break
        if king and king.gold > 0:
            kr, kc = king_pos
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = kr + dr, kc + dc
                    if board.in_bounds(nr, nc) and self.board[nr][nc] is None:
                        for p_type in ['P', 'N', 'B', 'R', 'Q']:
                            cost = board.PIECE_COST.get(p_type)
                            if cost and king.gold >= cost:
                                if p_type == 'P' and (nr == 0 or nr == board.BOARD_SIZE - 1):
                                    continue
                                self.board[nr][nc] = board.Piece(p_type, king.color)
                                if not self.is_in_check(king.color):
                                    legal_actions.append(("purchase", king_pos, (nr, nc), p_type))
                                self.board[nr][nc] = None

        # 4. Gold transfers (not in check, piece with gold > 0)
        if not self.is_in_check(self.turn):
            for r in range(board.BOARD_SIZE):
                for c in range(board.BOARD_SIZE):
                    piece = self.board[r][c]
                    if piece and piece.color == self.turn and piece.gold > 0:
                        visible = board.get_visible_squares(piece, (r, c), self.board)
                        for target in visible:
                            legal_actions.append(("transfer_gold", (r, c), target, None))

        if not legal_actions:
            return None
        return random.choice(legal_actions)
```

**Step 4: Run tests**

Run: `pytest tests/test_game.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/game.py tests/test_game.py
git commit -m "fix: get_random_move returns 4-tuple with all action types"
```

---

### Task 6: Fix get_training_example

**Files:**
- Modify: `src/game.py:387-454`
- Test: `tests/test_game.py`

**Step 1: Write failing test**

```python
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
```

**Step 2: Run to verify it may fail**

Run: `pytest tests/test_game.py::test_training_example_only_legal_actions -v`

The current code unconditionally adds collect_gold, which may cause the test to pass but with incorrect legal actions. The test checks format; we verify correctness manually.

**Step 3: Rewrite get_training_example**

Replace the entire method to mirror the legal action enumeration from `get_model_move`:

```python
    def get_training_example(self):
        """
        Returns (board_state, policy_target, player) where policy_target is
        a uniform distribution over all legal actions in the 8513-dim action space.
        """
        board_state = self.encode_board_state()
        total_actions = 8513
        policy_target = np.zeros(total_actions, dtype=np.float32)
        legal_actions = []

        board_size = board.BOARD_SIZE

        # 1. Standard moves (including promotions)
        for r in range(board_size):
            for c in range(board_size):
                piece = self.board[r][c]
                if piece and piece.color == self.turn:
                    moves = board.get_valid_moves(piece, (r, c), self.board, self.en_passant)
                    for move in moves:
                        if self.simulate_move_is_safe((r, c), move):
                            if piece.type == 'P' and self.move_leads_to_promotion(piece, (r, c), move):
                                for promo in ['Q', 'R', 'B', 'N']:
                                    legal_actions.append(("move", (r, c), move, promo))
                            else:
                                legal_actions.append(("move", (r, c), move, None))

        # 2. Gold collection (pawns, not in check)
        if not self.is_in_check(self.turn):
            for r in range(board_size):
                for c in range(board_size):
                    piece = self.board[r][c]
                    if piece and piece.color == self.turn and piece.type == 'P':
                        legal_actions.append(("collect_gold", (r, c), None, None))

        # 3. Purchase actions (based on king gold and adjacency)
        king, king_pos = None, None
        for r in range(board_size):
            for c in range(board_size):
                piece = self.board[r][c]
                if piece and piece.color == self.turn and piece.type == 'K':
                    king, king_pos = piece, (r, c)
                    break
            if king_pos:
                break
        if king and king.gold > 0:
            kr, kc = king_pos
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = kr + dr, kc + dc
                    if board.in_bounds(nr, nc) and self.board[nr][nc] is None:
                        for p_type in ['P', 'N', 'B', 'R', 'Q']:
                            cost = board.PIECE_COST.get(p_type)
                            if cost and king.gold >= cost:
                                if p_type == 'P' and (nr == 0 or nr == board.BOARD_SIZE - 1):
                                    continue
                                self.board[nr][nc] = board.Piece(p_type, king.color)
                                if not self.is_in_check(king.color):
                                    legal_actions.append(("purchase", king_pos, (nr, nc), p_type))
                                self.board[nr][nc] = None

        # 4. Gold transfers (not in check, pieces with gold)
        if not self.is_in_check(self.turn):
            for r in range(board_size):
                for c in range(board_size):
                    piece = self.board[r][c]
                    if piece and piece.color == self.turn and piece.gold > 0:
                        visible = board.get_visible_squares(piece, (r, c), self.board)
                        for target in visible:
                            legal_actions.append(("transfer_gold", (r, c), target, None))

        # Uniform distribution over legal actions
        if legal_actions:
            probability = 1.0 / len(legal_actions)
            for action in legal_actions:
                action_type, src, dst, pt = action
                index = self.move_to_index(action_type, src, dst, pt)
                policy_target[index] = probability

        return board_state, policy_target, self.turn
```

**Step 4: Run tests**

Run: `pytest tests/test_game.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/game.py tests/test_game.py
git commit -m "fix: get_training_example enumerates real legal actions"
```

---

### Task 7: Fix ai.py import and main.py path

**Files:**
- Modify: `src/ai.py:1-5` (wrap torch import)
- Modify: `src/main.py:1-19` (fix checkpoint path)

**Step 1: Fix ai.py import**

Replace the imports at the top of `src/ai.py`:

```python
import random
import os

try:
    import torch
    from training.model import ChessNet
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
```

Remove `import pygame` (unused) and `from .game import Game` (unused).

**Step 2: Fix main.py checkpoint path**

In `src/main.py`, replace line 18:
```python
    checkpoint_path = "../models/chess_model_checkpoint.pt"
```
with:
```python
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, '..', 'models', 'chess_model_checkpoint.pt')
```

Also add `import os` if not already imported (check existing imports).

**Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: All pass.

**Step 4: Commit**

```bash
git add src/ai.py src/main.py
git commit -m "fix: wrap torch import in ai.py, fix checkpoint path in main.py"
```

---

### Task 8: Fix loss function and unify checkpoints

**Files:**
- Modify: `training/train.py:10-44`
- Modify: `training/main.py:76-108`
- Modify: `training/interative_training.py:42-66`

**Step 1: Fix train.py — cross-entropy policy loss + unified checkpoint**

In `training/train.py`, replace the train function:

```python
def train(model, optimizer, dataloader, device, start_epoch, num_epochs, checkpoint_path):
    model.train()
    loss_fn_value = torch.nn.MSELoss()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        for batch_idx, (states, policy_targets, value_targets) in enumerate(dataloader):
            states = states.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            policy_pred, value_pred = model(states)

            # Cross-entropy loss for policy (soft targets)
            log_probs = torch.nn.functional.log_softmax(policy_pred, dim=1)
            loss_policy = -torch.sum(policy_targets * log_probs) / policy_targets.shape[0]
            loss_value = loss_fn_value(value_pred, value_targets)
            loss = loss_policy + loss_value

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}: Loss {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

        torch.save({
            'iteration': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")
```

Also fix the `main()` function's checkpoint loading to use unified key:

```python
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('iteration', checkpoint.get('epoch', 0))
        print(f"Resuming training from epoch {start_epoch}")
```

And fix the path:

```python
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, '..', 'models', 'chess_model_checkpoint.pt')
```

**Step 2: Fix training/main.py — same loss fix**

In the `TrainingWorker.run()` method, replace the loss computation (around lines 83-85):

```python
                    log_probs = torch.nn.functional.log_softmax(policy_pred, dim=1)
                    loss_policy = -torch.sum(policy_targets * log_probs) / policy_targets.shape[0]
                    loss_value = torch.nn.functional.mse_loss(value_pred, value_targets)
```

Also add `map_location=device` to the `torch.load` call on line 35:

```python
            checkpoint = torch.load(checkpoint_path, map_location=device)
```

**Step 3: Fix training/interative_training.py — same loss fix**

Replace the loss lines (around lines 52-54):

```python
                loss_policy_fn = torch.nn.functional.log_softmax(policy_pred, dim=1)
                loss_policy = -torch.sum(policy_targets * loss_policy_fn) / policy_targets.shape[0]
                loss_value = torch.nn.functional.mse_loss(value_pred, value_targets)
```

And add `map_location=device` to line 23:

```python
        checkpoint = torch.load(checkpoint_path, map_location=device)
```

**Step 4: Run tests**

Run: `pytest tests/ -v`
Expected: All pass (training code isn't tested by pytest but game tests should still pass).

**Step 5: Commit**

```bash
git add training/train.py training/main.py training/interative_training.py
git commit -m "fix: cross-entropy policy loss, unified checkpoint format, map_location"
```

---

### Task 9: Fix selfplay.py and dataset.py paths

**Files:**
- Modify: `training/selfplay.py:2,56,89`
- Modify: `training/dataset.py:8`

**Step 1: Fix selfplay.py save path and None-move guard**

At the top of `training/selfplay.py`, add path computation after the imports:

```python
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_SCRIPT_DIR, '..', 'training_data.npz')
```

In the game loop (line 56), add a None guard before apply_move:

```python
            if move is None:
                # No legal moves — game should end
                game_instance.game_over = True
                game_instance.winner = "draw"
                break

            # Record a training example before applying the move.
            example = game_instance.get_training_example()
            game_examples.append(example)

            game_instance.apply_move(move)
```

Replace the save line (89):
```python
    np.savez(_DATA_PATH, states=states, policy_targets=policy_targets, value_targets=value_targets)
```

**Step 2: Fix dataset.py to use matching path**

Change the default data_file:

```python
class ChessDataset(Dataset):
    def __init__(self, data_file=None):
        if data_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(script_dir, '..', 'training_data.npz')
        if os.path.exists(data_file):
```

**Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: All pass.

**Step 4: Commit**

```bash
git add training/selfplay.py training/dataset.py
git commit -m "fix: script-relative paths in selfplay and dataset, None-move guard"
```

---

### Task 10: Add integration tests for the training pipeline

**Files:**
- Create: `tests/test_training.py`

**Step 1: Write training pipeline tests**

```python
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
    for i in range(max_moves):
        if g.is_game_over():
            break
        move = g.get_random_move()
        if move is None:
            break
        g.apply_move(move)
    # Game should have ended or we hit max moves
    assert i > 0  # At least one move was made


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
```

**Step 2: Run tests**

Run: `pytest tests/test_training.py -v`
Expected: All pass.

**Step 3: Commit**

```bash
git add tests/test_training.py
git commit -m "test: add integration tests for training pipeline"
```

---

### Task 11: Smoke test the full pipeline

**Files:** None created (manual verification).

**Step 1: Run selfplay for 5 games**

Run from project root:
```bash
python -c "
import pygame
pygame.init()
from training.selfplay import generate_selfplay_data
generate_selfplay_data(num_games=5, model=None, device=None)
print('Selfplay complete')
"
```

Expected: Completes without crash, prints game info, creates/updates `training_data.npz`.

**Step 2: Verify training data shapes**

```bash
python -c "
import numpy as np
data = np.load('training_data.npz')
print('States:', data['states'].shape)
print('Policy:', data['policy_targets'].shape)
print('Values:', data['value_targets'].shape)
print('Policy sum sample:', data['policy_targets'][0].sum())
"
```

Expected: States `(N, 13, 8, 8)`, Policy `(N, 8513)`, Values `(N, 1)`, sum ~1.0.

**Step 3: Run 1 epoch of training**

```bash
python -c "
import torch
from training.model import ChessNet
from training.dataset import ChessDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device('cpu')
model = ChessNet(num_channels=13, policy_size=8513).to(device)
dataset = ChessDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for states, policy_targets, value_targets in loader:
    states = states.to(device)
    policy_targets = policy_targets.to(device)
    value_targets = value_targets.to(device)
    optimizer.zero_grad()
    policy_pred, value_pred = model(states)
    log_probs = F.log_softmax(policy_pred, dim=1)
    loss_policy = -torch.sum(policy_targets * log_probs) / policy_targets.shape[0]
    loss_value = F.mse_loss(value_pred, value_targets)
    loss = loss_policy + loss_value
    loss.backward()
    optimizer.step()
    print(f'Loss: {loss.item():.4f}')
    break
print('Training smoke test passed')
"
```

Expected: Prints loss value, no NaN, no crash.

**Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass.

**Step 5: Commit (if any minor fixes were needed)**

```bash
git add -A
git commit -m "chore: verify full pipeline works end-to-end"
```
