# ChessBuilder Bugfix & ML Pipeline Overhaul

## Approach

Layer-by-layer: fix game logic foundations first (since ML pipeline depends on correct game state), then fix ML pipeline on top. Targeted tests for critical fixes.

## Section 1: Core Game Fixes

### 1a. Unify move format — fix `get_random_move()`
Returns 2-tuple `((src), (dst))`. Fix to return standard 4-tuple `(action_type, src, dst, purchase_type)`. Enumerate all action types (gold collection, purchases, transfers), not just piece moves.

### 1b. Fix `apply_move()` promotion handling
When pawn reaches final rank during `apply_move()` (including simulate=True), promote it. For AI/simulation, auto-promote to queen. Repurpose `purchase_type` field as `promote_type` for move actions.

### 1c. Add purchase adjacency validation in `apply_move()`
Currently only UI enforces purchased pieces must be adjacent to king. `apply_move()` needs same check.

### 1d. Remove duplicate checkmate/stalemate block
Delete repeated check in `end_turn()` (~lines 1382-1388).

### 1e. Initialize `placement_mode` in `__init__()`
Add `self.placement_mode = False` alongside other mode flags.

## Section 2: Game Rule Fixes

### 2a. Verify gold transfer for pawns
`get_visible_squares()` limits pawns to 1-square diagonals (same as capture). Verify this is correct and no multi-square transfer is possible.

### 2b. Fix `has_insufficient_material()`
- 2 pieces (both kings) -> insufficient
- 3 pieces (2 kings + 1 bishop or knight) -> insufficient
- King+bishop vs king+bishop (same color diagonals) -> insufficient
- Everything else -> sufficient

### 2c. Verify threefold repetition
Position key includes gold per piece (correct for this variant). Verify `position_history` clears on `new_game()`.

### 2d. Handle `get_random_move()` returning None
Add safety checks in callers (AI turn in `main.py`, selfplay) to skip applying None moves.

## Section 3: ML Pipeline Fixes

### 3a. Fix `get_training_example()`
- Only add `collect_gold` when a pawn exists and side isn't in check
- Generate purchase actions from actual gold/king adjacency, not `self.purchase_mode` flag
- Generate transfer actions from actual board state

### 3b. Fix action encoding consistency
Verify `move_to_index()` and `index_to_move()` are perfect inverses across all 4 action types.

### 3c. Fix selfplay.py end-to-end
- Use fixed `get_random_move()` (4-tuple)
- Add None-move guard before `apply_move()`
- Fix save path to script-relative
- Add max-move limit per game

### 3d. Unify checkpoint format
Standardize: `{"model_state_dict", "optimizer_state_dict", "iteration", "loss"}`. All scripts load with fallback for missing keys.

### 3e. Fix loss function
Policy head: cross-entropy. Value head: MSE.

### 3f. Fix path handling
All training scripts use `os.path.dirname(__file__)`-relative paths.

### 3g. Fix `ai.py` import
Wrap `from training.model import ChessNet` in try/except since torch is optional.

## Section 4: Integration

### 4a. Selfplay smoke test
Run ~10 games with random moves. Verify no crashes, games terminate, `training_data.npz` has correct shapes.

### 4b. Training smoke test
Load data, run 1 epoch, save checkpoint. Verify loads back, correct loss functions, no NaN/Inf.

### 4c. AI inference test
Load checkpoint, play game vs random. Verify valid moves, no crashes, game terminates.

### 4d. Targeted tests
- `test_get_random_move_returns_4_tuple`
- `test_apply_move_handles_promotion`
- `test_purchase_adjacency_enforced`
- `test_insufficient_material_cases`
- `test_training_example_only_legal_actions`
