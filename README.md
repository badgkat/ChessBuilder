# Chess Builder

A custom chess variant in Pygame featuring:
- **Gold Collection**: Pawns can accumulate gold when it’s your turn and your side is not in check.
- **Piece Purchases**: Spend gold to buy and place new pieces near your king.
- **Promotion Choices**: Pawns can promote with an interactive menu.
- **Optional Time Controls**: A built-in chess clock for timed games.

## Features

1. **Gold Accumulation**  
   - When it’s your turn and you’re not in check, click a pawn you’ve already selected once to collect gold.  
   - Captured pieces add their accumulated gold to the capturing piece.

2. **Purchases**  
   - Kings can open a purchase overlay by clicking on them again (when it’s your turn).  
   - Spend gold to place new pieces next to your king if it does not leave you in check.  
   - Each piece has a cost (Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9).

3. **Promotion Menu**  
   - When a pawn reaches the last rank, a promotion menu appears so you can choose a new piece.

4. **Time Controls (Optional)**  
   - A chess clock can be activated, with multiple standard time options.  
   - Each clock runs only during its respective turn.

5. **Draw/Win Conditions**  
   - Threefold repetition and the 50-move rule trigger draws.  
   - Standard checkmate/stalemate detection for wins/losses.

## Installation

1. **Install Python** (3.7+ recommended).
2. **Install Pygame**:
   ```bash
   pip install pygame
   ```
3. **Clone or Download** this repository:
   ```bash
   git clone https://github.com/YourUsername/ChessBuilder.git
   cd ChessBuilder
   ```

## Running

Within the project directory, run:

```bash
python ChessBuilder.py
```

(or `python3 ChessBuilder.py` on some systems)

## Usage

- **Start New Game**  
  Press *New Game* in the pause menu (hit `Esc`). You may be prompted to select a time control if you’ve enabled the chess clock.
- **Moving Pieces**  
  Click a piece to see valid moves/captures.  
- **Collecting Gold**  
  If the currently moving side is not in check, and you have a pawn selected, click the same pawn again to collect gold.
- **Buying Pieces**  
  Select your king, then click the king again to open the purchase overlay if you have enough gold.
- **Promotion**  
  When a pawn reaches the last rank, a promotion menu appears so you can choose a new piece.
- **Pausing/Exiting**  
  Press `Esc` to toggle the pause menu or exit the game.

## Controls

- **Esc**: Toggles the pause menu or closes overlays.  
- **C**: Copies the move log to your clipboard.  
- **Mouse Wheel** (over the right panel): Scrolls the move log.

## Chess Clock (Optional)

- At game start, select a time format (e.g. “3|2”) if time control is enabled.
- Each player’s clock runs only during their turn.
- When you end your turn, any increment is added to the side that just moved.

## Code Organization

- **`board.py`**: Contains constants for the board (dimensions, colors), plus the `Piece` class, and movement logic.  
- **`clock.py`** *(optional)*: Implements a chess clock with starting time, increment, and methods for updating.  
- **`game.py`**: Holds the primary `Game` class, controlling board state, gold mechanics, promotions, overlays, and drawing calls.  
- **`ChessBuilder.py`**: The main entry point with the game loop (`main()`).

## To Do
- General UI beautifying.
- Adding Animations.
- Add an AI.
- Add non-local play.

## Known Limitations

- No castling. As the king must move to build a rook there is no situation where castling would be possible. 
- En Passsant logic only covers standard piece movement, not placed pawns. 

## Contributing

Pull requests and suggestions welcome! If you have ideas or new features, open an issue or submit a PR.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
See the [LICENSE](LICENSE) file for details.

---

*Enjoy building armies of gold-accumulating pieces and exploring this unique twist on classic chess!*
