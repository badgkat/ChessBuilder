import pygame, sys
from pathlib import Path

# Add parent directory to path when running directly
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))
    from src import board  # Import the module, not a class
    from src.game import Game
else:
    # Use relative imports when running as a package
    from . import board    # Import the module, not a class
    from .game import Game

def main():
    pygame.init()
    screen = pygame.display.set_mode((board.WINDOW_WIDTH, board.WINDOW_HEIGHT))
    pygame.display.set_caption("Chess Builder with Draw Rules")
    game = Game(screen)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if game.purchase_mode:
                        if game.pre_purchase_state:
                            game.board = copy.deepcopy(game.pre_purchase_state['board'])
                            game.move_log = game.pre_purchase_state['move_log'][:]
                            game.selected_piece_pos = game.pre_purchase_state['selected']
                            game.turn = game.pre_purchase_state['turn']
                            game.en_passant = game.pre_purchase_state['en_passant']
                            game.halfmove_clock = game.pre_purchase_state['halfmove_clock']
                            game.valid_purchase_placement = game.pre_purchase_state['valid_purchase_placement'][:]
                            game.error_message = game.pre_purchase_state['error_message']
                        game.purchase_mode = False
                    elif game.promotion_mode:
                        if game.pre_promotion_state:
                            game.board = copy.deepcopy(game.pre_promotion_state['board'])
                            game.move_log = game.pre_promotion_state['move_log'][:]
                            game.promotion_pos = game.pre_promotion_state['promotion_pos']
                            game.promotion_color = game.pre_promotion_state['promotion_color']
                            game.selected_piece_pos = game.pre_promotion_state['selected']
                            game.en_passant = game.pre_promotion_state['en_passant']
                            game.halfmove_clock = game.pre_promotion_state['halfmove_clock']
                            game.turn = game.pre_promotion_state['turn']
                        game.promotion_mode = False
                    else:
                        game.toggle_pause_menu()
                elif event.key == pygame.K_c:
                    log_text = "\n".join(game.get_move_log_lines())
                    pyperclip.copy(log_text)
                    game.error_message = "Game log copied to clipboard."
            elif event.type == pygame.MOUSEWHEEL:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if mouse_x >= (board.MARGIN_LEFT + board.BOARD_WIDTH):
                    game.move_log_scroll -= event.y * 20
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                game.handle_board_click(pos)

        game.update()

if __name__ == "__main__":
    main()
