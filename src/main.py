import pygame, sys, copy, pyperclip, torch
from pathlib import Path
from . import assets
from . import board
from . import game
from .game import Game
from .clock import ChessClock
from .ai import AI  # Our separate AI class

def main():
    pygame.init()
    screen = pygame.display.set_mode((board.WINDOW_WIDTH, board.WINDOW_HEIGHT))

    # Initially, create a game instance.
    game_instance = Game(screen)
    # Pre-load AI instance.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "../models/chess_model_checkpoint.pt"
    ai_instance = AI(checkpoint_path, device)

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Handle overlays: purchase or promotion overlays are canceled by restoring state.
                    if game_instance.purchase_mode:
                        if game_instance.pre_purchase_state:
                            game_instance.board = copy.deepcopy(game_instance.pre_purchase_state['board'])
                            game_instance.move_log = game_instance.pre_purchase_state['move_log'][:]
                            game_instance.selected_piece_pos = game_instance.pre_purchase_state['selected']
                            game_instance.turn = game_instance.pre_purchase_state['turn']
                            game_instance.en_passant = game_instance.pre_purchase_state['en_passant']
                            game_instance.halfmove_clock = game_instance.pre_purchase_state['halfmove_clock']
                            game_instance.valid_purchase_placement = game_instance.pre_purchase_state['valid_purchase_placement'][:]
                            game_instance.error_message = game_instance.pre_purchase_state['error_message']
                        game_instance.purchase_mode = False
                    elif game_instance.promotion_mode:
                        if game_instance.pre_promotion_state:
                            game_instance.board = copy.deepcopy(game_instance.pre_promotion_state['board'])
                            game_instance.move_log = game_instance.pre_promotion_state['move_log'][:]
                            game_instance.promotion_pos = game_instance.pre_promotion_state['promotion_pos']
                            game_instance.promotion_color = game_instance.pre_promotion_state['promotion_color']
                            game_instance.selected_piece_pos = game_instance.pre_promotion_state['selected']
                            game_instance.en_passant = game_instance.pre_promotion_state['en_passant']
                            game_instance.halfmove_clock = game_instance.pre_promotion_state['halfmove_clock']
                            game_instance.turn = game_instance.pre_promotion_state['turn']
                        game_instance.promotion_mode = False
                    else:
                        # Toggle the pause menu.
                        game_instance.toggle_pause_menu()

                elif event.key == pygame.K_c:
                    log_text = "\n".join(game_instance.get_move_log_lines())
                    pyperclip.copy(log_text)
                    game_instance.error_message = "Game log copied to clipboard."

            elif event.type == pygame.MOUSEWHEEL:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if mouse_x >= (board.MARGIN_LEFT + board.BOARD_WIDTH):
                    game_instance.move_log_scroll -= event.y * 20

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                # If any pause menu overlay is active, let the game instance process it.
                if game_instance.pause_menu:
                    game_instance.process_pause_menu_click(pos)
                    continue
                else:
                    # Otherwise, process a normal board click.
                    game_instance.handle_board_click(pos)


        # AI turn handling.
        if game_instance.ai_enabled and game_instance.turn == game_instance.ai_color and not game_instance.game_over:
            ai_move = ai_instance.get_move(game_instance)
            if ai_move is not None:
                game_instance.apply_move(ai_move)

        game_instance.update()
        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()
