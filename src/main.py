import pygame, sys, pyperclip
from pathlib import Path
from . import assets
from . import board
from . import game
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
                        game.restore_purchase_state()
                        game.purchase_mode = False
                    elif game.promotion_mode:
                        game.restore_promotion_state()
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
