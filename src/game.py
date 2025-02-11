import pygame, sys, copy, os
import board  # <-- We import our new board module
from clock import ChessClock
from clock import format_time
import importlib.resources as pkg_resources
from . import assets  # assets folder should be a package

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont(None, 24)
        self.load_images()
        self.new_game()

        self.selected_piece_pos = None
        self.valid_move_squares = []
        self.valid_capture_squares = []
        self.valid_gold_transfer_squares = []

        self.purchase_mode = False
        self.purchase_options = []
        self.purchase_overlay_rect = None
        self.purchase_selected_type = None
        self.valid_purchase_placement = []
        self.purchase_king_color = None
        self.pre_purchase_state = None

        self.promotion_mode = False
        self.promotion_options = []
        self.promotion_overlay_rect = None
        self.promotion_pos = None
        self.promotion_color = None
        self.pre_promotion_state = None

        self.pause_menu = False
        self.pause_menu_options = []

        self.game_over = False
        self.winner = None
        self.move_log = []
        self.error_message = ""
        self.move_log_scroll = 0
        self.en_passant = None
        self.halfmove_clock = 0
        self.position_history = {}
        
        self.time_control_mode = True
        self.time_control_options = []
        self.time_control_overlay_rect = None
        self.chess_clock = None
        self.create_time_control_options()

    def create_time_control_options(self):
        """Create dynamically sized time control overlay."""
        margin = 10
        btn_h = 30
        btn_spacing = 5
        
        # Calculate required height and width based on number of options
        choices = [
            ('1 min', 60, 60, 0),
            ('3|2', 180, 180, 2),
            ('5 min', 300, 300, 0),
            ('10 min', 600, 600, 0),
            ('15|10', 900, 900, 10),
        ]
        
        # Calculate minimum width needed for text
        text_surfaces = [self.font.render(label, True, (0,0,0)) for label, *_ in choices]
        min_btn_width = max(surf.get_width() for surf in text_surfaces) + margin * 4
        btn_w = max(200, min_btn_width)  # minimum width of 200px
        
        overlay_w = btn_w + margin * 2
        overlay_h = (btn_h * len(choices)) + (btn_spacing * (len(choices) - 1)) + margin * 2
        
        # Center the overlay on screen
        ox = (board.WINDOW_WIDTH - overlay_w) // 2
        oy = (board.WINDOW_HEIGHT - overlay_h) // 2
        
        self.time_control_overlay_rect = pygame.Rect(ox, oy, overlay_w, overlay_h)
        
        # Create buttons
        y_cursor = oy + margin
        self.time_control_options = []
        for label, wtime, btime, inc in choices:
            rect = pygame.Rect(ox + margin, y_cursor, btn_w, btn_h)
            self.time_control_options.append((rect, label, wtime, btime, inc))
            y_cursor += btn_h + btn_spacing

    def draw_time_control_overlay(self):
        """Draw the time control overlay with clickable options."""
        if not self.time_control_mode or not self.time_control_overlay_rect:
            return
        pygame.draw.rect(self.screen, (220,220,220), self.time_control_overlay_rect)
        pygame.draw.rect(self.screen, (0,0,0), self.time_control_overlay_rect, 2)

        for rect, label, _, _, _ in self.time_control_options:
            pygame.draw.rect(self.screen, (150,150,150), rect)
            txt_surf = self.font.render(label, True, (0,0,0))
            self.screen.blit(txt_surf, txt_surf.get_rect(center=rect.center))


    def to_display_coords(self, r, c):
        if self.turn == 'white':
            return r, c
        else:
            return (board.BOARD_SIZE - 1 - r), (board.BOARD_SIZE - 1 - c)

    def from_display_coords(self, dr, dc):
        if self.turn == 'white':
            return dr, dc
        else:
            return (board.BOARD_SIZE - 1 - dr), (board.BOARD_SIZE - 1 - dc)

    def get_file_label(self, c):
        files = ['a','b','c','d','e','f','g','h']
        return files[c] if self.turn == 'white' else files[7 - c]

    def get_rank_label(self, r):
        ranks = ['1','2','3','4','5','6','7','8']
        return ranks[7 - r] if self.turn == 'white' else ranks[r]

    def load_images(self):
        self.images = {}
        # Use package resources instead of direct file paths
        pieces = ['K','Q','R','B','N','P']
        for color in ['white', 'black']:
            for p in pieces:
                try:
                    with pkg_resources.path('src.assets', f"{color}_{p}.png") as path:
                        img = pygame.image.load(str(path))
                        img = pygame.transform.scale(img, (board.SQUARE_SIZE, board.SQUARE_SIZE))
                        self.images[(color, p)] = img
                except Exception as e:
                    print(f"Error loading {color}_{p}.png: {e}")

    def new_game(self):
        self.board = [[None for _ in range(board.BOARD_SIZE)] for _ in range(board.BOARD_SIZE)]
        self.board[7][4] = board.Piece('K', 'white')
        self.board[6][4] = board.Piece('P', 'white')
        self.board[0][4] = board.Piece('K', 'black')
        self.board[1][4] = board.Piece('P', 'black')
        self.turn = 'white'
        self.selected_piece_pos = None
        self.clear_valid_actions()
        self.purchase_mode = False
        self.placement_mode = False
        self.pause_menu = False
        self.game_over = False
        self.winner = None
        self.move_log = []
        self.error_message = ""
        self.move_log_scroll = 0
        self.en_passant = None
        self.halfmove_clock = 0
        self.position_history = {}
        self.promotion_mode = False
        self.promotion_pos = None
        self.promotion_color = None
        self.pre_purchase_state = None
        self.pre_promotion_state = None
        # Reset time control related attributes
        self.time_control_mode = True
        self.chess_clock = None

    def clear_valid_actions(self):
        self.valid_move_squares = []
        self.valid_capture_squares = []
        self.valid_gold_transfer_squares = []

    def get_move_log_lines(self):
        lines = []
        for i in range(0, len(self.move_log), 2):
            turn_num = i // 2 + 1
            white_move = self.move_log[i]
            black_move = self.move_log[i+1] if i+1 < len(self.move_log) else ""
            line = f"{turn_num}. {white_move} {black_move}"
            lines.append(line)
        return lines

    def get_king_pos(self, color, custom_board=None):
        board_data = custom_board if custom_board else self.board
        for r in range(board.BOARD_SIZE):
            for c in range(board.BOARD_SIZE):
                piece = board_data[r][c]
                if piece and piece.type == 'K' and piece.color == color:
                    return (r, c)
        return None

    def is_in_check(self, color, custom_board=None):
        bdata = custom_board if custom_board else self.board
        king_pos = self.get_king_pos(color, bdata)
        if king_pos is None:
            return False
        for r in range(board.BOARD_SIZE):
            for c in range(board.BOARD_SIZE):
                piece = bdata[r][c]
                if piece and piece.color != color:
                    moves = board.get_valid_moves(piece, (r, c), bdata, self.en_passant)
                    if king_pos in moves:
                        return True
        return False

    def simulate_move_is_safe(self, src, dst):
        sr, sc = src
        dr, dc = dst
        mover = self.board[sr][sc]
        captured = self.board[dr][dc]
        self.board[dr][dc] = mover
        self.board[sr][sc] = None
        safe = not self.is_in_check(mover.color)
        self.board[sr][sc] = mover
        self.board[dr][dc] = captured
        return safe

    def has_any_legal_moves(self, color):
        for r in range(board.BOARD_SIZE):
            for c in range(board.BOARD_SIZE):
                piece = self.board[r][c]
                if piece and piece.color == color:
                    moves = board.get_valid_moves(piece, (r, c), self.board, self.en_passant)
                    for m in moves:
                        if self.simulate_move_is_safe((r, c), m):
                            return True
        return False

    def get_position_key(self):
        rows = []
        for r in range(board.BOARD_SIZE):
            row_str = ""
            for c in range(board.BOARD_SIZE):
                piece = self.board[r][c]
                if piece:
                    row_str += f"{piece.color[0]}{piece.type}{piece.gold}"
                else:
                    row_str += "."
            rows.append(row_str)
        key = "|".join(rows) + f"_{self.turn}"
        if self.en_passant:
            key += f"_ep{self.en_passant[0]}"
        else:
            key += "_epNone"
        return key

    def update_valid_actions(self, row, col):
        self.clear_valid_actions()
        p = self.board[row][col]
        if not p:
            return
        moves = board.get_valid_moves(p, (row, col), self.board, self.en_passant)
        for (r, c) in moves:
            if not self.simulate_move_is_safe((row, col), (r, c)):
                continue
            if self.board[r][c] is None:
                self.valid_move_squares.append((r, c))
            elif self.board[r][c].color != p.color:
                self.valid_capture_squares.append((r, c))
        if p.gold > 0 and not self.is_in_check(p.color):
            transfers = board.get_visible_squares(p, (row, col), self.board)
            self.valid_gold_transfer_squares = transfers

    def move_piece(self, src, dst):
        sr, sc = src
        dr, dc = dst
        mover = self.board[sr][sc]
        # En passant check
        if mover.type == 'P' and self.en_passant is not None and dst == self.en_passant[0]:
            captured_pos = self.en_passant[1]
            self.board[captured_pos[0]][captured_pos[1]] = None
            notation = f"{mover.type}x{board.square_to_notation(dr, dc)} (e.p.)"
            self.move_log.append(notation)
            self.board[dr][dc] = mover
            self.board[sr][sc] = None
            self.halfmove_clock = 0
        else:
            # Normal move
            self.board[dr][dc] = mover
            self.board[sr][sc] = None
            notation = f"{mover.type}{board.square_to_notation(dr, dc)}"
            self.move_log.append(notation)
            self.halfmove_clock = 0 if mover.type == 'P' else self.halfmove_clock + 1

        # Promotion check
        if mover.type == 'P':
            final_rank = (mover.color == 'white' and dr == 0) or \
                         (mover.color == 'black' and dr == board.BOARD_SIZE - 1)
            if final_rank:
                self.pre_promotion_state = {
                    'board': copy.deepcopy(self.board),
                    'move_log': self.move_log[:],
                    'promotion_pos': (dr, dc),
                    'promotion_color': mover.color,
                    'selected': self.selected_piece_pos,
                    'en_passant': self.en_passant,
                    'halfmove_clock': self.halfmove_clock,
                    'turn': self.turn,
                }
                self.promotion_mode = True
                self.promotion_pos = (dr, dc)
                self.promotion_color = mover.color
                self.create_promotion_options()
                self.selected_piece_pos = None
                self.clear_valid_actions()
                self.en_passant = None
                return
            # If it was a 2-step move
            if abs(sr - dr) == 2:
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
        self.end_turn()

    def capture_piece(self, src, dst):
        sr, sc = src
        dr, dc = dst
        mover = self.board[sr][sc]
        target = self.board[dr][dc]
        if target:
            mover.gold += target.gold
        self.board[dr][dc] = mover
        self.board[sr][sc] = None
        notation = f"{mover.type}x{board.square_to_notation(dr, dc)}"
        self.move_log.append(notation)
        self.halfmove_clock = 0
        self.selected_piece_pos = None
        self.clear_valid_actions()
        self.en_passant = None
        self.end_turn()

    def create_purchase_options(self, row, col):
        """Create dynamically sized purchase overlay."""
        margin = 10
        btn_size = 50
        btn_spacing = 5
        
        purchase_list = ['P', 'N', 'B', 'R', 'Q']
        
        overlay_w = (btn_size * len(purchase_list)) + (btn_spacing * (len(purchase_list) - 1)) + margin * 2
        overlay_h = btn_size + margin * 2
        
        # Center on the board (not the whole window)
        ox = board.MARGIN_LEFT + (board.BOARD_WIDTH - overlay_w) // 2
        oy = (board.BOARD_HEIGHT - overlay_h) // 2
        
        self.purchase_overlay_rect = pygame.Rect(ox, oy, overlay_w, overlay_h)
        
        x_cursor = ox + margin
        y_cursor = oy + margin
        
        king = self.board[row][col]
        self.purchase_king_color = king.color
        
        self.pre_purchase_state = {
            'board': copy.deepcopy(self.board),
            'move_log': self.move_log[:],
            'selected': self.selected_piece_pos,
            'turn': self.turn,
            'en_passant': self.en_passant,
            'halfmove_clock': self.halfmove_clock,
            'valid_purchase_placement': self.valid_purchase_placement[:],
            'error_message': self.error_message,
        }
        
        self.purchase_options = []
        for p_type in purchase_list:
            rect = pygame.Rect(x_cursor, y_cursor, btn_size, btn_size)
            self.purchase_options.append((rect, p_type))
            x_cursor += btn_size + btn_spacing

    def create_promotion_options(self):
        """Create dynamically sized promotion overlay."""
        margin = 10
        btn_size = 50
        btn_spacing = 5
        
        promo_list = ['Q', 'R', 'B', 'N']
        
        overlay_w = (btn_size * len(promo_list)) + (btn_spacing * (len(promo_list) - 1)) + margin * 2
        overlay_h = btn_size + margin * 2
        
        # Center on the board (not the whole window)
        ox = board.MARGIN_LEFT + (board.BOARD_WIDTH - overlay_w) // 2
        oy = (board.BOARD_HEIGHT - overlay_h) // 2
        
        self.promotion_overlay_rect = pygame.Rect(ox, oy, overlay_w, overlay_h)
        
        x_cursor = ox + margin
        y_cursor = oy + margin
        
        self.promotion_options = []
        for p_type in promo_list:
            rect = pygame.Rect(x_cursor, y_cursor, btn_size, btn_size)
            self.promotion_options.append((rect, p_type))
            x_cursor += btn_size + btn_spacing

    def create_pause_menu_options(self):
        """Create dynamically sized pause menu overlay."""
        margin = 10
        btn_h = 40
        btn_spacing = 5
        
        options = [('New Game', 'new_game'), ('Quit', 'quit')]
        
        # Calculate minimum width needed for text
        text_surfaces = [self.font.render(label, True, (0,0,0)) for label, _ in options]
        min_btn_width = max(surf.get_width() for surf in text_surfaces) + margin * 4
        btn_w = max(200, min_btn_width)  # minimum width of 200px
        
        overlay_w = btn_w + margin * 2
        overlay_h = (btn_h * len(options)) + (btn_spacing * (len(options) - 1)) + margin * 2
        
        # Center on screen
        ox = (board.WINDOW_WIDTH - overlay_w) // 2
        oy = (board.WINDOW_HEIGHT - overlay_h) // 2
        
        self.pause_overlay_rect = pygame.Rect(ox, oy, overlay_w, overlay_h)
        
        y_cursor = oy + margin
        self.pause_menu_options = []
        for label, action in options:
            rect = pygame.Rect(ox + margin, y_cursor, btn_w, btn_h)
            self.pause_menu_options.append((rect, action, label))
            y_cursor += btn_h + btn_spacing

    def toggle_pause_menu(self):
        self.pause_menu = not self.pause_menu
        if self.pause_menu:
            self.create_pause_menu_options()

    def draw_board(self):
        board_x = board.MARGIN_LEFT
        board_y = 0
        for r in range(board.BOARD_SIZE):
            for c in range(board.BOARD_SIZE):
                dr, dc = self.to_display_coords(r, c)
                rect_x = board_x + dc * board.SQUARE_SIZE
                rect_y = board_y + dr * board.SQUARE_SIZE
                rect = pygame.Rect(rect_x, rect_y, board.SQUARE_SIZE, board.SQUARE_SIZE)

                color = board.WHITE_SQ if (r + c) % 2 == 0 else board.DARK_SQ
                pygame.draw.rect(self.screen, color, rect)

                # highlight selection
                if self.selected_piece_pos == (r, c):
                    select_overlay = pygame.Surface((board.SQUARE_SIZE, board.SQUARE_SIZE), pygame.SRCALPHA)
                    select_overlay.fill(board.SELECT_HIGHLIGHT_COLOR)
                    self.screen.blit(select_overlay, (rect_x, rect_y))

                # highlight valid moves, captures, gold transfers
                if (r, c) in self.valid_move_squares:
                    pygame.draw.rect(self.screen, board.BLUE, rect, 3)
                if (r, c) in self.valid_capture_squares:
                    pygame.draw.rect(self.screen, board.RED, rect, 3)
                if (r, c) in self.valid_gold_transfer_squares:
                    pygame.draw.rect(self.screen, board.GREEN, rect, 3)

                piece = self.board[r][c]
                if piece:
                    img = self.images.get((piece.color, piece.type))
                    if img:
                        self.screen.blit(img, rect)
                    if piece.gold > 0:
                        # draw gold circle
                        center_x = rect_x + board.SQUARE_SIZE - 15
                        center_y = rect_y + 15
                        radius = 12
                        pygame.draw.circle(self.screen, board.GOLD_CIRCLE_COLOR, (center_x, center_y), radius)
                        gold_txt = self.font.render(str(piece.gold), True, board.GOLD_TEXT_COLOR)
                        gold_rect = gold_txt.get_rect(center=(center_x, center_y))
                        self.screen.blit(gold_txt, gold_rect)

        # Notation side panels, move logs, etc.
        file_rect = pygame.Rect(0, 0, board.MARGIN_LEFT, board.BOARD_HEIGHT)
        pygame.draw.rect(self.screen, board.NOTATION_PANEL_COLOR, file_rect)

        rank_rect = pygame.Rect(0, board.BOARD_HEIGHT, board.BOARD_WIDTH + board.MARGIN_LEFT, board.MARGIN_BOTTOM)
        pygame.draw.rect(self.screen, board.NOTATION_PANEL_COLOR, rank_rect)

        # file labels
        for c in range(board.BOARD_SIZE):
            label = self.get_file_label(c)
            t_surf = self.font.render(label, True, (0, 0, 0))
            t_rect = t_surf.get_rect()
            t_rect.centerx = board_x + c * board.SQUARE_SIZE + (board.SQUARE_SIZE // 2)
            t_rect.centery = board_y + board.BOARD_HEIGHT + (board.MARGIN_BOTTOM // 2)
            self.screen.blit(t_surf, t_rect)

        # rank labels
        for r in range(board.BOARD_SIZE):
            label = self.get_rank_label(r)
            t_surf = self.font.render(label, True, (0, 0, 0))
            t_rect = t_surf.get_rect()
            t_rect.centery = board_y + r * board.SQUARE_SIZE + (board.SQUARE_SIZE // 2)
            t_rect.centerx = board_x - (board.MARGIN_LEFT // 2)
            self.screen.blit(t_surf, t_rect)

        panel_rect = pygame.Rect(board.BOARD_WIDTH + board.MARGIN_LEFT, 0, board.RIGHT_PANEL_WIDTH, board.BOARD_HEIGHT + board.MARGIN_BOTTOM)
        pygame.draw.rect(self.screen, board.RIGHT_PANEL_COLOR, panel_rect)

        lines = self.get_move_log_lines()
        line_height = 20
        total_height = len(lines) * line_height
        visible_height = board.BOARD_HEIGHT + board.MARGIN_BOTTOM
        max_scroll = max(0, total_height - visible_height)
        self.move_log_scroll = max(0, min(self.move_log_scroll, max_scroll))
        y_offset = 10 - self.move_log_scroll
        for line in lines:
            txt = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(txt, (board.BOARD_WIDTH + board.MARGIN_LEFT + 10, y_offset))
            y_offset += line_height

        bottom_rect = pygame.Rect(0, board_y + board.BOARD_HEIGHT + board.MARGIN_BOTTOM, board.WINDOW_WIDTH, board.BOTTOM_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, board.BOTTOM_PANEL_COLOR, bottom_rect)

        if self.error_message:
            err_txt = self.font.render(self.error_message, True, (255, 0, 0))
            self.screen.blit(err_txt, (10, board.BOARD_HEIGHT + board.MARGIN_BOTTOM + 34))

        # Purchase overlay
        if self.purchase_mode and self.purchase_overlay_rect:
            pygame.draw.rect(self.screen, (200, 200, 200), self.purchase_overlay_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), self.purchase_overlay_rect, 2)
            for rect_, p_type in self.purchase_options:
                pygame.draw.rect(self.screen, (150,150,150), rect_)
                if self.purchase_king_color:
                    icon = self.images.get((self.purchase_king_color, p_type))
                    if icon:
                        small_icon = pygame.transform.scale(icon, (rect_.width, rect_.height))
                        self.screen.blit(small_icon, rect_)
                cost_str = str(board.PIECE_COST[p_type])
                cost_txt = self.font.render(cost_str, True, (255, 215, 0))
                self.screen.blit(cost_txt, (rect_.x+2, rect_.y+2))

        # Promotion overlay
        if self.promotion_mode and self.promotion_overlay_rect:
            pygame.draw.rect(self.screen, (200, 200, 200), self.promotion_overlay_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), self.promotion_overlay_rect, 2)
            for rect_, p_type in self.promotion_options:
                pygame.draw.rect(self.screen, (150,150,150), rect_)
                if self.promotion_color:
                    icon = self.images.get((self.promotion_color, p_type))
                    if icon:
                        small_icon = pygame.transform.scale(icon, (rect_.width, rect_.height))
                        self.screen.blit(small_icon, rect_)

        # Pause menu
        if self.pause_menu:
            pygame.draw.rect(self.screen, (180, 180, 180), self.pause_overlay_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), self.pause_overlay_rect, 2)
            for rect_, action, label in self.pause_menu_options:
                pygame.draw.rect(self.screen, (150, 150, 150), rect_)
                ts = self.font.render(label, True, (0, 0, 0))
                self.screen.blit(ts, ts.get_rect(center=rect_.center))

        # Game Over overlay
        if self.game_over:
            overlay = pygame.Surface((board.WINDOW_WIDTH, board.WINDOW_HEIGHT))
            overlay.set_alpha(200)
            overlay.fill((50, 50, 50))
            self.screen.blit(overlay, (0, 0))
            text = f"Game Over: {self.winner} wins!" if self.winner != "draw" else "Game Over: Draw"
            ts = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(ts, ts.get_rect(center=(board.WINDOW_WIDTH // 2, board.WINDOW_HEIGHT // 2)))

        if self.chess_clock:
                white_str = format_time(self.chess_clock.white_time)
                black_str = format_time(self.chess_clock.black_time)
                turn_display = f"Turn: {self.turn} | White: {white_str}  Black: {black_str}"
        else:

                turn_display = f"Turn: {self.turn}"

        turn_txt = self.font.render(turn_display, True, (0, 0, 0))
        self.screen.blit(turn_txt, (board.MARGIN_LEFT + 10, board.BOARD_HEIGHT + board.MARGIN_BOTTOM + 10))

            # Then also draw your time control overlay (if open):
        self.draw_time_control_overlay()
    
    def handle_board_click(self, pos):
        if self.time_control_mode:
            if self.time_control_overlay_rect and self.time_control_overlay_rect.collidepoint(pos):
                # Check which option was clicked
                for rect, label, wsecs, bsecs, inc in self.time_control_options:
                    if rect.collidepoint(pos):
                        # user picked that time
                        self.chess_clock = ChessClock(wsecs, bsecs, inc)
                        self.chess_clock.start('white')  # White moves first
                        self.time_control_mode = False
                        return
            # clicked outside => close or do nothing
            self.time_control_mode = False
            return

        # handle promotion overlay        
        if self.promotion_mode:
            if self.promotion_overlay_rect and self.promotion_overlay_rect.collidepoint(pos):
                for rect_, p_type in self.promotion_options:
                    if rect_.collidepoint(pos):
                        pr, pc = self.promotion_pos
                        self.board[pr][pc].type = p_type
                        self.move_log[-1] = self.move_log[-1] + f"={p_type}"
                        self.promotion_mode = False
                        self.promotion_pos = None
                        self.promotion_color = None
                        self.clear_valid_actions()
                        self.end_turn()
            else:
                if self.pre_promotion_state:
                    self.board = copy.deepcopy(self.pre_promotion_state['board'])
                    self.move_log = self.pre_promotion_state['move_log'][:]
                    self.promotion_pos = self.pre_promotion_state['promotion_pos']
                    self.promotion_color = self.pre_promotion_state['promotion_color']
                    self.selected_piece_pos = self.pre_promotion_state['selected']
                    self.en_passant = self.pre_promotion_state['en_passant']
                    self.halfmove_clock = self.pre_promotion_state['halfmove_clock']
                    self.turn = self.pre_promotion_state['turn']
                self.promotion_mode = False
                self.promotion_pos = None
                self.promotion_color = None
                self.clear_valid_actions()
            return

        board_x = board.MARGIN_LEFT
        board_y = 0

        # handle pause menu
        if self.pause_menu:
            for rect_, action, lbl in self.pause_menu_options:
                if rect_.collidepoint(pos):
                    if action == 'new_game':
                        self.new_game()
                        self.pause_menu = False
                    elif action == 'quit':
                        pygame.quit()
                        sys.exit()
            return

        # handle purchase overlay
        if self.purchase_mode and self.purchase_overlay_rect:
            if self.purchase_overlay_rect.collidepoint(pos):
                for rect_, p_type in self.purchase_options:
                    if rect_.collidepoint(pos):
                        king_pos = self.selected_piece_pos
                        if not king_pos:
                            self.purchase_mode = False
                            return
                        king = self.board[king_pos[0]][king_pos[1]]
                        if king.gold >= board.PIECE_COST[p_type]:
                            self.purchase_selected_type = p_type
                            self.valid_purchase_placement = []
                            r0, c0 = king_pos
                            for dr2 in [-1, 0, 1]:
                                for dc2 in [-1, 0, 1]:
                                    if dr2 == 0 and dc2 == 0:
                                        continue
                                    nr, nc = r0 + dr2, c0 + dc2
                                    if board.in_bounds(nr, nc) and self.board[nr][nc] is None:
                                        # skip if pawn is row0 or row7
                                        if p_type == 'P' and (nr == 0 or nr == board.BOARD_SIZE-1):
                                            continue
                                        self.board[nr][nc] = board.Piece(p_type, king.color)
                                        if not self.is_in_check(king.color):
                                            self.valid_purchase_placement.append((nr, nc))
                                        self.board[nr][nc] = None
                            if not self.valid_purchase_placement:
                                self.error_message = "Purchase doesn't resolve check or can't place pawn on rank 1/8."
                                return
                            self.purchase_mode = False
                            self.placement_mode = True
                        else:
                            self.error_message = "Not enough gold for purchase."
            else:
                if self.pre_purchase_state:
                    self.board = copy.deepcopy(self.pre_purchase_state['board'])
                    self.move_log = self.pre_purchase_state['move_log'][:]
                    self.selected_piece_pos = self.pre_purchase_state['selected']
                    self.turn = self.pre_purchase_state['turn']
                    self.en_passant = self.pre_purchase_state['en_passant']
                    self.halfmove_clock = self.pre_purchase_state['halfmove_clock']
                    self.valid_purchase_placement = self.pre_purchase_state['valid_purchase_placement'][:]
                    self.error_message = self.pre_purchase_state['error_message']
                self.purchase_mode = False
            return

        # check if clicked on the board
        if not (board_x <= pos[0] < board_x + board.BOARD_WIDTH and board_y <= pos[1] < board_y + board.BOARD_HEIGHT):
            return

        adjusted_x = pos[0] - board_x
        adjusted_y = pos[1] - board_y
        dc = adjusted_x // board.SQUARE_SIZE
        dr = adjusted_y // board.SQUARE_SIZE
        row, col = self.from_display_coords(dr, dc)

        # if in placement mode
        if self.placement_mode:
            if (row, col) in self.valid_purchase_placement:
                king_pos = self.selected_piece_pos
                if king_pos is None:
                    self.placement_mode = False
                    return
                king = self.board[king_pos[0]][king_pos[1]]
                king.gold -= board.PIECE_COST[self.purchase_selected_type]
                self.board[row][col] = board.Piece(self.purchase_selected_type, king.color)
                notation = f"${self.purchase_selected_type}{board.square_to_notation(row, col)}"
                self.move_log.append(notation)
                self.halfmove_clock += 1
                self.placement_mode = False
                self.selected_piece_pos = None
                self.clear_valid_actions()
                self.end_turn()
            else:
                if self.pre_purchase_state:
                    self.board = copy.deepcopy(self.pre_purchase_state['board'])
                    self.move_log = self.pre_purchase_state['move_log'][:]
                    self.selected_piece_pos = self.pre_purchase_state['selected']
                    self.turn = self.pre_purchase_state['turn']
                    self.en_passant = self.pre_purchase_state['en_passant']
                    self.halfmove_clock = self.pre_purchase_state['halfmove_clock']
                    self.valid_purchase_placement = self.pre_purchase_state['valid_purchase_placement'][:]
                    self.error_message = self.pre_purchase_state['error_message']
                self.placement_mode = False
            return

        # normal board click
        if self.selected_piece_pos is None:
            p = self.board[row][col]
            if p and p.color == self.turn:
                self.selected_piece_pos = (row, col)
                self.update_valid_actions(row, col)
        else:
            sp = self.selected_piece_pos
            if (row, col) == sp:
                piece = self.board[row][col]
                if piece.type == 'P':
                    if self.is_in_check(self.turn):
                        self.error_message = "Cannot collect gold while in check."
                        return
                    piece.gold += 1
                    self.move_log.append(f"{piece.type}+{board.square_to_notation(row, col)}")
                    self.halfmove_clock = 0
                    self.selected_piece_pos = None
                    self.clear_valid_actions()
                    self.end_turn()
                elif piece.type == 'K':
                    self.purchase_mode = True
                    self.create_purchase_options(row, col)
                else:
                    self.selected_piece_pos = None
                    self.clear_valid_actions()
            else:
                if (row, col) in self.valid_move_squares:
                    self.move_piece(self.selected_piece_pos, (row, col))
                elif (row, col) in self.valid_capture_squares:
                    self.capture_piece(self.selected_piece_pos, (row, col))
                elif (row, col) in self.valid_gold_transfer_squares:
                    src = self.selected_piece_pos
                    target = self.board[row][col]
                    src_piece = self.board[src[0]][src[1]]
                    if src_piece.gold > 0:
                        target.gold += src_piece.gold
                        src_piece.gold = 0
                        self.move_log.append(f"{src_piece.type}G{board.square_to_notation(row, col)}")
                        self.halfmove_clock += 1
                        self.selected_piece_pos = None
                        self.clear_valid_actions()
                        self.end_turn()
                else:
                    p2 = self.board[row][col]
                    if p2 and p2.color == self.turn:
                        self.selected_piece_pos = (row, col)
                        self.update_valid_actions(row, col)
                    else:
                        self.selected_piece_pos = None
                        self.clear_valid_actions()

    def end_turn(self):
        self.turn = 'black' if self.turn == 'white' else 'white'
        if self.chess_clock:
            self.chess_clock.switch_turn()
        self.selected_piece_pos = None
        self.clear_valid_actions()
        self.purchase_mode = False
        self.placement_mode = False
        self.error_message = ""

        pos_key = self.get_position_key()
        self.position_history[pos_key] = self.position_history.get(pos_key, 0) + 1
        if self.position_history[pos_key] >= 3:
            self.game_over = True
            self.winner = "draw"
            return

        if self.halfmove_clock >= 100:
            self.game_over = True
            self.winner = "draw"
            return

        if not self.has_any_legal_moves(self.turn):
            if self.is_in_check(self.turn):
                self.game_over = True
                self.winner = 'black' if self.turn == 'white' else 'white'
            else:
                self.game_over = True
                self.winner = "draw"

    def update(self):
        self.screen.fill((0,0,0))

        # If we have a clock, update it
        if self.chess_clock and not self.time_control_mode and not self.game_over:
            self.chess_clock.update()
            # If time hits 0, you could auto-end
            if self.chess_clock.white_time <= 0:
                self.game_over = True
                self.winner = 'black'
            elif self.chess_clock.black_time <= 0:
                self.game_over = True
                self.winner = 'white'

        # Draw everything in one go using draw_board() which already includes all drawing logic
        self.draw_board()
        
        # Draw game state overlays
        if self.game_over:
            s = pygame.Surface((board.WINDOW_WIDTH, board.WINDOW_HEIGHT))
            s.set_alpha(128)
            s.fill((128, 128, 128))
            self.screen.blit(s, (0, 0))
            
            font = pygame.font.Font(None, 74)
            if self.winner and self.winner != "draw":
                text = font.render(f"{self.winner} wins!", True, (255, 255, 255))
            else:
                text = font.render("Draw!", True, (255, 255, 255))
            text_rect = text.get_rect(center=(board.WINDOW_WIDTH/2, board.WINDOW_HEIGHT/2))
            self.screen.blit(text, text_rect)
        
        # Draw pause menu last so it's always on top
        if self.pause_menu:
            self.draw_pause_menu()
        
        pygame.display.flip()
        self.clock.tick(30)