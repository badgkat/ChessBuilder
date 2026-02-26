import pygame, sys, copy, os, random
import numpy as np    
from . import board  # Changed to relative import
from .clock import ChessClock  # Changed to relative import
from .clock import format_time  # Changed to relative import
import importlib.resources as pkg_resources
from . import assets  # assets folder should be a package
import torch
import torch.nn.functional as F

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont(None, 24)
        self.load_images()
        
        self.ai_enabled = False
        self.ai_color = None

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
        self.placement_mode = False
        self.pre_purchase_state = None

        self.promotion_mode = False
        self.promotion_options = []
        self.promotion_overlay_rect = None
        self.promotion_pos = None
        self.promotion_color = None
        self.pre_promotion_state = None

        self.pause_menu = False
        self.pause_menu_options = []
        self.show_ai_submenu = False

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

        self.max_moves_reached = False  # Initialize flag here

    def create_time_control_submenu_options(self):
        margin = 10
        btn_h = 40
        btn_spacing = 5
        # Time control options including a "None" option.
        options = [
            ('None', 'none'),
            ('1 min', '1min'),
            ('3|2', '3|2'),
            ('5 min', '5min'),
            ('10 min', '10min'),
            ('15|10', '15|10')
        ]
        
        text_surfaces = [self.font.render(label, True, (0,0,0)) for label, _ in options]
        min_btn_width = max(surf.get_width() for surf in text_surfaces) + margin * 4
        btn_w = max(200, min_btn_width)
        
        overlay_w = btn_w + margin * 2
        overlay_h = (btn_h * len(options)) + (btn_spacing * (len(options) - 1)) + margin * 2
        
        ox = (board.WINDOW_WIDTH - overlay_w) // 2
        oy = (board.WINDOW_HEIGHT - overlay_h) // 2
        
        self.time_control_overlay_rect = pygame.Rect(ox, oy, overlay_w, overlay_h)
        
        y_cursor = oy + margin
        self.time_control_options = []
        # For each option, we store the action string and the corresponding time parameters.
        # Adjust the time parameters as needed.
        for label, action in options:
            if action == 'none':
                # "None" option: no time control.
                time_params = (None, None, None)
            elif action == '1min':
                time_params = (60, 60, 0)
            elif action == '3|2':
                time_params = (180, 180, 2)
            elif action == '5min':
                time_params = (300, 300, 0)
            elif action == '10min':
                time_params = (600, 600, 0)
            elif action == '15|10':
                time_params = (900, 900, 10)
            rect = pygame.Rect(ox + margin, y_cursor, btn_w, btn_h)
            self.time_control_options.append((rect, label, *time_params))
            y_cursor += btn_h + btn_spacing

    def get_model_move(self, model, device, temperature=1.0, use_dirichlet=False, epsilon=0.25, alpha=0.3, sample=False):
        """
        Selects a move using the model with adjustable exploration.
        ...
        """
        # 1. Encode board state and get model prediction.
        board_state = self.encode_board_state()  # Shape: (num_channels, BOARD_SIZE, BOARD_SIZE)
        state_tensor = torch.tensor(board_state).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            # Assuming model outputs logits for the policy.
            logits, _ = model(state_tensor)
        logits = logits.squeeze(0)
        
        # Apply temperature scaling.
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Optionally add Dirichlet noise.
        if use_dirichlet:
            noise = np.random.dirichlet([alpha] * probs.shape[-1])
            noise_tensor = torch.tensor(noise, dtype=torch.float32, device=device)
            probs = (1 - epsilon) * probs + epsilon * noise_tensor

        # 2. Build the list of legal actions.
        board_size = board.BOARD_SIZE
        legal_actions = []
        
        # a. Standard moves (including captures and promotions).
        for r in range(board_size):
            for c in range(board_size):
                piece = self.board[r][c]
                if piece and piece.color == self.turn:
                    moves = board.get_valid_moves(piece, (r, c), self.board, self.en_passant)
                    for move in moves:
                        candidate = ("move", (r, c), move, None)
                        if self.is_move_legal(candidate):
                            if piece.type == 'P' and self.move_leads_to_promotion(piece, (r, c), move):
                                for promo in ['Q', 'R', 'B', 'N']:
                                    candidate_promo = ("move", (r, c), move, promo)
                                    if self.is_move_legal(candidate_promo):
                                        legal_actions.append(candidate_promo)
                            else:
                                legal_actions.append(candidate)
        
        # b. Gold collection actions.
        for r in range(board_size):
            for c in range(board_size):
                piece = self.board[r][c]
                if piece and piece.color == self.turn and piece.type == 'P':
                    candidate = ("collect_gold", (r, c), None, None)
                    if self.is_move_legal(candidate):
                        legal_actions.append(candidate)
        
        # c. Purchase actions.
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
            r, c = king_pos
            adjacent = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if board.in_bounds(nr, nc) and self.board[nr][nc] is None:
                        adjacent.append((nr, nc))
            purchase_squares = [sq for sq in adjacent if self.board[sq[0]][sq[1]] is None]
            for candidate_square in purchase_squares:
                for p_type in ['P', 'N', 'B', 'R', 'Q']:
                    cost = board.PIECE_COST.get(p_type)
                    candidate = ("purchase", king_pos, candidate_square, p_type)
                    if cost is not None and king.gold >= cost and self.is_move_legal(candidate):
                        legal_actions.append(candidate)
        
        # d. Transfer gold actions.
        # For each friendly piece with gold, allow transferring its entire gold to another friendly piece
        # that is on a square visible from the source piece.
        for r in range(board_size):
            for c in range(board_size):
                piece = self.board[r][c]
                if piece and piece.color == self.turn and piece.gold > 0:
                    # Get squares that this piece can "see".
                    visible_squares = board.get_visible_squares(piece, (r, c), self.board)
                    for target in visible_squares:
                        tr, tc = target
                        target_piece = self.board[tr][tc]
                        # Only allow transfer if the target square has a friendly piece.
                        if target_piece and target_piece.color == self.turn:
                            candidate = ("transfer_gold", (r, c), (tr, tc), None)
                            if self.is_move_legal(candidate):
                                legal_actions.append(candidate)
        
        # 3. Map each legal action to its corresponding index.
        action_indices = []
        for action in legal_actions:
            action_type, src, dst, purchase_type = action
            try:
                idx = self.move_to_index(action_type, src, dst, purchase_type)
                action_indices.append((idx, action))
            except Exception as e:
                # Skip any problematic action.
                pass
            #print(action_indices) #debug print to make sure the ai sees correct avilable moves           
        if not action_indices:
            # Fallback: if no legal actions, return a random move.
            return self.get_random_move()
        
        # 4. Create lists for legal indices and actions.
        legal_idx_list = [idx for idx, action in action_indices]
        legal_actions_list = [action for idx, action in action_indices]
        
        # Extract the probabilities for legal moves.
        policy_probs = probs.cpu().numpy()
        legal_probs = policy_probs[legal_idx_list]
        
        # If sampling is requested, sample based on the legal probabilities.
        if sample:
            sum_prob = legal_probs.sum()
            if sum_prob > 0:
                normalized_probs = legal_probs / sum_prob
                chosen_idx = np.random.choice(len(legal_actions_list), p=normalized_probs)
                chosen_action = legal_actions_list[chosen_idx]
            else:
                # Fallback to best move if probabilities sum to zero.
                chosen_action = legal_actions_list[np.argmax(legal_probs)]
        else:
            # Deterministic: select the action with the highest probability.
            chosen_action = legal_actions_list[np.argmax(legal_probs)]
        
        return chosen_action

    def is_move_legal(self, move):
        """
        Returns True if applying the move does not leave the king in check.
        Uses a simulation on a copied minimal game state.
        """
        simulated_game = self.copy_for_simulation()
        try:
            simulated_game.apply_move(move, simulate=True)
        except Exception as e:
            print("Simulation error:", e)
            return False
        return not simulated_game.is_in_check(self.turn, custom_board=simulated_game.board)

    def move_leads_to_promotion(self, piece, src, dst):
        """
        Determines if moving a pawn from src to dst leads to promotion.
        Assumes standard chess rules: white promotes on row 0, black on the last row.
        """
        if piece.type != 'P':
            return False
        promotion_row = 0 if piece.color == 'white' else board.BOARD_SIZE - 1
        return dst[0] == promotion_row
    
    def copy_for_simulation(self):
        """
        Returns a copy of the game state for simulation.
        Only copies picklable attributes needed for move validation.
        Non-picklable attributes (like screen, clock, and pygame.Rect objects) are omitted.
        """
        sim = Game.__new__(Game)  # create a new instance without calling __init__
    
        # Essential game state:
        sim.board = copy.deepcopy(self.board)
        sim.turn = self.turn
        sim.en_passant = self.en_passant
        
        # Purchase-related state:
        sim.purchase_mode = self.purchase_mode
        sim.purchase_options = copy.deepcopy(self.purchase_options)
        sim.purchase_selected_type = self.purchase_selected_type
        sim.valid_purchase_placement = copy.deepcopy(self.valid_purchase_placement)
        sim.purchase_king_color = self.purchase_king_color
        sim.pre_purchase_state = copy.deepcopy(self.pre_purchase_state)
        
        # Promotion-related state:
        sim.promotion_mode = self.promotion_mode
        sim.promotion_options = copy.deepcopy(self.promotion_options)
        sim.promotion_pos = self.promotion_pos
        sim.promotion_color = self.promotion_color
        sim.pre_promotion_state = copy.deepcopy(self.pre_promotion_state)
        
        # Pause/menu state:
        sim.pause_menu = self.pause_menu
        sim.pause_menu_options = copy.deepcopy(self.pause_menu_options)
        sim.show_ai_submenu = self.show_ai_submenu
        
        # Game outcome and logging:
        sim.game_over = self.game_over
        sim.winner = self.winner
        sim.move_log = self.move_log[:]  # shallow copy works for list of strings
        sim.error_message = self.error_message
        sim.move_log_scroll = self.move_log_scroll
        sim.position_history = copy.deepcopy(self.position_history)
        
        # Move validation helpers:
        sim.valid_move_squares = copy.deepcopy(self.valid_move_squares)
        sim.valid_capture_squares = copy.deepcopy(self.valid_capture_squares)
        sim.valid_gold_transfer_squares = copy.deepcopy(self.valid_gold_transfer_squares)
        sim.halfmove_clock = self.halfmove_clock
        
        # Time control (excluding non-picklable overlays/chess_clock)
        sim.time_control_mode = self.time_control_mode
        sim.time_control_options = copy.deepcopy(self.time_control_options)
        sim.chess_clock = None  # Set chess_clock to a default (None) for simulation.
        
        # AI related state:
        sim.ai_enabled = self.ai_enabled
        sim.ai_color = self.ai_color
        sim.selected_piece_pos = self.selected_piece_pos
        
        # Exclude non-picklable objects:
        # sim.screen, sim.clock, sim.font, sim.purchase_overlay_rect, sim.promotion_overlay_rect,
        # and sim.time_control_overlay_rect are omitted.
        
        return sim

    def encode_board_state(self):
        """
        Encodes the current board state into a numpy array of shape 
        (13, BOARD_SIZE, BOARD_SIZE).
        
        Channels 0-5: Presence of white pieces:
            0: White King, 1: White Queen, 2: White Rook, 
            3: White Bishop, 4: White Knight, 5: White Pawn.
        Channels 6-11: Presence of black pieces:
            6: Black King, 7: Black Queen, 8: Black Rook, 
            9: Black Bishop, 10: Black Knight, 11: Black Pawn.
        Channel 12: Gold amount on each square (raw value, or normalized if preferred).
        """
        board_size = board.BOARD_SIZE
        # Create a tensor for piece presence (12 channels)
        piece_tensor = np.zeros((12, board_size, board_size), dtype=np.float32)
        # Create a separate tensor for gold values (1 channel)
        gold_tensor = np.zeros((1, board_size, board_size), dtype=np.float32)
        
        # Mapping from (color, piece type) to channel index
        piece_to_channel = {
            ('white', 'K'): 0,
            ('white', 'Q'): 1,
            ('white', 'R'): 2,
            ('white', 'B'): 3,
            ('white', 'N'): 4,
            ('white', 'P'): 5,
            ('black', 'K'): 6,
            ('black', 'Q'): 7,
            ('black', 'R'): 8,
            ('black', 'B'): 9,
            ('black', 'N'): 10,
            ('black', 'P'): 11,
        }
        
        for r in range(board_size):
            for c in range(board_size):
                piece = self.board[r][c]
                if piece:
                    key = (piece.color, piece.type)
                    channel = piece_to_channel.get(key)
                    if channel is not None:
                        piece_tensor[channel, r, c] = 1.0
                    # Assume that piece.gold holds the gold value (you can normalize if needed)
                    gold_tensor[0, r, c] = piece.gold
                    
        # Concatenate the piece tensor and gold tensor along the channel axis
        encoded_state = np.concatenate([piece_tensor, gold_tensor], axis=0)
        return encoded_state
    
    def get_training_example(self):
        """
        Returns a tuple (board_state, policy_target, player) where:
        - board_state is a numpy array encoding the current board configuration.
        - policy_target is a 1D numpy array representing a probability distribution over the entire action space.
        - player is a string ("white" or "black") representing the current player.
        
        The action space includes:
        - Standard moves (4096 indices for an 8x8 board)
        - 1 index for gold collection
        - 5 * 64 indices for purchase actions
        - 4096 indices for gold transfers (from a selected piece to a target square)
        """
        # Encode the board state (assume your encode_board_state() includes the gold channel)
        board_state = self.encode_board_state()
        
        board_size = board.BOARD_SIZE
        standard_moves = board_size * board_size * board_size * board_size  # 4096
        purchase_actions = 5 * (board_size * board_size)  # 320
        transfer_actions = board_size * board_size * board_size * board_size  # 4096
        total_actions = standard_moves + 1 + purchase_actions + transfer_actions  # standard + collect_gold + purchase + transfer_gold
        
        policy_target = np.zeros(total_actions, dtype=np.float32)
        legal_actions = []  # Each element is a tuple: (action_type, src, dst, purchase_type)
        
        # 1. Legal standard moves.
        for r in range(board_size):
            for c in range(board_size):
                piece = self.board[r][c]
                if piece and piece.color == self.turn:
                    moves = board.get_valid_moves(piece, (r, c), self.board, self.en_passant)
                    for move in moves:
                        if self.simulate_move_is_safe((r, c), move):
                            legal_actions.append(("move", (r, c), move, None))
                            
        # 2. Gold collection action.
        # Add the collect_gold action once per turn (if applicable by your rules).
        legal_actions.append(("collect_gold", None, None, None))
        
        # 3. Purchase actions (if in purchase mode).
        if self.purchase_mode:
            king_pos = self.selected_piece_pos  # or another method to determine the king’s position
            if king_pos:
                for r in range(board_size):
                    for c in range(board_size):
                        if (r, c) in self.valid_purchase_placement:
                            for p_type in ['P', 'N', 'B', 'R', 'Q']:
                                legal_actions.append(("purchase", None, (r, c), p_type))
                                
        # 4. Gold transfer actions.
        # If there's a selected piece with gold > 0, add legal transfers.
        if self.selected_piece_pos is not None:
            src = self.selected_piece_pos
            piece = self.board[src[0]][src[1]]
            if piece and piece.gold > 0 and self.valid_gold_transfer_squares:
                for target in self.valid_gold_transfer_squares:
                    legal_actions.append(("transfer_gold", src, target, None))
        
        # Assign uniform probability to each legal action.
        if legal_actions:
            probability = 1.0 / len(legal_actions)
            for action in legal_actions:
                action_type, src, dst, purchase_type = action
                index = self.move_to_index(action_type, src, dst, purchase_type)
                policy_target[index] = probability
                
        # Return the encoded board state, the policy target vector, and the current player's perspective.
        return board_state, policy_target, self.turn

    def apply_move(self, move, simulate=False):
        """
        Applies a move according to its action type.
        If simulate=True, the move is applied on the board but doesn't
        call end_turn() or update logs, so you can use it to validate moves.
        """
        action_type, src, dst, purchase_type = move

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

        elif action_type == "collect_gold":
            if src is not None:
                piece = self.board[src[0]][src[1]]
                if piece:
                    piece.gold += 1
                    self.move_log.append(f"{piece.type}+{board.square_to_notation(src[0], src[1])} (gold collected)")
                    self.halfmove_clock = 0
                else:
                    self.error_message = "No piece found to collect gold."
            else:
                self.error_message = "Invalid source for gold collection."
            if not simulate:
                self.end_turn()
                #print("turn ended now ", self.turn, "'s turn")
        elif action_type == "purchase":
            # Get the current king.
            king_pos, king = None, None
            for r in range(board.BOARD_SIZE):
                for c in range(board.BOARD_SIZE):
                    piece = self.board[r][c]
                    if piece and piece.color == self.turn and piece.type == 'K':
                        king_pos, king = (r, c), piece
                        break
                if king_pos:
                    break

            if not king:
                self.error_message = "No king available for purchase action."
            else:
                cost = board.PIECE_COST.get(purchase_type)
                if cost is None:
                    self.error_message = "Invalid purchase type."
                elif king.gold < cost:
                    self.error_message = "Not enough gold for purchase."
                else:
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

            if not simulate and not self.error_message:
                self.end_turn()
                #print("turn ended now ", self.turn, "'s turn")

        elif action_type == "transfer_gold":
            if src is not None:
                src_piece = self.board[src[0]][src[1]]
                target_piece = self.board[dst[0]][dst[1]]
                if src_piece and target_piece:
                    target_piece.gold += src_piece.gold
                    self.move_log.append(f"{src_piece.type}G{board.square_to_notation(dst[0], dst[1])}")
                    src_piece.gold = 0
                    self.halfmove_clock += 1
                else:
                    self.error_message = "Invalid source or target for gold transfer."
            else:
                self.error_message = "No source provided for gold transfer."
            if not simulate:
                self.end_turn()
                #print("turn ended now ", self.turn, "'s turn")
        else:
            raise ValueError(f"Unknown action type: {action_type}")

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
        
        # White always moves first.
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
        self.max_moves_reached = False  
        # Integrate AI settings (set via the pause menu AI submenu)
        # Expect that, if AI is enabled, self.ai_enabled and self.ai_color have been set.
        if hasattr(self, 'ai_enabled') and self.ai_enabled:
            # Ensure ai_color is valid; default to black if missing.
            if not hasattr(self, 'ai_color') or self.ai_color not in ['white', 'black']:
                self.ai_color = 'black'
            #print(f"New game started: AI enabled. AI plays as {self.ai_color}.")
        #else:
            #print("New game started: Human vs Human.")

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

    def purchase_piece(self, dst, purchase_type):
        """
        Places a new piece of the given purchase_type on the board at the destination (dst).
        Assumes that the destination is a valid empty square.
        """
        r, c = dst
        # Create a new piece with the same color as the current turn (i.e., the king's color).
        new_piece = board.Piece(purchase_type, self.turn)
        self.board[r][c] = new_piece

    def move_piece(self, src, dst):
        orig_pos = self.selected_piece_pos
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
        #print("turn ended now ", self.turn, "'s turn")

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
        #print("tun ended now ", self.turn, "'s turn")

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
        margin = 10
        btn_h = 40
        btn_spacing = 5
        # Main pause menu: New Game and Quit.
        options = [('New Game', 'new_game'), ('Quit', 'quit')]
        
        # Calculate minimum button width.
        text_surfaces = [self.font.render(label, True, (0, 0, 0)) for label, _ in options]
        min_btn_width = max(surf.get_width() for surf in text_surfaces) + margin * 4
        btn_w = max(200, min_btn_width)
        
        overlay_w = btn_w + margin * 2
        overlay_h = (btn_h * len(options)) + (btn_spacing * (len(options) - 1)) + margin * 2
        
        ox = (board.WINDOW_WIDTH - overlay_w) // 2
        oy = (board.WINDOW_HEIGHT - overlay_h) // 2
        
        self.pause_overlay_rect = pygame.Rect(ox, oy, overlay_w, overlay_h)
        
        y_cursor = oy + margin
        self.pause_menu_options = []
        for label, action in options:
            rect = pygame.Rect(ox + margin, y_cursor, btn_w, btn_h)
            self.pause_menu_options.append((rect, action, label))
            y_cursor += btn_h + btn_spacing

    def create_new_game_options(self):
        margin = 10
        btn_h = 40
        btn_spacing = 5
        # New Game options: local hotseat and AI options.
        options = [
            ('Local Hotseat', 'hotseat'),
            ('Play as White', 'ai_black'),
            ('Play as Black', 'ai_white'),
            ('Random (AI)', 'ai_random')
        ]
        
        text_surfaces = [self.font.render(label, True, (0, 0, 0)) for label, _ in options]
        min_btn_width = max(surf.get_width() for surf in text_surfaces) + margin * 4
        btn_w = max(200, min_btn_width)
        
        overlay_w = btn_w + margin * 2
        overlay_h = (btn_h * len(options)) + (btn_spacing * (len(options) - 1)) + margin * 2
        
        ox = (board.WINDOW_WIDTH - overlay_w) // 2
        oy = (board.WINDOW_HEIGHT - overlay_h) // 2
        
        self.new_game_overlay_rect = pygame.Rect(ox, oy, overlay_w, overlay_h)
        
        y_cursor = oy + margin
        self.new_game_options = []
        for label, action in options:
            rect = pygame.Rect(ox + margin, y_cursor, btn_w, btn_h)
            self.new_game_options.append((rect, action, label))
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

                # Highlight selection.
                if self.selected_piece_pos == (r, c):
                    select_overlay = pygame.Surface((board.SQUARE_SIZE, board.SQUARE_SIZE), pygame.SRCALPHA)
                    select_overlay.fill(board.SELECT_HIGHLIGHT_COLOR)
                    self.screen.blit(select_overlay, (rect_x, rect_y))

                # Highlight valid moves, captures, gold transfers.
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
                        # Draw gold circle.
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

        # File labels.
        for c in range(board.BOARD_SIZE):
            label = self.get_file_label(c)
            t_surf = self.font.render(label, True, (0, 0, 0))
            t_rect = t_surf.get_rect()
            t_rect.centerx = board_x + c * board.SQUARE_SIZE + (board.SQUARE_SIZE // 2)
            t_rect.centery = board_y + board.BOARD_HEIGHT + (board.MARGIN_BOTTOM // 2)
            self.screen.blit(t_surf, t_rect)

        # Rank labels.
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

        # Purchase overlay.
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

        # Promotion overlay.
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

        # Pause menu.
        if self.pause_menu:
            pygame.draw.rect(self.screen, (180, 180, 180), self.pause_overlay_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), self.pause_overlay_rect, 2)
            for rect_, action, label in self.pause_menu_options:
                pygame.draw.rect(self.screen, (150, 150, 150), rect_)
                ts = self.font.render(label, True, (0, 0, 0))
                self.screen.blit(ts, ts.get_rect(center=rect_.center))
        if self.pause_menu:
                # Draw New Game submenu overlay if active.
                if hasattr(self, 'new_game_overlay_rect') and self.new_game_overlay_rect:
                    pygame.draw.rect(self.screen, (180, 180, 180), self.new_game_overlay_rect)
                    pygame.draw.rect(self.screen, (0, 0, 0), self.new_game_overlay_rect, 2)
                    for rect, action, label in self.new_game_options:
                        pygame.draw.rect(self.screen, (150,150,150), rect)
                        ts = self.font.render(label, True, (0, 0, 0))
                        self.screen.blit(ts, ts.get_rect(center=rect.center))
                # Else if the Time Control submenu is active, draw that.
                elif hasattr(self, 'time_control_overlay_rect') and self.time_control_overlay_rect:
                    pygame.draw.rect(self.screen, (180, 180, 180), self.time_control_overlay_rect)
                    pygame.draw.rect(self.screen, (0, 0, 0), self.time_control_overlay_rect, 2)
                    for rect, label, wsecs, bsecs, inc in self.time_control_options:
                        pygame.draw.rect(self.screen, (150,150,150), rect)
                        ts = self.font.render(label, True, (0, 0, 0))
                        self.screen.blit(ts, ts.get_rect(center=rect.center))
                # Otherwise, draw the main pause menu.
                else:
                    pygame.draw.rect(self.screen, (220,220,220), self.pause_overlay_rect)
                    pygame.draw.rect(self.screen, (0, 0, 0), self.pause_overlay_rect, 2)
                    for rect, action, label in self.pause_menu_options:
                        pygame.draw.rect(self.screen, (200,200,200), rect)
                        ts = self.font.render(label, True, (0, 0, 0))
                        self.screen.blit(ts, ts.get_rect(center=rect.center))

        # Draw turn and time info.
        if self.chess_clock:
            white_str = format_time(self.chess_clock.white_time)
            black_str = format_time(self.chess_clock.black_time)
            turn_display = f"Turn: {self.turn} | White: {white_str}  Black: {black_str}"
        else:
            # Display default clock values or a prompt that time control is not active.
            turn_display = f"Turn: {self.turn} | Clock: N/A"
        turn_txt = self.font.render(turn_display, True, (0, 0, 0))
        self.screen.blit(turn_txt, (board.MARGIN_LEFT + 10, board.BOARD_HEIGHT + board.MARGIN_BOTTOM + 10))

    def handle_board_click(self, pos):
        # handle promotion overlay        
        if self.promotion_mode:
            if self.promotion_overlay_rect and self.promotion_overlay_rect.collidepoint(pos):
                # Player selected a promotion option.
                for rect_, p_type in self.promotion_options:
                    if rect_.collidepoint(pos):
                        pr, pc = self.promotion_pos
                        # Update the pawn to the chosen piece type.
                        self.board[pr][pc].type = p_type
                        self.move_log[-1] = self.move_log[-1] + f"={p_type}"
                        self.promotion_mode = False
                        self.promotion_pos = None
                        self.promotion_color = None
                        self.clear_valid_actions()
                        self.end_turn()
                        #print("turn ended now ", self.turn, "'s turn")
                        return
            else:
                # Player clicked off the promotion overlay: cancel the promotion.
                if self.pre_promotion_state:
                    self.board = copy.deepcopy(self.pre_promotion_state['board'])
                    self.move_log = self.pre_promotion_state['move_log'][:]
                    # Restore the pawn to its original position stored in 'selected'
                    orig_pos = self.pre_promotion_state['selected']
                    if orig_pos:
                        pr, pc = orig_pos
                        # Find the pawn at the promotion square.
                        promo_pos = self.pre_promotion_state['promotion_pos']
                        pawn = self.board[promo_pos[0]][promo_pos[1]]
                        self.board[pr][pc] = pawn  # move pawn back
                        self.board[promo_pos[0]][promo_pos[1]] = None
                    self.turn = self.pre_promotion_state['turn']
                    self.en_passant = self.pre_promotion_state['en_passant']
                    self.halfmove_clock = self.pre_promotion_state['halfmove_clock']
                self.selected_piece_pos = None
                self.promotion_mode = False
                self.promotion_pos = None
                self.promotion_color = None
                self.clear_valid_actions()
                return

        board_x = board.MARGIN_LEFT
        board_y = 0


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
                #print("turn ended now ", self.turn, "'s turn")
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
                    #print("turn ended now ", self.turn, "'s turn")
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
                        #print("turn ended now ", self.turn, "'s turn")
                else:
                    p2 = self.board[row][col]
                    if p2 and p2.color == self.turn:
                        self.selected_piece_pos = (row, col)
                        self.update_valid_actions(row, col)
                    else:
                        self.selected_piece_pos = None
                        self.clear_valid_actions()

    def process_pause_menu_click(self, pos):
        """
        Process clicks on any active pause menu overlays:
        - New Game submenu (local hotseat or AI)
        - Time control submenu (including "None")
        - Main pause menu (New Game / Quit)
        """
        # Process New Game submenu if active.
        if hasattr(self, 'new_game_overlay_rect') and self.new_game_overlay_rect:
            for rect, action, label in self.new_game_options:
                if rect.collidepoint(pos):
                    if action == 'hotseat':
                        self.ai_enabled = False
                        self.new_game()  # Start local hotseat game.
                        self.new_game_overlay_rect = None
                        self.pause_menu = False
                    elif action in ['ai_black', 'ai_white', 'ai_random']:
                        self.ai_enabled = True
                        if action == 'ai_black':
                            chosen_color = 'black'
                        elif action == 'ai_white':
                            chosen_color = 'white'
                        else:
                            chosen_color = random.choice(['black', 'white'])
                        self.ai_color = chosen_color
                        # Optionally, set the AI instance's color as well.
                        # Now show time control submenu.
                        self.create_time_control_submenu_options()
                        self.new_game_overlay_rect = None
                    return

        # Process Time Control submenu if active.
        elif hasattr(self, 'time_control_overlay_rect') and self.time_control_overlay_rect:
            for rect, label, wsecs, bsecs, inc in self.time_control_options:
                if rect.collidepoint(pos):
                    if label == 'None':
                        self.chess_clock = None
                    else:
                        self.chess_clock = ChessClock(wsecs, bsecs, inc)
                        self.chess_clock.start('white')
                        #print("Chess clock started:", self.chess_clock.white_time, self.chess_clock.black_time)
                    # Start new game after time control selection.
                    self.new_game()
                    self.pause_menu = False
                    self.time_control_overlay_rect = None
                    return

        # Otherwise, process the main pause menu.
        else:
            if self.pause_overlay_rect and self.pause_overlay_rect.collidepoint(pos):
                for rect, action, label in self.pause_menu_options:
                    if rect.collidepoint(pos):
                        if action == 'new_game':
                            self.create_new_game_options()
                        elif action == 'quit':
                            pygame.quit()
                            sys.exit()
                        break
            else:
                # Click outside the pause overlay closes it.
                self.pause_menu = False
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
        # New insufficient material check.
        if self.has_insufficient_material():
            self.game_over = True
            self.winner = "draw"
            return

    def update(self):
        self.screen.fill((0,0,0))

        # If we have a clock, update it
        if self.chess_clock and not self.game_over:
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
        
       # Draw game state overlays only if pause menu is not active.
        if self.game_over and not self.pause_menu:
            s = pygame.Surface((board.WINDOW_WIDTH, board.WINDOW_HEIGHT))
            s.set_alpha(128)
            s.fill((128, 128, 128))
            self.screen.blit(s, (0, 0))
            
            font = pygame.font.Font(None, 74)
            if self.winner and self.winner != "draw":
                text = font.render(f"{self.winner} wins!", True, (255, 255, 255))
                print(f"{self.winner} wins!")
            else:
                text = font.render("Draw!", True, (255, 255, 255))
                print("Draw!")
            text_rect = text.get_rect(center=(board.WINDOW_WIDTH/2, board.WINDOW_HEIGHT/2))
            self.screen.blit(text, text_rect)
            self.clock.tick(30)

    def move_to_index(self, action_type, src, dst, purchase_type):
        board_size = board.BOARD_SIZE  # typically 8
        standard_move_space = board_size * board_size * board_size * board_size  # 4096

        if action_type == "move":
            # Map a standard move: (src, dst) -> index in [0, 4095]
            src_index = src[0] * board_size + src[1]
            dst_index = dst[0] * board_size + dst[1]
            return src_index * (board_size * board_size) + dst_index

        elif action_type == "collect_gold":
            # Reserve the index immediately after standard moves.
            return standard_move_space  # index 4096

        elif action_type == "purchase":
            # Reserve a block for purchase actions.
            purchase_types = ['P', 'N', 'B', 'R', 'Q']
            if purchase_type not in purchase_types:
                raise ValueError("Invalid purchase type")
            purchase_index = purchase_types.index(purchase_type)
            purchase_block_start = standard_move_space + 1  # after collect_gold
            # Map the destination square for purchase.
            dst_index = dst[0] * board_size + dst[1]
            return purchase_block_start + purchase_index * (board_size * board_size) + dst_index

        elif action_type == "transfer_gold":
            # Reserve a block for gold transfers after the previous actions.
            purchase_block_size = 5 * (board_size * board_size)
            transfer_block_start = standard_move_space + 1 + purchase_block_size
            # Map the gold transfer as a function of both the source and target squares.
            src_index = src[0] * board_size + src[1]
            dst_index = dst[0] * board_size + dst[1]
            return transfer_block_start + src_index * (board_size * board_size) + dst_index

        else:
            raise ValueError("Unknown action type")

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

    def is_game_over(self):
        return self.game_over

    def get_outcome(self):
        if not self.game_over:
            return None  # Game not finished yet
        if self.winner == "draw":
            # Check if the draw resulted from reaching max moves
            if hasattr(self, "max_moves_reached") and self.max_moves_reached:
                return -5    # Strongly discourage a forced draw
            else:
                return -0.5  # Only slightly discourage a natural draw
        # For wins, reward white wins strongly and punish losses moderately.
        return 5 if self.winner == "white" else -3
        