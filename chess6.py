import random
import copy

# ======================
# BOARD INITIALIZATION
# ======================

def initial_board():
    return [
        ['br','bn','bb','bq','bk','bb','bn','br'],
        ['bp','bp','bp','bp','bp','bp','bp','bp'],
        ['0','0','0','0','0','0','0','0'],
        ['0','0','0','0','0','0','0','0'],
        ['0','0','0','0','0','0','0','0'],
        ['0','0','0','0','0','0','0','0'],
        ['wp','wp','wp','wp','wp','wp','wp','wp'],
        ['wr','wn','wb','wq','wk','wb','wn','wr']
    ]

def print_board(board):
    print("   a   b   c   d   e   f   g   h")
    for i, row in enumerate(board):
        print(8 - i, end=" ")
        for piece in row:
            print(f"{piece:^3}", end=" ")
        print(8 - i)
    print("   a   b   c   d   e   f   g   h")


# ======================
# BOARD HASHING
# ======================

def board_to_hash(board):
    """Convert board state to a hashable string for memoization."""
    return ''.join(''.join(row) for row in board)


# ======================
# MOVE LOGIC
# ======================

def algebraic_to_indices(move):
    if len(move) != 4:
        return None
    try:
        start_col, start_row, end_col, end_row = move
        if start_col not in 'abcdefgh' or end_col not in 'abcdefgh':
            return None
        if start_row not in '12345678' or end_row not in '12345678':
            return None
        return 8 - int(start_row), ord(start_col) - 97, 8 - int(end_row), ord(end_col) - 97
    except:
        return None


def is_valid_move(board, move, color):
    if len(move) != 4:
        return False

    res = algebraic_to_indices(move)
    if not res:
        return False

    sr, sc, er, ec = res
    if sr not in range(8) or er not in range(8) or sc not in range(8) or ec not in range(8):
        return False

    piece = board[sr][sc]
    if piece == '0' or piece[0] != color:
        return False

    target = board[er][ec]
    if target != '0' and target[0] == color:
        return False

    ptype = piece[1].lower()
    dr, dc = er - sr, ec - sc

    # Pawn movement
    if ptype == 'p':
        direction = -1 if color == 'w' else 1
        start_row = 6 if color == 'w' else 1

        if dc == 0 and board[er][ec] == '0':
            if dr == direction:
                return True
            if sr == start_row and dr == 2 * direction and board[sr + direction][sc] == '0':
                return True
        if abs(dc) == 1 and dr == direction and board[er][ec] != '0' and board[er][ec][0] != color:
            return True
        return False

    # Rook movement
    if ptype == 'r':
        if sr != er and sc != ec:
            return False
        step_r = (er - sr) and (1 if er > sr else -1)
        step_c = (ec - sc) and (1 if ec > sc else -1)
        r, c = sr + step_r if sr != er else sr, sc + step_c if sc != ec else sc
        while (r != er or c != ec):
            if board[r][c] != '0':
                return False
            if sr != er: r += step_r
            if sc != ec: c += step_c
        return True

    # Bishop movement
    if ptype == 'b':
        if abs(dr) != abs(dc):
            return False
        step_r = 1 if dr > 0 else -1
        step_c = 1 if dc > 0 else -1
        r, c = sr + step_r, sc + step_c
        while (r != er and c != ec):
            if board[r][c] != '0':
                return False
            r += step_r
            c += step_c
        return True

    # Queen movement
    if ptype == 'q':
        if sr == er or sc == ec:
            step_r = (er - sr) and (1 if er > sr else -1)
            step_c = (ec - sc) and (1 if ec > sc else -1)
            r, c = sr + step_r if sr != er else sr, sc + step_c if sc != ec else sc
            while (r != er or c != ec):
                if board[r][c] != '0':
                    return False
                if sr != er: r += step_r
                if sc != ec: c += step_c
            return True
        elif abs(dr) == abs(dc):
            step_r = 1 if dr > 0 else -1
            step_c = 1 if dc > 0 else -1
            r, c = sr + step_r, sc + step_c
            while (r != er and c != ec):
                if board[r][c] != '0':
                    return False
                r += step_r
                c += step_c
            return True
        return False

    # Knight movement
    if ptype == 'n':
        return (abs(dr), abs(dc)) in [(2, 1), (1, 2)]

    # King movement
    if ptype == 'k':
        return abs(dr) <= 1 and abs(dc) <= 1

    return False


def make_move(board, move, promotion_piece=None):
    sr, sc, er, ec = algebraic_to_indices(move)
    new_board = copy.deepcopy(board)
    piece = new_board[sr][sc]
    
    # Check for pawn promotion
    if piece[1].lower() == 'p':
        if (piece[0] == 'w' and er == 0) or (piece[0] == 'b' and er == 7):
            if promotion_piece:
                new_board[er][ec] = piece[0] + promotion_piece.lower()
            else:
                # Default to queen if no promotion piece specified
                new_board[er][ec] = piece[0] + 'q'
            new_board[sr][sc] = '0'
            return new_board
    
    new_board[er][ec] = new_board[sr][sc]
    new_board[sr][sc] = '0'
    return new_board

def check_for_winner(board):
    has_white_king = any('wk' in row for row in board)
    has_black_king = any('bk' in row for row in board)
    if not has_white_king:
        print("Black wins! The white king was captured.")
        return "black"
    if not has_black_king:
        print("White wins! The black king was captured.")
        return "white"
    return None

# ======================
# OPENING BOOK
# ======================

# Format: tuple of move sequence -> response move
openings = {
    # Italian Game
    ('e2e4',): 'e7e5',
    ('e2e4', 'e7e5', 'g1f3'): 'b8c6',
    ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4'): 'f8c5',
    ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5', 'd2d3'): 'g8f6',
    
    # Ruy Lopez
    ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5'): 'a7a6',
    ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4'): 'g8f6',
    ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'g8f6', 'e1g1'): 'f8e7',
    
    # Sicilian Defense
    ('e2e4', 'c7c5'): 'g1f3',
    ('e2e4', 'c7c5', 'g1f3'): 'd7d6',
    ('e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4'): 'c5d4',
    ('e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4', 'c5d4', 'f3d4'): 'g8f6',
    
    # French Defense
    ('e2e4', 'e7e6'): 'd2d4',
    ('e2e4', 'e7e6', 'd2d4'): 'd7d5',
    ('e2e4', 'e7e6', 'd2d4', 'd7d5', 'b1c3'): 'g8f6',
    
    # Caro-Kann Defense
    ('e2e4', 'c7c6'): 'd2d4',
    ('e2e4', 'c7c6', 'd2d4'): 'd7d5',
    ('e2e4', 'c7c6', 'd2d4', 'd7d5', 'b1c3'): 'd5e4',
    
    # Queen's Gambit
    ('d2d4',): 'd7d5',
    ('d2d4', 'd7d5', 'c2c4'): 'e7e6',
    ('d2d4', 'd7d5', 'c2c4', 'e7e6', 'b1c3'): 'g8f6',
    ('d2d4', 'd7d5', 'c2c4', 'e7e6', 'b1c3', 'g8f6', 'g1f3'): 'f8e7',
    
    # Queen's Gambit Declined
    ('d2d4', 'd7d5', 'c2c4', 'c7c6'): 'g1f3',
    ('d2d4', 'd7d5', 'c2c4', 'c7c6', 'g1f3'): 'g8f6',
    
    # King's Indian Defense
    ('d2d4', 'g8f6'): 'c2c4',
    ('d2d4', 'g8f6', 'c2c4'): 'g7g6',
    ('d2d4', 'g8f6', 'c2c4', 'g7g6', 'b1c3'): 'f8g7',
    
    # Nimzo-Indian Defense
    ('d2d4', 'g8f6', 'c2c4', 'e7e6'): 'b1c3',
    ('d2d4', 'g8f6', 'c2c4', 'e7e6', 'b1c3'): 'f8b4',
}

def get_opening_move(history):
    """Check if current position matches any opening book line."""
    # Convert history to tuple for lookup
    history_tuple = tuple(history)
    
    # Check if exact sequence exists in opening book
    if history_tuple in openings:
        return openings[history_tuple]
    
    return None


# ======================
# POSITIONAL EVALUATION
# ======================

piece_values = {
    'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000
}

pawn_table = [
    [0,0,0,0,0,0,0,0],
    [50,50,50,50,50,50,50,50],
    [10,10,20,30,30,20,10,10],
    [5,5,10,25,25,10,5,5],
    [0,0,0,20,20,0,0,0],
    [5,-5,-10,0,0,-10,-5,5],
    [5,10,10,-20,-20,10,10,5],
    [0,0,0,0,0,0,0,0]
]
knight_table = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,0,5,5,0,-20,-40],
    [-30,5,10,15,15,10,5,-30],
    [-30,0,15,20,20,15,0,-30],
    [-30,5,15,20,20,15,5,-30],
    [-30,0,10,15,15,10,0,-30],
    [-40,-20,0,0,0,0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
]
bishop_table = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,0,0,0,0,0,0,-10],
    [-10,0,5,10,10,5,0,-10],
    [-10,5,5,10,10,5,5,-10],
    [-10,0,10,10,10,10,0,-10],
    [-10,10,10,10,10,10,10,-10],
    [-10,5,0,0,0,0,5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
]
rook_table = [
    [0,0,0,5,5,0,0,0],
    [-5,0,0,0,0,0,0,-5],
    [-5,0,0,0,0,0,0,-5],
    [-5,0,0,0,0,0,0,-5],
    [-5,0,0,0,0,0,0,-5],
    [-5,0,0,0,0,0,0,-5],
    [5,10,10,10,10,10,10,5],
    [0,0,0,0,0,0,0,0]
]
queen_table = [
    [-20,-10,-10,-5,-5,-10,-10,-20],
    [-10,0,0,0,0,0,0,-10],
    [-10,0,5,5,5,5,0,-10],
    [-5,0,5,5,5,5,0,-5],
    [0,0,5,5,5,5,0,-5],
    [-10,5,5,5,5,5,0,-10],
    [-10,0,5,0,0,0,0,-10],
    [-20,-10,-10,-5,-5,-10,-10,-20]
]
king_table = [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [20,20,0,0,0,0,20,20],
    [20,30,10,0,0,10,30,20]
]

piece_tables = {
    'p': pawn_table,
    'n': knight_table,
    'b': bishop_table,
    'r': rook_table,
    'q': queen_table,
    'k': king_table
}

def evaluate_king_safety(board, color):
    safety_score = 0
    king_pos = None
    for r in range(8):
        for c in range(8):
            if board[r][c] == f"{color}k":
                king_pos = (r, c)
                break
        if king_pos:
            break
    
    if not king_pos:
        return 0
    
    kr, kc = king_pos
    direction = -1 if color == 'w' else 1
    
    for dc in [-1, 0, 1]:
        check_col = kc + dc
        if 0 <= check_col < 8:
            for dr in [direction, direction * 2]:
                check_row = kr + dr
                if 0 <= check_row < 8:
                    if board[check_row][check_col] == f"{color}p":
                        safety_score += 10
    
    if 2 <= kc <= 5:
        safety_score -= 20
    
    return safety_score

def evaluate_development(board, color):
    back_rank = 7 if color == 'w' else 0
    development_score = 0

    for sc in range(8):
        piece = board[back_rank][sc]
        if piece != '0' and piece[0] == color:
            ptype = piece[1].lower()
            if ptype in ['n', 'b']:
                development_score -= 15

    return development_score

def evaluate_mobility(board, color):
    legal_moves = get_all_legal_moves(board, color)
    return len(legal_moves) * 5

def evaluate_center_control(board, color):
    center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
    center_score = 0

    for sr in range(8):
        for sc in range(8):
            piece = board[sr][sc]
            if piece == '0' or piece[0] != color:
                continue

            for cr, cc in center_squares:
                if sr == cr and sc == cc:
                    center_score += 20
                else:
                    if 0 <= sr < 8 and 0 <= sc < 8 and 0 <= cr < 8 and 0 <= cc < 8:
                        move = f"{chr(sc + 97)}{8 - sr}{chr(cc + 97)}{8 - cr}"
                        if is_valid_move(board, move, color):
                            center_score += 10

    return center_score

def detect_forks(board, color):
    fork_score = 0
    enemy_color = 'b' if color == 'w' else 'w'

    for sr in range(8):
        for sc in range(8):
            piece = board[sr][sc]
            if piece == '0' or piece[0] != color:
                continue

            attacked_pieces = []
            for er in range(8):
                for ec in range(8):
                    target = board[er][ec]
                    if target != '0' and target[0] == enemy_color:
                        if 0 <= sr < 8 and 0 <= sc < 8 and 0 <= er < 8 and 0 <= ec < 8:
                            move = f"{chr(sc+97)}{8-sr}{chr(ec+97)}{8-er}"
                            if is_valid_move(board, move, color):
                                attacked_pieces.append(target)

            if len(attacked_pieces) >= 2:
                fork_value = sum(piece_values[p[1]] for p in attacked_pieces)
                fork_score += fork_value * 0.3

    return fork_score

def evaluate_pawn_structure(board, color):
    structure_score = 0
    pawn_columns = {i: [] for i in range(8)}
    
    for r in range(8):
        for c in range(8):
            if board[r][c] == f"{color}p":
                pawn_columns[c].append(r)
    
    for col, pawns in pawn_columns.items():
        if len(pawns) > 1:
            structure_score -= 20 * (len(pawns) - 1)
        
        if len(pawns) > 0:
            has_neighbor = False
            for adj_col in [col - 1, col + 1]:
                if 0 <= adj_col < 8 and len(pawn_columns[adj_col]) > 0:
                    has_neighbor = True
                    break
            if not has_neighbor:
                structure_score -= 15
    
    return structure_score

def evaluate_board_advanced(board):
    score = 0

    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece == '0':
                continue
            color = piece[0]
            ptype = piece[1]
            value = piece_values.get(ptype, 0)
            table = piece_tables.get(ptype.lower(), [[0]*8 for _ in range(8)])
            table_bonus = table[r][c]
            if color == 'w':
                score += value + table_bonus
            else:
                score -= value + table_bonus

    score += detect_forks(board, 'w')
    score += evaluate_center_control(board, 'w')
    score += evaluate_mobility(board, 'w')
    score += evaluate_development(board, 'w')
    score += evaluate_king_safety(board, 'w')
    score += evaluate_pawn_structure(board, 'w')

    score -= detect_forks(board, 'b')
    score -= evaluate_center_control(board, 'b')
    score -= evaluate_mobility(board, 'b')
    score -= evaluate_development(board, 'b')
    score -= evaluate_king_safety(board, 'b')
    score -= evaluate_pawn_structure(board, 'b')

    return score


# ======================
# AI MOVE SELECTION WITH MEMOIZATION
# ======================

# Global memoization cache - persists across all moves
global_memo = {}

def get_all_legal_moves(board, color):
    moves = []
    for sr in range(8):
        for sc in range(8):
            if board[sr][sc] != '0' and board[sr][sc][0] == color:
                for er in range(8):
                    for ec in range(8):
                        move = f"{chr(sc+97)}{8-sr}{chr(ec+97)}{8-er}"
                        if is_valid_move(board, move, color):
                            moves.append(move)
    return moves

def minimax(board, depth, maximizing_player, memo, alpha=float('-inf'), beta=float('inf')):
    """
    Minimax with memoization and alpha-beta pruning.
    alpha: best score maximizer can guarantee
    beta: best score minimizer can guarantee
    """
    # Create cache key
    board_hash = board_to_hash(board)
    cache_key = (board_hash, depth, maximizing_player)
    
    # Check if we've already computed this position
    if cache_key in memo:
        return memo[cache_key]
    
    # Base case: reached max depth
    if depth == 0:
        result = evaluate_board_advanced(board)
        memo[cache_key] = result
        return result
    
    if maximizing_player:  # White's turn
        max_eval = float('-inf')
        for move in get_all_legal_moves(board, 'w'):
            new_board = make_move(board, move)
            eval = minimax(new_board, depth - 1, False, memo, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff - prune remaining branches
        memo[cache_key] = max_eval
        return max_eval
    else:  # Black's turn
        min_eval = float('inf')
        for move in get_all_legal_moves(board, 'b'):
            new_board = make_move(board, move)
            eval = minimax(new_board, depth - 1, True, memo, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff - prune remaining branches
        memo[cache_key] = min_eval
        return min_eval

def ai_move_with_minimax(board, history, depth=3):
    opening_response = get_opening_move(history)
    if opening_response:
        return opening_response, None
    
    moves = get_all_legal_moves(board, 'b')
    if not moves:
        return None, None
    
    best_move = None
    best_promotion = None
    best_score = float('inf')
    
    # Use global memoization dictionary
    global global_memo
    
    # Track cache stats for this move
    cache_size_before = len(global_memo)
    
    alpha = float('-inf')
    beta = float('inf')
    
    for move in moves:
        sr, sc, er, ec = algebraic_to_indices(move)
        piece = board[sr][sc]
        
        # Check if this is a pawn promotion move
        if piece[1].lower() == 'p' and ((piece[0] == 'b' and er == 7)):
            # Evaluate all promotion options
            for promo in ['q', 'r', 'n', 'b']:
                new_board = make_move(board, move, promo)
                score = minimax(new_board, depth - 1, True, global_memo, alpha, beta)
                
                if score < best_score:
                    best_score = score
                    best_move = move
                    best_promotion = promo
                
                beta = min(beta, score)
        else:
            new_board = make_move(board, move)
            score = minimax(new_board, depth - 1, True, global_memo, alpha, beta)
            
            if score < best_score:
                best_score = score
                best_move = move
                best_promotion = None
            
            beta = min(beta, score)
    
    # Print cache statistics
    new_entries = len(global_memo) - cache_size_before
    print(f"Cache: {len(global_memo)} total positions ({new_entries} new this move)")
    
    return best_move, best_promotion


# ======================
# GAME LOOP
# ======================

board = initial_board()
history = []

while True:
    print_board(board)
    move = input("Your move (e.g. e2e4 or e7e8q for promotion): ").strip()
    if move.lower() == "quit":
        print("Game ended.")
        break

    # Parse promotion piece if specified
    promotion_piece = None
    if len(move) == 5:
        promotion_piece = move[4].lower()
        if promotion_piece not in ['q', 'r', 'n', 'b']:
            print("Invalid promotion piece. Use q, r, n, or b.")
            continue
        move = move[:4]
    
    if not is_valid_move(board, move, 'w'):
        print("Illegal move. Try again.")
        continue

    # Check if this is a pawn promotion
    sr, sc, er, ec = algebraic_to_indices(move)
    piece = board[sr][sc]
    if piece[1].lower() == 'p' and er == 0 and not promotion_piece:
        promotion_piece = input("Promote to (q/r/n/b, default q): ").strip().lower()
        if promotion_piece not in ['q', 'r', 'n', 'b']:
            promotion_piece = 'q'
    
    board = make_move(board, move, promotion_piece)
    history.append(move)
    
    if check_for_winner(board) != None:
        break
    
    ai_response, ai_promotion = ai_move_with_minimax(board, history)

    if not ai_response:
        print("AI has no moves. Game over.")
        break

    if ai_promotion:
        print(f"AI plays: {ai_response} (promotes to {ai_promotion.upper()})")
    else:
        print(f"AI plays: {ai_response}")
    
    board = make_move(board, ai_response, ai_promotion)
    history.append(ai_response)