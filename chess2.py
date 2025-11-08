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
            # Make every square take up 3 spaces for perfect alignment
            print(f"{piece:^3}", end=" ")
        print(8 - i)
    print("   a   b   c   d   e   f   g   h")



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

    # -------------------------------
    # PIECE-SPECIFIC MOVEMENT RULES
    # -------------------------------

    # Pawn movement
    if ptype == 'p':
        direction = -1 if color == 'w' else 1
        start_row = 6 if color == 'w' else 1

        # forward move
        if dc == 0 and board[er][ec] == '0':
            if dr == direction:
                return True
            if sr == start_row and dr == 2 * direction and board[sr + direction][sc] == '0':
                return True
        # diagonal capture
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
        if sr == er or sc == ec:  # rook-like
            step_r = (er - sr) and (1 if er > sr else -1)
            step_c = (ec - sc) and (1 if ec > sc else -1)
            r, c = sr + step_r if sr != er else sr, sc + step_c if sc != ec else sc
            while (r != er or c != ec):
                if board[r][c] != '0':
                    return False
                if sr != er: r += step_r
                if sc != ec: c += step_c
            return True
        elif abs(dr) == abs(dc):  # bishop-like
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


def make_move(board, move):
    sr, sc, er, ec = algebraic_to_indices(move)
    new_board = copy.deepcopy(board)
    new_board[er][ec] = new_board[sr][sc]
    new_board[sr][sc] = '0'
    return new_board

def check_for_winner(board):
    """Return 'white', 'black', or None depending on whether a king is missing."""
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

openings = {
    # King's Pawn Opening
    (('e2e4',),): 'e7e5',
    (('e2e4','e7e5'), ('g1f3',)): 'b8c6',
    (('e2e4','e7e5','g1f3','b8c6'), ('f1c4',)): 'g8f6',

    # Queen's Gambit
    (('d2d4',),): 'd7d5',
    (('d2d4','d7d5'), ('c2c4',)): 'e7e6',
}

def get_opening_move(history):
    for key, response in openings.items():
        if history[-len(key[0]):] == list(key[0]):
            return response
    return None


# ======================
# POSITIONAL EVALUATION
# ======================

piece_values = {
    'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000
}

# (same tables from before)
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
    """
    Check if king has pawn shield and isn't exposed.
    """
    safety_score = 0
    
    # Find king position
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
    
    # Check for pawn shield (pawns in front of king)
    for dc in [-1, 0, 1]:
        check_col = kc + dc
        if 0 <= check_col < 8:
            for dr in [direction, direction * 2]:
                check_row = kr + dr
                if 0 <= check_row < 8:
                    if board[check_row][check_col] == f"{color}p":
                        safety_score += 10
    
    # Penalize king in center during opening/middlegame
    if 2 <= kc <= 5:
        safety_score -= 20
    
    return safety_score
def evaluate_development(board, color):
    """
    Encourage moving knights and bishops off the back rank.
    """
    back_rank = 7 if color == 'w' else 0
    development_score = 0

    for sc in range(8):
        piece = board[back_rank][sc]
        if piece != '0' and piece[0] == color:  # color should be 'w' or 'b'
            ptype = piece[1].lower()  # assuming pieces are like 'wn', 'bb', etc.
            # Penalize knights and bishops still on back rank
            if ptype in ['n', 'b']:
                development_score -= 15

    return development_score

def evaluate_mobility(board, color):
    """
    Count legal moves available. More mobility = better position.
    """
    legal_moves = get_all_legal_moves(board, color)
    return len(legal_moves) * 5  # 5 points per legal move
def evaluate_center_control(board, color):
    """
    Bonus for controlling center squares (d5, e5, d4, e4).
    Assumes board is an 8x8 list-of-lists, empty squares == '0',
    and pieces are strings like 'wn' or 'bp' where piece[0] is color.
    """
    # (row, col) pairs: d5, e5, d4, e4
    center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
    center_score = 0

    for sr in range(8):
        for sc in range(8):
            piece = board[sr][sc]
            # skip empty squares and junk strings; check piece color properly
            if piece == '0' or piece[0] != color:
                continue

            # Occupy / attack each center square
            for cr, cc in center_squares:
                if sr == cr and sc == cc:
                    # occupying center
                    center_score += 20
                else:
                    # attacking center: build algebraic-like move and ask validator
                    if 0 <= sr < 8 and 0 <= sc < 8 and 0 <= cr < 8 and 0 <= cc < 8:
                        move = f"{chr(sc + 97)}{8 - sr}{chr(cc + 97)}{8 - cr}"
                        if is_valid_move(board, move, color):
                            center_score += 10


    return center_score

def detect_forks(board, color):
    """
    Detect forks: when one piece attacks 2+ valuable enemy pieces.
    Returns a score bonus for the forking side.
    """
    fork_score = 0
    enemy_color = 'b' if color == 'w' else 'w'

    for sr in range(8):
        for sc in range(8):
            piece = board[sr][sc]
            if piece == '0' or piece[0] != color:
                continue

            # Find all squares this piece attacks
            attacked_pieces = []
            for er in range(8):
                for ec in range(8):
                    target = board[er][ec]
                    if target != '0' and target[0] == enemy_color:
                        # Ensure coordinates are valid before building move string
                        if 0 <= sr < 8 and 0 <= sc < 8 and 0 <= er < 8 and 0 <= ec < 8:
                            move = f"{chr(sc+97)}{8-sr}{chr(ec+97)}{8-er}"
                            if is_valid_move(board, move, color):
                                attacked_pieces.append(target)



            # If attacking 2+ pieces, it's a fork
            if len(attacked_pieces) >= 2:
                # Calculate total value of forked pieces
                fork_value = sum(piece_values[p[1]] for p in attacked_pieces)
                fork_score += fork_value * 0.3  # 30% bonus for fork threat

    return fork_score


def evaluate_pawn_structure(board, color):
    """
    Penalize doubled and isolated pawns.
    """
    structure_score = 0
    pawn_columns = {i: [] for i in range(8)}
    
    # Map pawns to columns
    for r in range(8):
        for c in range(8):
            if board[r][c] == f"{color}p":
                pawn_columns[c].append(r)
    
    for col, pawns in pawn_columns.items():
        # Doubled pawns penalty
        if len(pawns) > 1:
            structure_score -= 20 * (len(pawns) - 1)
        
        # Isolated pawns (no friendly pawns on adjacent columns)
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
    """
    Comprehensive evaluation combining all factors.
    Positive = White advantage, Negative = Black advantage
    """
    score = 0  # ← Start from zero instead of calling itself

    # === Base material evaluation ===
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece == '0':
                continue
            color = piece[0]
            ptype = piece[1]
            value = piece_values.get(ptype, 0)
            # Use piece-square table
            table = piece_tables.get(ptype.lower(), [[0]*8 for _ in range(8)])
            table_bonus = table[r][c]
            if color == 'w':
                score += value + table_bonus
            else:
                score -= value + table_bonus

    # === Add positional and strategic heuristics ===
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
# AI MOVE SELECTION
# ======================

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

def minimax(board, depth, maximizing_player):
    """
    Recursively search 'depth' moves ahead.
    maximizing_player: True for White, False for Black
    """
    if depth == 0:
        return evaluate_board_advanced(board)
    
    if maximizing_player:  # White's turn
        max_eval = float('-inf')
        for move in get_all_legal_moves(board, 'w'):
            new_board = make_move(board, move)
            eval = minimax(new_board, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:  # Black's turn
        min_eval = float('inf')
        for move in get_all_legal_moves(board, 'b'):
            new_board = make_move(board, move)
            eval = minimax(new_board, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval

def ai_move_with_minimax(board, history, depth=3):
    opening_response = get_opening_move(history)
    if opening_response:
        return opening_response
    
    moves = get_all_legal_moves(board, 'b')
    if not moves:
        return None
    
    best_move = None
    best_score = float('inf')
    
    for move in moves:
        new_board = make_move(board, move)
        score = minimax(new_board, depth - 1, True)  # ← Look ahead
        
        if score < best_score:
            best_score = score
            best_move = move
    
    return best_move



# ======================
# GAME LOOP
# ======================

board = initial_board()
history = []

while True:
    print_board(board)
    move = input("Your move (e.g. e2e4): ").strip()
    if move.lower() == "quit":
        print("Game ended.")
        break

    if not is_valid_move(board, move, 'w'):
        print("Illegal move. Try again.")
        continue

    board = make_move(board, move)
    history.append(move)
    

    ai_response = ai_move_with_minimax(board, history)

    if check_for_winner(board) != None:
        break
    if not ai_response:
        print("AI has no moves. Game over.")
        break

    print(f"AI plays: {ai_response}")
    board = make_move(board, ai_response)
    history.append(ai_response)
print('test')
