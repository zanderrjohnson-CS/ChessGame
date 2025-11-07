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
    try:
        start_col, start_row = move[0], move[1]
        end_col, end_row = move[2], move[3]
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

def evaluate_board(board):
    """Evaluate the board. Positive means white is better; negative means black is better."""
    total = 0
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece == '0':
                continue

            color = piece[0]
            ptype = piece[1]
            base_value = piece_values[ptype]

            # lookup table for this piece
            table = piece_tables[ptype]

            # for white, use the table as-is; for black, mirror it vertically
            if color == 'w':
                positional = table[i][j]
                total += base_value + positional
            else:
                positional = table[7 - i][j]
                total -= base_value + positional

    return total



# ======================
# AI MOVE SELECTION
# ======================

def get_all_legal_moves(board, color):
    moves = []
    for sr in range(8):
        for sc in range(8):
            if board[sr][sc][0] == color:
                for er in range(8):
                    for ec in range(8):
                        move = f"{chr(sc+97)}{8-sr}{chr(ec+97)}{8-er}"
                        if is_valid_move(board, move, color):
                            moves.append(move)
    return moves

def ai_move(board, history):
    # Check if an opening move is available
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
        score = evaluate_board(new_board)
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
    

    ai_response = ai_move(board, history)
    if check_for_winner(board) != None:
        break
    if not ai_response:
        print("AI has no moves. Game over.")
        break

    print(f"AI plays: {ai_response}")
    board = make_move(board, ai_response)
    history.append(ai_response)
