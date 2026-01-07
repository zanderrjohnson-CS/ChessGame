# ===============================
# Imports
# ===============================
import chess
import numpy as np
import requests
import chess
import chess.pgn
import numpy as np
import io


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ChessAI/1.0; +https://example.com)"
}

# ===============================
# Board → Tensor Encoding
# ===============================

PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_tensor(board: chess.Board):
    """
    Converts a python-chess Board into a (16, 8, 8) tensor.
    """
    tensor = np.zeros((16, 8, 8), dtype=np.float32)

    # Piece placement
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = PIECE_TO_CHANNEL[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6

            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)

            tensor[channel, row, col] = 1

    # Side to move
    tensor[12, :, :] = int(board.turn)

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1

    # En passant square
    if board.ep_square is not None:
        r = 7 - chess.square_rank(board.ep_square)
        c = chess.square_file(board.ep_square)
        tensor[15, r, c] = 1

    return tensor


# ===============================
# Sanity Check: Board Tensor
# ===============================

board = chess.Board()
x = board_to_tensor(board)
print("Board tensor shape:", x.shape)  # Expected: (16, 8, 8)


# ===============================
# Chess.com API Utilities
# ===============================

BASE_URL = "https://api.chess.com/pub/player"

PLAYERS = [
    "MagnusCarlsen",
    "Hikaru",
    "Firouzja2003",
    "FabianoCaruana"
]

def get_available_archives(username):
    url = f"https://api.chess.com/pub/player/{username.lower()}/games/archives"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()["archives"]

def fetch_games_from_archive(archive_url):
    resp = requests.get(archive_url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json().get("games", [])


def filter_archives_by_year(archives, year):
    """
    Filters archive URLs by year.
    """
    return [url for url in archives if f"/{year}/" in url]


# ===============================
# Fetch Games (No ML)
# ===============================

all_games = {}

for player in PLAYERS:
    print(f"\nFetching games for {player}...")
    archives = get_available_archives(player)

    # Change year here if desired
    year_archives = filter_archives_by_year(archives, 2024)

    games = []
    for archive_url in year_archives:
        games.extend(fetch_games_from_archive(archive_url))

    all_games[player] = games
    print(f"{player}: {len(games)} games loaded")


# ===============================
# Done
# ===============================

def generate_move_to_index():
    files = 'abcdefgh'
    ranks = '12345678'
    moves = []

    for f1 in files:
        for r1 in ranks:
            for f2 in files:
                for r2 in ranks:
                    # Normal move
                    moves.append(f"{f1}{r1}{f2}{r2}")
                    # Pawn promotion (only makes sense for ranks 2->8 or 7->1)
                    if r2 == '8':
                        moves.append(f"{f1}{r1}{f2}{r2}q")
                        moves.append(f"{f1}{r1}{f2}{r2}r")
                        moves.append(f"{f1}{r1}{f2}{r2}b")
                        moves.append(f"{f1}{r1}{f2}{r2}n")
                    if r2 == '1':
                        moves.append(f"{f1}{r1}{f2}{r2}q")
                        moves.append(f"{f1}{r1}{f2}{r2}r")
                        moves.append(f"{f1}{r1}{f2}{r2}b")
                        moves.append(f"{f1}{r1}{f2}{r2}n")
    move_to_index = {m: i for i, m in enumerate(moves)}
    index_to_move = {i: m for m, i in move_to_index.items()}
    return move_to_index, index_to_move

MOVE_TO_INDEX, INDEX_TO_MOVE = generate_move_to_index()
print("Total moves:", len(MOVE_TO_INDEX))

def parse_game_to_examples(pgn_text):
    """
    Given a PGN string, yields (board_tensor, move_index) for each move.
    """
    examples = []

    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        return examples

    board = game.board()
    for move in game.mainline_moves():
        # Encode current board
        board_tensor = board_to_tensor(board)
        # Encode move as integer
        move_str = move.uci()
        if move_str not in MOVE_TO_INDEX:
            # Sometimes weird promotion moves appear; skip them
            board.push(move)
            continue
        move_idx = MOVE_TO_INDEX[move_str]

        examples.append((board_tensor, move_idx))
        board.push(move)

    return examples
all_examples = []

for player, games in all_games.items():
    print(f"Processing games for {player} ({len(games)} games)...")
    for game in games:
        pgn_text = game.get("pgn", "")
        examples = parse_game_to_examples(pgn_text)
        all_examples.extend(examples)

print("Total training examples collected:", len(all_examples))
board_tensor, move_idx = all_examples[0]
print("Board tensor shape:", board_tensor.shape)
print("Move index:", move_idx)
print("Move UCI:", INDEX_TO_MOVE[move_idx])

def save_training_examples(examples, filename="chess_policy_data.npz"):
    """
    Saves training examples to disk as NumPy arrays.
    """
    X = np.stack([ex[0] for ex in examples])   # board tensors
    y = np.array([ex[1] for ex in examples])   # move indices

    np.savez_compressed(filename, X=X, y=y)

    print(f"Saved {len(examples)} examples to {filename}")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
save_training_examples(all_examples, "chess_policy_2024_elite.npz")
data = np.load("chess_policy_2024_elite.npz")
X_loaded = data["X"]
y_loaded = data["y"]

print("Reloaded X shape:", X_loaded.shape)
print("Reloaded y shape:", y_loaded.shape)
print("First move UCI:", INDEX_TO_MOVE[y_loaded[0]])
