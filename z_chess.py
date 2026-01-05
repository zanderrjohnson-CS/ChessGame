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

print("\nData fetch complete.")
print(get_available_archives("MagnusCarlsen")[-3:])
