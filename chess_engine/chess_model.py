# ===============================
# chess_engine/chess_model.py

# ===============================
# Imports
# ===============================
import chess
import chess.pgn
import numpy as np
import requests
import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ===============================
# CONFIG
# ===============================
COLLECT_DATA = False        # <-- SET TRUE ONLY WHEN YOU WANT TO RE-FETCH GAMES
DATASET_PATH = "chess_policy_2024_elite.npz"


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ChessAI/1.0)"
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
    tensor = np.zeros((16, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            channel = PIECE_TO_CHANNEL[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6

            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            tensor[channel, row, col] = 1

    tensor[12, :, :] = int(board.turn)

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1

    if board.ep_square is not None:
        r = 7 - chess.square_rank(board.ep_square)
        c = chess.square_file(board.ep_square)
        tensor[15, r, c] = 1

    return tensor


# ===============================
# Move Indexing
# ===============================
def generate_move_to_index():
    files = "abcdefgh"
    ranks = "12345678"
    moves = []

    for f1 in files:
        for r1 in ranks:
            for f2 in files:
                for r2 in ranks:
                    moves.append(f"{f1}{r1}{f2}{r2}")
                    if r2 in ("1", "8"):
                        for p in "qrbn":
                            moves.append(f"{f1}{r1}{f2}{r2}{p}")

    move_to_index = {m: i for i, m in enumerate(moves)}
    index_to_move = {i: m for m, i in move_to_index.items()}
    return move_to_index, index_to_move


MOVE_TO_INDEX, INDEX_TO_MOVE = generate_move_to_index()


# ===============================
# Chess.com API
# ===============================
BASE_URL = "https://api.chess.com/pub/player"

PLAYERS = [
    "MagnusCarlsen",
    "Hikaru",
    "Firouzja2003",
    "FabianoCaruana",
]

def get_available_archives(username):
    url = f"{BASE_URL}/{username.lower()}/games/archives"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()["archives"]

def fetch_games_from_archive(url):
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json().get("games", [])

def filter_archives_by_year(archives, year):
    return [u for u in archives if f"/{year}/" in u]


# ===============================
# PGN → Training Examples
# ===============================

def legal_move_mask(board):
    mask = np.full(len(MOVE_TO_INDEX), -1e9, dtype=np.float32)
    for move in board.legal_moves:
        uci = move.uci()
        if uci in MOVE_TO_INDEX:
            mask[MOVE_TO_INDEX[uci]] = 0.0
    return mask



def parse_game_to_examples(pgn_text):
    examples = []
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return examples

    board = game.board()

    for move in game.mainline_moves():
        board_tensor = board_to_tensor(board)
        move_str = move.uci()

        if move_str in MOVE_TO_INDEX:
            mask = legal_move_mask(board)
            examples.append((board_tensor, MOVE_TO_INDEX[move_str], mask))

        board.push(move)

    return examples


def save_training_examples(examples, path):
    X = np.stack([e[0] for e in examples])
    y = np.array([e[1] for e in examples])
    M = np.stack([e[2] for e in examples])

    np.savez_compressed(path, X=X, y=y, M=M)
    print(f"Saved {len(y)} examples to {path}")



# ===============================
# DATA COLLECTION (OPTION B)
# ===============================
if COLLECT_DATA:
    all_examples = []

    for player in PLAYERS:
        print(f"Fetching {player}...")
        archives = get_available_archives(player)
        archives = filter_archives_by_year(archives, 2024)

        for archive in archives:
            games = fetch_games_from_archive(archive)
            for game in games:
                pgn = game.get("pgn", "")
                all_examples.extend(parse_game_to_examples(pgn))

    save_training_examples(all_examples, DATASET_PATH)

else:
    print("Skipping data collection.")


# ===============================
# Dataset + Dataloader
# ===============================
data = np.load(DATASET_PATH)
print(data["X"].shape, data["y"].shape, data["M"].shape)

# ===============================
# DEBUG: LIMIT DATASET SIZE
# ===============================
MAX_EXAMPLES = 50_000   

X = data["X"][:MAX_EXAMPLES]
y = data["y"][:MAX_EXAMPLES]

print(f"Using {len(y)} training examples for debugging")


class ChessPolicyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = ChessPolicyDataset(X, y)
print("Dataset size:", len(dataset))
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0   # IMPORTANT on macOS
)

@torch.no_grad()
def select_move(model, board):
    model.eval()

    x = torch.tensor(
        board_to_tensor(board),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    logits = model(x).squeeze(0)

    mask = legal_move_mask(board, device)
    masked_logits = logits + mask

    probs = torch.softmax(masked_logits, dim=0)

    move_idx = torch.argmax(probs).item()
    return chess.Move.from_uci(INDEX_TO_MOVE[move_idx])


# ===============================
# Policy Network
# ===============================
class PolicyNetwork(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_moves)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ===============================
# Training Loop
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PolicyNetwork(len(MOVE_TO_INDEX)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print("Dataset size:", len(dataset))
def train():
    # everything in your training loop goes here
    for epoch in range(50):
        total_loss = 0.0
        print(f"Starting epoch {epoch+1}...")

        for boards, moves in dataloader:
            boards = boards.to(device)
            moves = moves.to(device)
            

            optimizer.zero_grad()
            logits = model(boards)
            
            loss = criterion(logits, moves)
            loss.backward()
            optimizer.step()


            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader):.4f}")    

    torch.save(model.state_dict(), "policy_net.pt")


if __name__ == "__main__":
    train() 

model = PolicyNetwork(len(MOVE_TO_INDEX)).to(device)
model.load_state_dict(torch.load("policy_net.pt", map_location=device, weights_only=True))
model.eval()

