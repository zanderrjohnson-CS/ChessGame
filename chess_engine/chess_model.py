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

#collect data flag
COLLECT_DATA = False        

#initialize dataset names
DATASET_PATH = "chess_policy_2024_elite.npz"
VALUE_DATASET_PATH = "chess_value_2024_elite.npz"

#for data fetching 
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ChessAI/1.0)"
}


# ===============================
# Board → Tensor Encoding
# ===============================

#struct for encoding board as tensor
PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_tensor(board: chess.Board):
    #initialize tensor of 16x8x8 for 16 layers and 8x8 board. 16 layers for pieces, turn, castling rights, en passant
    tensor = np.zeros((16, 8, 8), dtype=np.float32)

    #64 iterations
    for square in chess.SQUARES:
        #piece at square
        piece = board.piece_at(square)
        #if there is a piece on that square
        if piece:
            #gets piece nunmber for encoding
            channel = PIECE_TO_CHANNEL[piece.piece_type]
            #white = 0-5, black = 6-11
            if piece.color == chess.BLACK:
                channel += 6
            #row and column for tensor
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            #3d tensor for convolutional NN. Conv nn recognizes patterns. Uses layers to combine patterns. 
            tensor[channel, row, col] = 1
    #
    tensor[12, :, :] = int(board.turn)

    #Other channeling info. Need to specifiy queenside/kingside for both colors. 
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


def parse_game_to_policy_examples(pgn_text, start_move=0, end_move=999):
    """Parse game for policy network with move range"""
    examples = []
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return examples

    board = game.board()
    move_num = 0

    for move in game.mainline_moves():
        if start_move <= move_num < end_move:
            board_tensor = board_to_tensor(board)
            move_str = move.uci()

            if move_str in MOVE_TO_INDEX:
                mask = legal_move_mask(board)
                examples.append((board_tensor, MOVE_TO_INDEX[move_str], mask))

        board.push(move)
        move_num += 1

    return examples


def save_policy_examples(examples, path):
    X = np.stack([e[0] for e in examples])
    y = np.array([e[1] for e in examples])
    M = np.stack([e[2] for e in examples])

    np.savez_compressed(path, X=X, y=y, M=M)
    print(f"Saved {len(y)} policy examples to {path}")


# ===============================
# DATA COLLECTION
# ===============================
if COLLECT_DATA:
    all_policy_examples = []

    for player in PLAYERS:
        print(f"Fetching {player}...")
        archives = get_available_archives(player)
        archives = filter_archives_by_year(archives, 2024)

        for archive in archives:
            games = fetch_games_from_archive(archive)
            for game in games:
                pgn = game.get("pgn", "")
                
                # Policy examples - emphasize endgame
                # 15k early game (moves 0-15)
                early_examples = parse_game_to_policy_examples(pgn, start_move=0, end_move=15)
                if early_examples and len(all_policy_examples) < 15000:
                    all_policy_examples.extend(early_examples)
                
                # 60k mid to endgame (moves 15-60) - more emphasis on late game
                late_examples = parse_game_to_policy_examples(pgn, start_move=15, end_move=60)
                all_policy_examples.extend(late_examples)

    save_policy_examples(all_policy_examples, DATASET_PATH)

else:
    print("Skipping data collection.")


# ===============================
# Dataset + Dataloader
# ===============================
policy_data = np.load(DATASET_PATH)
print("Policy data:", policy_data["X"].shape, policy_data["y"].shape)

# Use more training examples
MAX_POLICY_EXAMPLES = 100_000  # Increased from 40k

X_policy = policy_data["X"][:MAX_POLICY_EXAMPLES]
y_policy = policy_data["y"][:MAX_POLICY_EXAMPLES]

print(f"Using {len(y_policy)} policy training examples")


class ChessPolicyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


policy_dataset = ChessPolicyDataset(X_policy, y_policy)
policy_dataloader = DataLoader(
    policy_dataset,
    batch_size=32,  # Increased batch size for faster training
    shuffle=True,
    num_workers=0
)


# ===============================
# IMPROVED Policy Network (Deeper, Residual Blocks)
# ===============================
class ResidualBlock(nn.Module):
    """Residual block like AlphaZero"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = torch.relu(out)
        return out


class PolicyNetwork(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        
        # Initial convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(16, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Residual tower (like AlphaZero)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),  # 6 residual blocks
        )
        
        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(256, 128, 1),  # 1x1 conv
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.policy_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, num_moves)
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.residual_blocks(x)
        x = self.policy_conv(x)
        x = x.view(x.size(0), -1)
        return self.policy_fc(x)


# ===============================
# Training Loop with Learning Rate Scheduling
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

policy_model = PolicyNetwork(len(MOVE_TO_INDEX)).to(device)
policy_criterion = nn.CrossEntropyLoss()
policy_optimizer = optim.Adam(policy_model.parameters(), lr=2e-3)  # Higher initial LR

# Learning rate scheduler - reduce LR when loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    policy_optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5, 
    verbose=True
)


def train():
    print("\n=== Training Improved Policy Network ===")
    print(f"Network parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    
    best_loss = float('inf')
    
    for epoch in range(100):  # 100 epochs as requested
        total_loss = 0.0
        policy_model.train()
        
        for boards, moves in policy_dataloader:
            boards = boards.to(device)
            moves = moves.to(device)
            
            policy_optimizer.zero_grad()
            logits = policy_model(boards)
            
            loss = policy_criterion(logits, moves)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            
            policy_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(policy_dataloader)
        print(f"Epoch {epoch+1}/100 | Loss: {avg_loss:.4f} | LR: {policy_optimizer.param_groups[0]['lr']:.6f}")
        
        # Adjust learning rate based on loss
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy_model.state_dict(), "policy_net_best.pt")
            print(f"  → New best model saved! (loss: {best_loss:.4f})")
    
    # Save final model
    torch.save(policy_model.state_dict(), "policy_net.pt")
    print("\n✓ Training complete!")
    print(f"Final model saved as policy_net.pt")
    print(f"Best model saved as policy_net_best.pt (loss: {best_loss:.4f})")


if __name__ == "__main__":
    train()


# Load model for inference
policy_model = PolicyNetwork(len(MOVE_TO_INDEX)).to(device)
try:
    policy_model.load_state_dict(torch.load("policy_net.pt", map_location=device, weights_only=True))
    policy_model.eval()
    print("Loaded policy_net.pt")
except:
    print("No policy_net.pt found - train first!")