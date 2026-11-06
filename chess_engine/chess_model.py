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
# Board → 16x8x8 Tensor Encoding
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

    #Other board state encoding. Need to specifiy queenside/kingside for both colors. 
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

# Converts moves to output slots for the NN (and back).
#
# Old version: every from-square x to-square pair (64*64) plus promotions = 8192
# slots, but ~78% of them could never be played (a1a1, a1b3 for a rook...).
#
# New version: only moves that SOME piece could make on an empty board:
#   - "queen lines": any number of squares straight or diagonally
#     (this covers rook, bishop, queen, king and pawn moves, castling as e1g1 too)
#   - knight jumps
# That's 1456 + 336 = 1792 slots (Leela Chess Zero uses 1858; the extra 66 are
# underpromotions).
#
# Promotions: a pawn reaching the last rank is stored in the same slot as its
# from->to squares (e7e8q -> slot "e7e8"), i.e. we assume it promotes to a queen.
# Underpromotions (to rook/bishop/knight) get no slot. The search still tries
# them, because it generates moves with python-chess, not from these slots.
#
# Note a slot doesn't know WHICH piece is moving: "a1a8" is a rook move in one
# game and a queen move in another, so we can't make per-piece slot lists.
KNIGHT_JUMPS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
QUEEN_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def generate_move_to_index():
    moves = []
    for from_sq in chess.SQUARES:
        f, r = chess.square_file(from_sq), chess.square_rank(from_sq)
        targets = []
        for df, dr in QUEEN_DIRECTIONS:
            for dist in range(1, 8):
                tf, tr = f + df * dist, r + dr * dist
                if not (0 <= tf < 8 and 0 <= tr < 8):
                    break
                targets.append(chess.square(tf, tr))
        for df, dr in KNIGHT_JUMPS:
            tf, tr = f + df, r + dr
            if 0 <= tf < 8 and 0 <= tr < 8:
                targets.append(chess.square(tf, tr))
        for to_sq in targets:
            moves.append(chess.square_name(from_sq) + chess.square_name(to_sq))

    move_to_index = {m: i for i, m in enumerate(moves)}
    index_to_move = {i: m for m, i in move_to_index.items()}
    return move_to_index, index_to_move


def move_to_index(move: chess.Move):
    """Output slot for a move, or None for an underpromotion (no slot)."""
    if move.promotion is not None and move.promotion != chess.QUEEN:
        return None
    return MOVE_TO_INDEX[chess.square_name(move.from_square) + chess.square_name(move.to_square)]


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Slot -> move on this board, adding '=Q' when a pawn reaches the last rank."""
    move = chess.Move.from_uci(INDEX_TO_MOVE[idx])
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.PAWN and chess.square_rank(move.to_square) in (0, 7):
        move.promotion = chess.QUEEN
    return move


MOVE_TO_INDEX, INDEX_TO_MOVE = generate_move_to_index()


# ===============================
# Chess.com API
# ===============================

# fetching games from chess.com for data collection. Has information on game accuracy, who won, moves, etc. Need to still clean data.
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
    #set all moves to -1e9 (illegal)
    mask = np.full(len(MOVE_TO_INDEX), -1e9, dtype=np.float32)
    for move in board.legal_moves:
        #move.uci() returns i.e. e2e4 

        idx = move_to_index(move)
        if idx is not None:
            #move_to_index maps the move to its slot, which is set to 0 (legal)
            mask[idx] = 0.0
    #mask is applied to NN output logits to filter illegal moves
    return mask


def parse_game_to_policy_examples(pgn_text, start_move=0, end_move=999):

    """Parse game for policy network with move range"""
    #list for storing examples
    examples = []
    #read pgn game. cleans game info and returns move history of game.
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    #if no game, return empty list
    if game is None:
        return examples
    #starting board position
    board = game.board()
    move_num = 0

    #iterate through all moves of the game
    for move in game.mainline_moves():
        #specify range for dynamic training example makeup (i.e. 70% endgame, 30% opening)
        if start_move <= move_num < end_move:
            #convert board to tensor
            board_tensor = board_to_tensor(board)
            #get the move's output slot (None for rare underpromotions, which we skip)
            move_idx = move_to_index(move)
            if move_idx is not None:
                mask = legal_move_mask(board)
                #Could maybe remove mask to save 5x storage.
                examples.append((board_tensor, move_idx, mask))

        board.push(move)
        move_num += 1
    #examples is huge list (2.4 GBs for 100k examples)
    return examples

##################################################################
###### Examples for how for how data is structured.  #############
##################################################################

# ##### np.stack example  ######
# X = [
#     array(16x8x8),  # board 1
#     array(16x8x8),  # board 2
#     array(16x8x8),  # board 3
# ]
# np.stack(X)  # → array(3, 16, 8, 8)  — stacks them into one big array


# ###### np.array example  ######
# y = [1234, 5678, 91]
# np.array(y)  # → array([1234, 5678, 91])

# ##### examples varible structure  ######
# all_policy_examples = [
#     (board_tensor_1, move_idx_1, mask_1),  # Position 1 from game 1
#     (board_tensor_2, move_idx_2, mask_2),  # Position 2 from game 1
#     (board_tensor_3, move_idx_3, mask_3),  # Position 3 from game 1
#     ...
#     (board_tensor_N, move_idx_N, mask_N),  # Position from game 500
# ]

# ##### save_policy_examples function  ######
# X = np.stack([e[0] for e in examples])  # All board tensors → (100000, 16, 8, 8)
# y = np.array([e[1] for e in examples])  # All move indices  → (100000,)
# M = np.stack([e[2] == 0.0 for e in examples])  # Legal-move masks (True/False) → (100000, 1792)

# np.savez_compressed(path, X=X, y=y, M=M)  # Save to disk


def save_policy_examples(examples, game_ids, path):
    X = np.stack([e[0] for e in examples])
    y = np.array([e[1] for e in examples])
    # Masks saved as True/False (1 byte per slot) instead of 0/-1e9 floats (4 bytes).
    M = np.stack([e[2] == 0.0 for e in examples])
    # G = which game each position came from, so we can split train/validation
    # by GAME (positions from one game are too similar to be in both).
    G = np.array(game_ids)

    np.savez_compressed(path, X=X, y=y, M=M, G=G)
    print(f"Saved {len(y)} policy examples from {len(set(game_ids))} games to {path}")


def load_policy_examples(path):
    """Load a dataset; datasets saved before the 1792-slot move index are
    re-labelled automatically. Returns X, y, M (bool legal masks), G (game ids or None)."""
    from chess_engine.convert_weights import OLD_NUM_MOVES, convert_dataset_labels

    data = np.load(path)
    X, y, M = data["X"], data["y"], data["M"]
    G = data["G"] if "G" in data.files else None

    if M.shape[1] == OLD_NUM_MOVES:
        print("Dataset uses the old 8192-move index; re-labelling it...")
        keep, y, M = convert_dataset_labels(y, M)
        X = X[keep]
        G = G[keep] if G is not None else None
        print(f"  dropped {int((~keep).sum())} underpromotion examples")
    elif M.dtype != bool:
        M = M == 0.0
    return X, y, M, G



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
if __name__ == "__main__":

    # ===============================
    # DATA COLLECTION
    # ===============================
    if COLLECT_DATA:
        all_policy_examples = []
        all_game_ids = []
        early_count = 0  # opening positions collected so far
        MAX_EARLY_EXAMPLES = 15_000
        game_id = 0

        for player in PLAYERS:
            print(f"Fetching {player}...")
            archives = get_available_archives(player)
            archives = filter_archives_by_year(archives, 2024)

            for archive in archives:
                games = fetch_games_from_archive(archive)
                for game in games:
                    pgn = game.get("pgn", "")

                    # Opening positions (moves 0-15), capped at 15k total.
                    # (Bug fix: this used to check len(all_policy_examples), which
                    # counts ALL examples, so openings stopped after a few games.)
                    if early_count < MAX_EARLY_EXAMPLES:
                        early_examples = parse_game_to_policy_examples(pgn, start_move=0, end_move=15)
                        all_policy_examples.extend(early_examples)
                        all_game_ids.extend([game_id] * len(early_examples))
                        early_count += len(early_examples)

                    # Middlegame and endgame positions (moves 15-60)
                    late_examples = parse_game_to_policy_examples(pgn, start_move=15, end_move=60)
                    all_policy_examples.extend(late_examples)
                    all_game_ids.extend([game_id] * len(late_examples))
                    game_id += 1

        save_policy_examples(all_policy_examples, all_game_ids, DATASET_PATH)

    else:
        print("Skipping data collection.")


    # ===============================
    # Dataset: sample, then split into TRAIN and VALIDATION
    # ===============================
    # X = board tensors (inputs), y = the move the GM played (labels),
    # M = legal-move masks, G = which game each position came from.
    X_all, y_all, M_all, G_all = load_policy_examples(DATASET_PATH)
    print("Policy data:", X_all.shape, y_all.shape)

    rng = np.random.default_rng(seed=0)  # fixed seed so the split is the same every run

    # Take a RANDOM sample, not the first N rows (the first rows are all one player's games).
    MAX_POLICY_EXAMPLES = 100_000
    if len(y_all) > MAX_POLICY_EXAMPLES:
        sample = np.sort(rng.choice(len(y_all), MAX_POLICY_EXAMPLES, replace=False))
        X_all, y_all, M_all = X_all[sample], y_all[sample], M_all[sample]
        G_all = G_all[sample] if G_all is not None else None

    # Validation set: ~10% of positions the network NEVER trains on. Its loss tells
    # us whether the net is learning chess or just memorising the training games.
    # We hold out whole GAMES: neighbouring positions of one game are nearly
    # identical, so splitting them would let the net "cheat".
    VAL_FRACTION = 0.10
    if G_all is not None:
        games = np.unique(G_all)
        val_games = rng.choice(games, max(1, int(len(games) * VAL_FRACTION)), replace=False)
        is_val = np.isin(G_all, val_games)
    else:
        print("No game ids in this dataset (collected before they were saved); "
              "splitting by position instead. Re-collect data for a cleaner split.")
        is_val = rng.random(len(y_all)) < VAL_FRACTION

    X_train, y_train = X_all[~is_val], y_all[~is_val]
    X_val, y_val, M_val = X_all[is_val], y_all[is_val], M_all[is_val]
    print(f"Train: {len(y_train)} positions | Validation: {len(y_val)} positions")


    ##### Batch size #####
    # higher batch size can speed up training but requires more memory. 32 is a common choice for CNNs, but can adjust based on hardware.
    # higher batch size has more stable gradients but may get stuck in local minima. lower batch size has more noisy gradients which can help escape local minima but may require more epochs to converge.
    ##############################################################################
    class ChessPolicyDataset(Dataset):
        def __init__(self, X, y, M=None):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
            self.M = torch.tensor(M, dtype=torch.bool) if M is not None else None

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            if self.M is None:
                return self.X[idx], self.y[idx]
            return self.X[idx], self.y[idx], self.M[idx]

    policy_dataloader = DataLoader(
        ChessPolicyDataset(X_train, y_train),
        batch_size=32,
        shuffle=True, #randomize order of examples each epoch for better generalization
        num_workers=0 #Could set to >0 for faster data loading, but may cause issues on some platforms
    )
    val_dataloader = DataLoader(ChessPolicyDataset(X_val, y_val, M_val), batch_size=256, shuffle=False)


    # ===============================
    # Training Loop with Learning Rate Scheduling
    # ===============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    policy_model = PolicyNetwork(len(MOVE_TO_INDEX)).to(device)
    policy_criterion = nn.CrossEntropyLoss()
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=2e-3)  # Higher initial LR

    # Learning rate scheduler - halve the LR when VALIDATION loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        policy_optimizer,
        mode='min',
        factor=0.5,
        patience=3,
    )

    MAX_EPOCHS = 100
    EARLY_STOP_PATIENCE = 8  # stop if validation loss hasn't improved for this many epochs

    def evaluate_on_validation():
        """Validation loss, plus how often the net's favourite LEGAL move is the
        move the grandmaster played (this is where the saved masks M are used)."""
        policy_model.eval()  # turns dropout off
        total_loss, correct, count = 0.0, 0, 0
        with torch.no_grad():  # no gradients needed: we're only measuring
            for boards, moves, masks in val_dataloader:
                boards, moves, masks = boards.to(device), moves.to(device), masks.to(device)
                logits = policy_model(boards)
                total_loss += policy_criterion(logits, moves).item() * len(moves)
                legal_logits = logits.masked_fill(~masks, float("-inf"))
                correct += (legal_logits.argmax(dim=1) == moves).sum().item()
                count += len(moves)
        return total_loss / count, correct / count

    ### Training:
    # model guesses move then calculate loss based off how far away from correct move.
    # Once loss is calculated, backpropagate through network to update parameters based on how much each parameter contributed to the error.
    #update parameters with optimizer. Adam is a good default choice for CNNs.
    def train():
        print("\n=== Training Policy Network ===")
        print(f"Network parameters: {sum(p.numel() for p in policy_model.parameters()):,}")

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(MAX_EPOCHS):
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

            train_loss = total_loss / len(policy_dataloader)
            val_loss, val_acc = evaluate_on_validation()
            print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Train loss: {train_loss:.4f} | "
                  f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.1%} | "
                  f"LR: {policy_optimizer.param_groups[0]['lr']:.6f}")

            # Train loss going down while val loss goes UP = overfitting (memorising).
            scheduler.step(val_loss)

            # Keep the model that does best on positions it has never seen.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(policy_model.state_dict(), "policy_net_best.pt")
                print(f"  → New best model saved! (val loss: {best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f"Validation loss hasn't improved for {EARLY_STOP_PATIENCE} epochs; stopping early.")
                    break

        # Save final model
        torch.save(policy_model.state_dict(), "policy_net.pt")
        print("\n✓ Training complete!")
        print(f"Final model saved as policy_net.pt")
        print(f"Best model saved as policy_net_best.pt (val loss: {best_val_loss:.4f})")
        print("Copy policy_net_best.pt over policy_net.pt to play with the best one.")


    train()

""" 
# Load model for inference
load_policy = True  # Set to False if you want to skip loading 
if load_policy:
    policy_model = PolicyNetwork(len(MOVE_TO_INDEX)).to(device)
    try:
        policy_model.load_state_dict(torch.load("policy_net.pt", map_location=device, weights_only=True))
        policy_model.eval()
        print("Loaded policy_net.pt")
    except:
        print("No policy_net.pt found - train first!") """