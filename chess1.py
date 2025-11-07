import chess
import torch
import torch.nn as nn
import torch.optim as optim

piece_map = {
    None: 0,
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

def encode_board(board: chess.Board):
    """Return a numeric tensor representation of the board state."""
    encoded = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            encoded.append(0)
        else:
            value = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                value = -value
            encoded.append(value)
    return torch.tensor(encoded, dtype=torch.float32)

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 4096)  # 64*64 possible moves

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)  # probability distribution

def move_to_index(move):
    return move.from_square * 64 + move.to_square

def index_to_move(index):
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)

model = ChessNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.NLLLoss()

# Example training on random boards (in practice, load PGN database)
for epoch in range(10):
    board = chess.Board()
    encoded = encode_board(board).unsqueeze(0)  # batch size 1
    move = chess.Move.from_uci("a2a4")  # pretend target move
    target = torch.tensor([move_to_index(move)], dtype=torch.long)

    optimizer.zero_grad()
    output = model(encoded)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

def choose_move(board, model):
    encoded = encode_board(board).unsqueeze(0)
    with torch.no_grad():
        probs = model(encoded).exp().squeeze()  # convert log-softmax back to probs
    
    # Mask illegal moves
    legal_moves = list(board.legal_moves)
    legal_indices = [move_to_index(m) for m in legal_moves]
    mask = torch.zeros(4096)
    mask[legal_indices] = 1.0
    probs = probs * mask
    move_index = torch.argmax(probs).item()
    
    return index_to_move(move_index)
import chess

# Each opening is a sequence of moves in UCI format
openings = [
    ["e2e4"],
    ["d2d4"],
    ["c2c4"],
    ["e2e4", "c7c5"],
    ["e2e4", "e7e6"],
    ["e2e4", "c7c6"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    ["d2d4", "d7d5", "c2c4"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    ["e2e4", "d7d5"],
]

training_examples = []

for opening in openings:
    board = chess.Board()
    for move_uci in opening:
        move = chess.Move.from_uci(move_uci)
        # For each move, store the current board and the correct move
        training_examples.append((board.copy(), move))
        board.push(move)
best_move = choose_move(board, model)
print("Model suggests:", best_move.uci())
 
# Example usage
board = chess.Board()
best_move = choose_move(board, model)
print("Model suggests:", best_move.uci())

print("test")