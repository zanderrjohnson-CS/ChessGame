# chess_engine/move_select.py
import chess
import torch
from chess_engine.chess_model import PolicyNetwork, MOVE_TO_INDEX, board_to_tensor
from chess_engine.move_mask import select_legal_move

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model once when module is imported
model = PolicyNetwork(len(MOVE_TO_INDEX)).to(device)
model.load_state_dict(torch.load("policy_net.pt", map_location=device, weights_only=True))
model.eval()

def get_best_move(fen: str) -> str:
    board = chess.Board(fen)

    # 1) Encode board (NumPy: 16×8×8)
    x_np = board_to_tensor(board)

    # 2) Convert to torch + add batch dim → (1, 16, 8, 8)
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)

    # 3) Forward pass
    with torch.no_grad():
        logits = model(x).squeeze(0)   # (num_moves,)

    # 4) Mask + select legal move
    move = select_legal_move(logits, board)

    return move.uci()