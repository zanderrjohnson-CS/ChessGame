# engine/engine.py
import chess
from chess_engine.chess_model import model, board_to_tensor as encode_board
from chess_engine.move_mask import select_legal_move

def get_best_move(fen: str) -> str:
    board = chess.Board(fen)

    # Encode board
    x = encode_board(board)          # shape: (1, C, 8, 8)

    # Predict logits
    logits = model(x)                # shape: (1, num_moves)

    # Select legal move (mask at inference)
    move = select_legal_move(logits[0], board)

    return move.uci()
