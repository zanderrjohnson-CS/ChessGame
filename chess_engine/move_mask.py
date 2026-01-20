# engine/move_select.py
import torch
import chess
from chess_engine.chess_model import MOVE_TO_INDEX, INDEX_TO_MOVE

def select_legal_move(logits, board: chess.Board):
    legal_moves = list(board.legal_moves)

    mask = torch.full_like(logits, float("-inf"))

    for move in legal_moves:
        idx = MOVE_TO_INDEX(move)  # your existing mapping
        mask[idx] = 0.0

    masked_logits = logits + mask
    move_idx = torch.argmax(masked_logits).item()

    return INDEX_TO_MOVE(move_idx, board)
