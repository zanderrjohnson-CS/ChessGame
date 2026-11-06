# chess_engine/move_mask.py
import torch
import chess
from chess_engine.chess_model import move_to_index, index_to_move

def select_legal_move(logits: torch.Tensor, board: chess.Board) -> chess.Move:
    mask = torch.full_like(logits, float("-inf"))

    for move in board.legal_moves:
        idx = move_to_index(move)
        if idx is not None:
            mask[idx] = 0.0

    move_idx = torch.argmax(logits + mask).item()
    return index_to_move(move_idx, board)
