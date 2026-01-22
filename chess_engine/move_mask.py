# chess_engine/move_mask.py
import torch
import chess
from chess_engine.chess_model import MOVE_TO_INDEX, INDEX_TO_MOVE

def select_legal_move(logits: torch.Tensor, board: chess.Board) -> chess.Move:
    mask = torch.full_like(logits, float("-inf"))

    for move in board.legal_moves:
        idx = MOVE_TO_INDEX.get(move.uci())
        if idx is not None:
            mask[idx] = 0.0

    move_idx = torch.argmax(logits + mask).item()
    return chess.Move.from_uci(INDEX_TO_MOVE[move_idx])
