# tests/test_move_index.py
import random

import chess

from chess_engine.chess_model import MOVE_TO_INDEX, index_to_move, move_to_index


def test_index_size():
    # 1456 queen-line moves + 336 knight jumps
    assert len(MOVE_TO_INDEX) == 1792


def test_no_impossible_slots():
    assert "a1a1" not in MOVE_TO_INDEX
    assert "a1b3" in MOVE_TO_INDEX      # knight jump
    assert "a1c4" not in MOVE_TO_INDEX  # no piece moves like that


def test_every_legal_move_round_trips():
    random.seed(0)
    for _ in range(30):
        board = chess.Board()
        for _ in range(120):
            moves = list(board.legal_moves)
            if not moves:
                break
            for move in moves:
                idx = move_to_index(move)
                if idx is None:
                    assert move.promotion in (chess.ROOK, chess.BISHOP, chess.KNIGHT)
                else:
                    assert index_to_move(idx, board) == move
            board.push(random.choice(moves))


def test_queen_promotion_shares_the_plain_slot():
    board = chess.Board("8/4P1k1/8/8/8/8/6K1/8 w - - 0 1")
    promo = chess.Move.from_uci("e7e8q")
    assert move_to_index(promo) == MOVE_TO_INDEX["e7e8"]
    assert index_to_move(MOVE_TO_INDEX["e7e8"], board) == promo
    assert move_to_index(chess.Move.from_uci("e7e8n")) is None
