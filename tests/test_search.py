# tests/test_search.py
# Run with:  python -m pytest tests
# These tests use only evaluate.py + search.py, so they don't need the
# 160 MB policy network file.
import chess

from chess_engine.evaluate import evaluate
from chess_engine.search import MATE_SCORE, Searcher


def best_move(fen, depth=3):
    board = chess.Board(fen)
    results, _ = Searcher(time_limit=10).run(board, list(board.legal_moves), depth)
    move, score = max(((m, s) for m, s in results if s is not None), key=lambda r: r[1])
    return move.uci(), score


def test_start_position_is_balanced():
    assert evaluate(chess.Board()) == 0


def test_evaluation_is_symmetric():
    # Same position with colors swapped should give the same score
    # for the side to move.
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    assert evaluate(board) == evaluate(board.mirror())


def test_finds_mate_in_one():
    move, score = best_move("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    assert move == "e1e8"
    assert score >= MATE_SCORE - 10


def test_finds_mate_in_two():
    # 1. Rd8+ Rxd8 2. Rxd8#  (doubled rooks beat the lone back-rank defender)
    move, score = best_move("r5k1/5ppp/8/8/8/8/3R1PPP/3R2K1 w - - 0 1", depth=4)
    assert move == "d2d8"
    assert score == MATE_SCORE - 3  # mate in 3 plies = 2 of our moves


def test_does_not_take_defended_pawn_with_queen():
    # Qxd5?? loses the queen to ...Qxd5 / exd5 style recaptures. Here the
    # d5 pawn is defended by the e6 pawn.
    move, _ = best_move("rnbqkbnr/ppp2ppp/4p3/3p4/8/3Q4/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    assert move != "d3d5"


def test_search_leaves_board_unchanged_after_timeout():
    board = chess.Board()
    Searcher(time_limit=0.05).run(board, list(board.legal_moves), 10)
    assert board.fen() == chess.Board().fen()
