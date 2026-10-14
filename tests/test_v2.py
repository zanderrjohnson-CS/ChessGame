# Run with:  python -m unittest tests.test_v2
import random
import unittest

import chess

from chess_engine.v2.evaluate import evaluate, evaluate_material
from chess_engine.v2.search import Searcher, MATE_SCORE


def best(fen, time_limit=3.0):
    return Searcher().search(chess.Board(fen), time_limit=time_limit)


class TestEvaluate(unittest.TestCase):
    def test_start_position_is_even(self):
        self.assertEqual(evaluate(chess.Board()), 0)

    def test_symmetry(self):
        # A position and its color-flipped mirror must score the same for the side
        # to move. Catches mistakes in how the piece-square tables are indexed.
        rng = random.Random(0)
        for _ in range(50):
            board = chess.Board()
            for _ in range(rng.randint(0, 60)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            self.assertEqual(evaluate(board), evaluate(board.mirror()), board.fen())
            self.assertEqual(evaluate_material(board), evaluate_material(board.mirror()))

    def test_extra_queen_is_good(self):
        board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
        self.assertGreater(evaluate(board), 800)
        board.turn = chess.BLACK
        self.assertLess(evaluate(board), -800)


class TestSearch(unittest.TestCase):
    def test_mate_in_one(self):
        r = best("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        self.assertEqual(r["move"].uci(), "h5f7")
        self.assertEqual(r["score"], MATE_SCORE - 1)

    def test_mate_in_two(self):
        # Rook ladder: Ra7, then Rb8#.
        r = best("6k1/8/8/8/8/8/R7/1R4K1 w - - 0 1")
        self.assertEqual(r["score"], MATE_SCORE - 3)

    def test_takes_free_queen(self):
        r = best("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", time_limit=1.0)
        self.assertEqual(r["move"].uci(), "d2d5")

    def test_saves_queen_not_knight(self):
        # Queen on d4 and knight on e5 are both attacked by pawns (v1 moved the
        # knight in a position like this). v2 must not leave the queen en prise.
        fen = "rnbqkbnr/pp3ppp/3p4/2p1N3/3Q4/8/PPPPPPPP/RNB1KB1R w KQkq - 0 1"
        board = chess.Board(fen)
        board.push(best(fen)["move"])
        queen_squares = board.pieces(chess.QUEEN, chess.WHITE)
        for reply in board.legal_moves:
            self.assertNotIn(reply.to_square, queen_squares, f"queen hangs after {board.peek()}")

    def test_avoids_stalemate_when_winning(self):
        # Q+K vs K: Qb6 would be stalemate. Any other sensible move keeps winning.
        r = best("k7/8/2K5/8/8/8/8/1Q6 w - - 0 1", time_limit=1.0)
        board = chess.Board("k7/8/2K5/8/8/8/8/1Q6 w - - 0 1")
        board.push(r["move"])
        self.assertFalse(board.is_stalemate())


if __name__ == "__main__":
    unittest.main()
