# Run with:  python -m unittest tests.test_v2
import random
import unittest

import chess

from chess_engine.v2.evaluate import evaluate, evaluate_material, evaluate_pst
from chess_engine.v2.features import extract_features, F
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
            mirror = board.mirror()
            self.assertEqual(evaluate(board), evaluate(mirror), board.fen())
            self.assertEqual(evaluate_pst(board), evaluate_pst(mirror), board.fen())
            self.assertEqual(extract_features(board), [-x for x in extract_features(mirror)])
            self.assertEqual(evaluate_material(board), evaluate_material(board.mirror()))

    def test_extra_queen_is_good(self):
        board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
        self.assertGreater(evaluate(board), 800)
        board.turn = chess.BLACK
        self.assertLess(evaluate(board), -800)


class TestFeatures(unittest.TestCase):
    def feats(self, fen):
        return extract_features(chess.Board(fen))

    def test_pawn_structure(self):
        # White: doubled + isolated c-pawns, passed a-pawn on the 6th. Black: h-pawn only.
        f = self.feats("4k3/7p/P7/8/2P5/2P5/8/4K3 w - - 0 1")
        self.assertEqual(f[F["doubled_pawn"]], 1)
        self.assertEqual(f[F["isolated_pawn"]], 3 - 1)  # a + two c-pawns vs black h-pawn
        self.assertEqual(f[F["passed_pawn_r6"]], 1)

    def test_rooks(self):
        # White rook on open d-file and on the 7th; black rook on a semi-open file.
        f = self.feats("r3k3/1R6/8/8/8/8/P7/3RK3 w - - 0 1")
        self.assertEqual(f[F["rook_open_file"]], 2)   # both white rooks (b- and d-files) are open
        self.assertEqual(f[F["rook_on_7th"]], 1)
        self.assertEqual(f[F["rook_semi_open_file"]], -1)

    def test_fianchetto_and_castled_king(self):
        f = self.feats("4k3/8/8/8/8/6P1/5PBP/6K1 w - - 0 1")
        self.assertEqual(f[F["fianchetto"]], 1)
        self.assertEqual(f[F["king_castled"]], 1)
        self.assertEqual(f[F["king_pawn_shield"]], 3)  # f2, g3, h2


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
