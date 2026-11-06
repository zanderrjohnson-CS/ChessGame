# chess_engine/search.py
"""
Alpha-beta search: the "calculation" half of the bot.

The policy network is like a strong player's intuition: it glances at the
board and says which moves look natural. It never checks what happens next,
so it can miss a mate in one or walk into a tactic.

Search fixes that by actually trying moves, the opponent's replies, our
replies to those, and so on, then scoring the final positions with
evaluate(). This is the same split AlphaZero and Leela use: a neural net for
intuition, a search for calculation. (They use Monte-Carlo Tree Search; we use
classic alpha-beta because it is simpler to read.)
"""
import time

import chess

from chess_engine.evaluate import PIECE_VALUES, evaluate

MATE_SCORE = 100_000  # bigger than any material score
INFINITY = 10 * MATE_SCORE

# Root moves scoring within this many centipawns of the best are treated as
# "equally good" by search, and the policy network picks among them.
TIE_MARGIN = 25


class SearchTimeout(Exception):
    pass


def _capture_order_key(board: chess.Board, move: chess.Move) -> int:
    """MVV-LVA: try 'Most Valuable Victim, Least Valuable Attacker' first,
    e.g. pawn takes queen before queen takes pawn. Good moves first means
    alpha-beta can skip (prune) more of the tree."""
    if board.is_en_passant(move):
        victim = PIECE_VALUES[chess.PAWN]
    else:
        captured = board.piece_at(move.to_square)
        victim = PIECE_VALUES[captured.piece_type] if captured else 0
    attacker = PIECE_VALUES[board.piece_at(move.from_square).piece_type]
    promo = PIECE_VALUES[move.promotion] if move.promotion else 0
    return victim * 10 - attacker + promo


def _ordered_moves(board: chess.Board):
    moves = list(board.legal_moves)
    moves.sort(
        key=lambda m: (board.is_capture(m) or m.promotion is not None,
                       _capture_order_key(board, m)),
        reverse=True,
    )
    return moves


def _sort_key(result):
    score = result[1]
    return -INFINITY if score is None else score


class Searcher:
    def __init__(self, time_limit: float = 2.0):
        self.time_limit = time_limit
        self.nodes = 0
        self.deadline = None

    def _check_time(self):
        self.nodes += 1
        # Only look at the clock every 1024 nodes; time.time() is not free.
        if self.nodes % 1024 == 0 and time.time() > self.deadline:
            raise SearchTimeout()

    def quiescence(self, board: chess.Board, alpha: int, beta: int) -> int:
        """Keep searching captures only, until the position is 'quiet'.

        Without this the search suffers from the horizon effect: at depth 3
        it might see 'I take your queen!' and stop, never noticing that the
        queen is defended. This replaces the old simulate_exchange() filter
        and is more general: it sees captures anywhere on the board, not
        only on the square we just moved to.
        """
        self._check_time()
        stand_pat = evaluate(board)  # "what if I just don't capture?"
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        for move in _ordered_moves(board):
            if not (board.is_capture(move) or move.promotion):
                break  # ordered: captures/promotions come first
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        """Alpha-beta in 'negamax' form.

        Negamax trick: a position that is +3 for me is -3 for you, so one
        function serves both players by flipping the sign each ply.
        alpha = the best score I'm already guaranteed elsewhere.
        beta  = the best score my opponent is already guaranteed elsewhere.
        If a move scores >= beta, the opponent would never allow this line,
        so we stop looking at it (a 'beta cutoff'). That pruning is what lets
        alpha-beta search far fewer positions than plain minimax.
        """
        self._check_time()

        if board.is_checkmate():
            # Being mated is the worst result. Subtracting ply makes the
            # engine prefer FASTER mates (and slower losses).
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material() \
                or board.can_claim_draw():
            return 0
        if depth == 0:
            return self.quiescence(board, alpha, beta)

        best = -INFINITY
        for move in _ordered_moves(board):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # beta cutoff
        return best

    def search_root(self, board: chess.Board, root_moves, max_depth: int = 4):
        """Iterative deepening: search depth 1, then 2, then 3... until time
        runs out, and keep the result of the deepest search that finished.

        root_moves should be ordered best-first (we pass the policy network's
        ranking). Returns a list of (move, score) for the deepest completed
        depth, plus that depth.
        """
        self.deadline = time.time() + self.time_limit
        self.nodes = 0
        completed = [(m, 0) for m in root_moves]  # if even depth 1 times out
        completed_depth = 0

        for depth in range(1, max_depth + 1):
            results = []
            alpha = -INFINITY
            try:
                for move in root_moves:
                    board.push(move)
                    # The window lets any move within TIE_MARGIN of the best
                    # so far get an exact score (so the policy can break
                    # near-ties); clearly worse moves get cut off early.
                    window_floor = alpha - TIE_MARGIN
                    score = -self.negamax(board, depth - 1, -INFINITY, -window_floor, 1)
                    board.pop()
                    # A score at or below the floor is only an upper bound
                    # ("at most this good"), not an exact value.
                    exact = score > window_floor
                    results.append((move, score if exact else None))
                    alpha = max(alpha, score)
            except SearchTimeout:
                # Undo whatever pushes the interrupted search left behind.
                while len(board.move_stack) > self._root_stack_len:
                    board.pop()
                break
            completed = results
            completed_depth = depth
            # Search the previous best move first next time (better pruning).
            root_moves = [m for m, _ in sorted(results, key=_sort_key, reverse=True)]
            if max(s for _, s in results if s is not None) >= MATE_SCORE - 100:
                break  # found a forced mate; no need to look deeper
        return completed, completed_depth

    def run(self, board: chess.Board, root_moves, max_depth: int = 4):
        self._root_stack_len = len(board.move_stack)
        return self.search_root(board, root_moves, max_depth)
