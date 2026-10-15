# ===============================
# chess_engine/v2/search.py
# ===============================
# Look-ahead search: try moves, try the opponent's best replies, and so on,
# then pick the move that leads to the best position *assuming both sides
# play well*. The evaluator (evaluate.py) only scores the positions at the
# end of each line ("leaves").
#
# Pieces, roughly in order of importance:
#   - Negamax alpha-beta: minimax written so a single function handles both
#     sides (my score = -opponent's score), with pruning of lines that can't
#     change the result.
#   - Quiescence search: at the end of the normal depth, keep searching
#     captures until the position is "quiet". Without it the bot would stop
#     in the middle of a trade and think it won a queen (the "horizon effect").
#   - Iterative deepening: search depth 1, then 2, then 3... until time runs
#     out. Each pass makes the next one faster through better move ordering.
#   - Transposition table: a cache of positions already searched (the same
#     position is often reached by different move orders).
#   - Move ordering: alpha-beta prunes much more when good moves are tried
#     first. Order: cached best move, captures (most valuable victim / least
#     valuable attacker), killer moves, then quiet moves by history score.
#   - Null-move pruning and late move reductions: two standard tricks that
#     skip work on lines that are very likely bad.
# ===============================
import time

import chess

from chess_engine.v2.evaluate import evaluate, SIMPLE_VALUE


MATE_SCORE = 100_000
MATE_THRESHOLD = MATE_SCORE - 1_000  # scores beyond this mean "mate in N"
INF = 1_000_000

# Transposition table entry flags: was the stored score exact, or only a bound?
EXACT, LOWER, UPPER = 0, 1, 2


class SearchTimeout(Exception):
    pass


def _key(board: chess.Board):
    # Hashable identity of a position (pieces, side to move, castling, en passant).
    # python-chess exposes this as a private helper; it's much faster than
    # chess.polyglot.zobrist_hash in pure Python.
    return board._transposition_key()


class Searcher:
    def __init__(self, eval_fn=evaluate, use_null_move=True, use_lmr=True):
        self.eval_fn = eval_fn
        self.use_null_move = use_null_move
        self.use_lmr = use_lmr
        self.tt = {}

    # ---------------------------
    # Public entry point
    # ---------------------------
    def search(self, board: chess.Board, time_limit: float = 2.0, max_depth: int = 64,
               verbose: bool = False):
        """Iterative deepening. Returns a dict with move, score, depth, nodes, pv."""
        board = board.copy()
        self.nodes = 0
        self.deadline = time.time() + time_limit
        self.killers = [[None, None] for _ in range(128)]
        self.history = {}
        if len(self.tt) > 2_000_000:
            self.tt.clear()

        # Positions already played in the game, for repetition detection.
        self.game_keys = []
        tmp = board.copy()
        while tmp.move_stack:
            tmp.pop()
            self.game_keys.append(_key(tmp))
        self.path_keys = []

        legal = list(board.legal_moves)
        if not legal:
            return {"move": None, "score": 0, "depth": 0, "nodes": 0, "pv": []}

        best_move, best_score, done_depth = legal[0], 0, 0
        start = time.time()

        for depth in range(1, max_depth + 1):
            try:
                score, move = self._root(board, depth, best_move)
            except SearchTimeout:
                # The previous best move is searched first in each iteration,
                # so a move that already beat it in this partial pass is safe to use.
                if self._partial_best is not None and self._partial_best[1] != best_move \
                        and self._partial_best[0] > best_score:
                    best_score, best_move = self._partial_best
                break
            best_move, best_score, done_depth = move, score, depth
            if verbose:
                print(f"depth {depth:2d}  score {format_score(score):>8}  nodes {self.nodes:8d}  "
                      f"time {time.time() - start:5.2f}s  pv {' '.join(m.uci() for m in self._pv(board, depth))}")
            # Found a forced mate: no point searching deeper.
            if abs(score) >= MATE_THRESHOLD:
                break
            # The next depth usually takes several times longer than this one.
            # If more than half the budget is gone, it almost certainly won't finish.
            if time.time() - start > time_limit * 0.5:
                break

        return {
            "move": best_move,
            "score": best_score,
            "depth": done_depth,
            "nodes": self.nodes,
            "time": time.time() - start,
            "pv": self._pv(board, done_depth),
        }

    # ---------------------------
    # Root search
    # ---------------------------
    def _root(self, board, depth, prev_best):
        alpha, beta = -INF, INF
        best_move, best_score = None, -INF
        self._partial_best = None

        moves = self._order_moves(board, list(board.legal_moves), prev_best, 0)
        for i, move in enumerate(moves):
            self.path_keys.append(_key(board))
            board.push(move)
            if i == 0:
                score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            else:
                # Principal variation search: prove this move is worse with a
                # cheap "null window" search; only re-search if it might be better.
                score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                if alpha < score < beta:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            self.path_keys.pop()

            if score > best_score:
                best_score, best_move = score, move
                self._partial_best = (score, move)
            if score > alpha:
                alpha = score

        self.tt[_key(board)] = (depth, best_score, EXACT, best_move)
        return best_score, best_move

    # ---------------------------
    # Main alpha-beta search
    # ---------------------------
    def _negamax(self, board, depth, alpha, beta, ply):
        self.nodes += 1
        if self.nodes & 2047 == 0 and time.time() > self.deadline:
            raise SearchTimeout()

        key = _key(board)

        # Draws by repetition or the 50-move rule.
        if board.halfmove_clock >= 100 or key in self.path_keys or key in self.game_keys:
            return 0

        in_check = board.is_check()
        # Check extension: don't let a check push a tactic past the horizon.
        if in_check:
            depth += 1

        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply)

        # Transposition table lookup.
        tt_move = None
        entry = self.tt.get(key)
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = entry
            if tt_depth >= depth:
                tt_score = _score_from_tt(tt_score, ply)
                if tt_flag == EXACT:
                    return tt_score
                if tt_flag == LOWER and tt_score >= beta:
                    return tt_score
                if tt_flag == UPPER and tt_score <= alpha:
                    return tt_score

        # Null-move pruning: if we could skip our turn and still be winning by
        # more than beta, this position is too good; the opponent won't allow it.
        # Unsafe in check and in pawn endgames (zugzwang), so skip those.
        if (self.use_null_move and depth >= 3 and not in_check and beta < MATE_THRESHOLD
                and _has_non_pawn_material(board, board.turn)
                and self.eval_fn(board) >= beta):
            board.push(chess.Move.null())
            self.path_keys.append(key)
            score = -self._negamax(board, depth - 3, -beta, -beta + 1, ply + 1)
            self.path_keys.pop()
            board.pop()
            if score >= beta:
                return beta

        moves = list(board.legal_moves)
        if not moves:
            # Checkmate (worse the sooner it happens) or stalemate.
            return -MATE_SCORE + ply if in_check else 0

        moves = self._order_moves(board, moves, tt_move, ply)
        original_alpha = alpha
        best_score, best_move = -INF, None

        self.path_keys.append(key)
        for i, move in enumerate(moves):
            is_quiet = not board.is_capture(move) and move.promotion is None
            board.push(move)
            gives_check = board.is_check()

            if i == 0:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Late move reductions: quiet moves ordered late are usually bad,
                # so search them shallower first and only search fully if they surprise us.
                reduction = 0
                if (self.use_lmr and depth >= 3 and i >= 3 and is_quiet
                        and not in_check and not gives_check):
                    reduction = 1 if i < 8 else 2
                score = -self._negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                if score > alpha and reduction:
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if score > best_score:
                best_score, best_move = score, move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                # Beta cutoff: this move is so good the opponent won't allow this line.
                if is_quiet:
                    self._store_killer(move, ply)
                    self.history[(board.turn, move.from_square, move.to_square)] = \
                        self.history.get((board.turn, move.from_square, move.to_square), 0) + depth * depth
                break
        self.path_keys.pop()

        if best_score <= original_alpha:
            flag = UPPER
        elif best_score >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self.tt[key] = (depth, _score_to_tt(best_score, ply), flag, best_move)
        return best_score

    # ---------------------------
    # Quiescence search: only captures (or check evasions) until the position is quiet
    # ---------------------------
    def _quiesce(self, board, alpha, beta, ply):
        self.nodes += 1
        if self.nodes & 2047 == 0 and time.time() > self.deadline:
            raise SearchTimeout()

        in_check = board.is_check()
        if in_check:
            # In check you can't "stand pat": every evasion must be considered.
            moves = list(board.legal_moves)
            if not moves:
                return -MATE_SCORE + ply
            best = -INF
        else:
            # "Stand pat": the side to move can usually decline to capture,
            # so the static eval is a lower bound on its score.
            stand_pat = self.eval_fn(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            best = stand_pat
            moves = list(board.generate_legal_captures())
            # Add quiet queen promotions too; they swing the eval as much as a capture.
            moves += [m for m in board.legal_moves
                      if m.promotion == chess.QUEEN and not board.is_capture(m)]

        moves.sort(key=lambda m: self._capture_score(board, m), reverse=True)
        for move in moves:
            if not in_check:
                # Delta pruning: even winning this piece (plus a margin) can't raise alpha.
                victim = board.piece_type_at(move.to_square)
                gain = SIMPLE_VALUE[victim] if victim else (100 if board.is_en_passant(move) else 0)
                if move.promotion:
                    gain += SIMPLE_VALUE[move.promotion] - 100
                if stand_pat + gain + 200 < alpha:
                    continue
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best

    # ---------------------------
    # Move ordering
    # ---------------------------
    def _capture_score(self, board, move):
        # MVV-LVA: capture the most valuable victim with the least valuable attacker first.
        victim = board.piece_type_at(move.to_square)
        if victim is None:
            victim_value = 100 if board.is_en_passant(move) else 0
        else:
            victim_value = SIMPLE_VALUE[victim]
        attacker_value = SIMPLE_VALUE[board.piece_type_at(move.from_square)] or 1000  # king
        score = victim_value * 10 - attacker_value // 10
        if move.promotion:
            score += SIMPLE_VALUE[move.promotion] * 10
        return score

    def _order_moves(self, board, moves, tt_move, ply):
        killers = self.killers[ply] if ply < len(self.killers) else (None, None)
        turn = board.turn

        def score(move):
            if move == tt_move:
                return 10_000_000
            if board.is_capture(move) or move.promotion:
                return 1_000_000 + self._capture_score(board, move)
            if move == killers[0]:
                return 900_000
            if move == killers[1]:
                return 800_000
            return self.history.get((turn, move.from_square, move.to_square), 0)

        return sorted(moves, key=score, reverse=True)

    def _store_killer(self, move, ply):
        if ply >= len(self.killers):
            return
        k = self.killers[ply]
        if k[0] != move:
            k[1] = k[0]
            k[0] = move

    def _pv(self, board, depth):
        """Reconstruct the expected line of play from the transposition table."""
        pv, b, seen = [], board.copy(), set()
        for _ in range(depth):
            entry = self.tt.get(_key(b))
            if entry is None or entry[3] is None or entry[3] not in b.legal_moves:
                break
            k = _key(b)
            if k in seen:
                break
            seen.add(k)
            pv.append(entry[3])
            b.push(entry[3])
        return pv


# ---------------------------
# Helpers
# ---------------------------
def _has_non_pawn_material(board, color):
    return bool(board.occupied_co[color] & ~board.pawns & ~board.kings)


# Mate scores are stored relative to the node (not the root) in the cache, so
# "mate in 3 from here" stays correct when the position is reached at another depth.
def _score_to_tt(score, ply):
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _score_from_tt(score, ply):
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


def format_score(score):
    if score >= MATE_THRESHOLD:
        return f"M{(MATE_SCORE - score + 1) // 2}"
    if score <= -MATE_THRESHOLD:
        return f"-M{(MATE_SCORE + score) // 2}"
    return f"{score / 100:+.2f}"
