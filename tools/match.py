# ===============================
# tools/match.py
# ===============================
# Play two engines against each other to measure which is stronger.
# A handful of games proves nothing; run at least ~50-100 per comparison.
#
#   python -m tools.match v2 v1 --games 40 --time 0.5
#   python -m tools.match v2 v2-material --games 100 --time 0.2
#
# Engines: v2 (material + piece-square tables), v2-material (material only),
#          v1 (policy network + blunder filter; needs torch and policy_net.pt), random
# ===============================
import argparse
import math
import random
from multiprocessing import Pool

import chess

# Short, varied openings so the games aren't all identical.
OPENINGS = [
    "", "e2e4 e7e5", "e2e4 c7c5", "d2d4 d7d5", "d2d4 g8f6", "c2c4 e7e5",
    "e2e4 e7e6", "e2e4 c7c6", "g1f3 d7d5", "d2d4 g8f6 c2c4 e7e6",
    "e2e4 e7e5 g1f3 b8c6", "e2e4 d7d5", "d2d4 f7f5", "e2e4 g7g6",
    "c2c4 c7c5", "e2e4 e7e5 f1c4 g8f6",
]


def make_engine(name, think_time):
    if name == "random":
        return lambda board: random.choice(list(board.legal_moves))
    if name == "v1":
        from chess_engine.move_select import get_best_move as v1_move
        import contextlib, io
        def v1(board):
            with contextlib.redirect_stdout(io.StringIO()):  # v1 prints a lot
                return chess.Move.from_uci(v1_move(board.fen()))
        return v1

    from chess_engine.v2.search import Searcher
    from chess_engine.v2.evaluate import evaluate, evaluate_material
    eval_fn = {"v2": evaluate, "v2-material": evaluate_material}[name]
    searcher = Searcher(eval_fn=eval_fn)
    return lambda board: searcher.search(board, time_limit=think_time)["move"]


def play_game(args):
    white_name, black_name, opening, think_time, seed = args
    random.seed(seed)
    engines = {chess.WHITE: make_engine(white_name, think_time),
               chess.BLACK: make_engine(black_name, think_time)}
    board = chess.Board()
    for uci in opening.split():
        board.push_uci(uci)
    while not board.is_game_over(claim_draw=True) and board.ply() < 300:
        board.push(engines[board.turn](board))
    result = board.result(claim_draw=True)
    if result == "*":
        result = "1/2-1/2"  # move limit reached
    return white_name, black_name, result


def elo_diff(score):
    score = min(max(score, 0.001), 0.999)
    return -400 * math.log10(1 / score - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_a")
    parser.add_argument("engine_b")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--time", type=float, default=0.5, help="seconds per move for v2")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = []
    for i in range(args.games):
        opening = OPENINGS[(i // 2) % len(OPENINGS)]
        # Each opening is played twice with colors swapped, to be fair.
        white, black = (args.engine_a, args.engine_b) if i % 2 == 0 else (args.engine_b, args.engine_a)
        jobs.append((white, black, opening, args.time, i))

    wins = draws = losses = 0
    with Pool(args.workers) as pool:
        for white, black, result in pool.imap_unordered(play_game, jobs):
            a_is_white = white == args.engine_a
            if result == "1/2-1/2":
                draws += 1
            elif (result == "1-0") == a_is_white:
                wins += 1
            else:
                losses += 1
            print(f"{white:>12} vs {black:<12} {result:8}  running: +{wins} ={draws} -{losses}", flush=True)

    n = wins + draws + losses
    score = (wins + 0.5 * draws) / n
    print(f"\n{args.engine_a} vs {args.engine_b}: +{wins} ={draws} -{losses}  "
          f"score {score:.1%}  ≈ {elo_diff(score):+.0f} Elo for {args.engine_a}")


if __name__ == "__main__":
    main()
