# ===============================
# tools/tune.py  —  Texel tuning: learn the evaluator's feature weights from game results
# ===============================
# The idea (named after the Texel engine, whose author published it):
#
#   1. Take lots of positions from real games, each labeled with how the game
#      ended: 1 = White won, 0.5 = draw, 0 = Black won.
#   2. The evaluator gives each position a score in centipawns. Squash it into a
#      predicted "chance White wins" with a sigmoid:
#          p = 1 / (1 + 10^(-K * score / 400))
#      (+400 centipawns ≈ 91% for K=1, the same curve the Elo system uses.)
#   3. The loss is the mean squared error between p and the actual result.
#   4. Gradient descent adjusts the weights to make the loss smaller.
#
# This is logistic regression: a linear model (score = sum of count * weight)
# passed through a sigmoid. Same training loop as a neural network (forward
# pass, loss, gradient, update, watch validation loss), just one layer deep.
#
# Usage:
#   python -m tools.tune data/quiet-labeled.epd          # EPD: FEN + "1-0" / "0-1" / "1/2-1/2"
#   python -m tools.tune games.pgn --pgn                 # or your own PGN games
#
# Writes chess_engine/v2/weights.json, which evaluate.py loads automatically.
# ===============================
import argparse
import json
import math
import os
import random
import time
from multiprocessing import Pool

import numpy as np
import chess
import chess.pgn

from chess_engine.v2.evaluate import (HAND_WEIGHTS, MAX_PHASE, WEIGHTS_PATH,
                                      evaluate_pst, pst_score_and_phase)
from chess_engine.v2.features import FEATURE_NAMES, NUM_FEATURES, extract_features

RESULTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


# ===============================
# 1. Load labeled positions
# ===============================
def load_epd(path, limit=None):
    data = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            result = line.split('"')[1] if '"' in line else parts[-1].strip(";")
            if result not in RESULTS:
                continue
            data.append((" ".join(parts[:4]) + " 0 1", RESULTS[result]))
            if limit and len(data) >= limit:
                break
    return data


def load_pgn(path, limit=None, skip_plies=10):
    """Sample quiet positions from PGN games (not in check, and no capture
    sequence changes the static eval, i.e. quiescence search == static eval)."""
    from chess_engine.v2.search import Searcher, INF
    searcher = Searcher(eval_fn=evaluate_pst)
    searcher.nodes, searcher.deadline = 0, float("inf")
    data = []
    with open(path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            result = game.headers.get("Result")
            if result not in RESULTS:
                continue
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                if ply < skip_plies or board.is_check() or random.random() > 0.25:
                    continue
                if searcher._quiesce(board, -INF, INF, 0) == evaluate_pst(board):
                    data.append((board.fen(), RESULTS[result]))
            if limit and len(data) >= limit:
                break
    return data


# ===============================
# 2. Turn positions into numbers (features)
# ===============================
def featurize(item):
    fen, result = item
    board = chess.Board(fen)
    mg, eg, phase = pst_score_and_phase(board)
    return mg, eg, phase, extract_features(board), result


def build_dataset(data, workers):
    with Pool(workers) as pool:
        rows = pool.map(featurize, data, chunksize=2000)
    base_mg = np.array([r[0] for r in rows], dtype=np.float64)
    base_eg = np.array([r[1] for r in rows], dtype=np.float64)
    phase = np.array([r[2] for r in rows], dtype=np.float64) / MAX_PHASE
    feats = np.array([r[3] for r in rows], dtype=np.float32)
    y = np.array([r[4] for r in rows], dtype=np.float64)
    return base_mg, base_eg, phase, feats, y


# ===============================
# 3. The model: score = PST score + X @ w,  prediction = sigmoid(score)
# ===============================
# Each feature has a middlegame and an endgame weight, so the design matrix X
# has two columns per feature: count * phase and count * (1 - phase).
# Tapering is then just part of the input, and the model stays linear in w.
def design_matrix(feats, phase):
    return np.hstack([feats * phase[:, None], feats * (1 - phase)[:, None]])


def predict(base, X, w, k):
    score = base + X @ w
    return 1.0 / (1.0 + np.power(10.0, -k * score / 400.0))


def mse(base, X, w, y, k):
    return float(np.mean((predict(base, X, w, k) - y) ** 2))


def fit_k(base, X, w, y):
    """Find the sigmoid scale K that best fits the CURRENT eval to the results.
    (Tuning K first keeps the weights in centipawns rather than arbitrary units.)"""
    best_k, best_loss = 1.0, float("inf")
    for k in np.arange(0.2, 3.0, 0.05):
        loss = mse(base, X, w, y, k)
        if loss < best_loss:
            best_k, best_loss = float(k), loss
    return best_k


# ===============================
# 4. Training loop: full-batch gradient descent with the Adam optimizer
# ===============================
def train(base, X, y, w, k, val, epochs, lr, l2):
    base_v, X_v, y_v = val
    c = k * math.log(10) / 400.0  # derivative of the sigmoid's exponent w.r.t. the score
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    n = len(y)

    for epoch in range(1, epochs + 1):
        p = predict(base, X, w, k)
        # d(loss)/d(w) by the chain rule: loss -> p -> score -> w
        err = (p - y) * p * (1 - p) * c
        grad = (2.0 / n) * (X.T @ err) + 2 * l2 * w

        # Adam: momentum (m) plus a per-weight step size (v)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** epoch)
        v_hat = v / (1 - beta2 ** epoch)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if epoch == 1 or epoch % 100 == 0:
            print(f"  epoch {epoch:5d}   train loss {mse(base, X, w, y, k):.6f}   "
                  f"validation loss {mse(base_v, X_v, w, y_v, k):.6f}", flush=True)
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="EPD file (or PGN with --pgn)")
    parser.add_argument("--pgn", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="max positions to load")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.5, help="Adam step size, in centipawns")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 penalty pulling weights toward 0")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default=WEIGHTS_PATH)
    args = parser.parse_args()

    t0 = time.time()
    data = load_pgn(args.path, args.limit) if args.pgn else load_epd(args.path, args.limit)
    print(f"Loaded {len(data):,} positions in {time.time() - t0:.0f}s")

    t0 = time.time()
    base_mg, base_eg, phase, feats, y = build_dataset(data, args.workers)
    base = base_mg * phase + base_eg * (1 - phase)
    X = design_matrix(feats, phase)
    print(f"Computed features in {time.time() - t0:.0f}s  ({X.shape[0]:,} rows x {X.shape[1]} columns)")

    # Hold out 10% of positions to detect overfitting. Validation positions are
    # never used to compute gradients.
    idx = np.random.default_rng(0).permutation(len(y))
    n_val = len(y) // 10
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr = (base[tr_idx], X[tr_idx], y[tr_idx])
    val = (base[val_idx], X[val_idx], y[val_idx])

    zero = np.zeros(2 * NUM_FEATURES)
    hand = np.array([HAND_WEIGHTS[n][0] for n in FEATURE_NAMES] +
                    [HAND_WEIGHTS[n][1] for n in FEATURE_NAMES], dtype=np.float64)

    k = fit_k(*tr[:2], zero, tr[2])
    print(f"Sigmoid scale K = {k:.2f}\n")

    print("Validation loss before training:")
    print(f"  piece-square tables only     {mse(*val[:2], zero, val[2], k):.6f}")
    print(f"  + hand-picked feature weights {mse(*val[:2], hand, val[2], k):.6f}\n")

    print("Training (starting from the hand-picked weights):")
    w = train(*tr, hand.copy(), k, val, args.epochs, args.lr, args.l2)

    tuned_val = mse(*val[:2], w, val[2], k)
    print(f"\n  tuned feature weights        {tuned_val:.6f}\n")

    print(f"{'feature':<22}{'hand mg':>9}{'hand eg':>9}{'tuned mg':>10}{'tuned eg':>10}")
    for i, name in enumerate(FEATURE_NAMES):
        h = HAND_WEIGHTS[name]
        print(f"{name:<22}{h[0]:>9}{h[1]:>9}{w[i]:>10.0f}{w[NUM_FEATURES + i]:>10.0f}")

    out = {
        "weights": {name: [round(float(w[i])), round(float(w[NUM_FEATURES + i]))]
                    for i, name in enumerate(FEATURE_NAMES)},
        "info": {
            "dataset": os.path.basename(args.path),
            "positions": int(len(y)),
            "K": k,
            "validation_mse_pst_only": mse(*val[:2], zero, val[2], k),
            "validation_mse_hand": mse(*val[:2], hand, val[2], k),
            "validation_mse_tuned": tuned_val,
            "epochs": args.epochs,
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved weights to {args.out}")


if __name__ == "__main__":
    main()
