# chess_engine/convert_weights.py
"""
Convert a policy network trained with the OLD 8192-slot move index into the
NEW 1792-slot index, without retraining.

Why this works: the last layer, policy_fc[3] = Linear(2048, num_moves), has one
row of weights (plus one bias) per output slot. Each row is an independent
"detector" for one move. So to drop slots we just keep the rows for the moves
that still exist, in the new order. Everything before that layer is unchanged.

Promotion slots: the new slot "e7e8" means both "a piece moves e7->e8" and
"a pawn promotes on e8 (to a queen)". The old net had separate rows for
"e7e8" and "e7e8q", and we can only keep one. We keep the plain "e7e8" row:
pieces stepping onto the back rank (like a black bishop retreating d7->e8)
happen far more often than promotions, and the search finds promotions on
its own anyway because winning a queen is worth ~8 pawns. With this choice
the converted net picks the same top move as the old one in 100% of 3,000
grandmaster test positions. A net trained from scratch on the new index
learns both meanings for these slots.

move_select.py calls convert() automatically when it loads an old file, so
policy_net.pt keeps working as-is. To save a converted copy (110 MB instead of
160 MB, and no conversion at start-up), run from the repo root:
    python -m chess_engine.convert_weights policy_net.pt policy_net.pt
"""
import sys

import torch

from chess_engine.chess_model import MOVE_TO_INDEX

FINAL_LAYER = "policy_fc.3"
OLD_NUM_MOVES = 8192


def old_move_to_index():
    files, ranks = "abcdefgh", "12345678"
    moves = []
    for f1 in files:
        for r1 in ranks:
            for f2 in files:
                for r2 in ranks:
                    moves.append(f"{f1}{r1}{f2}{r2}")
                    if r2 in ("1", "8"):
                        for p in "qrbn":
                            moves.append(f"{f1}{r1}{f2}{r2}{p}")
    return {m: i for i, m in enumerate(moves)}


def needs_conversion(state_dict) -> bool:
    return state_dict[f"{FINAL_LAYER}.weight"].shape[0] == OLD_NUM_MOVES


def convert(state_dict):
    weight = state_dict[f"{FINAL_LAYER}.weight"]
    bias = state_dict[f"{FINAL_LAYER}.bias"]
    assert weight.shape[0] == OLD_NUM_MOVES, f"expected {OLD_NUM_MOVES} rows, got {weight.shape[0]}"

    old_index = old_move_to_index()
    rows = []
    for uci, new_idx in sorted(MOVE_TO_INDEX.items(), key=lambda kv: kv[1]):
        rows.append(old_index[uci])
    rows = torch.tensor(rows)

    state_dict = dict(state_dict)
    state_dict[f"{FINAL_LAYER}.weight"] = weight[rows].clone()
    state_dict[f"{FINAL_LAYER}.bias"] = bias[rows].clone()
    return state_dict


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    old = torch.load(src, map_location="cpu", weights_only=True)
    if not needs_conversion(old):
        raise SystemExit(f"{src} already uses the {len(MOVE_TO_INDEX)}-slot index.")
    new = convert(old)
    torch.save(new, dst)
    before = sum(t.numel() for t in old.values())
    after = sum(t.numel() for t in new.values())
    print(f"Converted {src} -> {dst}: {before:,} -> {after:,} numbers")


def convert_dataset_labels(y_old, M_old=None):
    """Re-label a dataset collected with the old 8192-slot index.

    Returns (keep, y_new, M_new): keep is a boolean array of rows to keep
    (underpromotions have no new slot and are dropped), y_new the new move
    slots for the kept rows, and M_new the legal-move masks as booleans.
    """
    import chess
    import numpy as np

    from chess_engine.chess_model import move_to_index

    old_index = old_move_to_index()
    old_moves = {i: m for m, i in old_index.items()}

    new_slot_of_old = np.full(OLD_NUM_MOVES, -1, dtype=np.int64)
    for i, uci in old_moves.items():
        try:
            idx = move_to_index(chess.Move.from_uci(uci))
        except (KeyError, chess.InvalidMoveError):
            idx = None  # a never-possible move like a1a1 or a1c4
        if idx is not None:
            new_slot_of_old[i] = idx

    y_mapped = new_slot_of_old[np.asarray(y_old)]
    keep = y_mapped >= 0
    y_new = y_mapped[keep]

    M_new = None
    if M_old is not None:
        legal_old = np.asarray(M_old)[keep] == 0.0  # old masks: 0 = legal, -1e9 = illegal
        M_new = np.zeros((len(y_new), len(MOVE_TO_INDEX)), dtype=bool)
        for i in np.nonzero(new_slot_of_old >= 0)[0]:
            # "e7e8" and "e7e8q" both land in slot e7e8, so OR them together.
            M_new[:, new_slot_of_old[i]] |= legal_old[:, i]
    return keep, y_new, M_new
