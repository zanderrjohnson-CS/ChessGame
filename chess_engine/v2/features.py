# ===============================
# chess_engine/v2/features.py
# ===============================
# Relational features: things about a piece that depend on the OTHER pieces
# (is this pawn passed? is this rook on an open file? is the king sheltered?).
# Piece-square tables can't see any of this; they only know "piece X on square Y".
#
# Every feature is a count, computed as (White's count - Black's count).
# For example, if White has 2 doubled pawns and Black has 1, DOUBLED = +1.
#
# The evaluator turns counts into a score with a weight per feature:
#     score = sum(count_i * weight_i)
# Each feature has two weights, one for the middlegame and one for the endgame
# (a passed pawn matters much more in the endgame), blended by game phase.
#
# Because the score is a weighted sum (linear in the weights), the weights can be
# learned from data with logistic regression. That is what tools/tune.py does.
# ===============================
import chess


FEATURE_NAMES = [
    # Material adjustments on top of the piece-square tables' built-in values
    "pawn", "knight", "bishop", "rook", "queen",
    # Pawn structure
    "doubled_pawn", "isolated_pawn", "supported_pawn", "phalanx_pawn",
    "passed_pawn_r2", "passed_pawn_r3", "passed_pawn_r4",
    "passed_pawn_r5", "passed_pawn_r6", "passed_pawn_r7",
    # Minor pieces
    "bishop_pair", "fianchetto", "knight_outpost",
    # Rooks
    "rook_open_file", "rook_semi_open_file", "rook_on_7th", "connected_rooks",
    # King safety
    "king_castled", "king_pawn_shield", "king_zone_attacks",
    # Mobility: squares each piece can move to that aren't guarded by enemy pawns
    "knight_mobility", "bishop_mobility", "rook_mobility", "queen_mobility",
]
NUM_FEATURES = len(FEATURE_NAMES)
F = {name: i for i, name in enumerate(FEATURE_NAMES)}


# ===============================
# Precomputed bitboard masks
# ===============================
# A bitboard is a 64-bit integer with one bit per square. Bitwise AND/OR on them
# answers questions like "are there enemy pawns in front of this pawn?" in one step.

FILE_MASKS = [chess.BB_FILES[f] for f in range(8)]
ADJACENT_FILES = [
    (FILE_MASKS[f - 1] if f > 0 else 0) | (FILE_MASKS[f + 1] if f < 7 else 0)
    for f in range(8)
]


def _ranks_ahead(color, rank):
    """Bitboard of every rank strictly in front of `rank` from `color`'s side."""
    mask = 0
    ranks = range(rank + 1, 8) if color == chess.WHITE else range(0, rank)
    for r in ranks:
        mask |= chess.BB_RANKS[r]
    return mask


# PASSED_MASK[color][sq]: squares where an enemy pawn would stop a pawn on sq
# from being passed (same file + adjacent files, in front of it).
PASSED_MASK = {c: [0] * 64 for c in chess.COLORS}
# OUTPOST_MASK[color][sq]: squares where an enemy pawn could later attack sq from
# (adjacent files, in front). If none of them has an enemy pawn, it's an outpost.
OUTPOST_MASK = {c: [0] * 64 for c in chess.COLORS}
for _c in chess.COLORS:
    for _sq in chess.SQUARES:
        _f, _r = chess.square_file(_sq), chess.square_rank(_sq)
        _ahead = _ranks_ahead(_c, _r)
        PASSED_MASK[_c][_sq] = _ahead & (FILE_MASKS[_f] | ADJACENT_FILES[_f])
        OUTPOST_MASK[_c][_sq] = _ahead & ADJACENT_FILES[_f]


def _relative_rank(color, sq):
    r = chess.square_rank(sq)
    return r if color == chess.WHITE else 7 - r


def _pawn_attacks(pawns, color):
    """All squares attacked by a set of pawns of `color`."""
    if color == chess.WHITE:
        return ((pawns << 7) & ~chess.BB_FILE_H | (pawns << 9) & ~chess.BB_FILE_A) & chess.BB_ALL
    return (pawns >> 9) & ~chess.BB_FILE_H | (pawns >> 7) & ~chess.BB_FILE_A


# Fianchetto: bishop on its long-diagonal square with the knight-file pawn pushed
# one square in front of it (b2+b3 / g2+g3 for White, b7+b6 / g7+g6 for Black).
FIANCHETTO = {
    chess.WHITE: [(chess.B2, chess.B3), (chess.G2, chess.G3)],
    chess.BLACK: [(chess.B7, chess.B6), (chess.G7, chess.G6)],
}


# ===============================
# Feature extraction
# ===============================
def _side_features(board, color, out, sign):
    """Add `sign` * this side's feature counts into `out`."""
    us_pawns = board.pieces_mask(chess.PAWN, color)
    them_pawns = board.pieces_mask(chess.PAWN, not color)
    us_occ = board.occupied_co[color]
    them_attacks_by_pawns = _pawn_attacks(them_pawns, not color)
    occupied = board.occupied

    # --- Material counts ---
    out[F["pawn"]] += sign * chess.popcount(us_pawns)
    out[F["knight"]] += sign * chess.popcount(board.pieces_mask(chess.KNIGHT, color))
    bishops = board.pieces_mask(chess.BISHOP, color)
    out[F["bishop"]] += sign * chess.popcount(bishops)
    rooks = board.pieces_mask(chess.ROOK, color)
    out[F["rook"]] += sign * chess.popcount(rooks)
    out[F["queen"]] += sign * chess.popcount(board.pieces_mask(chess.QUEEN, color))

    # --- Pawn structure ---
    for f in range(8):
        n = chess.popcount(us_pawns & FILE_MASKS[f])
        if n:
            if n > 1:
                out[F["doubled_pawn"]] += sign * (n - 1)
            if not us_pawns & ADJACENT_FILES[f]:
                out[F["isolated_pawn"]] += sign * n

    our_pawn_attacks = _pawn_attacks(us_pawns, color)
    for sq in chess.scan_forward(us_pawns):
        bb = chess.BB_SQUARES[sq]
        if bb & our_pawn_attacks:
            out[F["supported_pawn"]] += sign
        # A friendly pawn directly beside this one (counted once per pair: look right only)
        if chess.square_file(sq) < 7 and us_pawns & chess.BB_SQUARES[sq + 1]:
            out[F["phalanx_pawn"]] += sign
        if not them_pawns & PASSED_MASK[color][sq]:
            rr = _relative_rank(color, sq)
            if 1 <= rr <= 6:
                out[F[f"passed_pawn_r{rr + 1}"]] += sign

    # --- Minor pieces ---
    if chess.popcount(bishops) >= 2:
        out[F["bishop_pair"]] += sign
    for bishop_sq, pawn_sq in FIANCHETTO[color]:
        if bishops & chess.BB_SQUARES[bishop_sq] and us_pawns & chess.BB_SQUARES[pawn_sq]:
            out[F["fianchetto"]] += sign

    king_sq = board.king(color)
    them_king_sq = board.king(not color)
    them_king_zone = chess.BB_KING_ATTACKS[them_king_sq] | chess.BB_SQUARES[them_king_sq] \
        if them_king_sq is not None else 0
    zone_attacks = 0
    mobility_area = ~us_occ & ~them_attacks_by_pawns

    for sq in chess.scan_forward(board.pieces_mask(chess.KNIGHT, color)):
        attacks = chess.BB_KNIGHT_ATTACKS[sq]
        out[F["knight_mobility"]] += sign * chess.popcount(attacks & mobility_area)
        zone_attacks += chess.popcount(attacks & them_king_zone)
        if (3 <= _relative_rank(color, sq) <= 5 and chess.BB_SQUARES[sq] & our_pawn_attacks
                and not them_pawns & OUTPOST_MASK[color][sq]):
            out[F["knight_outpost"]] += sign

    for sq in chess.scan_forward(bishops):
        attacks = chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occupied]
        out[F["bishop_mobility"]] += sign * chess.popcount(attacks & mobility_area)
        zone_attacks += chess.popcount(attacks & them_king_zone)

    # --- Rooks ---
    seventh = chess.BB_RANK_7 if color == chess.WHITE else chess.BB_RANK_2
    for sq in chess.scan_forward(rooks):
        attacks = (chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occupied]
                   | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occupied])
        out[F["rook_mobility"]] += sign * chess.popcount(attacks & mobility_area)
        zone_attacks += chess.popcount(attacks & them_king_zone)
        file_mask = FILE_MASKS[chess.square_file(sq)]
        if not us_pawns & file_mask:
            if not them_pawns & file_mask:
                out[F["rook_open_file"]] += sign
            else:
                out[F["rook_semi_open_file"]] += sign
        if chess.BB_SQUARES[sq] & seventh:
            out[F["rook_on_7th"]] += sign
        if attacks & rooks:
            out[F["connected_rooks"]] += sign  # counted by both rooks of the pair

    for sq in chess.scan_forward(board.pieces_mask(chess.QUEEN, color)):
        attacks = (chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occupied]
                   | chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occupied]
                   | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occupied])
        out[F["queen_mobility"]] += sign * chess.popcount(attacks & mobility_area)
        zone_attacks += chess.popcount(attacks & them_king_zone)

    # Attacks on the enemy king's surroundings are good for us.
    out[F["king_zone_attacks"]] += sign * zone_attacks

    # --- Our king's safety ---
    if king_sq is not None:
        rr = _relative_rank(color, king_sq)
        kf = chess.square_file(king_sq)
        if rr == 0 and kf in (1, 2, 6, 7):
            out[F["king_castled"]] += sign
        if rr <= 1:
            shield_files = FILE_MASKS[kf] | ADJACENT_FILES[kf]
            step = 1 if color == chess.WHITE else -1
            for d in (1, 2):
                r = chess.square_rank(king_sq) + step * d
                if 0 <= r <= 7:
                    out[F["king_pawn_shield"]] += sign * chess.popcount(
                        us_pawns & shield_files & chess.BB_RANKS[r])


def extract_features(board: chess.Board) -> list[int]:
    """Feature counts for this position, from White's point of view (White - Black)."""
    out = [0] * NUM_FEATURES
    _side_features(board, chess.WHITE, out, +1)
    _side_features(board, chess.BLACK, out, -1)
    return out
