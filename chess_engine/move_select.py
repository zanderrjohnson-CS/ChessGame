# chess_engine/move_select.py
"""
Move selection: combines the two halves of the bot.

1. Intuition  - the policy network (chess_model.py) ranks every legal move.
2. Calculation - alpha-beta search (search.py) checks those moves, in the
   policy's order, several moves deep.

Final choice: the move search scores best. If several moves are within
TIE_MARGIN centipawns of the best, we trust the network and pick the one it
liked most. So search vetoes tactical mistakes, and the network decides
everything search can't tell apart (style, plans, openings).
"""
import chess
import torch

from chess_engine.chess_model import PolicyNetwork, MOVE_TO_INDEX, board_to_tensor, move_to_index
from chess_engine.convert_weights import convert, needs_conversion
from chess_engine.search import MATE_SCORE, TIE_MARGIN, Searcher, _sort_key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load policy network. Files trained with the old 8192-slot move index are
# converted to the new 1792-slot index on the fly (see convert_weights.py).
state_dict = torch.load("policy_net.pt", map_location=device, weights_only=True)
if needs_conversion(state_dict):
    state_dict = convert(state_dict)
    print("Converted policy_net.pt from the old 8192-move index")
policy_model = PolicyNetwork(len(MOVE_TO_INDEX)).to(device)
policy_model.load_state_dict(state_dict)
policy_model.eval()
print("Loaded policy network")

SEARCH_TIME_SECONDS = 2.0
SEARCH_MAX_DEPTH = 4


def get_policy_ranked_moves(board):
    """Get all legal moves ranked by policy network preference"""
    board_tensor = torch.tensor(
        board_to_tensor(board),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = policy_model(board_tensor).squeeze(0)

    # Get scores for all legal moves. Underpromotions have no slot; give them
    # a very low score so they go last, but the search still checks them.
    legal_moves = []
    for move in board.legal_moves:
        idx = move_to_index(move)
        score = logits[idx].item() if idx is not None else float("-inf")
        legal_moves.append((move, score))

    # Sort by score (highest first)
    legal_moves.sort(key=lambda x: x[1], reverse=True)
    return legal_moves


def _fmt(score):
    """Centipawns -> '+1.25' (pawns), or 'M2' for mate in 2."""
    if abs(score) >= MATE_SCORE - 1000:
        plies = MATE_SCORE - abs(score)
        return f"{'' if score > 0 else '-'}M{(plies + 1) // 2}"
    return f"{score / 100:+.2f}"


def get_best_move(fen: str, time_limit: float = SEARCH_TIME_SECONDS) -> str:
    """Find best move: policy network for ordering, alpha-beta search to verify."""
    board = chess.Board(fen)
    if board.is_game_over():
        return None

    ranked = get_policy_ranked_moves(board)
    policy_rank = {move: i for i, (move, _) in enumerate(ranked)}

    searcher = Searcher(time_limit=time_limit)
    results, depth = searcher.run(board, [m for m, _ in ranked], SEARCH_MAX_DEPTH)

    best_score = max(score for _, score in results if score is not None)
    near_best = [m for m, score in results
                 if score is not None and score > best_score - TIE_MARGIN]
    choice = min(near_best, key=lambda m: policy_rank[m])

    print(f"\nSearched depth {depth}, {searcher.nodes} positions")
    for move, score in sorted(results, key=_sort_key, reverse=True)[:5]:
        if score is None:
            break
        mark = "<-" if move == choice else ""
        print(f"  {move.uci()}: search={_fmt(score)}, policy rank #{policy_rank[move] + 1} {mark}")
    return choice.uci()


# Quick test
if __name__ == "__main__":
    board = chess.Board()
    print("Starting position:")
    print(board)
    print(f"\nBest move: {get_best_move(board.fen())}")
