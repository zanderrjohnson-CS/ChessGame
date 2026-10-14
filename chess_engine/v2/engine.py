# ===============================
# chess_engine/v2/engine.py
# ===============================
# What the API calls: FEN in, UCI move out.
import chess

from chess_engine.v2.search import Searcher, format_score

# One searcher for the whole server so its transposition table carries over
# between moves of a game.
_searcher = Searcher()


def get_best_move(fen: str, time_limit: float = 2.0, moves: list[str] | None = None) -> str | None:
    """Search the position for `time_limit` seconds and return the best move in UCI.

    `moves` (optional) is the game's move history from the starting position;
    passing it lets the engine detect and avoid/seek threefold repetitions.
    """
    if moves:
        board = chess.Board()
        for uci in moves:
            board.push_uci(uci)
    else:
        board = chess.Board(fen)

    result = _searcher.search(board, time_limit=time_limit)
    if result["move"] is None:
        return None
    print(f"v2: {result['move'].uci()}  eval {format_score(result['score'])}  "
          f"depth {result['depth']}  nodes {result['nodes']}  "
          f"pv {' '.join(m.uci() for m in result['pv'])}")
    return result["move"].uci()


if __name__ == "__main__":
    import sys
    fen = sys.argv[1] if len(sys.argv) > 1 else chess.STARTING_FEN
    board = chess.Board(fen)
    print(board, "\n")
    Searcher().search(board, time_limit=float(sys.argv[2]) if len(sys.argv) > 2 else 5.0, verbose=True)
