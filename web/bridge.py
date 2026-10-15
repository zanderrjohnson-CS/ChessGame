# Glue between the web page (JavaScript) and the v2 engine (Python), running
# in the browser through Pyodide. All chess rules come from python-chess.
import json

import chess

from chess_engine.v2.search import Searcher, format_score

_searcher = Searcher()


def _board(moves_json):
    board = chess.Board()
    for uci in json.loads(moves_json):
        board.push_uci(uci)
    return board


def _san_list(board):
    replay, sans = chess.Board(), []
    for move in board.move_stack:
        sans.append(replay.san(move))
        replay.push(move)
    return sans


def state(moves_json):
    board = _board(moves_json)
    outcome = board.outcome(claim_draw=True)
    pieces = [None] * 64
    for sq, piece in board.piece_map().items():
        pieces[sq] = piece.symbol()
    return json.dumps({
        "pieces": pieces,
        "turn": "w" if board.turn == chess.WHITE else "b",
        "legal": [m.uci() for m in board.legal_moves],
        "check": board.king(board.turn) if board.is_check() else None,
        "san": _san_list(board),
        "over": outcome is not None,
        "result": outcome.result() if outcome else None,
        "reason": outcome.termination.name.replace("_", " ").lower() if outcome else None,
    })


def think(moves_json, seconds):
    board = _board(moves_json)
    r = _searcher.search(board, time_limit=float(seconds))
    pv_san, b = [], board.copy()
    for m in r["pv"]:
        pv_san.append(b.san(m))
        b.push(m)
    score = r["score"] if board.turn == chess.WHITE else -r["score"]  # White's view for display
    return json.dumps({
        "move": r["move"].uci() if r["move"] else None,
        "eval": format_score(score),
        "depth": r["depth"],
        "nodes": r["nodes"],
        "time": round(r["time"], 2),
        "pv": pv_san,
    })
