#!/usr/bin/env python3
# ===============================
# uci.py  —  run v2 as a UCI engine
# ===============================
# UCI (Universal Chess Interface) is the text protocol chess GUIs and bot
# runners use to talk to engines over stdin/stdout. Speaking UCI lets v2 plug
# into lichess-bot (to play rated games on Lichess), cutechess, Arena, etc.
#
#   python uci.py
#   > uci
#   > position startpos moves e2e4
#   > go wtime 60000 btime 60000 winc 1000 binc 1000
#   < bestmove e7e5
# ===============================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess  # noqa: E402

from chess_engine.v2.search import Searcher, MATE_THRESHOLD, MATE_SCORE  # noqa: E402

ENGINE_NAME = "ZanderBot v2"
AUTHOR = "Zander Johnson"


def think_time(board, params):
    """How long to search, in seconds, given the clock info from a 'go' command."""
    if "movetime" in params:
        return max(params["movetime"] / 1000 - 0.05, 0.01)
    if "depth" in params or "infinite" in params:
        return 3600.0
    our_time = params.get("wtime" if board.turn == chess.WHITE else "btime")
    if our_time is None:
        return 2.0
    inc = params.get("winc" if board.turn == chess.WHITE else "binc", 0)
    moves_to_go = params.get("movestogo", 30)
    # Spend an even share of the remaining time plus most of the increment,
    # never more than a fifth of what's left, and keep a safety margin for lag.
    budget = our_time / max(moves_to_go, 1) + inc * 0.75
    budget = min(budget, our_time / 5)
    return max((budget - 100) / 1000, 0.02)


def uci_score(score):
    if score >= MATE_THRESHOLD:
        return f"mate {(MATE_SCORE - score + 1) // 2}"
    if score <= -MATE_THRESHOLD:
        return f"mate -{(MATE_SCORE + score) // 2}"
    return f"cp {score}"


def main():
    searcher = Searcher()
    board = chess.Board()

    for line in sys.stdin:
        tokens = line.split()
        if not tokens:
            continue
        cmd = tokens[0]

        if cmd == "uci":
            print(f"id name {ENGINE_NAME}")
            print(f"id author {AUTHOR}")
            print("uciok", flush=True)
        elif cmd == "isready":
            print("readyok", flush=True)
        elif cmd == "ucinewgame":
            searcher = Searcher()
        elif cmd == "position":
            if tokens[1] == "startpos":
                board = chess.Board()
                rest = tokens[2:]
            else:  # position fen <6 fields> [moves ...]
                board = chess.Board(" ".join(tokens[2:8]))
                rest = tokens[8:]
            if rest and rest[0] == "moves":
                for uci in rest[1:]:
                    board.push_uci(uci)
        elif cmd == "go":
            params = {}
            i = 1
            while i < len(tokens):
                if tokens[i] in ("infinite", "ponder"):
                    params[tokens[i]] = True
                    i += 1
                else:
                    if i + 1 < len(tokens):
                        params[tokens[i]] = int(tokens[i + 1])
                    i += 2
            max_depth = params.get("depth", 64)
            result = searcher.search(board, time_limit=think_time(board, params), max_depth=max_depth)
            if result["move"] is None:
                print("bestmove 0000", flush=True)
                continue
            nps = int(result["nodes"] / max(result["time"], 1e-3))
            pv = " ".join(m.uci() for m in result["pv"]) or result["move"].uci()
            print(f"info depth {result['depth']} score {uci_score(result['score'])} "
                  f"nodes {result['nodes']} nps {nps} time {int(result['time'] * 1000)} pv {pv}")
            print(f"bestmove {result['move'].uci()}", flush=True)
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
