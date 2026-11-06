#!/usr/bin/env python3
# uci_engine.py
"""
Lets other chess programs (like lichess-bot) talk to this bot using UCI,
the Universal Chess Interface.

UCI is a plain-text conversation over stdin/stdout. The GUI (or lichess-bot)
sends commands, one per line, and the engine answers:

    GUI -> engine                         engine -> GUI
    uci                                   id name ... / uciok
    isready                               readyok
    position startpos moves e2e4 e7e5     (nothing: just remember the position)
    go wtime 60000 btime 60000 winc 1000  info depth 3 score cp 25 ... / bestmove g1f3
    quit                                  (exit)

lichess-bot handles everything about Lichess itself (logging in, accepting
challenges, sending moves over the internet). This file only answers
"here's a position, what's your move?".

Important detail: in UCI, stdout belongs to the protocol. Any stray print()
would be read by lichess-bot as a (broken) command, so all of the bot's
normal debug prints are sent to stderr instead.
"""
import os
import sys

# Make "policy_net.pt" and "chess_engine" resolve no matter where we're started from.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

UCI_OUT = sys.stdout   # the real stdout: only UCI messages go here
sys.stdout = sys.stderr  # everything else (prints in move_select.py etc.) goes to stderr

import chess  # noqa: E402

from chess_engine.search import MATE_SCORE  # noqa: E402

_choose_move = None


def load_engine():
    """Load the neural network the first time it's needed. Answering "uci"
    right away and loading later matters: programs like lichess-bot give an
    engine only a few seconds to reply to "uci", and loading PyTorch plus a
    160 MB network can take longer. "isready" exists for exactly this: the GUI
    waits for "readyok" before sending a position."""
    global _choose_move
    if _choose_move is None:
        from chess_engine.move_select import choose_move
        _choose_move = choose_move
    return _choose_move

ENGINE_NAME = "ChessGame policy+search"
DEFAULT_MOVE_TIME = 2.0   # seconds, when the GUI gives no clock
MAX_MOVE_TIME = 10.0

# UCI options we announce. Programs like lichess-bot only send options an
# engine announces (python-chess refuses unknown ones), and lichess-bot's
# default config sends all of these. Format: name -> (UCI type line, default).
OPTIONS = {
    # Milliseconds to keep in reserve per move for network lag. Used below.
    "Move Overhead": ("type spin default 100 min 0 max 5000", 100),
    # CPU threads PyTorch may use for the neural network. Used below.
    "Threads": (f"type spin default {os.cpu_count() or 1} min 1 max 256", os.cpu_count() or 1),
    # The rest are accepted so lichess-bot's default config works, but our
    # engine has no use for them (no hash table, no endgame tablebases).
    "Hash": ("type spin default 16 min 1 max 65536", 16),
    "SyzygyPath": ("type string default <empty>", ""),
    "UCI_ShowWDL": ("type check default false", False),
}
settings = {name: default for name, (_, default) in OPTIONS.items()}


def send(line: str):
    UCI_OUT.write(line + "\n")
    UCI_OUT.flush()  # without flush, lichess-bot might wait forever for our answer


def parse_position(tokens):
    """'position startpos moves e2e4 e7e5' or 'position fen <6 fields> moves ...'"""
    if tokens[0] == "startpos":
        board = chess.Board()
        rest = tokens[1:]
    elif tokens[0] == "fen":
        board = chess.Board(" ".join(tokens[1:7]))
        rest = tokens[7:]
    else:
        raise ValueError(f"bad position command: {tokens}")
    if rest and rest[0] == "moves":
        for uci in rest[1:]:
            board.push_uci(uci)  # keeping the history lets search see repetitions
    return board


def set_option(args):
    """'setoption name Move Overhead value 300' -> settings["Move Overhead"] = 300.
    Option names can contain spaces, hence the search for 'value'."""
    if "name" not in args:
        return
    rest = args[args.index("name") + 1:]
    if "value" in rest:
        name, value = " ".join(rest[:rest.index("value")]), " ".join(rest[rest.index("value") + 1:])
    else:
        name, value = " ".join(rest), ""
    match = next((n for n in OPTIONS if n.lower() == name.lower()), None)
    if match is None:
        return  # unknown option: ignore it rather than crash
    default = OPTIONS[match][1]
    try:
        if isinstance(default, bool):
            settings[match] = value.lower() == "true"
        elif isinstance(default, int):
            settings[match] = int(value)
        else:
            settings[match] = value
    except ValueError:
        return
    if match == "Threads":
        import torch
        torch.set_num_threads(max(1, settings["Threads"]))


def time_for_move(board: chess.Board, args: dict) -> float:
    """Simple time management: spend about 1/30 of the remaining clock plus most
    of the increment, never more than 1/5 of what's left. 'Move Overhead' is
    taken off first so network lag can't make us lose on time."""
    overhead = settings["Move Overhead"] / 1000
    if "movetime" in args:
        return max(0.05, args["movetime"] / 1000 - overhead)
    mine, inc = ("wtime", "winc") if board.turn == chess.WHITE else ("btime", "binc")
    if mine not in args:
        return DEFAULT_MOVE_TIME
    remaining = max(0.0, args[mine] / 1000 - overhead)
    increment = args.get(inc, 0) / 1000
    budget = remaining / 30 + increment * 0.8
    # Leave room for network start-up and lag: never use more than 1/5 of the clock.
    return max(0.05, min(budget, remaining / 5, MAX_MOVE_TIME))


def format_score(score: int) -> str:
    if abs(score) >= MATE_SCORE - 1000:
        plies = MATE_SCORE - abs(score)
        moves = (plies + 1) // 2
        return f"mate {moves if score > 0 else -moves}"
    return f"cp {score}"


def main():
    board = chess.Board()
    for line in sys.stdin:
        tokens = line.split()
        if not tokens:
            continue
        cmd, args = tokens[0], tokens[1:]

        if cmd == "uci":
            send(f"id name {ENGINE_NAME}")
            send("id author zanderrjohnson-CS")
            for name, (type_line, _) in OPTIONS.items():
                send(f"option name {name} {type_line}")
            send("uciok")
        elif cmd == "isready":
            load_engine()
            send("readyok")
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            board = parse_position(args)
        elif cmd == "go":
            # "go wtime 60000 btime 58000 winc 1000 binc 1000" -> {"wtime": 60000, ...}
            numbers = {}
            for key, value in zip(args, args[1:]):
                if value.lstrip("-").isdigit():
                    numbers[key] = int(value)
            if board.is_game_over():
                send("bestmove 0000")
                continue
            move, score, depth, nodes = load_engine()(board, time_for_move(board, numbers))
            send(f"info depth {depth} score {format_score(score)} nodes {nodes} pv {move.uci()}")
            send(f"bestmove {move.uci()}")
        elif cmd == "setoption":
            set_option(args)
        elif cmd == "quit":
            break
        # Anything else (stop, ponderhit, debug...) is safe to ignore:
        # our search always finishes within its own time limit.


if __name__ == "__main__":
    main()
