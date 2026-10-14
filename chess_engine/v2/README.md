# v2: evaluator + search

No neural network. v2 looks ahead with alpha-beta search and scores the
positions at the end of each line with a hand-written evaluator.

| File | What it does |
|---|---|
| `evaluate.py` | Scores a position: material + piece-square tables, blended between middlegame and endgame |
| `search.py` | Alpha-beta search, quiescence search, iterative deepening, transposition table, move ordering |
| `engine.py` | `get_best_move(fen, time_limit)` used by the API |

## Try it

```bash
# Analyse a position (FEN, seconds) and print each search depth
python -m chess_engine.v2.engine "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4" 3

# Unit tests (mates, hanging pieces, eval symmetry)
python -m unittest tests.test_v2

# Engine vs engine match
python -m tools.match v2 v1 --games 40 --time 0.5

# Play in the browser (v2 is the default; ENGINE=v1 for the old bot)
uvicorn api.main:app --reload    # then open http://localhost:8000/
```

## Next steps

1. Add relational features to `evaluate.py` one at a time (passed/isolated/doubled
   pawns, bishop pair, rooks on open files, king pawn shield, mobility) and keep
   each one only if `tools/match.py` shows it helps.
2. Texel-tune the weights on the game file with logistic regression.
