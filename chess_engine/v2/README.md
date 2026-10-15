# v2: evaluator + search

No neural network. v2 looks ahead with alpha-beta search and scores the
positions at the end of each line with an evaluation function whose weights
were learned from game results.

| File | What it does |
|---|---|
| `evaluate.py` | Scores a position: material + piece-square tables (middlegame/endgame blend) + weighted relational features |
| `features.py` | 29 relational features: pawn structure, passed pawns, bishop pair, fianchetto, outposts, rook files, king safety, mobility |
| `weights.json` | Feature weights learned by `tools/tune.py` (loaded automatically) |
| `search.py` | Alpha-beta search, quiescence search, iterative deepening, transposition table, move ordering |
| `engine.py` | `get_best_move(fen, time_limit)` used by the API |

Other entry points: `uci.py` (UCI protocol, used by lichess-bot; see `lichess/README.md`),
`web/` (browser version, published to GitHub Pages by `.github/workflows/pages.yml`).

## Try it

```bash
# Analyse a position (FEN, seconds) and print each search depth
python -m chess_engine.v2.engine "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4" 3

# Unit tests (mates, hanging pieces, features, eval symmetry)
python -m unittest tests.test_v2

# Engine vs engine match: v2, v2-hand, v2-pst, v2-material, v1, random
python -m tools.match v2 v2-hand --games 60 --time 0.3

# Re-learn the feature weights (Texel tuning) from positions labeled with results
python -m tools.tune quiet-labeled.epd          # or: python -m tools.tune games.pgn --pgn

# Play in the browser locally
bash tools/build_site.sh && python -m http.server -d _site 8000
```

## Texel tuning in one paragraph

Each training position has a label: 1 if White won the game, 0.5 draw, 0 if Black
won. The evaluator's score is turned into a predicted win probability with a
sigmoid, `1 / (1 + 10^(-K * score / 400))`, and the loss is the mean squared error
against the labels. Because the score is a weighted sum of feature counts, this
is logistic regression, trained with the same loop as a neural network: forward
pass, loss, gradient, Adam update, and a held-out validation set to catch
overfitting. The weights stay readable: you can see what the data thinks a
bishop pair or an open-file rook is worth.

The current weights came from `quiet-labeled.epd` (725,000 quiet positions from
engine games, published by the Zurichess author): validation loss went from
0.06139 with hand-picked weights to 0.05981.
