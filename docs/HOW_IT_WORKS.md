# How the chess bot works

A beginner's tour of every part of this repo, in the order data flows through it.
The one-sentence version: **a neural network learned to guess which move a grandmaster
would play, and a search algorithm double-checks those guesses before the bot moves.**

```
 chess.com games ──► training examples ──► train PolicyNetwork ──► policy_net.pt
                      (chess_model.py)        (chess_model.py)            │
                                                                          ▼
 browser (index.html) ──FEN──► FastAPI (api/main.py) ──► get_best_move (move_select.py)
                                                           │  1. policy net ranks moves  (intuition)
                                                           │  2. alpha-beta search checks them (calculation)
                                         ◄──── "e2e4" ─────┘
```

---

## 1. Turning a board into numbers: `board_to_tensor` (chess_model.py)

A neural network only understands arrays of numbers, so the board becomes a
**16 × 8 × 8 tensor**: sixteen 8×8 "pictures" (called *channels*) stacked on top of each other.

| Channels | What a `1` means |
|---|---|
| 0–5 | a white pawn / knight / bishop / rook / queen / king is on this square |
| 6–11 | the same for black |
| 12 | whole plane is 1 if it's white's turn |
| 13, 14 | white / black still has some castling right |
| 15 | the en-passant target square |

This style (one yes/no plane per piece type) is called **one-hot encoding**. It's what
AlphaZero uses too. Keeping the 8×8 shape lets the network see which pieces are *next to*
each other, which matters a lot in chess.

## 2. Turning moves into numbers: `generate_move_to_index`

The network's output is a list of scores, one per possible move. Every
"from-square → to-square" pair (64 × 64 = 4096) plus every promotion variant
gets a fixed slot, **8192 slots in total**. `MOVE_TO_INDEX["e2e4"]` gives the slot number,
`INDEX_TO_MOVE` goes back. Most slots are moves that can never happen (a1→a1),
which is wasteful but simple, and it works.

## 3. Getting training data: the chess.com section

`get_available_archives` / `fetch_games_from_archive` download every 2024 game played by
Magnus Carlsen, Hikaru, Firouzja and Caruana. `parse_game_to_policy_examples` replays each
game and, **at every position, records (board tensor, move the GM actually played)**.

That pair is one *training example*: the input, and the "right answer" (the **label**).
Learning from labelled examples like this is called **supervised learning**: the network
is learning to imitate grandmasters.

## 4. The neural network: `PolicyNetwork`

A **policy** network answers "which move should I play here?" It has three parts:

1. **`input_conv`**: a *convolution* slides a small 3×3 window over the board looking for
   local patterns (a pawn chain, a knight fork shape). It turns the 16 input channels into
   256 "feature" channels.
2. **`residual_blocks`** (6 of them): more convolutions stacked deep, so later layers can
   combine simple patterns into complex ones ("bishop + queen aimed at h7").
   The *skip connection* (`out += residual`) adds each block's input back onto its output.
   That keeps information flowing through deep networks and makes them much easier to train.
   This is the idea behind ResNet and AlphaZero's network.
   **BatchNorm** keeps the numbers in a sensible range; **ReLU** (`max(0, x)`) is the
   non-linearity that lets a network learn anything more interesting than a straight line.
3. **The policy head** (`policy_conv` + `policy_fc`): squashes the features down and uses
   fully-connected layers to produce one score (a **logit**) for each of the 8192 move slots.
   **Dropout(0.3)** randomly switches off 30% of neurons during training so the
   network can't just memorise the training games.

Size: about **40.7 million parameters** (numbers the training adjusts). Note that 33.5M of
those are in `policy_fc` alone, which is why `policy_net.pt` is 160 MB.

## 5. Training: `train()`

Each loop over the whole dataset is an **epoch**; the data is fed in **batches** of 32.
For every batch:

1. **Forward pass**: the network scores all 8192 moves for each board.
2. **Loss**: `CrossEntropyLoss` measures how much probability the network gave to the move
   the grandmaster really played. Low loss = good guesses.
3. **Backward pass** (`loss.backward()`): *backpropagation* works out, for every one of the
   40M parameters, which direction would have reduced the loss.
4. **Optimizer step**: `Adam` nudges every parameter a little in that direction.
   The *learning rate* (`lr`) is how big a nudge.

Extras: **gradient clipping** stops a single weird batch from making a huge jump, and
**ReduceLROnPlateau** halves the learning rate when the loss stops improving, so training
can settle into finer adjustments.

## 6. Picking a move: `move_select.py` *(changed in step 1)*

1. **Ask the network.** `get_policy_ranked_moves` runs the board through the network and
   ranks the *legal* moves by their logit. (Illegal moves are simply skipped. This is called
   **masking**.)
2. **Check with search.** The ranked moves go to the alpha-beta search (below).
3. **Choose.** Play whichever move search scores best. If several moves are within
   0.25 pawns of the best, let the network pick among them, because it knows things the
   simple evaluation doesn't (opening theory, plans, style).

## 7. Search: `search.py` + `evaluate.py` *(new in step 1)*

**Why it was needed.** The network gives *intuition* but never *calculates*. Before this
change, the bot played 1.d4 instead of mate in one in a Scholar's-mate position,
and shuffled its rook sideways instead of playing a back-rank mate. The old safety net (`simulate_exchange`)
only checked trades on the square the piece moved to, so it couldn't see mates, forks,
or pieces left hanging elsewhere.

- **`evaluate.py`** gives a score for "who's winning" in **centipawns** (100 = one pawn):
  material (pawn 100, knight 320, ... queen 900) plus **piece-square tables**, small
  bonuses for good squares (knights in the centre, king tucked behind pawns).
- **Minimax**: "I pick the move that's best for me, assuming you then pick the reply that's
  best for you, and so on." `negamax` is the same idea written once for both sides by
  flipping the sign (+3 for me = −3 for you).
- **Alpha-beta pruning**: once a line is proven worse than one we already have, stop
  looking at it. Same answer as minimax but far fewer positions, *especially if good moves
  are tried first*. This is where the network helps search: its ranking is the
  move order at the root.
- **Quiescence search**: at the end of a line, keep playing out captures until things are
  calm, so the bot doesn't stop calculating right after "I win your queen!" and miss the
  recapture (the *horizon effect*).
- **Iterative deepening**: search 1 move deep, then 2, then 3... until the 2-second
  budget runs out, keeping the deepest finished answer.
- **Mate scores**: checkmate scores ±100,000 minus the distance, so the bot prefers faster
  mates and slower losses.

**Result:** in an 8-game match (new bot at 0.5 s/move vs. the old version), the new bot
won 6 and drew 2, with no losses. It now finds the mates it used to miss (see `tests/test_search.py`).

This "neural net for intuition + search for calculation" combination is exactly how
AlphaZero and Leela Chess Zero work (they use a different search called MCTS, and a learned
evaluation instead of hand-written tables, which is a possible next step).

## 8. The web app: `api/main.py` and `api/static/index.html`

`FastAPI` serves the page at `/` and answers `POST /move` with `{"fen": ...}` →
`{"move": "e2e4"}`. **FEN** is a one-line text description of a chess position.
The page uses chess.js to track the game and calls the API after each of your moves.

Run it from the repo root with `uvicorn api.main:app --reload` and open http://127.0.0.1:8000.

---

## Things noticed for later steps

- **No validation set.** "Best model" is chosen by *training* loss, which always keeps
  going down, even when the network starts memorising instead of learning. Holding out
  ~10% of positions and tracking loss/accuracy on them would show whether it's improving.
- **Opening data bug.** `len(all_policy_examples) < 15000` counts *all* examples, so once
  the list passes 15k (after a handful of games) no more opening positions are added.
  And `[:MAX_POLICY_EXAMPLES]` takes the first 100k in order (mostly one player's games)
  rather than a random sample.
- **Castling channels** merge kingside and queenside rights into one plane per colour,
  so the network can't tell which side it may castle on.
- **Perspective.** The board isn't flipped for black, so the network must learn every
  pattern twice (once per colour). Flipping so "side to move" is always at the bottom is a
  standard trick.
- **Stored masks (`M`)** take 8192 floats per example and are never used in training.
- **A value network** (`.gitignore` already mentions `value_net.pt`) could replace the
  hand-written `evaluate.py` with a learned evaluation.
- `move_mask.py` isn't used anywhere (`move_select.py` does its own masking).
- Housekeeping: `__pycache__/` and `.DS_Store` are committed; `test.html` duplicates
  `api/static/index.html`.
