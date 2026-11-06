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

## 2. Turning moves into numbers: `generate_move_to_index` *(changed in step 2)*

The network's output is a list of scores, one per possible move, so every move needs a
fixed **slot** number. `move_to_index(move)` gives a move's slot; `index_to_move(slot, board)`
goes back.

**Before (8,192 slots):** every from-square → to-square pair (64 × 64 = 4096) plus four
promotion versions of every move onto the first or last rank. About 78% of those slots were
moves no piece can ever make (a1→a1, a1→c4). The network still had to learn to give them
a score of "never", and every slot cost 2,049 weights in the last layer.

**After (1,792 slots):** only moves that *some* piece could make on an empty board:

- **Queen lines**: 1 to 7 squares in any straight or diagonal direction (1,456 slots). This
  covers rooks, bishops, queens, kings, pawns, and castling (`e1g1` is a 2-square king move).
- **Knight jumps** (336 slots).

Leela Chess Zero uses almost the same list (1,858 slots; the extra 66 are underpromotions).

**Why not a separate list per piece** (rooks only straight, bishops only diagonal)? A slot
doesn't know which piece is moving. `a1a8` is a rook move in one game and a queen move in
another, and the network learns from the board which piece is on a1. The queen-lines-plus-
knight list is exactly the set of moves *any* piece can make, so it's already as small as
a shared list can get. (AlphaZero went the other way, with 73 move types for each of the 64 squares,
because that layout suits convolutions. It's bigger, not smaller.)

**Promotions:** a pawn reaching the last rank uses its plain from→to slot (`e7e8q` goes into
slot `e7e8`), so the network assumes promotion to a queen. Rook, bishop and knight
promotions get no slot, so the network never suggests them and the training data skips
them (only 3 of the ~118,000 grandmaster moves I tested with were underpromotions). The search still tries them, because it generates moves with
python-chess, not from the slots, so the bot can still underpromote when it matters.

**Masking still happens.** Shrinking the list removes moves that are *never* possible. The
moves that are illegal *in this position* (a pinned piece, a blocked rook) still get
filtered out each time the bot picks a move.

**Your trained model still works.** The last layer, `Linear(2048, num_moves)`, has one row
of weights per slot, and each row is its own detector for one move. Keeping only the rows
whose moves still exist converts the old network without retraining
(`chess_engine/convert_weights.py`, done automatically when `move_select.py` loads an old
file). The converted network picks the same move as before in 100% of 3,000 test positions.

**Before vs. after (trained from scratch, same data, same random seed).** I trained both
versions on ~106,000 positions from 1,740 grandmaster games, holding out other games to
test on. (Your chess.com dataset isn't in the repo, so I used a public GM game collection.)

| Training steps | Old (8,192 slots) val loss / accuracy | New (1,792 slots) val loss / accuracy |
|---|---|---|
| 500 | 6.00 / 8.5% | **5.84 / 10.3%** |
| 1,000 | 5.80 / 11.9% | **5.70 / 12.2%** |
| 1,500 | 5.68 / 12.8% | **5.56** / 12.7% |
| 2,000 | 5.54 / 12.7% | **5.44 / 13.4%** |
| 2,500 | 5.47 / 14.1% | **5.36 / 15.0%** |

*Accuracy* = how often the net's top legal move is the move the GM played.

- **Learns a bit faster per step:** lower validation loss at every checkpoint (accuracy was a tie at 1,500), and after
  2,500 steps the new net beats where the old one was after 3,000 (5.40 / 14.9%). It wastes
  no effort pushing impossible moves down.
- **Each step is cheaper:** 27.6M parameters instead of 40.7M. In an isolated timing test a
  training step took 319 ms instead of 551 ms. In the full run it was closer to 10% faster,
  because loading data and other work don't shrink.
- **Smaller model file:** about 110 MB instead of 160 MB.

These are short runs, so the exact numbers are rough, but the loss was lower for the new version at
every checkpoint.

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
   fully-connected layers to produce one score (a **logit**) for each of the 1,792 move slots.
   **Dropout(0.3)** randomly switches off 30% of neurons during training so the
   network can't just memorise the training games.

Size: about **27.6 million parameters** (numbers the training adjusts), down from 40.7M
before step 2 shrank the move list. 20.5M of those are still in `policy_fc`, mostly the
`Linear(128*8*8, 2048)` layer, which is the next-biggest thing to slim down.

## 5. Training: `train()` *(changed in step 3)*

Each loop over the whole dataset is an **epoch**; the data is fed in **batches** of 32.
For every batch:

1. **Forward pass**: the network scores all 1,792 move slots for each board.
2. **Loss**: `CrossEntropyLoss` measures how much probability the network gave to the move
   the grandmaster really played. Low loss = good guesses.
3. **Backward pass** (`loss.backward()`): *backpropagation* works out, for every one of the
   27.6M parameters, which direction would have reduced the loss.
4. **Optimizer step**: `Adam` nudges every parameter a little in that direction.
   The *learning rate* (`lr`) is how big a nudge.

Extras: **gradient clipping** stops a single weird batch from making a huge jump, and
**ReduceLROnPlateau** halves the learning rate when the validation loss stops improving, so training
can settle into finer adjustments.

### Validation set and early stopping *(new in step 3)*

**The problem:** training loss *always* goes down if you train long enough, because the
network can simply memorise the 100k training positions. Before step 3, "best model" was
picked by training loss, so there was no way to tell learning from memorising.

**The fix:** before training, ~10% of the **games** are set aside as a **validation set**.
The network never trains on them. After every epoch it's tested on them:

- **Val loss**: the same cross-entropy score, but on unseen positions.
- **Val accuracy**: how often the network's favourite *legal* move (using the saved masks `M`,
  which finally have a job) is the move the grandmaster played.

Reading the log:

| Train loss | Val loss | What it means |
|---|---|---|
| ↓ | ↓ | Learning real chess. Keep going. |
| ↓ | → or ↑ | **Overfitting**: memorising the training games. |

Even a 2-epoch test run on 60 games showed it: train loss fell from 7.33 to 5.97 while val loss went
*up* (6.457 → 6.463).

Now:
- `policy_net_best.pt` is the epoch with the **lowest validation loss**.
- The learning-rate scheduler watches validation loss too.
- **Early stopping**: if validation loss hasn't improved for 8 epochs, training stops
  instead of running all 100.

**Why split by game, not by position?** Position 20 and position 21 of the same game are
almost identical. If one were in training and the other in validation, the network could
"recognise" the validation position, and the score would look better than it really is.
That's called **data leakage**. New datasets save a game id `G` for every position.
Older datasets have no `G`, so the script falls back to a random position split
and says so.

**Other data fixes:**
- **Opening bug:** the 15k cap on opening positions checked the size of the *whole* list,
  so after a few games no more openings were added. It now has its own counter.
- **Random sample:** `[:100_000]` used to take the first 100k rows, which come mostly from
  the first player fetched. It now takes a random 100k.
- **Smaller files:** masks are saved as True/False (1 byte per slot) instead of 4-byte floats.
- **Old datasets still load:** a dataset saved with the 8,192-slot index is re-labelled to
  the new 1,792-slot index automatically (`convert_dataset_labels`).

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

### Playing on Lichess: `uci_engine.py` *(new)*

`uci_engine.py` lets the bot speak **UCI**, the text protocol chess programs use to talk to
engines. [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) runs it and passes
positions back and forth with Lichess. Setup steps and how the pieces connect are in
[LICHESS.md](LICHESS.md).

---

## Things noticed for later steps

- **Castling channels** merge kingside and queenside rights into one plane per colour,
  so the network can't tell which side it may castle on.
- **Perspective.** The board isn't flipped for black, so the network must learn every
  pattern twice (once per colour). Flipping so "side to move" is always at the bottom is a
  standard trick.
- **A value network** (`.gitignore` already mentions `value_net.pt`) could replace the
  hand-written `evaluate.py` with a learned evaluation.
- `move_mask.py` isn't used anywhere (`move_select.py` does its own masking).
- Housekeeping: `__pycache__/` and `.DS_Store` are committed; `test.html` duplicates
  `api/static/index.html`.
