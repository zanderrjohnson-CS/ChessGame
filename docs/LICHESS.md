# Playing the bot on Lichess

## How the pieces connect

```
 Lichess servers  ◄──internet──►  lichess-bot  ◄──stdin/stdout (UCI)──►  uci_engine.py  ──►  choose_move()
 (games, clocks,                  (logs in with your                     (this repo:          (policy net +
  challenges)                      token, accepts                         translates UCI       alpha-beta
                                   challenges, sends moves)               to Python calls)     search)
```

- **[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot)** is an open-source program
  that does everything Lichess-related: it connects with your API token, accepts or sends
  challenges, follows the game, and tells Lichess which move you played.
- **UCI (Universal Chess Interface)** is the standard way chess programs talk to engines
  like Stockfish. It's plain text, one command per line. lichess-bot starts our engine as
  a separate program and writes lines like
  `position startpos moves e2e4 e7e5` and `go wtime 60000 btime 60000 winc 1000 binc 1000`.
  The engine answers `bestmove g1f3`.
- **`uci_engine.py`** (in this repo) is the adapter. It reads those lines, builds a
  `chess.Board` with the full move history, decides how long to think from the clock
  (about 1/30 of the remaining time plus most of the increment), calls `choose_move()`, and
  prints the answer. It also prints `info ... score cp 25` so lichess-bot can see the
  evaluation (which it uses for things like offering draws).

Because the engine speaks plain UCI, any UCI program can use it, not only lichess-bot.
For example, you could load it into a chess GUI like Cute Chess and play it against Stockfish.

Three details in `uci_engine.py` that matter:
- **It announces UCI options.** Engines list their settings when they reply to `uci`, and
  python-chess (which lichess-bot uses) refuses to send a setting the engine didn't
  announce. lichess-bot's default `config.yml` sends `Move Overhead`, `Threads`, `Hash`,
  `SyzygyPath` and `UCI_ShowWDL`, so the engine announces all of them. Two of them do
  something: **Move Overhead** (milliseconds kept back each move for network lag, so the bot
  doesn't lose on time) and **Threads** (CPU threads for the neural network). The others
  are accepted and ignored, because this engine has no hash table or endgame tablebases.
- **stdout is reserved for UCI.** lichess-bot reads everything the engine prints, so the
  bot's normal debug `print()`s are sent to stderr instead (you'll see them in the
  lichess-bot log).
- **The network loads lazily.** lichess-bot waits only a few seconds for the reply to `uci`,
  and loading PyTorch plus a 160 MB network can take longer. So the engine replies right
  away and loads the network on `isready`, which is the command that exists for slow
  start-ups.

## Setup in a GitHub Codespace

Everything below runs in the codespace terminal. Your repo is at `/workspaces/ChessGame`.

### 1. Make sure the model file is really there

```bash
cd /workspaces/ChessGame
ls -lh policy_net.pt          # should be ~160 MB
git lfs pull                  # run this if it's only a few hundred bytes (a Git LFS pointer)
pip install -r requirements.txt
```

If `pip install -r requirements.txt` fails on `torch==2.5.1` (that version has no build for
the newest Python releases), install the packages without the pins instead:
`pip install torch chess numpy fastapi uvicorn requests`.

### 2. Test the engine on its own (optional, takes 10 seconds)

```bash
printf 'uci\nisready\nposition startpos moves e2e4\ngo wtime 60000 btime 60000\nquit\n' | python3 uci_engine.py 2>/dev/null
```

You should see `uciok`, `readyok`, an `info ...` line and a `bestmove ...` line.

### 3. Get lichess-bot

```bash
cd /workspaces
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot
pip install -r requirements.txt
cp config.yml.default config.yml
```

lichess-bot lives next to ChessGame, not inside it, so its `config.yml` (which holds your
token) never gets committed to your repo.

### 4. Create a bot account and token

1. Make a **new** Lichess account for the bot. An account that has already played games
   can't become a bot.
2. Logged in as the bot account, create a token with the **bot:play** scope:
   <https://lichess.org/account/oauth/token/create?scopes[]=bot:play&description=lichess-bot>
3. Copy the token (Lichess shows it only once). Either put it in `config.yml` as `token:`,
   or (better) add it as a Codespaces secret named `LICHESS_BOT_TOKEN`. To do that, go to
   GitHub → Settings → Codespaces → Secrets and give the secret access to this repo.
   lichess-bot reads that environment variable automatically. Restart the codespace after
   adding the secret.

### 5. Point lichess-bot at the engine

Edit these parts of `/workspaces/lichess-bot/config.yml` and leave everything else as it is:

```yaml
engine:
  dir: "/workspaces/ChessGame/"        # folder that contains the engine
  name: "uci_engine.py"                # the engine file
  interpreter: "python3"               # run it with Python (it isn't a compiled program)
  working_dir: "/workspaces/ChessGame/"
  protocol: "uci"
  ponder: false                        # our engine doesn't think on the opponent's time

challenge:
  concurrency: 1                       # one game at a time: the search uses the whole CPU
  accept_bot: true
  time_controls:                       # ~2-3 s per move suits blitz and rapid best
    - blitz
    - rapid
  modes:
    - casual                           # start with unrated games
```

Leave the `uci_options:` block under `engine:` as it is (or uncomment it if you commented it
out). The engine accepts all of those settings. Raise `Move Overhead` (to 300, say) if the
bot ever loses on time, and set `Threads` to the number of CPU cores your codespace has
(`nproc` tells you).

### 6. Upgrade the account to a bot (one time, cannot be undone)

```bash
cd /workspaces/lichess-bot
python3 lichess-bot.py -u
```

This turns the account into a BOT account permanently. After that it starts playing.
From then on, just run `python3 lichess-bot.py`.

### 7. Play it against other bots

- **Challenge it yourself:** open `https://lichess.org/@/<your-bot-name>` from your normal
  account and click *Challenge to a game*.
- **Let it find opponents:** turn on matchmaking in `config.yml` and the bot will challenge
  other Lichess bots when it's idle:

  ```yaml
  matchmaking:
    allow_matchmaking: true
    challenge_variant: "standard"
    challenge_timeout: 1               # minutes idle before it sends a challenge
    challenge_initial_time:
      - 180                            # 3+2 blitz
    challenge_increment:
      - 2
    challenge_mode: "casual"
    opponent_rating_difference: 300
  ```

  You can see the bot list at <https://lichess.org/player/bots>.
- The bot's rating on its profile page is the easiest way to track whether a change made
  it stronger.

### Things to know

- **Codespaces go to sleep** after 30 idle minutes by default, and the bot goes offline
  with them. You can raise the idle timeout (up to 4 hours) under GitHub → Settings →
  Codespaces.
- Codespace CPUs are small, so the bot searches less deeply than on a fast laptop. Longer
  time controls give it more thinking time.
- Bot accounts can't play UltraBullet, and Lichess expects rated bot games to try to win.
