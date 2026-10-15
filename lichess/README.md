# Getting a Lichess rating

Lichess lets engines play as **BOT** accounts. The official
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) program connects a
UCI engine to Lichess, accepts challenges, and can seek games against other bots
by itself. Games are rated, so after ~20 games the account has a real rating.
It runs on your Mac, so the bot is online only while the program is running.

## 1. Create the bot account (one time)

1. Log out and create a **new** Lichess account for the bot, e.g. `ZanderBot`.
   It must never have played a game: only fresh accounts can become bots.
2. While logged in as the bot, create an API token at
   <https://lichess.org/account/oauth/token/create> with the scope
   **"Play games with the bot API"**. Copy it somewhere safe; never commit it.
3. Upgrade the account to a BOT. This is permanent for that account:

   ```bash
   curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer YOUR_TOKEN"
   ```

## 2. Install lichess-bot (one time)

```bash
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install chess          # needed by ZanderBot itself
cp config.yml.default config.yml
```

## 3. Point it at ZanderBot

Edit these keys in `lichess-bot/config.yml` (leave everything else as it is):

```yaml
token: "YOUR_TOKEN"

engine:
  dir: "/Users/YOU/path/to/ChessGame/lichess/"   # this folder
  name: "run_engine.sh"
  protocol: "uci"
  ponder: false

challenge:
  accept_bot: true
  time_controls: [blitz, rapid]
  modes: [casual, rated]

matchmaking:
  allow_matchmaking: true          # seek games against other bots automatically
  challenge_initial_time: [180, 300]
  challenge_increment: [2, 3]
  challenge_mode: "rated"
```

Check the engine starts on its own first:

```bash
/Users/YOU/path/to/ChessGame/lichess/run_engine.sh
uci          # should print "uciok"
quit
```

## 4. Run it

```bash
cd lichess-bot && source venv/bin/activate
python3 lichess-bot.py
```

Games appear on `https://lichess.org/@/ZanderBot`. Leave it running (and the Mac
awake, e.g. `caffeinate -i python3 lichess-bot.py`) until the rating stops being
provisional, which is the "?" after it.

People can also challenge the bot directly from its profile page.
