#!/usr/bin/env bash
# Starts ZanderBot v2 as a UCI engine. lichess-bot runs this file.
cd "$(dirname "$0")/.."
if [ -x .venv/bin/python ]; then exec .venv/bin/python uci.py; fi
exec python3 uci.py
