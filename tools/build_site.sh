#!/usr/bin/env bash
# Assemble the GitHub Pages site into _site/ (used by .github/workflows/pages.yml).
# Local test:  bash tools/build_site.sh && python -m http.server -d _site 8000
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf _site
mkdir -p _site/py/chess_engine/v2
cp web/index.html web/engine-worker.js _site/
cp web/bridge.py _site/py/
cp chess_engine/__init__.py _site/py/chess_engine/
cp chess_engine/v2/*.py chess_engine/v2/weights.json _site/py/chess_engine/v2/
pip wheel --quiet --no-deps --wheel-dir _site/py chess==1.11.2
ls _site/py/chess-1.11.2-py3-none-any.whl > /dev/null
echo "Site built in _site/"
