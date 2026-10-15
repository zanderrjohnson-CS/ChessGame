// Runs Python (Pyodide) in a background thread so the page stays responsive
// while the engine thinks.
const PY_FILES = [
  "chess_engine/__init__.py",
  "chess_engine/v2/__init__.py",
  "chess_engine/v2/engine.py",
  "chess_engine/v2/evaluate.py",
  "chess_engine/v2/features.py",
  "chess_engine/v2/search.py",
  "chess_engine/v2/weights.json",
  "bridge.py",
];
const CHESS_WHEEL = "py/chess-1.11.2-py3-none-any.whl";

let pyodide = null;
let bridge = null;

async function init(pyodideBase) {
  importScripts(pyodideBase + "pyodide.js");
  pyodide = await loadPyodide({ indexURL: pyodideBase });

  // python-chess is pure Python, so its wheel can be unpacked straight into site-packages.
  const wheel = await (await fetch(CHESS_WHEEL)).arrayBuffer();
  pyodide.unpackArchive(wheel, "wheel");

  pyodide.FS.mkdirTree("/home/pyodide/chess_engine/v2");
  for (const path of PY_FILES) {
    const text = await (await fetch("py/" + path)).text();
    pyodide.FS.writeFile("/home/pyodide/" + path, text);
  }
  bridge = pyodide.pyimport("bridge");
}

self.onmessage = async (e) => {
  const { id, cmd, args } = e.data;
  try {
    let result = null;
    if (cmd === "init") await init(args.pyodideBase);
    else if (cmd === "state") result = JSON.parse(bridge.state(JSON.stringify(args.moves)));
    else if (cmd === "think") result = JSON.parse(bridge.think(JSON.stringify(args.moves), args.seconds));
    self.postMessage({ id, result });
  } catch (err) {
    self.postMessage({ id, error: String(err) });
  }
};
