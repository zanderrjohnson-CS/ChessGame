# api/main.py
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ENGINE=v1 uses the original policy network (needs torch + policy_net.pt).
# Default is v2: evaluator + search, no neural network.
ENGINE = os.environ.get("ENGINE", "v2")
if ENGINE == "v1":
    from chess_engine.move_select import get_best_move
else:
    from chess_engine.v2 import get_best_move

# Seconds v2 thinks per move.
THINK_TIME = float(os.environ.get("THINK_TIME", "2.0"))


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MoveRequest(BaseModel):
    fen: str

class MoveResponse(BaseModel):
    move: str

@app.post("/move")
def get_move(req: MoveRequest):
    try:
        if ENGINE == "v1":
            return {"move": get_best_move(req.fen)}
        return {"move": get_best_move(req.fen, time_limit=THINK_TIME)}
    except Exception as e:
        return {"error": str(e)}
    
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    with open("api/static/index.html", "r", encoding="utf-8") as f:
        return f.read()



