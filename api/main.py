# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from chess_engine.move_select import get_best_move

app = FastAPI()

class MoveRequest(BaseModel):
    fen: str

class MoveResponse(BaseModel):
    move: str

@app.post("/move", response_model=MoveResponse)
def get_move(req: MoveRequest):
    move = get_best_move(req.fen)
    return {"move": move}
