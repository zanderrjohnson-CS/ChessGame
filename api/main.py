# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from chess_engine.move_select import get_best_move

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chess_engine.move_select import get_best_move




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
        return {"move": get_best_move(req.fen)}
    except Exception as e:
        return {"error": str(e)}
    
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    with open("api/static/index.html", "r", encoding="utf-8") as f:
        return f.read()



