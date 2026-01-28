# chess_engine/move_select.py
import chess
import torch
from chess_engine.chess_model import PolicyNetwork, MOVE_TO_INDEX, INDEX_TO_MOVE, board_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model once when module is imported
model = PolicyNetwork(len(MOVE_TO_INDEX)).to(device)
model.load_state_dict(torch.load("policy_net.pt", map_location=device, weights_only=True))
model.eval()

def evaluate_position(board):
    """Simple material evaluation"""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            score += value if piece.color == chess.BLACK else -value
    return score

def get_move_with_lookahead(board, depth=2):
    """Evaluate moves with minimax lookahead"""
    
    def minimax(board, depth, maximizing_for_black):
        if depth == 0 or board.is_game_over():
            return evaluate_position(board)
        
        legal_moves = list(board.legal_moves)
        
        if maximizing_for_black:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval_score = minimax(board, depth - 1, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = minimax(board, depth - 1, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
            return min_eval
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    
    best_move = None
    best_score = float('-inf')
    
    is_black = board.turn == chess.BLACK
    
    for move in legal_moves:
        board.push(move)
        
        # Evaluate this position with minimax
        material_score = minimax(board, depth - 1, not is_black)
        
        board.pop()
        
        # Get neural net preference
        board_tensor = torch.tensor(
            board_to_tensor(board),
            dtype=torch.float32
        ).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(board_tensor).squeeze(0)
        
        move_idx = MOVE_TO_INDEX.get(move.uci())
        nn_score = logits[move_idx].item() if move_idx else -100
        
        # Combined score: 90% material safety, 10% neural net
        combined_score = 0.9 * (material_score if is_black else -material_score) + 0.1 * nn_score
        
        if combined_score > best_score:
            best_score = combined_score
            best_move = move
    
    return best_move

def get_best_move(fen: str) -> str:
    board = chess.Board(fen)
    move = get_move_with_lookahead(board, depth=2)
    return move.uci() if move else list(board.legal_moves)[0].uci()