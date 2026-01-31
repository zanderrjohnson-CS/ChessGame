# chess_engine/move_select.py
import chess
import torch
from chess_engine.chess_model import PolicyNetwork, MOVE_TO_INDEX, INDEX_TO_MOVE, board_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load policy network
policy_model = PolicyNetwork(len(MOVE_TO_INDEX)).to(device)
policy_model.load_state_dict(torch.load("policy_net_best.pt", map_location=device, weights_only=True))
policy_model.eval()
print("Loaded policy network")

# Piece values for blunder detection
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # King can't be captured
}


def get_material_balance(board):
    """Calculate material balance (positive = white ahead)"""
    balance = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES[piece.piece_type]
            balance += value if piece.color == chess.WHITE else -value
    return balance


def simulate_exchange(board, move):
    """
    Simulate a capture sequence and return material change for the moving side.
    Returns the net material gain/loss after all sensible recaptures.
    """
    moving_color = board.turn
    initial_balance = get_material_balance(board)
    
    # Make the move
    board_copy = board.copy()
    board_copy.push(move)
    
    # Simulate capture sequence on the destination square
    target_square = move.to_square
    
    # Keep trading while captures are available on that square
    while True:
        # Find captures to the target square
        captures = [m for m in board_copy.legal_moves if m.to_square == target_square and board_copy.is_capture(m)]
        
        if not captures:
            break
        
        # Pick the least valuable attacker (standard exchange logic)
        best_capture = None
        best_attacker_value = float('inf')
        
        for capture in captures:
            attacker = board_copy.piece_at(capture.from_square)
            if attacker and PIECE_VALUES[attacker.piece_type] < best_attacker_value:
                best_attacker_value = PIECE_VALUES[attacker.piece_type]
                best_capture = capture
        
        if best_capture:
            board_copy.push(best_capture)
        else:
            break
    
    final_balance = get_material_balance(board_copy)
    
    # Return material change from the perspective of the moving side
    if moving_color == chess.WHITE:
        return final_balance - initial_balance
    else:
        return initial_balance - final_balance


def is_blunder(board, move, threshold=-1):
    """
    Check if a move is a blunder.
    A blunder loses material after the exchange sequence.
    threshold=-1 means losing more than 1 pawn worth is a blunder.
    Equal trades (0) or slight losses (-1) are allowed.
    """
    material_change = simulate_exchange(board, move)
    return material_change < threshold


def get_policy_ranked_moves(board):
    """Get all legal moves ranked by policy network preference"""
    board_tensor = torch.tensor(
        board_to_tensor(board),
        dtype=torch.float32
    ).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = policy_model(board_tensor).squeeze(0)
    
    # Get scores for all legal moves
    legal_moves = []
    for move in board.legal_moves:
        uci = move.uci()
        if uci in MOVE_TO_INDEX:
            idx = MOVE_TO_INDEX[uci]
            legal_moves.append((move, logits[idx].item()))
    
    # Sort by score (highest first)
    legal_moves.sort(key=lambda x: x[1], reverse=True)
    return legal_moves


def get_best_move(fen: str) -> str:
    """Find best move using policy network with blunder filter"""
    board = chess.Board(fen)
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    
    # Get moves ranked by policy network
    ranked_moves = get_policy_ranked_moves(board)
    
    print(f"\nEvaluating {len(ranked_moves)} moves...")
    
    # Try moves in order of policy preference
    for i, (move, score) in enumerate(ranked_moves):
        material_change = simulate_exchange(board, move)
        is_bad = is_blunder(board, move)
        
        # Debug: show top moves being considered
        if i < 5:
            status = "BLUNDER" if is_bad else "OK"
            print(f"  {move.uci()}: policy={score:.2f}, material_change={material_change:+d}, {status}")
        
        if not is_bad:
            print(f"Selected: {move.uci()} (rank #{i+1})")
            return move.uci()
    
    # If ALL moves are blunders (rare), pick the least bad one
    print("All moves are blunders, picking least bad...")
    least_bad_move = max(ranked_moves, key=lambda x: simulate_exchange(board, x[0]))
    print(f"Selected: {least_bad_move[0].uci()} (least bad option)")
    return least_bad_move[0].uci()


# Quick test
if __name__ == "__main__":
    board = chess.Board()
    print("Starting position:")
    print(board)
    print(f"\nBest move: {get_best_move(board.fen())}")