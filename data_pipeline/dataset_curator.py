import os
import asyncio
import pandas as pd
import chess
import csv
import re
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from token_bucket import TokenBucket
from prompts import reasoning_prompt

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("GEMINI_API_KEY")  # You'd need your own API key here if you want to curate your own data, but the data I've curated already exists on HF
MODEL = "gemini-2.5-flash"
INPUT_CSV = "tactic_evals_balanced.csv"
OUTPUT_CSV = "chess_reasoning_dataset.csv"
MAX_RETRIES = 20

# Rate Limit: 15 RPM = 1 request every 4 seconds
RATE_LIMIT_CAPACITY = 1
RATE_LIMIT_REFILL_RATE = 5.0 / 60.0 

client = genai.Client(api_key=API_KEY)

# --- HELPER FUNCTIONS ---

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def get_king_context(board, color):
    # 1. Where is the King?
    king_square = board.king(color)
    if king_square is None: return "King captured (Invalid Board)"
    
    sq_name = chess.square_name(king_square)
    color_name = "White" if color == chess.WHITE else "Black"
    
    # 2. What are the rights?
    can_castle_k = board.has_kingside_castling_rights(color)
    can_castle_q = board.has_queenside_castling_rights(color)
    
    # 3. Derive Status
    status = []
    
    # CASE A: King is on standard Castled squares
    if color == chess.WHITE:
        if sq_name in ["g1", "h1"]: status.append(f"The {color_name} King is tucked away on the Kingside ({sq_name}).")
        elif sq_name in ["c1", "b1", "a1"]: status.append(f"The {color_name} King is tucked away on the Queenside ({sq_name}).")
        elif sq_name == "e1": status.append(f"The {color_name} King is in the center (e1).")
        else: status.append(f"The {color_name} King is exposed/active on {sq_name}.")
    else: # Black
        if sq_name in ["g8", "h8"]: status.append(f"The {color_name} King is tucked away on the Kingside ({sq_name}).")
        elif sq_name in ["c8", "b8", "a8"]: status.append(f"The {color_name} King is tucked away on the Queenside ({sq_name}).")
        elif sq_name == "e8": status.append(f"The {color_name} King is in the center (e8).")
        else: status.append(f"The {color_name} King is exposed/active on {sq_name}.")

    # CASE B: Castling Rights
    rights_str = []
    if can_castle_k: rights_str.append("Kingside")
    if can_castle_q: rights_str.append("Queenside")
    
    if rights_str:
        status.append(f"It still has the right to castle: {' & '.join(rights_str)}.")
    else:
        is_start_sq = (sq_name == "e1" if color == chess.WHITE else sq_name == "e8")
        if is_start_sq:
            status.append("It has lost the right to castle (Stuck in center).")
            
    return " ".join(status)

def fen_to_natural_language(fen):
    try:
        board = chess.Board(fen)
        text_output = []
        
        piece_names = {
            chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
            chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King"
        }
        
        # 1. Piece Locations
        priority = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
        for color, color_name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
            pieces_list = []
            for piece_type in priority:
                squares = board.pieces(piece_type, color)
                for sq in squares:
                    sq_name = chess.square_name(sq)
                    pieces_list.append(f"{piece_names[piece_type]} on {sq_name}")
            
            if pieces_list:
                text_output.append(f"{color_name}'s Layout: {', '.join(pieces_list)}.")
            else:
                text_output.append(f"{color_name} has no pieces.")

        # Note: I removed the old get_castling_status call here because I have get_king_context
        return "\n".join(text_output)
    except Exception:
        return "Board description unavailable."

def get_tactical_analysis(fen, move_uci):
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        
        piece_moved = board.piece_at(move.from_square)
        piece_captured = board.piece_at(move.to_square)
        
        piece_map = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}
        
        board.push(move)
        is_check = board.is_check()
        is_mate = board.is_checkmate()
        
        facts = []
        if is_mate:
            facts.append(f"CRITICAL: The move {move_uci} results in CHECKMATE.")
        elif is_check:
            facts.append(f"The move {move_uci} delivers a DIRECT CHECK to the opponent's King.")
        else:
            facts.append(f"The move {move_uci} is a quiet move (no immediate check).")
            
        if piece_captured:
            captured_str = piece_map.get(piece_captured.piece_type, "Piece")
            facts.append(f"MATERIAL: It captures the opponent's {captured_str} on square {chess.square_name(move.to_square)}.")
        else:
            facts.append("MATERIAL: It does not capture any piece immediately.")
            
        return " ".join(facts)
    except Exception:
        return "Tactical analysis unavailable."

def get_detailed_material_analysis(fen, move_uci):
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        
        def count_material(b):
            w = sum(len(b.pieces(p, chess.WHITE)) * v for p, v in values.items())
            bl = sum(len(b.pieces(p, chess.BLACK)) * v for p, v in values.items())
            return w, bl

        # 1. BEFORE
        w_start, b_start = count_material(board)
        
        # 2. AFTER
        board.push(move)
        w_end, b_end = count_material(board)
        
        # 3. WHO IS WINNING NOW?
        final_diff = w_end - b_end
        if final_diff > 0: leader = f"White leads by +{final_diff}"
        elif final_diff < 0: leader = f"Black leads by +{abs(final_diff)}"
        else: leader = "Material is exactly equal"

        # 4. CAPTURE DETAILS
        w_lost = w_start - w_end
        b_lost = b_start - b_end
        
        action = "No material captured."
        if b_lost > 0: action = f"White captured {b_lost} points of material."
        if w_lost > 0: action = f"Black captured {w_lost} points of material."

        return (f"SCOREBOARD BEFORE: White {w_start} - Black {b_start}. "
                f"ACTION: {action} "
                f"SCOREBOARD AFTER: White {w_end} - Black {b_end}. "
                f"CURRENT STATE: {leader}.")

    except Exception:
        return "Material analysis unavailable."

def clean_output(text):
    match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if match: return match.group(1).strip()
    return None

async def process_row(row, bucket, stats):
    row_id = row['id']
    input_fen = row['FEN']
    input_move = row['Move']
    
    # --- 1. CALCULATE FACTS (EXACTLY AS IN TEST SCRIPT) ---
    
    # Tactics
    tactical_analysis = get_tactical_analysis(input_fen, input_move)
    
    # Material
    material_count = get_detailed_material_analysis(input_fen, input_move)
    
    # Board Visuals
    board_description = fen_to_natural_language(input_fen)
    
    # King Context
    board = chess.Board(input_fen)
    white_king_status = get_king_context(board, chess.WHITE)
    black_king_status = get_king_context(board, chess.BLACK)
    king_summary = f"{white_king_status}\n{black_king_status}"
    
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            await bucket.consume()
            
            # --- 2. CALL API ---
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL,
                contents=reasoning_prompt.format(
                    material_count=material_count,       # As used in test script
                    tactical_facts=tactical_analysis,    # Mapped: variable tactical_analysis -> key tactical_facts
                    board_description=board_description, # As used in test script
                    king_summary=king_summary,           # As used in test script
                    input_fen=input_fen,
                    input_move=input_move,
                ),
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            
            # --- 3. VALIDATE ---
            reasoning = clean_output(response.text)
            
            if reasoning and len(reasoning) > 50:
                print(f"[{get_timestamp()}] [ID:{row_id}] ✅ Success.")
                return (row_id, input_fen, input_move, reasoning, "success")
            else:
                print(f"[{get_timestamp()}] [ID:{row_id}] ⚠️ Validation Failed")
                raise ValueError("Validation failed")

        except Exception as e:
            retries += 1
            print(f"[{get_timestamp()}] [ID:{row_id}] ❌ Error: {str(e)}")
            if retries > MAX_RETRIES:
                return (row_id, input_fen, input_move, None, f"failed: {str(e)}")
            await asyncio.sleep(2 ** retries)

# --- MAIN ENGINE ---

async def main():
    bucket = TokenBucket(capacity=RATE_LIMIT_CAPACITY, refill_rate=RATE_LIMIT_REFILL_RATE)
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return
        
    df = pd.read_csv(INPUT_CSV)
    
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV, on_bad_lines='skip')
            if 'id' in existing_df.columns:
                processed_ids = set(existing_df['id'].unique())
                print(f"Resuming... Found {len(processed_ids)} completed rows.")
        except: pass
    else:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['id', 'FEN', 'Best Move', 'Reasoning'])

    rows_to_process = df[~df['id'].isin(processed_ids)]
    if rows_to_process.empty:
        print("Job done!")
        return

    print(f"Starting job on {len(rows_to_process)} rows...")
    
    BATCH_SIZE = 50 
    pbar = tqdm(total=len(rows_to_process), desc="Reasoning", unit="row")
    stats = {'success': 0, 'failed': 0}

    for i in range(0, len(rows_to_process), BATCH_SIZE):
        batch = rows_to_process.iloc[i : i + BATCH_SIZE]
        batch_tasks = [process_row(row, bucket, stats) for _, row in batch.iterrows()]
        
        for task in asyncio.as_completed(batch_tasks):
            row_id, fen, move, reasoning, status = await task
            
            if status == "success":
                stats['success'] += 1
                with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([row_id, fen, move, reasoning])
            else:
                stats['failed'] += 1
            
            pbar.set_postfix(success=stats['success'], fail=stats['failed'])
            pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")