import os
import asyncio
import pandas as pd
import chess
import csv
import re
import aiohttp
import logging
import psutil
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, 
    TimeElapsedColumn, MofNCompleteColumn
)
from rich.logging import RichHandler
from dataset_curator import (
    get_king_context, 
    fen_to_natural_language, 
    get_detailed_material_analysis
)
from prompts import terminal_reasoning_prompt

# --- CONFIGURATION ---
@dataclass
class Config:
    INPUT_CSV: str = "sft_sampled_dataset.csv"      # The raw sampled source
    OUTPUT_CSV: str = "terminal_dataset.csv"        # The new 2k file
    LOG_FILE: str = "terminal_ops.log"
    
    MODEL_ID: str = "qwen2.5:7b" 
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    
    MAX_RETRIES: int = 3
    CONCURRENCY_LIMIT: int = 2
    GPU_TEMP_THRESHOLD: float = 82.0
    GPU_VRAM_THRESHOLD: float = 96.0

config = Config()

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        RichHandler(rich_tracebacks=True, show_path=False, console=Console(stderr=True))
    ]
)
logger = logging.getLogger("terminal_curator")
console = Console()

# --- GPU MONITOR ---
USE_NVML = False
try:
    import pynvml
    pynvml.nvmlInit()
    USE_NVML = True
except Exception: pass

class ResourceMonitor:
    def get_status(self) -> Dict[str, float]:
        stats = {"vram": 0.0, "temp": 0.0}
        if USE_NVML:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                stats["vram"] = (mem.used / mem.total) * 100
                stats["temp"] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except: pass
        else:
            stats["vram"] = psutil.virtual_memory().percent
        return stats

    def is_healthy(self) -> bool:
        s = self.get_status()
        return s["temp"] < config.GPU_TEMP_THRESHOLD and s["vram"] < config.GPU_VRAM_THRESHOLD

monitor = ResourceMonitor()

# --- HELPER: GET TERMINAL FACTS ---
def get_terminal_facts(board: chess.Board) -> str:
    """Generates specific facts about why the game is over."""
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        loser = "Black" if board.turn == chess.BLACK else "White"
        
        # Find who is giving check
        king_sq = board.king(board.turn)
        attackers = board.attackers(not board.turn, king_sq)
        attacker_names = [chess.piece_name(board.piece_at(sq).piece_type) for sq in attackers]
        attacker_str = ", ".join(attacker_names)
        
        return f"Game Over: {winner} wins by Checkmate. The {loser} King is in check by {attacker_str} and has 0 legal moves."
        
    elif board.is_stalemate():
        return "Game Over: Draw by Stalemate. The King is NOT in check, but has 0 legal moves."
        
    elif board.is_insufficient_material():
        return "Game Over: Draw by Insufficient Material."
        
    return "Game Over."

def validate_response(text: str, expected: str) -> bool:
    match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
    if not match: return False
    return match.group(1).strip() == expected

def clean_xml_output(text: str) -> Optional[str]:
    match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    return match.group(1).strip() if match else None

# --- PROCESS ROW ---
async def process_row(session, row, semaphore):
    row_id = row['id']
    fen = row['FEN']
    
    try:
        # 1. Calculate Facts (CPU)
        board = chess.Board(fen)
        
        # Determine strict output
        if board.is_checkmate():
            expected_output = "CHECKMATE"
        elif board.is_stalemate():
            expected_output = "STALEMATE"
        else:
            expected_output = "DRAW" # Fallback
            
        # Gather Intelligence
        terminal_facts = get_terminal_facts(board)
        material = get_detailed_material_analysis(fen, None) # No move needed
        board_desc = fen_to_natural_language(fen)
        king_summary = f"White: {get_king_context(board, chess.WHITE)}\nBlack: {get_king_context(board, chess.BLACK)}"
        
        prompt = terminal_reasoning_prompt.format(
            input_fen=fen,
            terminal_facts=terminal_facts,
            material_count=material,
            board_description=board_desc,
            king_summary=king_summary,
            expected_output=expected_output
        )
        
    except Exception as e:
        return {"status": "fail", "reason": str(e)}

    # 2. Inference (GPU)
    async with semaphore:
        while not monitor.is_healthy():
            await asyncio.sleep(30)

        payload = {
            "model": config.MODEL_ID,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 2048}
        }

        for attempt in range(config.MAX_RETRIES):
            try:
                async with session.post(config.OLLAMA_URL, json=payload) as resp:
                    if resp.status != 200: raise Exception("API Error")
                    data = await resp.json()
                    raw = data.get("response", "")
                    
                    if validate_response(raw, expected_output):
                        thinking = clean_xml_output(raw)
                        if thinking:
                            return {
                                "id": row_id, "FEN": fen, 
                                "Best Move": expected_output, 
                                "Reasoning": thinking, "status": "success"
                            }
            except:
                await asyncio.sleep(1)
                
        return {"status": "fail_retries"}

# --- MAIN ---
async def main():
    console.rule("[bold red]Terminal State Curator[/bold red]")
    
    # Load Source (SFT Sampled)
    if not os.path.exists(config.INPUT_CSV):
        console.print(f"[red]Input file {config.INPUT_CSV} not found![/red]")
        return

    df = pd.read_csv(config.INPUT_CSV)
    
    # Filter ONLY Game Over rows (NaN moves)
    # This checks if 'Move' is NaN/None/Empty
    todo_df = df[df['Move'].isna() | (df['Move'].astype(str).str.lower() == 'nan')]
    
    console.print(f"Found {len(todo_df)} Terminal Rows to process.")
    
    # Resume Check
    if os.path.exists(config.OUTPUT_CSV):
        done = pd.read_csv(config.OUTPUT_CSV)
        done_ids = set(done['id'])
        todo_df = todo_df[~todo_df['id'].isin(done_ids)]
    else:
        with open(config.OUTPUT_CSV, 'w', newline='') as f:
            csv.writer(f).writerow(['id', 'FEN', 'Best Move', 'Reasoning'])
            
    if len(todo_df) == 0: 
        console.print("[green]Job Complete![/green]")
        return

    semaphore = asyncio.Semaphore(config.CONCURRENCY_LIMIT)
    
    async with aiohttp.ClientSession() as session:
        # RESTORED PROGRESS BAR FORMATTING HERE
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[stats]}"), 
            console=console
        ) as progress:
            
            task_id = progress.add_task("[cyan]Processing...", total=len(todo_df), stats="Init...")
            
            # Simple chunking
            BATCH_SIZE = 50
            chunks = [todo_df[i:i+BATCH_SIZE] for i in range(0, len(todo_df), BATCH_SIZE)]
            
            for chunk in chunks:
                tasks = [process_row(session, row, semaphore) for _, row in chunk.iterrows()]
                
                for coro in asyncio.as_completed(tasks):
                    res = await coro
                    
                    if res.get("status") == "success":
                        with open(config.OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerow([res['id'], res['FEN'], res['Best Move'], res['Reasoning']])
                    
                    # Update Stats in Progress Bar
                    gpu = monitor.get_status()
                    stats_str = f"[blue]GPU: {gpu['temp']}°C / VRAM: {gpu['vram']:.0f}%[/blue]"
                    progress.update(task_id, advance=1, stats=stats_str)

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())