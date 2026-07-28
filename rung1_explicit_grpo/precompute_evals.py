"""
precompute_evals.py

Precomputes best_eval_cp for every row of grpo_training_data.csv using the
EC2-local Stockfish at fixed depth. This becomes the reference eval that the
dense centipawn reward subtracts against during GRPO training.

Method (matches training-time scoring in rewards.py EXACTLY):
  1. Load FEN, note side to move (the mover).
  2. Push the CSV's best move.
  3. engine.analyse() the resulting position, Limit(depth=DEPTH, time=MAX_SECONDS).
  4. Score from the mover's POV, mate_score=MATE_SCORE.
  Single-threaded engine for determinism.

Resumable: progress saved per-row to PROGRESS_JSONL. Rerun to continue.
On completion, writes CSV_OUT = original CSV + best_eval_cp column.
Original CSV is never modified.
"""

import csv
import json
import os
import shutil
import sys

import chess
import chess.engine
from tqdm import tqdm

# ============================== CONFIG ==============================
CSV_IN = "/home/ubuntu/grpo_training_data.csv"
CSV_OUT = "/home/ubuntu/grpo_training_data_evals.csv"
PROGRESS_JSONL = "/home/ubuntu/precompute_evals_progress.jsonl"

FEN_COL = "FEN"
MOVE_COL = "Best Move"

STOCKFISH_PATH = shutil.which("stockfish") or "/usr/games/stockfish"
DEPTH = 12                  # frozen for the whole project, do not change mid-run
MAX_SECONDS_PER_EVAL = 10.0 # safety cap only; depth 12 finishes way before this
MATE_SCORE = 1500           # forced mate reads as +/-1500cp
ENGINE_THREADS = 1          # single thread = deterministic evals
ENGINE_HASH_MB = 256
# ====================================================================


def open_engine():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": ENGINE_THREADS, "Hash": ENGINE_HASH_MB})
    return engine


def eval_after_best_move(engine, fen, best_move_uci):
    """Returns (cp_int, error_str). cp is from the mover's perspective."""
    board = chess.Board(fen)
    mover = board.turn
    try:
        move = chess.Move.from_uci(best_move_uci.strip())
    except ValueError:
        return None, f"unparseable move: {best_move_uci!r}"
    if move not in board.legal_moves:
        return None, f"CSV best move is illegal in position: {best_move_uci!r}"
    board.push(move)
    if board.is_checkmate():
        return MATE_SCORE, None  # best move delivers mate, no engine needed
    limit = chess.engine.Limit(depth=DEPTH, time=MAX_SECONDS_PER_EVAL)
    info = engine.analyse(board, limit)
    cp = info["score"].pov(mover).score(mate_score=MATE_SCORE)
    return cp, None


def load_progress():
    done = {}
    if os.path.exists(PROGRESS_JSONL):
        with open(PROGRESS_JSONL, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["idx"]] = rec
    return done


def main():
    if not os.path.exists(STOCKFISH_PATH):
        print(f"Stockfish not found at {STOCKFISH_PATH}. Run: sudo apt install -y stockfish")
        sys.exit(1)

    with open(CSV_IN, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if FEN_COL not in fieldnames or MOVE_COL not in fieldnames:
        print(f"Column mismatch. Expected {FEN_COL!r} and {MOVE_COL!r}.")
        print(f"Actual header: {fieldnames}")
        print("Fix FEN_COL / MOVE_COL at the top of this script and rerun.")
        sys.exit(1)

    done = load_progress()
    todo = [i for i in range(len(rows)) if i not in done]
    print(f"{len(rows)} rows total, {len(done)} done, {len(todo)} to go.")

    engine = open_engine()
    errors = 0

    progress_f = open(PROGRESS_JSONL, "a")
    try:
        for i in tqdm(todo, desc="Evaluating", unit="pos"):
            fen = rows[i][FEN_COL]
            best = rows[i][MOVE_COL]
            try:
                cp, err = eval_after_best_move(engine, fen, best)
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
                # engine died, restart once and retry
                try:
                    engine.quit()
                except Exception:
                    pass
                engine = open_engine()
                try:
                    cp, err = eval_after_best_move(engine, fen, best)
                except Exception as e:
                    cp, err = None, f"engine failed twice: {e}"

            if err is not None:
                errors += 1
                tqdm.write(f"[row {i}] {err}")

            progress_f.write(json.dumps({"idx": i, "best_eval_cp": cp, "error": err}) + "\n")
            progress_f.flush()
    finally:
        progress_f.close()
        try:
            engine.quit()
        except Exception:
            pass

    done = load_progress()
    if len(done) < len(rows):
        print(f"\nIncomplete: {len(rows) - len(done)} rows remaining. Rerun to resume.")
        sys.exit(0)

    # merge and write final CSV
    n_null = 0
    out_fields = list(fieldnames) + ["best_eval_cp"]
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            cp = done[i]["best_eval_cp"]
            if cp is None:
                n_null += 1
            row["best_eval_cp"] = "" if cp is None else cp
            writer.writerow(row)

    print(f"\nWrote {CSV_OUT}")
    print(f"Rows with failed evals (empty best_eval_cp): {n_null}")
    if n_null > 0:
        print("These rows will be DROPPED by train_grpo.py's loader. "
              "If the count is more than a handful, investigate before training.")


if __name__ == "__main__":
    print(f"Stockfish: {STOCKFISH_PATH}, depth {DEPTH}, threads {ENGINE_THREADS}, mate_score {MATE_SCORE}")
    main()