"""
score_cp.py — Stockfish centipawn-loss analysis over eval_completions_rung1.csv
================================================================================
For every legal (non-CHECKMATE) parsed move: cp_loss = max(0, best_eval_cp - move_eval),
using the frozen procedure (depth 12, single thread, mover POV, mate_score 1500).
best_eval_cp comes from the precomputed grpo_training_data_evals.csv.
Prints per-model stats and paired comparisons on shared-legal positions.
Writes cp_scores_rung1.csv (per-move) for the paper.
"""

import shutil
import chess
import chess.engine
import pandas as pd
from tqdm import tqdm

# ── CONFIG ─────────────
COMPLETIONS_CSV = "eval_completions_rung1.csv"
EVALS_CSV       = "grpo_training_data_evals.csv"
OUT_CSV         = "cp_scores_rung1.csv"

STOCKFISH_PATH = shutil.which("stockfish") or "/usr/games/stockfish"
DEPTH = 12
MAX_SECONDS_PER_EVAL = 10.0
MATE_SCORE = 1500
ENGINE_THREADS = 1
ENGINE_HASH_MB = 256

MODELS = ["random_legal", "base", "sft", "grpo_v1", "grpo_v2"]
PAIRS  = [("sft", "grpo_v2"), ("grpo_v1", "grpo_v2"), ("sft", "grpo_v1")]
GOOD_MOVE_CP = 50  # cp_loss <= this counts as a "good move"


def open_engine():
    e = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    e.configure({"Threads": ENGINE_THREADS, "Hash": ENGINE_HASH_MB})
    return e


def main():
    comp = pd.read_csv(COMPLETIONS_CSV)
    evals = pd.read_csv(EVALS_CSV)
    best_cp = {r["FEN"]: r["best_eval_cp"] for _, r in evals.iterrows()
               if pd.notna(r["best_eval_cp"])}

    # rows we can score: legal, parsed, non-CHECKMATE, with a known best eval
    rows = comp[(comp["legal"] == True) & comp["parsed_move"].notna()].copy()
    rows = rows[rows["parsed_move"] != "CHECKMATE"]
    rows = rows[rows["fen"].isin(best_cp.keys())].reset_index(drop=True)
    print(f"{len(rows)} legal moves to score across {rows['model'].nunique()} models.")

    engine = open_engine()
    cp_losses = []
    try:
        for _, r in tqdm(rows.iterrows(), total=len(rows), desc="Scoring", unit="move"):
            fen, mv_uci = r["fen"], r["parsed_move"]
            board = chess.Board(fen)
            mover = board.turn
            mv = chess.Move.from_uci(mv_uci)
            board.push(mv)
            if board.is_checkmate():
                move_cp = MATE_SCORE
            else:
                try:
                    info = engine.analyse(board, chess.engine.Limit(depth=DEPTH, time=MAX_SECONDS_PER_EVAL))
                    move_cp = info["score"].pov(mover).score(mate_score=MATE_SCORE)
                except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
                    try:
                        engine.quit()
                    except Exception:
                        pass
                    engine = open_engine()
                    info = engine.analyse(board, chess.engine.Limit(depth=DEPTH, time=MAX_SECONDS_PER_EVAL))
                    move_cp = info["score"].pov(mover).score(mate_score=MATE_SCORE)
            cp_losses.append(max(0, int(best_cp[fen]) - move_cp))
    finally:
        try:
            engine.quit()
        except Exception:
            pass

    rows["cp_loss"] = cp_losses
    rows[["model", "fen", "parsed_move", "best_move", "cp_loss"]].to_csv(OUT_CSV, index=False)
    print(f"\nWrote per-move cp losses to {OUT_CSV}")

    # ── per-model stats ──
    print("\n" + "=" * 78)
    print(f"{'model':<14}{'n_legal':>8}{'median_cp':>11}{'mean_cp':>10}{'good<=' + str(GOOD_MOVE_CP) + 'cp%':>12}")
    print("=" * 78)
    for m in MODELS:
        sub = rows[rows["model"] == m]["cp_loss"]
        if len(sub) == 0:
            print(f"{m:<14}{0:>8}")
            continue
        good = 100.0 * (sub <= GOOD_MOVE_CP).mean()
        print(f"{m:<14}{len(sub):>8}{sub.median():>11.0f}{sub.mean():>10.0f}{good:>12.1f}")
    print("=" * 78)

    # ── paired comparisons on shared-legal positions ──
    for a, b in PAIRS:
        da = rows[rows["model"] == a].set_index("fen")["cp_loss"]
        db = rows[rows["model"] == b].set_index("fen")["cp_loss"]
        shared = da.index.intersection(db.index)
        if len(shared) == 0:
            print(f"\n{a} vs {b}: no shared-legal positions.")
            continue
        diff = da.loc[shared] - db.loc[shared]  # positive => b better (lower loss)
        a_wins = int((diff < 0).sum())
        b_wins = int((diff > 0).sum())
        ties = int((diff == 0).sum())
        print(f"\n{a} vs {b} on {len(shared)} shared-legal positions:")
        print(f"  {a} wins {a_wins}, {b} wins {b_wins}, ties {ties}")
        print(f"  median diff {diff.median():+.0f}cp, mean diff {diff.mean():+.0f}cp "
              f"(positive favors {b})")


if __name__ == "__main__":
    print(f"Stockfish: {STOCKFISH_PATH}, depth {DEPTH}, threads {ENGINE_THREADS}")
    main()
