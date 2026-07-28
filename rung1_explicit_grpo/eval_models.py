"""
eval_models.py — Greedy held-out evaluation for the chess paper (with per-example dump).
=========================================================================================
Scores base / SFT / GRPO-v1 / GRPO-v2 on the same 100 held-out positions for:
  format compliance, legal-move rate, exact-best-move accuracy, false-checkmate count.
Loads the base ONCE and swaps LoRA adapters to save memory. Greedy decoding.
Writes:
  eval_summary_rung1.csv      — aggregate metrics per model
  eval_completions_rung1.csv  — every position's full completion + flags
"""

import torch
import random
import pandas as pd
import chess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rewards import extract_move, is_move_legal, is_position_checkmate

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_MODEL      = "unsloth/qwen3-14b-bnb-4bit"
SFT_ADAPTER     = "qwen3_14b_sft_checkpoints/final_model_lora"
GRPO_V1_ADAPTER = "grpo_checkpoints/final_model_lora"
GRPO_V2_ADAPTER = "grpo_v2_checkpoints/final_model_lora"
TRAINING_DATA   = "grpo_training_data.csv"
SEED            = 3407
EVAL_SET_SIZE   = 100
MAX_NEW         = 512
COMPLETIONS_OUT = "eval_completions_rung1.csv"
SUMMARY_OUT     = "eval_summary_rung1.csv"

MODELS_IN_ORDER = ["random_legal", "base", "sft", "grpo_v1", "grpo_v2"]

SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""


def build_prompt(fen: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nFEN: {fen}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── SAME HELD-OUT 100 AS TRAINING (same shuffle seed + split) ───────────────
print("Loading held-out eval split (same seed/split as training)...")
df = pd.read_csv(TRAINING_DATA).sample(frac=1, random_state=SEED).reset_index(drop=True)
eval_df = df.iloc[:EVAL_SET_SIZE].reset_index(drop=True)
eval_rows = [(r["FEN"], str(r["Best Move"]).strip()) for _, r in eval_df.iterrows()]
print(f"  {len(eval_rows)} eval positions.")


# ── LOAD BASE ONCE, ATTACH ALL ADAPTERS ─────────────────────────────────────
print(f"\nLoading base model {BASE_MODEL} in 4-bit (cached)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, dtype=torch.bfloat16, device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
tokenizer.eos_token = "<|im_end|>"
tokenizer.eos_token_id = im_end
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Attaching SFT adapter as 'sft'...")
model = PeftModel.from_pretrained(base, SFT_ADAPTER, adapter_name="sft")
print("Attaching GRPO v1 adapter as 'grpo_v1'...")
model.load_adapter(GRPO_V1_ADAPTER, adapter_name="grpo_v1")
print("Attaching GRPO v2 adapter as 'grpo_v2'...")
model.load_adapter(GRPO_V2_ADAPTER, adapter_name="grpo_v2")
model.eval()
model.generation_config.eos_token_id = im_end
model.generation_config.pad_token_id = tokenizer.pad_token_id


# ── GENERATION + SCORING ────────────────────────────────────────────────────
@torch.no_grad()
def generate_one(fen: str) -> str:
    inputs = tokenizer(build_prompt(fen), return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW,
        do_sample=False,            # greedy
        use_cache=True,
        eos_token_id=im_end,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def score_model(label: str):
    n = len(eval_rows)
    fmt_ok = legal = correct = false_mate = 0
    records = []
    for i, (fen, best) in enumerate(eval_rows):
        comp = generate_one(fen)
        move = extract_move(comp)
        has_tags = ("<thinking>" in comp and "</thinking>" in comp
                    and "<output>" in comp and "</output>" in comp)
        is_format_ok = bool(has_tags and move is not None)
        is_legal = is_correct = is_false_mate = False

        if move is not None:
            if move == "CHECKMATE":
                if is_position_checkmate(fen):
                    is_legal = True
                    if best.upper() == "CHECKMATE":
                        is_correct = True
                else:
                    is_false_mate = True
            else:
                if is_move_legal(fen, move):
                    is_legal = True
                    if move == best:
                        is_correct = True

        fmt_ok     += int(is_format_ok)
        legal      += int(is_legal)
        correct    += int(is_correct)
        false_mate += int(is_false_mate)

        records.append({
            "model": label, "fen": fen, "best_move": best, "parsed_move": move,
            "format_ok": is_format_ok, "legal": is_legal,
            "correct": is_correct, "false_mate": is_false_mate, "completion": comp,
        })
        if (i + 1) % 20 == 0:
            print(f"  [{label}] {i + 1}/{n}")

    metrics = {
        "format": 100.0 * fmt_ok / n,
        "legal": 100.0 * legal / n,
        "accuracy": 100.0 * correct / n,
        "false_mate": false_mate,
    }
    return metrics, records


all_records = []
results = {}

print("\n=== BASE (no adapter) ===")
with model.disable_adapter():
    results["base"], recs = score_model("base")
    all_records += recs

print("\n=== SFT ===")
model.set_adapter("sft")
results["sft"], recs = score_model("sft")
all_records += recs

print("\n=== GRPO v1 (old 5-fn reward) ===")
model.set_adapter("grpo_v1")
results["grpo_v1"], recs = score_model("grpo_v1")
all_records += recs

print("\n=== GRPO v2 (Rung 1: gated + dense cp reward) ===")
model.set_adapter("grpo_v2")
results["grpo_v2"], recs = score_model("grpo_v2")
all_records += recs


# ── RANDOM-LEGAL BASELINE ───────────────────
random.seed(SEED)
rl_legal = rl_correct = 0
for fen, best in eval_rows:
    try:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        if not moves:
            continue
        mv = random.choice(moves).uci()
        rl_legal += 1
        is_corr = (mv == best)
        if is_corr:
            rl_correct += 1
        all_records.append({
            "model": "random_legal", "fen": fen, "best_move": best, "parsed_move": mv,
            "format_ok": None, "legal": True, "correct": is_corr,
            "false_mate": False, "completion": f"(random legal move: {mv})",
        })
    except Exception:
        pass
results["random_legal"] = {
    "format": float("nan"),
    "legal": 100.0 * rl_legal / len(eval_rows),
    "accuracy": 100.0 * rl_correct / len(eval_rows),
    "false_mate": 0,
}


# ── WRITE CSVs ──────────────────────────────────────────────────────────────
comp_df = pd.DataFrame(all_records, columns=[
    "model", "fen", "best_move", "parsed_move",
    "format_ok", "legal", "correct", "false_mate", "completion",
])
comp_df.to_csv(COMPLETIONS_OUT, index=False)
print(f"\nWrote {len(comp_df)} per-position rows to {COMPLETIONS_OUT}")

summary_df = pd.DataFrame([{"model": k, **results[k]} for k in MODELS_IN_ORDER])
summary_df.to_csv(SUMMARY_OUT, index=False)
print(f"Wrote aggregate metrics to {SUMMARY_OUT}")


# ── TABLE ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'model':<14}{'format%':>9}{'legal%':>9}{'accuracy%':>11}{'false_mate':>12}")
print("=" * 60)
for k in MODELS_IN_ORDER:
    r = results[k]
    fmt = "n/a" if r["format"] != r["format"] else f"{r['format']:.1f}"
    print(f"{k:<14}{fmt:>9}{r['legal']:>9.1f}{r['accuracy']:>11.1f}{r['false_mate']:>12}")
print("=" * 60)
print("\nGreedy decoding, 100 held-out positions. legal% counts correct CHECKMATE claims.")
