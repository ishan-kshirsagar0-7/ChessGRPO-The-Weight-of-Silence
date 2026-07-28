"""
eval_sft.py — Evaluate the SFT-trained chess reasoning model on unseen positions.
==================================================================================
Loads the trained LoRA model via HuggingFace (bypasses Unsloth inference bugs),
runs inference on FENs from tactic_evals.csv (excluding training data),
and shows full outputs + validation results.
"""

import chess
import re
import sys
import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from validator import validate_row


# ─────────────────────────────────────────────────────────────────────────────
# TEE — prints to both terminal and file simultaneously
# ─────────────────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


OUTPUT_FILE = "eval_results.txt"
tee = Tee(OUTPUT_FILE)
sys.stdout = tee


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "sft_v2_checkpoints/final_model_lora"
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
TACTIC_EVALS_CSV = "tactic_evals.csv"
TRAINING_DATA_CSV = "chess_reasoning_v2_dataset.csv"
NUM_TEST_POSITIONS = 20
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Must match training exactly
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (vanilla HuggingFace + PEFT, no Unsloth inference)
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading base model: {BASE_MODEL}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

print(f"Loading LoRA adapter from: {MODEL_PATH}...")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Model loaded.\n")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD TEST POSITIONS (unseen FENs from tactic_evals.csv)
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading test positions from {TACTIC_EVALS_CSV}...")

train_df = pd.read_csv(TRAINING_DATA_CSV)
training_fens = set(train_df["FEN"].unique())
print(f"  Training FENs to exclude: {len(training_fens)}")

evals_df = pd.read_csv(TACTIC_EVALS_CSV)
evals_df = evals_df[evals_df["Move"].notna()]
evals_df = evals_df[~evals_df["FEN"].isin(training_fens)]

evals_df = evals_df.sample(n=min(NUM_TEST_POSITIONS, len(evals_df)), random_state=SEED)
print(f"  Selected {len(evals_df)} unseen test positions.\n")


# ─────────────────────────────────────────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

results = []

for idx, (_, row) in enumerate(evals_df.iterrows()):
    fen = row["FEN"]
    expected_move = row["Move"]

    print(f"{'='*70}")
    print(f"TEST {idx+1}/{len(evals_df)}")
    print(f"FEN:           {fen}")
    print(f"Expected Move: {expected_move}")
    print(f"{'='*70}")

    # Build chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"FEN: {fen}"},
    ]

    # Tokenize
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            use_cache=True,
        )
    elapsed = time.time() - start_time

    # Decode only the generated part
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract thinking and output
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw_output, re.DOTALL)
    output_match = re.search(r"<output>(.*?)</output>", raw_output, re.DOTALL)

    thinking = thinking_match.group(1).strip() if thinking_match else "[NO THINKING TAGS]"
    model_move = output_match.group(1).strip() if output_match else "[NO OUTPUT TAGS]"

    # --- CHECK RESULTS ---

    # 1. Format check
    has_format = thinking_match is not None and output_match is not None

    # 2. Move legality check
    is_legal = False
    is_correct = False
    try:
        board = chess.Board(fen)
        if model_move in ("CHECKMATE", "STALEMATE", "DRAW"):
            is_legal = board.is_game_over()
            is_correct = (model_move == expected_move)
        else:
            move = chess.Move.from_uci(model_move)
            is_legal = move in board.legal_moves
            is_correct = (model_move == expected_move)
    except:
        is_legal = False
        is_correct = False

    # 3. Reasoning validation
    validation_pass = False
    fail_reasons = ""
    if thinking_match and model_move and model_move != "[NO OUTPUT TAGS]":
        report = validate_row(0, fen, model_move, thinking)
        validation_pass = (report.overall == "PASS")
        fail_reasons = report.fail_reasons

    # --- PRINT RESULTS ---
    print(f"\n📝 REASONING:")
    print(thinking)
    print(f"\n🎯 MODEL MOVE:    {model_move}")
    print(f"📌 EXPECTED MOVE: {expected_move}")
    print(f"⏱️  Generation:    {elapsed:.1f}s")
    print()
    print(f"  Format OK:        {'✅' if has_format else '❌'}")
    print(f"  Move Legal:       {'✅' if is_legal else '❌'}")
    print(f"  Move Correct:     {'✅' if is_correct else '❌'}")
    print(f"  Reasoning Valid:  {'✅' if validation_pass else '❌'}")
    if fail_reasons:
        print(f"  Fail Reasons:     {fail_reasons}")
    print()

    results.append({
        "fen": fen,
        "expected": expected_move,
        "model_move": model_move,
        "has_format": has_format,
        "is_legal": is_legal,
        "is_correct": is_correct,
        "reasoning_valid": validation_pass,
        "time": elapsed,
    })


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n\n{'='*70}")
print(f"EVALUATION SUMMARY — {len(results)} unseen positions")
print(f"{'='*70}")

format_ok = sum(1 for r in results if r["has_format"])
legal = sum(1 for r in results if r["is_legal"])
correct = sum(1 for r in results if r["is_correct"])
reasoning_ok = sum(1 for r in results if r["reasoning_valid"])
avg_time = sum(r["time"] for r in results) / len(results)

print(f"  Format compliance:   {format_ok}/{len(results)} ({format_ok/len(results)*100:.0f}%)")
print(f"  Legal moves:         {legal}/{len(results)} ({legal/len(results)*100:.0f}%)")
print(f"  Correct moves:       {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")
print(f"  Reasoning validated: {reasoning_ok}/{len(results)} ({reasoning_ok/len(results)*100:.0f}%)")
print(f"  Avg generation time: {avg_time:.1f}s")
print()

print("Per-position breakdown:")
for i, r in enumerate(results):
    fmt = "✅" if r["has_format"] else "❌"
    leg = "✅" if r["is_legal"] else "❌"
    cor = "✅" if r["is_correct"] else "❌"
    rea = "✅" if r["reasoning_valid"] else "❌"
    print(f"  {i+1:2d}. {fmt} Format  {leg} Legal  {cor} Correct  {rea} Reasoning  | {r['model_move']:10s} vs {r['expected']:10s}")

print(f"\nResults saved to {OUTPUT_FILE}")

# Restore stdout
sys.stdout = tee.terminal
tee.close()
print(f"All output saved to {OUTPUT_FILE}")