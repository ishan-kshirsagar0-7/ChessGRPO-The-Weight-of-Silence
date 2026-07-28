"""
eval_latent.py  --  Canonical evaluation of Rung 2 latent stage adapters.

Scores each latent stage adapter on the SAME 100 held-out positions the Rung 1
harness used (grpo_training_data.csv, same seed/split), so results are directly
comparable to grpo_v2's eval_summary_rung1.csv.

The ONLY difference from eval_models.py is generation: each stage adapter runs
the Coconut silent-thinking loop (N = 2*stage thoughts) before decoding, exactly
as the training gate does. Scoring reuses rewards.py verbatim, so legality and
checkmate logic is identical to Rung 1.

Greedy decoding, batch-1, matching the D5 latent-eval protocol.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rewards import extract_move, is_move_legal, is_position_checkmate

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_MODEL    = "unsloth/qwen3-14b-bnb-4bit"
TRAINING_DATA = "grpo_training_data.csv" 
SEED          = 3407
EVAL_SET_SIZE = 100
MAX_NEW       = 512
RUN_DIR       = "rung2_checkpoints"

# Which stage adapters to score. Each is scored with 2*stage silent thoughts.
THOUGHTS_PER_CHUNK = 2
STAGE_ADAPTERS = {
    1: os.path.join(RUN_DIR, "stage_1", "final"),
    2: os.path.join(RUN_DIR, "stage_2", "final"),
    3: os.path.join(RUN_DIR, "stage_3", "final"),
}

COMPLETIONS_OUT = "eval_completions_rung2.csv"
SUMMARY_OUT     = "eval_summary_rung2.csv"

BOT_TOKEN = "<|box_start|>"
EOT_TOKEN = "<|box_end|>"

SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""


def build_latent_prompt(fen: str) -> str:
    """Prefix only, ending at the BOT marker. Identical to the training gate's
    build_gate_prompt: the model must produce eot + reasoning + move itself."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nFEN: {fen}<|im_end|>\n"
        f"<|im_start|>assistant\n<thinking>\n{BOT_TOKEN}"
    )


# ── SAME HELD-OUT 100 AS RUNG 1 (same seed + split) ─────────────────────────
print("Loading held-out eval split (same seed/split as eval_models.py)...")
df = pd.read_csv(TRAINING_DATA).sample(frac=1, random_state=SEED).reset_index(drop=True)
eval_df = df.iloc[:EVAL_SET_SIZE].reset_index(drop=True)
eval_rows = [(r["FEN"], str(r["Best Move"]).strip()) for _, r in eval_df.iterrows()]
print(f"  {len(eval_rows)} eval positions (identical to Rung 1 harness).")


# ── LOAD BASE ONCE, ATTACH ALL STAGE ADAPTERS ───────────────────────────────
print(f"\nLoading base model {BASE_MODEL} in 4-bit...")
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

# marker sanity: they must be single-token, or the latent loop is malformed
for t in (BOT_TOKEN, EOT_TOKEN):
    ids = tokenizer.encode(t, add_special_tokens=False)
    if len(ids) != 1:
        raise SystemExit(f"FATAL: {t} not single-token: {ids}")

# Attach every existing stage adapter under its own name.
present = {}
first = True
model = None
for stage, path in STAGE_ADAPTERS.items():
    if not os.path.isdir(path):
        print(f"  stage {stage}: adapter not found at {path}, skipping.")
        continue
    name = f"stage_{stage}"
    if first:
        print(f"Attaching {name} from {path}...")
        model = PeftModel.from_pretrained(base, path, adapter_name=name)
        first = False
    else:
        print(f"Attaching {name} from {path}...")
        model.load_adapter(path, adapter_name=name)
    present[stage] = name

if model is None:
    raise SystemExit("No stage adapters found. Check RUN_DIR and STAGE_ADAPTERS.")

model.eval()
model.config.use_cache = False       # off during the manual thought loop
model.generation_config.eos_token_id = im_end
model.generation_config.pad_token_id = tokenizer.pad_token_id


# ── LATENT GENERATION ───────────────
@torch.no_grad()
def generate_latent(fen: str, n_thoughts: int) -> str:
    """Prefix -> n silent thoughts -> greedy decode. Returns decoded completion."""
    prompt = build_latent_prompt(fen)
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False).input_ids.to(model.device)
    mask = torch.ones_like(ids)

    embed = model.get_input_embeddings()
    cur_emb = embed(ids)
    cur_mask = mask

    # skip English for n_thoughts steps: grab hidden state, feed it back raw
    for _ in range(n_thoughts):
        out = model(inputs_embeds=cur_emb, attention_mask=cur_mask,
                    output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][:, -1:, :].to(cur_emb.dtype)
        cur_emb = torch.cat([cur_emb, h], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(cur_mask[:, :1])], dim=1)

    # now decode words normally. eval() disables the GC path that corrupts
    # train-mode generation (TRL #3089); KV cache is safe here.
    gen = model.generate(
        inputs_embeds=cur_emb,
        attention_mask=cur_mask,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        use_cache=True,
        eos_token_id=im_end,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(gen[0], skip_special_tokens=False)


# ── SCORING ─────────────────────────────
def score_stage(stage: int, adapter_name: str):
    n_thoughts = THOUGHTS_PER_CHUNK * stage
    model.set_adapter(adapter_name)
    n = len(eval_rows)
    fmt_ok = legal = correct = false_mate = 0
    records = []

    pbar = tqdm(eval_rows, desc=f"stage {stage} ({n_thoughts} thoughts)",
                unit="pos", total=n)
    for i, (fen, best) in enumerate(pbar):
        comp = generate_latent(fen, n_thoughts)
        move = extract_move(comp)
        has_tags = ("</thinking>" in comp and "<output>" in comp
                    and "</output>" in comp)
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
            "stage": stage, "n_thoughts": n_thoughts, "fen": fen,
            "best_move": best, "parsed_move": move, "format_ok": is_format_ok,
            "legal": is_legal, "correct": is_correct,
            "false_mate": is_false_mate, "completion": comp,
        })

        done = i + 1
        pbar.set_postfix(legal=f"{100*legal/done:.0f}%",
                         fmt=f"{100*fmt_ok/done:.0f}%",
                         fmate=false_mate)
    pbar.close()

    metrics = {
        "stage": stage,
        "n_thoughts": n_thoughts,
        "format": 100.0 * fmt_ok / n,
        "legal": 100.0 * legal / n,
        "accuracy": 100.0 * correct / n,
        "false_mate": false_mate,
    }
    print(f"  stage {stage} done: format {metrics['format']:.1f}%  "
          f"legal {metrics['legal']:.1f}%  accuracy {metrics['accuracy']:.1f}%  "
          f"false_mate {false_mate}", flush=True)
    return metrics, records


all_records = []
results = {}

for stage in sorted(present):
    name = present[stage]
    results[stage], recs = score_stage(stage, name)
    all_records += recs


# ── WRITE CSVs ──────────────────────────────────────────────────────────────
comp_df = pd.DataFrame(all_records, columns=[
    "stage", "n_thoughts", "fen", "best_move", "parsed_move",
    "format_ok", "legal", "correct", "false_mate", "completion",
])
comp_df.to_csv(COMPLETIONS_OUT, index=False)
print(f"\nWrote {len(comp_df)} per-position rows to {COMPLETIONS_OUT}")

summary_df = pd.DataFrame([results[s] for s in sorted(results)])
summary_df.to_csv(SUMMARY_OUT, index=False)
print(f"Wrote aggregate metrics to {SUMMARY_OUT}")


# ── TABLE ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 66)
print(f"{'stage':<7}{'thoughts':>9}{'format%':>10}{'legal%':>9}"
      f"{'accuracy%':>11}{'false_mate':>12}")
print("=" * 66)
for s in sorted(results):
    r = results[s]
    print(f"{s:<7}{r['n_thoughts']:>9}{r['format']:>10.1f}{r['legal']:>9.1f}"
          f"{r['accuracy']:>11.1f}{r['false_mate']:>12}")
print("=" * 66)
print("\nGreedy, batch-1, 100 held-out positions (same as Rung 1 harness).")
print("Compare legal%/accuracy% directly against eval_summary_rung1.csv (grpo_v2).")
print("For the paper: append grpo_v2's row so the plateau and the curriculum")
print("curve sit in one table.")