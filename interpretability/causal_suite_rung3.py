"""
causal_suite_rung3.py -- Six-condition causal test suite (baseline, substitute,
ablate, zero, noise, lenmatch_ablate) on RUNG-3, with a MODE switch:

  MODE = "smoke" -> N=20, nothing written to disk. Prints results + a linear
                     extrapolation to N=1000 at the end. Run this first to
                     measure real per-position cost on this box.
  MODE = "full"  -> N=1000, every result written to disk the moment it's
                     scored. Safe to kill (Ctrl-C, crash, instance hiccup)
                     and rerun unchanged: it scans the existing completions
                     CSV on startup, skips any (condition, position) pair
                     already done, and resumes mid-condition if needed.
                     Summary + pairwise CSVs are rewritten after EVERY
                     condition finishes, not just at the very end -- so a
                     post-completion crash (known harmless CUDA/bitsandbytes
                     teardown segfault, seen twice now) never costs you the
                     aggregate tables, only the pretty printed report at the
                     bottom, which is trivially rebuilt from the CSVs anyway.

Same shuffle/seed as every prior harness in this project, so the first 100
rows of any run here are byte-identical to the original rung-3 n=100 causal
battery -- directly comparable, not a fresh unrelated sample.

Per-position generation is wrapped in try/except: one bad completion (OOM,
weird decode, whatever) is logged and counted as unparseable, it does not
kill the run.
"""

import csv
import logging
import os
import time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rewards import extract_move, is_move_legal, is_position_checkmate

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODE = "full"                       # "smoke" or "full" -- the only line to change between runs

SMOKE_N = 20
FULL_N  = 1000

if MODE == "smoke":
    EVAL_SET_SIZE = SMOKE_N
    SAVE_OUTPUTS  = False
elif MODE == "full":
    EVAL_SET_SIZE = FULL_N
    SAVE_OUTPUTS  = True
else:
    raise SystemExit(f"FATAL: MODE must be 'smoke' or 'full', got {MODE!r}")

BASE_MODEL     = "unsloth/qwen3-14b-bnb-4bit"
RUNG3_ADAPTER  = "rung3_checkpoints/final_model_lora_150/policy"
TRAINING_DATA  = "grpo_training_data.csv"
SEED           = 3407               # same shuffle as every prior harness
T_STAR_N_REF   = 30                 # reference boards for the mean, disjoint from eval
NOISE_SEED     = 7
MAX_NEW        = 512
N_THOUGHTS     = 4

CONDITIONS = ["baseline", "substitute", "ablate", "zero", "noise", "lenmatch_ablate"]

OUT_DIR         = f"causal_rung3_{MODE}_results"
T_STAR_OUT      = os.path.join(OUT_DIR, "t_star_rung3.pt")
NOISE_VEC_OUT   = os.path.join(OUT_DIR, "noise_star_rung3.pt")
COMPLETIONS_OUT = os.path.join(OUT_DIR, "causal_completions_rung3.csv")
SUMMARY_OUT     = os.path.join(OUT_DIR, "causal_summary_rung3.csv")
PAIRED_OUT      = os.path.join(OUT_DIR, "causal_paired_rung3.csv")
LOG_OUT         = os.path.join(OUT_DIR, "causal_suite_rung3.log")

CSV_FIELDS = ["position_idx", "condition", "fen", "best_move", "parsed_move",
              "format_ok", "legal", "correct", "false_mate", "completion"]

BOT_TOKEN = "<|box_start|>"
EOT_TOKEN = "<|box_end|>"

SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""

# original rung-3 n=100 numbers, printed alongside this run's for direct comparison
N100_REFERENCE = {
    "baseline":        {"format": 100.0, "legal": 58.0, "accuracy": 9.0, "false_mate": 0},
    "substitute":      {"format": 100.0, "legal": 58.0, "accuracy": 9.0, "false_mate": 0},
    "ablate":          {"format": 100.0, "legal": 57.0, "accuracy": 9.0, "false_mate": 0},
    "zero":            {"format": 16.0,  "legal": 9.0,  "accuracy": 0.0, "false_mate": 0},
    "noise":           {"format": 100.0, "legal": 60.0, "accuracy": 8.0, "false_mate": 0},
    "lenmatch_ablate": {"format": 90.0,  "legal": 53.0, "accuracy": 9.0, "false_mate": 0},
}

# ═══════════════════════════════════════════════════════════════════════════

if SAVE_OUTPUTS:
    os.makedirs(OUT_DIR, exist_ok=True)


def setup_logging():
    handlers = [logging.StreamHandler()]
    if SAVE_OUTPUTS:
        handlers.append(logging.FileHandler(LOG_OUT))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("causal_rung3")


log = setup_logging()
log.info("MODE=%s  EVAL_SET_SIZE=%d  SAVE_OUTPUTS=%s", MODE, EVAL_SET_SIZE, SAVE_OUTPUTS)


def to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"


def normalize_move(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def build_latent_prompt(fen: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nFEN: {fen}<|im_end|>\n"
        f"<|im_start|>assistant\n<thinking>\n{BOT_TOKEN}"
    )


def fmt_hms(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ── SAME SHUFFLE/SEED AS EVERY PRIOR HARNESS ─────────────────────────────────
log.info("Loading dataset, same shuffle/seed as every prior harness...")
df = pd.read_csv(TRAINING_DATA).sample(frac=1, random_state=SEED).reset_index(drop=True)
eval_df = df.iloc[:EVAL_SET_SIZE].reset_index(drop=True)
eval_rows = [(i, r["FEN"], str(r["Best Move"]).strip()) for i, r in eval_df.iterrows()]
log.info("  %d eval positions (first 100 identical to the original rung-3 arc).", len(eval_rows))

ref_df = df.iloc[EVAL_SET_SIZE:EVAL_SET_SIZE + T_STAR_N_REF].reset_index(drop=True)
ref_fens = [r["FEN"] for _, r in ref_df.iterrows()]
log.info("  %d reference positions for T_star (disjoint from eval).", len(ref_fens))


# ── LOAD MODEL + RUNG-3 ADAPTER (ONE LOAD FOR ALL 6 CONDITIONS) ─────────────
log.info("Loading base model %s in 4-bit...", BASE_MODEL)
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

for t in (BOT_TOKEN, EOT_TOKEN):
    ids = tokenizer.encode(t, add_special_tokens=False)
    if len(ids) != 1:
        raise SystemExit(f"FATAL: {t} not single-token: {ids}")

log.info("Attaching RUNG-3 adapter from %s...", RUNG3_ADAPTER)
model = PeftModel.from_pretrained(base, RUNG3_ADAPTER)
model.eval()
model.config.use_cache = False
model.generation_config.eos_token_id = im_end
model.generation_config.pad_token_id = tokenizer.pad_token_id

embed = model.get_input_embeddings()
device = next(model.parameters()).device
hidden_dim = embed.embedding_dim


# ── SHARED THOUGHT LOOP + DECODE ─────────────────────────────────────────────
@torch.no_grad()
def run_thought_loop(fen: str):
    prompt = build_latent_prompt(fen)
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False).input_ids.to(device)
    cur_emb = embed(ids)
    cur_mask = torch.ones_like(ids)
    thought_vecs = []
    for _ in range(N_THOUGHTS):
        out = model(inputs_embeds=cur_emb, attention_mask=cur_mask,
                    output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][:, -1:, :].to(cur_emb.dtype)
        thought_vecs.append(h[0, 0].clone())
        cur_emb = torch.cat([cur_emb, h], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(cur_mask[:, :1])], dim=1)
    return cur_emb, cur_mask, thought_vecs


@torch.no_grad()
def decode(cur_emb, cur_mask) -> str:
    gen = model.generate(
        inputs_embeds=cur_emb, attention_mask=cur_mask, max_new_tokens=MAX_NEW,
        do_sample=False, use_cache=True, eos_token_id=im_end,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(gen[0], skip_special_tokens=False)


@torch.no_grad()
def generate_with_fixed_slots(fen: str, slots: torch.Tensor) -> str:
    prompt = build_latent_prompt(fen)
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False).input_ids.to(device)
    cur_emb = embed(ids)
    cur_mask = torch.ones_like(ids)
    slot_emb = slots.to(cur_emb.dtype).to(device).unsqueeze(0)
    cur_emb = torch.cat([cur_emb, slot_emb], dim=1)
    cur_mask = torch.cat([cur_mask, torch.ones(1, N_THOUGHTS, device=device,
                                                 dtype=cur_mask.dtype)], dim=1)
    return decode(cur_emb, cur_mask)


# ── T_STAR: mean thought vectors over the disjoint reference sample ─────────
def compute_t_star() -> torch.Tensor:
    if SAVE_OUTPUTS and os.path.isfile(T_STAR_OUT):
        log.info("Found cached T_star at %s, reusing.", T_STAR_OUT)
        return torch.load(T_STAR_OUT, map_location=device)

    log.info("Computing T_star over %d reference boards...", len(ref_fens))
    per_board = []
    for fen in tqdm(ref_fens, desc="collecting thoughts for T_star", unit="pos"):
        _, _, vecs = run_thought_loop(fen)
        per_board.append(vecs)

    stacked = torch.stack([torch.stack(b) for b in per_board])
    t_star = stacked.mean(dim=0).float()

    sims = torch.nn.functional.cosine_similarity(
        stacked.float(), t_star.unsqueeze(0).expand_as(stacked.float()), dim=-1)
    log.info("  T_star sanity, mean cosine of each reference board to T_star per slot:")
    for t in range(N_THOUGHTS):
        log.info("    slot %d: mean=%.3f  min=%.3f", t, sims[:, t].mean().item(), sims[:, t].min().item())

    if SAVE_OUTPUTS:
        torch.save(t_star, T_STAR_OUT)
        log.info("  Saved T_star to %s", T_STAR_OUT)
    return t_star


def build_noise_star(t_star: torch.Tensor) -> torch.Tensor:
    if SAVE_OUTPUTS and os.path.isfile(NOISE_VEC_OUT):
        log.info("Found cached noise vector at %s, reusing.", NOISE_VEC_OUT)
        return torch.load(NOISE_VEC_OUT, map_location=device)

    target_norms = t_star.norm(dim=-1)
    log.info("T_star per-slot norms (target magnitudes): %s",
              [f"{n:.2f}" for n in target_norms.tolist()])

    g = torch.Generator(device="cpu").manual_seed(NOISE_SEED)
    raw = torch.randn(t_star.shape, generator=g).to(device).float()
    unit = raw / raw.norm(dim=-1, keepdim=True)
    noise_star = unit * target_norms.unsqueeze(-1)

    cos_to_tstar = torch.nn.functional.cosine_similarity(noise_star, t_star, dim=-1)
    log.info("Cosine of noise vector to T_star per slot (should be near 0): %s",
              [f"{c:.3f}" for c in cos_to_tstar.tolist()])

    if SAVE_OUTPUTS:
        torch.save(noise_star, NOISE_VEC_OUT)
        log.info("  Saved noise vector to %s", NOISE_VEC_OUT)
    return noise_star


t_star = compute_t_star()
noise_star = build_noise_star(t_star)
pad_id = tokenizer.pad_token_id
pad_embed = embed(torch.tensor([[pad_id]], device=device))[0, 0].detach().float()
log.info("Pad embedding norm = %.2f (an ordinary in-vocab magnitude, content-free meaning).",
          pad_embed.norm().item())
zero_slots = torch.zeros(N_THOUGHTS, hidden_dim, device=device)


# ── CONDITION DISPATCH ───────────────────────────────────────────────────────
@torch.no_grad()
def generate_condition(fen: str, condition: str) -> str:
    if condition == "baseline":
        cur_emb, cur_mask, _ = run_thought_loop(fen)
        return decode(cur_emb, cur_mask)
    if condition == "substitute":
        return generate_with_fixed_slots(fen, t_star)
    if condition == "ablate":
        prompt = build_latent_prompt(fen)
        ids = tokenizer(prompt, return_tensors="pt",
                        add_special_tokens=False).input_ids.to(device)
        return decode(embed(ids), torch.ones_like(ids))
    if condition == "zero":
        return generate_with_fixed_slots(fen, zero_slots)
    if condition == "noise":
        return generate_with_fixed_slots(fen, noise_star)
    if condition == "lenmatch_ablate":
        pad_slots = pad_embed.view(1, -1).expand(N_THOUGHTS, -1)
        return generate_with_fixed_slots(fen, pad_slots)
    raise ValueError(f"unknown condition: {condition}")


def score_completion(fen, best, comp):
    move = extract_move(comp)
    has_tags = ("</thinking>" in comp and "<output>" in comp and "</output>" in comp)
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

    return move, is_format_ok, is_legal, is_correct, is_false_mate


# ── RESUME SUPPORT (full mode only) ──────────────────────────────────────────
def load_existing_completions():
    if not (SAVE_OUTPUTS and os.path.isfile(COMPLETIONS_OUT)):
        return {}
    try:
        existing = pd.read_csv(COMPLETIONS_OUT, on_bad_lines="skip")
    except Exception as e:
        log.warning("Could not parse existing completions file (%s) -- starting fresh.", e)
        return {}
    done = {}
    for _, row in existing.iterrows():
        try:
            key = (row["condition"], int(row["position_idx"]))
            done[key] = row.to_dict()
        except (ValueError, KeyError, TypeError):
            continue
    return done


# ── SCORE ONE CONDITION, RESUMABLE, INCREMENTAL WRITE ────────────────────────
def score_condition(condition: str, done_lookup: dict):
    n = len(eval_rows)
    records = []
    fmt_ok = legal = correct = false_mate = 0

    already_done = set()
    if SAVE_OUTPUTS:
        for i in range(n):
            key = (condition, i)
            if key in done_lookup:
                r = done_lookup[key]
                records.append(r)
                fmt_ok     += int(to_bool(r["format_ok"]))
                legal      += int(to_bool(r["legal"]))
                correct    += int(to_bool(r["correct"]))
                false_mate += int(to_bool(r["false_mate"]))
                already_done.add(i)

    pending = [i for i in range(n) if i not in already_done]
    if already_done:
        log.info("  %s: %d/%d already done on disk, resuming remaining %d.",
                  condition, len(already_done), n, len(pending))

    csv_file = csv_writer = None
    if SAVE_OUTPUTS and pending:
        write_header = (not os.path.isfile(COMPLETIONS_OUT)) or os.stat(COMPLETIONS_OUT).st_size == 0
        csv_file = open(COMPLETIONS_OUT, "a", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            csv_writer.writeheader()

    total_scored = len(already_done)
    pbar = tqdm(pending, desc=f"condition={condition}", unit="pos", total=n, initial=total_scored)
    for i in pbar:
        idx, fen, best = eval_rows[i]
        try:
            comp = generate_condition(fen, condition)
            move, is_format_ok, is_legal, is_correct, is_false_mate = score_completion(fen, best, comp)
        except Exception as e:
            log.warning("  Generation failed, condition=%s idx=%d fen=%.30s...: %s -- "
                        "counting as unparseable, continuing.", condition, idx, fen, e)
            comp = f"__GENERATION_ERROR__: {e}"
            move, is_format_ok, is_legal, is_correct, is_false_mate = None, False, False, False, False

        fmt_ok += int(is_format_ok); legal += int(is_legal)
        correct += int(is_correct); false_mate += int(is_false_mate)
        total_scored += 1

        record = {
            "position_idx": idx, "condition": condition, "fen": fen, "best_move": best,
            "parsed_move": move, "format_ok": is_format_ok, "legal": is_legal,
            "correct": is_correct, "false_mate": is_false_mate, "completion": comp,
        }
        records.append(record)
        if csv_writer:
            csv_writer.writerow(record)
            csv_file.flush()
            os.fsync(csv_file.fileno())

        pbar.set_postfix(legal=f"{100*legal/total_scored:.0f}%",
                          fmt=f"{100*fmt_ok/total_scored:.0f}%", fmate=false_mate)
    pbar.close()
    if csv_file:
        csv_file.close()

    metrics = {
        "condition": condition, "format": 100.0 * fmt_ok / n, "legal": 100.0 * legal / n,
        "accuracy": 100.0 * correct / n, "false_mate": false_mate,
    }
    log.info("  %s done: format %.1f%%  legal %.1f%%  accuracy %.1f%%  false_mate %d",
              condition, metrics["format"], metrics["legal"], metrics["accuracy"], false_mate)
    return metrics, records


# ── PAIRWISE EXACT-MOVE-MATCH ANALYSIS ───────────────────────────────────────
def pairwise_analysis(records_by_condition, conditions):
    move_by_cond = {}
    for c in conditions:
        move_by_cond[c] = {}
        for r in records_by_condition[c]:
            idx = int(r["position_idx"])
            move_by_cond[c][idx] = normalize_move(r["parsed_move"])

    indices = [idx for idx, _, _ in eval_rows]
    rows = []
    for i, condA in enumerate(conditions):
        for condB in conditions[i + 1:]:
            match = both_none = one_none = 0
            for idx in indices:
                mA, mB = move_by_cond[condA].get(idx), move_by_cond[condB].get(idx)
                if mA is None and mB is None:
                    both_none += 1
                elif mA is None or mB is None:
                    one_none += 1
                elif mA == mB:
                    match += 1
            n = len(indices)
            rows.append({
                "condition_a": condA, "condition_b": condB, "n": n,
                "match": match, "match_pct": round(100.0 * match / n, 1),
                "flips": n - match, "both_unparseable": both_none, "one_unparseable": one_none,
            })
    return pd.DataFrame(rows)


# ── RUN ALL 6 CONDITIONS ─────────────────────────────────────────────────────
log.info("")
log.info("Running %d conditions on RUNG-3, MODE=%s, N=%d: %s",
          len(CONDITIONS), MODE, EVAL_SET_SIZE, CONDITIONS)

done_lookup = load_existing_completions()
if done_lookup:
    log.info("Resume: found %d already-completed (condition, position) rows on disk.", len(done_lookup))

all_records, records_by_condition, results = [], {}, {}
summary_df, paired_df = None, None
t_run_start = time.time()

try:
    for idx, condition in enumerate(CONDITIONS):
        results[condition], recs = score_condition(condition, done_lookup)
        records_by_condition[condition] = recs
        all_records += recs
        elapsed = time.time() - t_run_start
        done_n = idx + 1
        avg = elapsed / done_n
        remaining = avg * (len(CONDITIONS) - done_n)
        log.info("  [%d/%d conditions done, elapsed %s, est. remaining %s, avg %s/condition]",
                  done_n, len(CONDITIONS), fmt_hms(elapsed), fmt_hms(remaining), fmt_hms(avg))

        # Incremental save: rebuild summary + pairwise CSVs after EVERY condition,
        # not just at the very end. If the known post-completion CUDA teardown
        # crash hits right after the last condition, these are already correct
        # and on disk -- nothing to reconstruct by hand.
        if SAVE_OUTPUTS:
            completed_so_far = CONDITIONS[:done_n]
            summary_df = pd.DataFrame([results[c] for c in completed_so_far])
            paired_df = pairwise_analysis(records_by_condition, completed_so_far)
            summary_df.to_csv(SUMMARY_OUT, index=False)
            paired_df.to_csv(PAIRED_OUT, index=False)
            log.info("  Updated %s and %s through condition '%s'.",
                      SUMMARY_OUT, PAIRED_OUT, condition)
except Exception:
    log.exception("FATAL error mid-run. If MODE=full, everything scored so far is already "
                  "safely on disk (including summary/paired through the last completed "
                  "condition) -- just rerun this script unchanged and it will resume "
                  "from exactly where it stopped.")
    raise

if summary_df is None:
    summary_df = pd.DataFrame([results[c] for c in CONDITIONS])
    paired_df = pairwise_analysis(records_by_condition, CONDITIONS)


# ── FINAL REPORT ──────────────────────────────────────────────────────────
total_elapsed = time.time() - t_run_start
log.info("")
log.info("=" * 78)
log.info("RUNG-3, MODE=%s, N=%d vs ORIGINAL RUNG-3 (N=100), same 6 conditions", MODE, EVAL_SET_SIZE)
log.info("=" * 78)
log.info(f"{'condition':<18}{'thisrun legal%':>16}{'n100 legal%':>13}{'thisrun fmt%':>14}{'n100 fmt%':>11}{'fmate':>8}")
for c in CONDITIONS:
    r = results[c]
    ref = N100_REFERENCE[c]
    log.info(f"{c:<18}{r['legal']:>16.1f}{ref['legal']:>13.1f}{r['format']:>14.1f}"
              f"{ref['format']:>11.1f}{r['false_mate']:>8}")
log.info("=" * 78)

log.info("")
log.info("Pairwise exact-move match (n=%d, no rows dropped):", len(eval_rows))
log.info("\n%s", paired_df.to_string(index=False))

log.info("")
log.info("TOTAL RUNTIME for N=%d, all 6 conditions: %s", EVAL_SET_SIZE, fmt_hms(total_elapsed))
if MODE == "smoke":
    log.info("Extrapolated to N=1000 (linear scaling, model-load cost excluded): %s",
              fmt_hms(total_elapsed * (1000.0 / EVAL_SET_SIZE)))
    log.info("Nothing was saved to disk (MODE=smoke). Switch MODE to 'full' to run for real.")
else:
    log.info("All output in %s -- scp -r that whole folder down.", OUT_DIR)