"""
causal_suite_stage2.py -- Six-condition causal test suite (baseline, substitute,
ablate, zero, noise, lenmatch_ablate), re-run against the STAGE-2 adapter instead of
rung-3. Same frozen 100-position harness, same rewards.py scoring, one model load,
one file, one run. Ends with a pairwise exact-move-match table across all conditions,
built without the dropna bug that inflated zero's match rate in the rung-3 arc.
"""

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
BASE_MODEL     = "unsloth/qwen3-14b-bnb-4bit"
STAGE2_ADAPTER = "rung2_checkpoints/stage_2/final"   # stage-2 launchpad, NOT rung-3
TRAINING_DATA  = "grpo_training_data.csv"
SEED           = 3407
EVAL_SET_SIZE  = 100                # same frozen 100 as every prior harness
T_STAR_N_REF   = 30                 # reference boards for the mean, disjoint from eval
NOISE_SEED     = 7
MAX_NEW        = 512
N_THOUGHTS     = 4                  # stage-2 uses 4 thoughts, matches rung-3's count

CONDITIONS = ["baseline", "substitute", "ablate", "zero", "noise", "lenmatch_ablate"]

OUT_DIR         = "causal_stage2_results"
T_STAR_OUT      = os.path.join(OUT_DIR, "t_star_stage2.pt")
NOISE_VEC_OUT   = os.path.join(OUT_DIR, "noise_star_stage2.pt")
COMPLETIONS_OUT = os.path.join(OUT_DIR, "causal_completions_stage2.csv")
SUMMARY_OUT     = os.path.join(OUT_DIR, "causal_summary_stage2.csv")
PAIRED_OUT      = os.path.join(OUT_DIR, "causal_paired_stage2.csv")
LOG_OUT         = os.path.join(OUT_DIR, "causal_suite_stage2.log")

BOT_TOKEN = "<|box_start|>"
EOT_TOKEN = "<|box_end|>"

SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""

# rung-3's known numbers, printed alongside stage-2's for direct comparison
RUNG3_REFERENCE = {
    "baseline":        {"format": 100.0, "legal": 58.0, "accuracy": 9.0, "false_mate": 0},
    "substitute":      {"format": 100.0, "legal": 58.0, "accuracy": 9.0, "false_mate": 0},
    "ablate":          {"format": 100.0, "legal": 57.0, "accuracy": 9.0, "false_mate": 0},
    "zero":            {"format": 16.0,  "legal": 9.0,  "accuracy": 0.0, "false_mate": 0},
    "noise":           {"format": 100.0, "legal": 60.0, "accuracy": 8.0, "false_mate": 0},
    "lenmatch_ablate": {"format": 90.0,  "legal": 53.0, "accuracy": 9.0, "false_mate": 0},
}

# ═══════════════════════════════════════════════════════════════════════════

os.makedirs(OUT_DIR, exist_ok=True)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(LOG_OUT), logging.StreamHandler()],
    )
    return logging.getLogger("causal_stage2")


log = setup_logging()


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


# ── SAME HELD-OUT 100 + REFERENCE SAMPLE AS THE RUNG-3 ARC ──────────────────
log.info("Loading dataset, same shuffle/seed as the rung-3 causal arc...")
df = pd.read_csv(TRAINING_DATA).sample(frac=1, random_state=SEED).reset_index(drop=True)
eval_df = df.iloc[:EVAL_SET_SIZE].reset_index(drop=True)
eval_rows = [(r["FEN"], str(r["Best Move"]).strip()) for _, r in eval_df.iterrows()]
log.info("  %d eval positions.", len(eval_rows))

ref_df = df.iloc[EVAL_SET_SIZE:EVAL_SET_SIZE + T_STAR_N_REF].reset_index(drop=True)
ref_fens = [r["FEN"] for _, r in ref_df.iterrows()]
log.info("  %d reference positions for T_star (disjoint from eval).", len(ref_fens))


# ── LOAD MODEL + STAGE-2 ADAPTER (ONE LOAD FOR ALL 6 CONDITIONS) ────────────
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

log.info("Attaching STAGE-2 adapter from %s...", STAGE2_ADAPTER)
model = PeftModel.from_pretrained(base, STAGE2_ADAPTER)
model.eval()
model.config.use_cache = False
model.generation_config.eos_token_id = im_end
model.generation_config.pad_token_id = tokenizer.pad_token_id

embed = model.get_input_embeddings()
device = next(model.parameters()).device
hidden_dim = embed.embedding_dim


# ── SHARED THOUGHT LOOP + DECODE (used by every condition below) ────────────
@torch.no_grad()
def run_thought_loop(fen: str):
    """Stage-2's own deterministic 4-step Coconut loop. Returns (cur_emb, cur_mask,
    thought_vecs); used both for baseline generation and for building T_star."""
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
    """slots: [N_THOUGHTS, hidden_dim]. Spliced in place of the thought loop, then
    decoded normally. Shared by substitute/zero/noise/lenmatch_ablate -- the four are
    mechanically identical, only which fixed vector goes in differs."""
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
    if os.path.isfile(T_STAR_OUT):
        log.info("Found cached stage-2 T_star at %s, reusing.", T_STAR_OUT)
        return torch.load(T_STAR_OUT, map_location=device)

    log.info("Computing stage-2 T_star over %d reference boards...", len(ref_fens))
    per_board = []
    for fen in tqdm(ref_fens, desc="collecting thoughts for T_star", unit="pos"):
        _, _, vecs = run_thought_loop(fen)
        per_board.append(vecs)

    stacked = torch.stack([torch.stack(b) for b in per_board])   # [n_ref, n_thoughts, d]
    t_star = stacked.mean(dim=0).float()

    sims = torch.nn.functional.cosine_similarity(
        stacked.float(), t_star.unsqueeze(0).expand_as(stacked.float()), dim=-1)
    log.info("  T_star sanity, mean cosine of each reference board to T_star per slot:")
    for t in range(N_THOUGHTS):
        log.info("    slot %d: mean=%.3f  min=%.3f", t, sims[:, t].mean().item(), sims[:, t].min().item())

    torch.save(t_star, T_STAR_OUT)
    log.info("  Saved T_star to %s", T_STAR_OUT)
    return t_star


def build_noise_star(t_star: torch.Tensor) -> torch.Tensor:
    if os.path.isfile(NOISE_VEC_OUT):
        log.info("Found cached stage-2 noise vector at %s, reusing.", NOISE_VEC_OUT)
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


# ── SCORING (identical logic across every condition) ─────────────────────────
def score_condition(condition: str):
    n = len(eval_rows)
    fmt_ok = legal = correct = false_mate = 0
    records = []

    pbar = tqdm(eval_rows, desc=f"condition={condition}", unit="pos", total=n)
    for fen, best in pbar:
        comp = generate_condition(fen, condition)
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

        fmt_ok += int(is_format_ok); legal += int(is_legal)
        correct += int(is_correct); false_mate += int(is_false_mate)

        records.append({
            "condition": condition, "fen": fen, "best_move": best,
            "parsed_move": move, "format_ok": is_format_ok, "legal": is_legal,
            "correct": is_correct, "false_mate": is_false_mate, "completion": comp,
        })
        done = len(records)
        pbar.set_postfix(legal=f"{100*legal/done:.0f}%", fmt=f"{100*fmt_ok/done:.0f}%", fmate=false_mate)
    pbar.close()

    metrics = {
        "condition": condition, "format": 100.0 * fmt_ok / n, "legal": 100.0 * legal / n,
        "accuracy": 100.0 * correct / n, "false_mate": false_mate,
    }
    log.info("  %s done: format %.1f%%  legal %.1f%%  accuracy %.1f%%  false_mate %d",
              condition, metrics["format"], metrics["legal"], metrics["accuracy"], false_mate)
    return metrics, records


# ── PAIRWISE EXACT-MOVE-MATCH ANALYSIS -- fixes the dropna bug ──────────────
def pairwise_analysis(records_by_condition, conditions):
    """Exact-move match rate between every pair of conditions, over ALL 100 positions,
    with nothing dropped. An unparseable completion (move=None) on either side always
    counts as a mismatch -- never silently excluded before counting. This is the fix
    for the bug where the old paired-analysis script's dropna() inflated zero's
    reported match rate to 87.5% when the true rate (nothing dropped) was ~14%."""
    move_by_cond = {c: {r["fen"]: r["parsed_move"] for r in records_by_condition[c]}
                    for c in conditions}
    fens = [fen for fen, _ in eval_rows]
    rows = []
    for i, condA in enumerate(conditions):
        for condB in conditions[i + 1:]:
            match = both_none = one_none = 0
            for fen in fens:
                mA, mB = move_by_cond[condA].get(fen), move_by_cond[condB].get(fen)
                if mA is None and mB is None:
                    both_none += 1
                elif mA is None or mB is None:
                    one_none += 1
                elif mA == mB:
                    match += 1
            n = len(fens)
            rows.append({
                "condition_a": condA, "condition_b": condB, "n": n,
                "match": match, "match_pct": round(100.0 * match / n, 1),
                "flips": n - match, "both_unparseable": both_none, "one_unparseable": one_none,
            })
    return pd.DataFrame(rows)


# ── RUN ALL 6 CONDITIONS ─────────────────────────────────────────────────────
log.info("")
log.info("Running %d conditions on the STAGE-2 adapter: %s", len(CONDITIONS), CONDITIONS)
all_records, records_by_condition, results = [], {}, {}
t_run_start = time.time()
for idx, condition in enumerate(CONDITIONS):
    t_cond_start = time.time()
    results[condition], recs = score_condition(condition)
    records_by_condition[condition] = recs
    all_records += recs
    elapsed = time.time() - t_run_start
    done_n = idx + 1
    avg = elapsed / done_n
    remaining = avg * (len(CONDITIONS) - done_n)
    log.info("  [%d/%d conditions done, elapsed %s, est. remaining %s]",
              done_n, len(CONDITIONS), fmt_hms(elapsed), fmt_hms(remaining))

comp_df = pd.DataFrame(all_records, columns=[
    "condition", "fen", "best_move", "parsed_move",
    "format_ok", "legal", "correct", "false_mate", "completion",
])
comp_df.to_csv(COMPLETIONS_OUT, index=False)
log.info("Wrote %d per-position rows to %s", len(comp_df), COMPLETIONS_OUT)

summary_df = pd.DataFrame([results[c] for c in CONDITIONS])
summary_df.to_csv(SUMMARY_OUT, index=False)
log.info("Wrote aggregate metrics to %s", SUMMARY_OUT)

paired_df = pairwise_analysis(records_by_condition, CONDITIONS)
paired_df.to_csv(PAIRED_OUT, index=False)
log.info("Wrote pairwise match analysis to %s", PAIRED_OUT)


# ── FINAL REPORT ──────────────────────────────────────────────────────────
log.info("")
log.info("=" * 78)
log.info("STAGE-2 vs RUNG-3, same 6 conditions, same frozen 100-position harness")
log.info("=" * 78)
log.info(f"{'condition':<18}{'stage2 legal%':>15}{'rung3 legal%':>14}{'stage2 fmt%':>13}{'rung3 fmt%':>12}{'fmate':>8}")
for c in CONDITIONS:
    r = results[c]
    ref = RUNG3_REFERENCE[c]
    log.info(f"{c:<18}{r['legal']:>15.1f}{ref['legal']:>14.1f}{r['format']:>13.1f}"
              f"{ref['format']:>12.1f}{r['false_mate']:>8}")
log.info("=" * 78)

log.info("")
log.info("Pairwise exact-move match (n=%d, no rows dropped):", len(eval_rows))
log.info("\n%s", paired_df.to_string(index=False))
log.info("")
log.info("If stage-2's baseline vs zero true match rate lands near rung-3's ~14/100")
log.info("(not the buggy 87.5%% from the old dropna summary), the zero-collapse pattern")
log.info("replicates on stage-2 too. If substitute/noise cluster tightly with baseline")
log.info("here the same way they did on rung-3, that's the content-invariance finding")
log.info("showing up BEFORE RL, not just after it.")
log.info("")
log.info("All output in %s -- scp -r that whole folder down.", OUT_DIR)