"""
07_train_latent.py  --  Rung 2: Coconut curriculum SFT on Qwen3-14B (4-bit).

Two modes, one code path. MODE="measure" runs the equivalence proof and a
throughput sweep, then exits without training. MODE="train" runs the curriculum.
Both use the identical latent_step, so measured numbers are real numbers.

Model loading is byte-identical to train_grpo.py so that stage adapters remain
comparable to grpo_v2 on the frozen harness (eval_models.py).

The in-script gate is a TRIPWIRE, not a paper number. It runs on 100 rows held
out of training, with this process's weights. Canonical numbers come from
eval_models.py, always.

Sized from 08_length_profile.py on the real data:
  prefix          146 tok median, 163 max
  stage-1 suffix  171 tok median, 288 max
  stage-1 total   315 tok median, 442 max
  stage-5 total   173 tok median, 190 max
"""

import os

# MUST be set before torch is imported. Reduces fragmentation across the
# 5 stages, whose sequence lengths shrink from ~440 tok down to ~190 tok.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import math
import re
import time
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

MODE = "train"                   # "measure" -> proof + sweep + exit. "train" -> curriculum.

BASE_MODEL = "unsloth/qwen3-14b-bnb-4bit"
SFT_ADAPTER_PATH = "qwen3_14b_sft_checkpoints/final_model_lora"
PARSED_JSONL = "parsed_dataset.jsonl"
RUN_DIR = "rung2_checkpoints"

# ---- curriculum ----
STAGES = [1, 2, 3, 4, 5]
THOUGHTS_PER_CHUNK = 2
EPOCHS_PER_STAGE = 1

ROWS_PER_STAGE = 4000

# ---- optimization ----
BATCH_SIZE = 4                   # measured: b8 OOMs at stage 5. b4 peaked 20.64 GB.
GRAD_ACCUM = 2                   # effective batch 8
LR = 1e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
SEED = 3407

# ---- data hygiene ----
GATE_N = 100                     # last N rows, held out of training, gate only
MAX_TOTAL_TOKENS = 460           # p100 of stage-1 totals is 442. This is a real guard now.

# ---- the tripwire ----
GATE_MIN_LEGAL = 0.20            # D5: below this, stop the curriculum
GATE_MAX_NEW_TOKENS = 512        # longest true stage-1 suffix is 288. Headroom for rambling.
GATE_SHOW_SAMPLES = 2

# ---- OOM resilience ----
MAX_OOM_FRACTION = 0.02          # abort the stage if >2% of micro-batches OOM

# ---- logging / resumability ----
LOG_EVERY = 10                   # micro-batches
SAVE_EVERY = 200                 # micro-batches
GATE_ON_STAGES = [1, 2, 3, 4, 5]

# ---- measure mode ----
MEASURE_STAGES = [1, 5]
MEASURE_BATCHES = [1, 2, 4, 8]
MEASURE_STEPS = 6                # first 2 discarded as warmup
MEASURE_PROOF_ROWS = 4

# ---- format (must match SFT byte-for-byte) ----
BOT_TOKEN = "<|box_start|>"
EOT_TOKEN = "<|box_end|>"

SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""

# ═══════════════════════════════════════════════════════════════════════════

try:
    import chess
except ImportError:
    raise SystemExit(
        "python-chess is not importable in this env. The gate needs it.\n"
        "Do NOT pip install into `chess` (patched site-packages). Investigate first."
    )


def banner(msg):
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78, flush=True)


# ───────────────────────────────────────────────────────────────────────────
# TRANSFORM
# ───────────────────────────────────────────────────────────────────────────

def get_chunks(row):
    s = row["section_texts"]
    return [
        row["preamble"] + s[0],
        s[1],
        s[2],
        s[3],
        s[4] + row["conclusion"],
    ]


def build_example(row, stage):
    chunks = get_chunks(row)
    spoken = "".join(chunks[stage:])
    shared_prefix = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nFEN: {row['fen']}<|im_end|>\n"
        f"<|im_start|>assistant\n<thinking>\n"
    )
    closing = f"\n</thinking>\n<output>\n{row['best_move']}\n</output><|im_end|>"
    if spoken:
        suffix = f"{EOT_TOKEN}\n{spoken}{closing}"
    else:
        suffix = f"{EOT_TOKEN}{closing}"
    return {
        "prefix": shared_prefix + BOT_TOKEN,
        "n_thoughts": THOUGHTS_PER_CHUNK * stage,
        "suffix": suffix,
    }


def build_gate_prompt(row, stage):
    """Prefix only. The model must produce eot + spoken + move on its own."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nFEN: {row['fen']}<|im_end|>\n"
        f"<|im_start|>assistant\n<thinking>\n{BOT_TOKEN}"
    )


# ───────────────────────────────────────────────────────────────────────────
# COLLATE
# ───────────────────────────────────────────────────────────────────────────

def positions_from_mask(mask):
    pos = mask.long().cumsum(dim=1) - 1
    return pos.clamp(min=0)


def collate(batch_rows, stage, tok, device, pad_id):
    exs = [build_example(r, stage) for r in batch_rows]
    n_thoughts = exs[0]["n_thoughts"]

    prefix_ids = [tok(e["prefix"], add_special_tokens=False).input_ids for e in exs]
    suffix_ids = [tok(e["suffix"], add_special_tokens=False).input_ids for e in exs]

    P = max(len(p) for p in prefix_ids)
    S = max(len(s) for s in suffix_ids)
    B = len(exs)

    prefix_t = torch.full((B, P), pad_id, dtype=torch.long)
    prefix_mask = torch.zeros((B, P), dtype=torch.long)
    for i, p in enumerate(prefix_ids):
        prefix_t[i, P - len(p):] = torch.tensor(p)          # LEFT pad
        prefix_mask[i, P - len(p):] = 1

    suffix_t = torch.full((B, S), pad_id, dtype=torch.long)
    suffix_labels = torch.full((B, S), -100, dtype=torch.long)
    suffix_mask = torch.zeros((B, S), dtype=torch.long)
    for i, s in enumerate(suffix_ids):
        suffix_t[i, :len(s)] = torch.tensor(s)              # RIGHT pad
        suffix_labels[i, :len(s)] = torch.tensor(s)
        suffix_mask[i, :len(s)] = 1

    return (prefix_t.to(device), prefix_mask.to(device),
            suffix_t.to(device), suffix_labels.to(device),
            suffix_mask.to(device), n_thoughts)


# ───────────────────────────────────────────────────────────────────────────
# THE LATENT STEP
# ───────────────────────────────────────────────────────────────────────────

def latent_step(model, prefix_t, prefix_mask, suffix_t,
                suffix_labels, suffix_mask, n_thoughts):
    """Batched latent training step. Loss on real suffix tokens only.

    Each thought costs a full forward pass over the sequence so far. n thoughts
    means n+1 forwards per step. This is inherent to Coconut, not a bug, and it
    is why stage 5 (10 thoughts, 173 tokens) is slower than stage 1
    (2 thoughts, 315 tokens).

    The .to(cur_emb.dtype) cast guards against prepare_model_for_kbit_training
    upcasting the final norm to fp32 while embeddings stay bf16. On this build
    both land in fp32 so it is a no-op, but it costs nothing and prevents a
    silent torch.cat dtype crash if the env ever changes.
    """
    embed = model.get_input_embeddings()
    cur_emb = embed(prefix_t)                               # [B, P, D]
    cur_mask = prefix_mask

    for _ in range(n_thoughts):
        pos = positions_from_mask(cur_mask)
        out = model(inputs_embeds=cur_emb, attention_mask=cur_mask,
                    position_ids=pos, output_hidden_states=True,
                    use_cache=False)
        h = out.hidden_states[-1][:, -1:, :]                # aligned via left-pad
        h = h.to(cur_emb.dtype)                             # NO detach. The whole point.
        cur_emb = torch.cat([cur_emb, h], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(cur_mask[:, :1])], dim=1)

    suffix_emb = embed(suffix_t)
    full_emb = torch.cat([cur_emb, suffix_emb], dim=1)
    full_mask = torch.cat([cur_mask, suffix_mask], dim=1)
    pos = positions_from_mask(full_mask)
    out = model(inputs_embeds=full_emb, attention_mask=full_mask,
                position_ids=pos, use_cache=False)

    P = prefix_t.shape[1]
    S = suffix_t.shape[1]
    logits = out.logits[:, P + n_thoughts - 1: P + n_thoughts - 1 + S, :]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        suffix_labels.reshape(-1),
        ignore_index=-100,
    )
    return loss


# ───────────────────────────────────────────────────────────────────────────
# SETUP
# ───────────────────────────────────────────────────────────────────────────

def load_rows():
    rows = []
    with open(PARSED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def containment_check():
    """The markers must NEVER appear in the real training corpus.
    check_special_tokens.py scanned chess_reasoning_final_dataset.csv, which is
    the trap file. Redo it against what we actually train on."""
    with open(PARSED_JSONL, "r", encoding="utf-8") as f:
        blob = f.read()
    for t in (BOT_TOKEN, EOT_TOKEN):
        if t in blob:
            raise SystemExit(
                f"FATAL: {t} appears inside {PARSED_JSONL}. "
                "The silence markers collide with real data. Pick different tokens."
            )
    print(f"Containment OK: neither marker appears in {PARSED_JSONL}.")


def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tok.unk_token_id:
        tok.eos_token = "<|im_end|>"
        tok.eos_token_id = im_end
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for t in (BOT_TOKEN, EOT_TOKEN):
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"FATAL: {t} is not single-token on {BASE_MODEL}: {ids}")
    print(f"Markers single-token: "
          f"{BOT_TOKEN}={tok.encode(BOT_TOKEN, add_special_tokens=False)[0]}, "
          f"{EOT_TOKEN}={tok.encode(EOT_TOKEN, add_special_tokens=False)[0]}")
    print(f"EOS={tok.eos_token_id} PAD={tok.pad_token_id}")
    return tok


def load_model(adapter_path):
    print(f"\nLoading {BASE_MODEL} in 4-bit...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    print(f"Loading adapter: {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()     # required: GC + PEFT + inputs_embeds
    model.config.use_cache = False
    model.train()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train:,} / {n_all:,} ({100*n_train/n_all:.3f}%)")

    print("\nLoRA sanity (lora_B norms must be > 0, else SFT was lost):")
    shown = 0
    for name, p in model.named_parameters():
        if "lora_B" in name:
            print(f"  {name}: {p.data.norm().item():.6f}")
            shown += 1
            if shown >= 3:
                break

    embed = model.get_input_embeddings()
    dev = embed.weight.device
    print(f"\nDtype report:")
    print(f"  embedding: {embed.weight.dtype} on {dev}")
    with torch.no_grad():
        probe = torch.tensor([[1, 2, 3]], device=dev)
        out = model(input_ids=probe, output_hidden_states=True, use_cache=False)
        print(f"  hidden_states[-1]: {out.hidden_states[-1].dtype}")
    del out
    torch.cuda.empty_cache()

    return model, dev


def filter_rows(rows, tok, stage_max):
    """Drop rows exceeding MAX_TOTAL_TOKENS at the longest stage (stage 1)."""
    keep, dropped = [], 0
    for r in tqdm(rows, desc="length filter", unit="row"):
        ex = build_example(r, 1)          # stage 1 = longest suffix
        n = (len(tok(ex["prefix"], add_special_tokens=False).input_ids)
             + THOUGHTS_PER_CHUNK * stage_max
             + len(tok(ex["suffix"], add_special_tokens=False).input_ids))
        if n <= MAX_TOTAL_TOKENS:
            keep.append(r)
        else:
            dropped += 1
    print(f"Length filter: kept {len(keep)}, dropped {dropped} "
          f"(> {MAX_TOTAL_TOKENS} tokens)")
    return keep


def subsample_for_stage(rows, stage):
    """Deterministic per-stage subset. Same stage always yields the same rows,
    so a mid-stage resume sees the identical data. Different stages see
    different subsets, so total coverage across the curriculum stays high."""
    if ROWS_PER_STAGE <= 0 or ROWS_PER_STAGE >= len(rows):
        return rows
    rng = np.random.default_rng(SEED + 7919 * stage)
    idx = rng.choice(len(rows), size=ROWS_PER_STAGE, replace=False)
    return [rows[i] for i in sorted(idx)]


# ───────────────────────────────────────────────────────────────────────────
# MEASURE MODE
# ───────────────────────────────────────────────────────────────────────────

def equivalence_proof(model, rows, tok, device, pad_id, stage):
    banner(f"EQUIVALENCE PROOF at stage {stage} on the REAL 14B (NF4)")
    print("Tolerance is NOT asserted. We report the number and read it.")
    test_rows = rows[:MEASURE_PROOF_ROWS]

    model.eval()
    with torch.no_grad():
        singles = []
        for r in tqdm(test_rows, desc="singles", unit="row"):
            singles.append(latent_step(model, *collate([r], stage, tok, device, pad_id)).item())
        batched = latent_step(model, *collate(test_rows, stage, tok, device, pad_id)).item()
    model.train()

    counts = [len(tok(build_example(r, stage)["suffix"], add_special_tokens=False).input_ids)
              for r in test_rows]
    expected = sum(l * c for l, c in zip(singles, counts)) / sum(counts)
    diff = abs(batched - expected)

    print(f"\nsingle losses:       {[f'{l:.4f}' for l in singles]}")
    print(f"token-weighted mean: {expected:.4f}")
    print(f"batched loss:        {batched:.4f}")
    print(f"abs diff:            {diff:.4f}  "
          f"({100*diff/expected:.2f}% relative)")
    print("\nfp32 control (06_prototype_batching.py, 0.6B): 1e-5 absolute.")
    print("bf16 0.6B: 0.7% relative. NF4 14B: expect ~2%. Quantization plus")
    print("kernel-order noise compounded through the thought loop. Not a bug.")
    print("It does prove batch-1 greedy eval is mandatory.")
    return diff


def throughput_sweep(model, rows, tok, device, pad_id, n_train):
    banner("THROUGHPUT SWEEP on the REAL 14B. GC ON. This sets BATCH_SIZE.")
    params = [p for p in model.parameters() if p.requires_grad]
    results = {}

    print(f"{'stage':>6} {'thoughts':>9} {'batch':>6} {'peak GB':>9} "
          f"{'s/step':>9} {'ex/s':>8}")
    print("-" * 78)

    for stage in MEASURE_STAGES:
        for B in MEASURE_BATCHES:
            optim = torch.optim.AdamW(params, lr=LR)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            times = []
            oom = False
            try:
                for step in range(MEASURE_STEPS):
                    batch_rows = [rows[(step * B + j) % len(rows)] for j in range(B)]
                    b = collate(batch_rows, stage, tok, device, pad_id)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    loss = latent_step(model, *b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    times.append(time.time() - t0)
            except torch.cuda.OutOfMemoryError:
                oom = True
            finally:
                del optim
                gc.collect()
                torch.cuda.empty_cache()

            if oom:
                print(f"{stage:>6} {2*stage:>9} {B:>6} {'OOM':>9} {'-':>9} {'-':>8}")
                continue

            steady = times[2:] if len(times) > 2 else times
            sps = sum(steady) / len(steady)
            peak = torch.cuda.max_memory_allocated() / 1e9
            results[(stage, B)] = sps
            print(f"{stage:>6} {2*stage:>9} {B:>6} {peak:>9.2f} "
                  f"{sps:>9.3f} {B/sps:>8.2f}", flush=True)

    banner("CURRICULUM COST PROJECTION")
    rows_used = ROWS_PER_STAGE if 0 < ROWS_PER_STAGE < n_train else n_train
    print(f"Linear fit of s/step against thought count, per batch size.")
    print(f"Assumes {rows_used} rows/stage, {EPOCHS_PER_STAGE} epoch(s) per stage.\n")
    print(f"{'batch':>6} {'s/step @2':>11} {'s/step @10':>12} {'total hours':>13}")
    print("-" * 78)
    for B in MEASURE_BATCHES:
        if (1, B) not in results or (5, B) not in results:
            continue
        s1, s5 = results[(1, B)], results[(5, B)]
        slope = (s5 - s1) / (10 - 2)
        intercept = s1 - slope * 2
        total = 0.0
        for stage in STAGES:
            t = THOUGHTS_PER_CHUNK * stage
            sps = intercept + slope * t
            steps = math.ceil(rows_used / B) * EPOCHS_PER_STAGE
            total += steps * sps
        print(f"{B:>6} {s1:>11.3f} {s5:>12.3f} {total/3600:>13.1f}")

    print("\nThe gate runs generation in the same process after each stage.")
    print("Leave headroom. Then set MODE='train' and rerun.")


# ───────────────────────────────────────────────────────────────────────────
# THE GATE  --  batch-1 greedy, D5 protocol. Tripwire only.
# ───────────────────────────────────────────────────────────────────────────

MOVE_RE = re.compile(r"<output>\s*(\S+?)\s*</output>", re.DOTALL)


@torch.no_grad()
def run_gate(model, gate_rows, stage, tok, device):
    banner(f"GATE after stage {stage}  --  {len(gate_rows)} held-out rows, "
           f"batch-1 greedy")
    was_training = model.training
    model.eval()
    torch.cuda.empty_cache()

    embed = model.get_input_embeddings()
    n_thoughts = THOUGHTS_PER_CHUNK * stage
    n_format = n_legal = n_exact = n_truncated = 0
    samples = []

    for i, row in enumerate(tqdm(gate_rows, desc=f"gate s{stage}", unit="pos")):
        prompt = build_gate_prompt(row, stage)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        mask = torch.ones_like(ids)

        cur_emb = embed(ids)
        cur_mask = mask
        for _ in range(n_thoughts):
            out = model(inputs_embeds=cur_emb, attention_mask=cur_mask,
                        output_hidden_states=True, use_cache=False)
            h = out.hidden_states[-1][:, -1:, :].to(cur_emb.dtype)
            cur_emb = torch.cat([cur_emb, h], dim=1)
            cur_mask = torch.cat([cur_mask, torch.ones_like(cur_mask[:, :1])], dim=1)

        # eval() disables the GC recompute path, which is what corrupts
        # generation in train() mode (train_grpo.py comment, TRL #3089).
        gen = model.generate(
            inputs_embeds=cur_emb,
            attention_mask=cur_mask,
            max_new_tokens=GATE_MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        if gen.shape[1] >= GATE_MAX_NEW_TOKENS:
            n_truncated += 1
        text = tok.decode(gen[0], skip_special_tokens=False)

        m = MOVE_RE.search(text)
        if m:
            n_format += 1
            uci = m.group(1)
            try:
                board = chess.Board(row["fen"])
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves:
                    n_legal += 1
                    if uci == row["best_move"]:
                        n_exact += 1
            except (ValueError, IndexError):
                pass

        if i < GATE_SHOW_SAMPLES:
            samples.append((row["fen"], row["best_move"], text[:700]))

    n = len(gate_rows)
    fmt, legal, exact = n_format / n, n_legal / n, n_exact / n

    for fen, best, text in samples:
        print(f"\n--- sample ---\nFEN:  {fen}\nBEST: {best}\nGEN:  {text!r}")

    print(f"\n  format ok  : {n_format:>4}/{n}  ({fmt:.1%})")
    print(f"  legal      : {n_legal:>4}/{n}  ({legal:.1%})")
    print(f"  exact      : {n_exact:>4}/{n}  ({exact:.1%})")
    print(f"  truncated  : {n_truncated:>4}/{n}   "
          f"(hit max_new_tokens={GATE_MAX_NEW_TOKENS})")
    if n_truncated > 0.05 * n:
        print("  WARNING: heavy truncation. The model is rambling, or "
              "GATE_MAX_NEW_TOKENS is too small. A low format score here may "
              "be an artifact, not a collapse.")
    print(f"\n  gate threshold: legal >= {GATE_MIN_LEGAL:.0%}")
    print(f"  VERDICT: {'PASS' if legal >= GATE_MIN_LEGAL else 'FAIL -> STOP'}",
          flush=True)

    if was_training:
        model.train()
    torch.cuda.empty_cache()
    return {"format": fmt, "legal": legal, "exact": exact, "truncated": n_truncated,
            "n_format": n_format, "n_legal": n_legal, "n_exact": n_exact, "n": n}


# ───────────────────────────────────────────────────────────────────────────
# CHECKPOINTING
# ───────────────────────────────────────────────────────────────────────────

def progress_path():
    return os.path.join(RUN_DIR, "progress.json")


def read_progress():
    if os.path.exists(progress_path()):
        with open(progress_path(), "r") as f:
            return json.load(f)
    return {"completed_stages": [], "gates": {}, "stopped": False}


def write_progress(p):
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(progress_path(), "w") as f:
        json.dump(p, f, indent=2)


def stage_dir(stage):
    return os.path.join(RUN_DIR, f"stage_{stage}")


def save_mid_stage(model, optim, sched, stage, epoch, micro_step):
    d = os.path.join(stage_dir(stage), "latest")
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d)
    torch.save({
        "stage": stage,
        "epoch": epoch,
        "micro_step": micro_step,
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, os.path.join(stage_dir(stage), "state.pt"))
    print(f"  [ckpt] stage {stage} epoch {epoch} micro_step {micro_step}", flush=True)


def resolve_resume(progress):
    """Returns (adapter_path, start_stage, mid_state_or_None)."""
    completed = progress["completed_stages"]

    for stage in STAGES:
        if stage in completed:
            continue
        state_p = os.path.join(stage_dir(stage), "state.pt")
        latest_p = os.path.join(stage_dir(stage), "latest")
        if os.path.exists(state_p) and os.path.isdir(latest_p):
            return latest_p, stage, torch.load(state_p, map_location="cpu",
                                               weights_only=False)
        break

    if not completed:
        return SFT_ADAPTER_PATH, STAGES[0], None

    last = max(completed)
    nxt = [s for s in STAGES if s > last]
    if not nxt:
        return None, None, None
    return os.path.join(stage_dir(last), "final"), nxt[0], None


# ───────────────────────────────────────────────────────────────────────────
# TRAIN MODE
# ───────────────────────────────────────────────────────────────────────────

def train_stage(model, tok, device, pad_id, pool_rows, gate_rows,
                stage, mid_state, progress):
    train_rows = subsample_for_stage(pool_rows, stage)
    n = len(train_rows)
    micro_per_epoch = math.ceil(n / BATCH_SIZE)
    total_micro = micro_per_epoch * EPOCHS_PER_STAGE
    total_optim = math.ceil(total_micro / GRAD_ACCUM)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=int(WARMUP_RATIO * total_optim),
        num_training_steps=total_optim,
    )

    start_epoch, start_micro = 0, 0
    if mid_state is not None:
        optim.load_state_dict(mid_state["optim"])
        sched.load_state_dict(mid_state["sched"])
        torch.set_rng_state(mid_state["torch_rng"])
        torch.cuda.set_rng_state_all(mid_state["cuda_rng"])
        start_epoch = mid_state["epoch"]
        start_micro = mid_state["micro_step"]
        print(f"Resuming stage {stage} at epoch {start_epoch}, "
              f"micro_step {start_micro}/{micro_per_epoch}")

    banner(f"STAGE {stage}  --  {THOUGHTS_PER_CHUNK * stage} silent thoughts, "
           f"{5 - stage} spoken chunk(s)")
    print(f"rows {n} (subsampled from {len(pool_rows)}) | "
          f"batch {BATCH_SIZE} x accum {GRAD_ACCUM} = effective {BATCH_SIZE * GRAD_ACCUM}")
    print(f"micro-batches/epoch {micro_per_epoch} | epochs {EPOCHS_PER_STAGE} "
          f"| optimizer steps {total_optim}")
    print(f"lr {LR} | warmup {int(WARMUP_RATIO * total_optim)} | "
          f"clip {MAX_GRAD_NORM} | forwards/step {THOUGHTS_PER_CHUNK * stage + 1}",
          flush=True)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    t_start = time.time()
    running = []
    n_oom = 0

    for epoch in range(start_epoch, EPOCHS_PER_STAGE):
        rng = np.random.default_rng(SEED + 1000 * stage + epoch)
        perm = rng.permutation(n)
        first = start_micro if epoch == start_epoch else 0

        pbar = tqdm(range(first, micro_per_epoch), desc=f"s{stage} e{epoch}",
                    unit="mb", total=micro_per_epoch - first)

        for micro in pbar:
            idx = perm[micro * BATCH_SIZE:(micro + 1) * BATCH_SIZE]
            batch_rows = [train_rows[i] for i in idx]
            if not batch_rows:
                continue

            try:
                b = collate(batch_rows, stage, tok, device, pad_id)
                loss = latent_step(model, *b)
                (loss / GRAD_ACCUM).backward()
                running.append(loss.item())
            except torch.cuda.OutOfMemoryError:
                n_oom += 1
                optim.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                lens = [len(tok(build_example(r, stage)["suffix"],
                                add_special_tokens=False).input_ids) for r in batch_rows]
                print(f"\n  [OOM] s{stage} mb {micro+1}, suffix lens {lens}. "
                      f"Batch skipped. Total OOMs: {n_oom}", flush=True)
                if n_oom > MAX_OOM_FRACTION * micro_per_epoch:
                    raise SystemExit(
                        f"OOM rate exceeded {MAX_OOM_FRACTION:.0%}. "
                        f"Lower BATCH_SIZE or MAX_TOTAL_TOKENS and resume."
                    )
                continue

            if (micro + 1) % GRAD_ACCUM == 0 or micro == micro_per_epoch - 1:
                torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)

            if (micro + 1) % LOG_EVERY == 0 and running:
                avg = sum(running[-LOG_EVERY:]) / min(LOG_EVERY, len(running))
                elapsed = time.time() - t_start
                done = micro - first + 1
                sps = elapsed / done
                remain = (micro_per_epoch - micro - 1) * sps
                peak = torch.cuda.max_memory_allocated() / 1e9
                pbar.set_postfix(loss=f"{avg:.4f}")
                print(f"  s{stage} e{epoch} mb {micro+1}/{micro_per_epoch} | "
                      f"loss {avg:.4f} | lr {sched.get_last_lr()[0]:.2e} | "
                      f"{sps:.2f}s/mb | eta {remain/3600:.2f}h | "
                      f"peak {peak:.1f}GB", flush=True)

            if (micro + 1) % SAVE_EVERY == 0:
                save_mid_stage(model, optim, sched, stage, epoch, micro + 1)

        pbar.close()
        start_micro = 0

    final = os.path.join(stage_dir(stage), "final")
    os.makedirs(final, exist_ok=True)
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"\nStage {stage} adapter saved: {final}")
    print(f"Stage {stage} wall time: {(time.time() - t_start)/3600:.2f}h | "
          f"OOM skips: {n_oom}", flush=True)

    del optim, sched
    gc.collect()
    torch.cuda.empty_cache()

    if stage in GATE_ON_STAGES:
        metrics = run_gate(model, gate_rows, stage, tok, device)
        metrics["oom_skips"] = n_oom
        progress["gates"][str(stage)] = metrics
        if metrics["legal"] < GATE_MIN_LEGAL:
            progress["stopped"] = True
            progress["stopped_at_stage"] = stage
            write_progress(progress)
            banner(f"GATE FAILED at stage {stage}. "
                   f"legal {metrics['legal']:.1%} < {GATE_MIN_LEGAL:.0%}")
            print("This is a RESULT, not a crash. The curriculum collapses here.")
            print("Adapter is saved. Run eval_models.py for canonical numbers.")
            return False

    progress["completed_stages"].append(stage)
    write_progress(progress)
    return True


# ───────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA.")

    banner(f"07_train_latent.py  --  MODE = {MODE}")
    print(f"base       : {BASE_MODEL}")
    print(f"sft        : {SFT_ADAPTER_PATH}")
    print(f"data       : {PARSED_JSONL}")
    print(f"run dir    : {RUN_DIR}")
    print(f"gpu        : {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GiB)")
    print(f"alloc conf : {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"rows/stage : {ROWS_PER_STAGE if ROWS_PER_STAGE > 0 else 'ALL'}")
    print(f"batch      : {BATCH_SIZE} x {GRAD_ACCUM} accum")

    containment_check()
    tok = build_tokenizer()
    pad_id = tok.pad_token_id

    all_rows = load_rows()
    print(f"\nLoaded {len(all_rows)} rows from {PARSED_JSONL}")
    gate_rows = all_rows[-GATE_N:]
    pool = all_rows[:-GATE_N]
    print(f"Held out last {len(gate_rows)} rows for the gate. "
          f"They are NEVER trained on.")

    pool_rows = filter_rows(pool, tok, stage_max=max(STAGES))

    progress = read_progress()
    if progress.get("stopped"):
        raise SystemExit(
            f"progress.json says the run stopped at stage "
            f"{progress.get('stopped_at_stage')}. Delete {RUN_DIR} to start over, "
            "or edit progress.json deliberately."
        )

    if MODE == "measure":
        model, device = load_model(SFT_ADAPTER_PATH)
        equivalence_proof(model, pool_rows, tok, device, pad_id, stage=5)
        throughput_sweep(model, pool_rows, tok, device, pad_id, n_train=len(pool_rows))
        banner("MEASURE COMPLETE. No weights were saved. Nothing was trained.")
        return

    if MODE != "train":
        raise SystemExit(f"MODE must be 'measure' or 'train', got {MODE!r}")

    adapter_path, start_stage, mid_state = resolve_resume(progress)
    if adapter_path is None:
        banner("All stages already complete. Nothing to do.")
        return
    print(f"\nResume plan: start at stage {start_stage}, adapter = {adapter_path}")
    if progress["completed_stages"]:
        print(f"Completed already: {sorted(progress['completed_stages'])}")

    model, device = load_model(adapter_path)

    for stage in STAGES:
        if stage < start_stage:
            continue
        ms = mid_state if stage == start_stage else None
        ok = train_stage(model, tok, device, pad_id, pool_rows, gate_rows,
                         stage, ms, progress)
        if not ok:
            return

    banner("CURRICULUM COMPLETE")
    for s in sorted(progress["gates"], key=int):
        g = progress["gates"][s]
        print(f"  stage {s}: format {g['format']:.1%}  "
              f"legal {g['legal']:.1%}  exact {g['exact']:.1%}")
    print(f"\nFinal adapter: {os.path.join(stage_dir(max(STAGES)), 'final')}")
    print("Now run eval_models.py against grpo_v2 for the canonical comparison.")


if __name__ == "__main__":
    main()