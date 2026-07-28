"""
train_rung3_gumbel.py -- Rung 3, Experiment 2: Gumbel-Softmax straight-through latent
thoughts, replacing Experiment 1's deterministic raw-hidden-state thoughts so GRPO's
gradient can actually reach thought generation, not just spoken-word phrasing. Same
launchpad, reward, data, and KL leash as Experiment 1; only the thought mechanism changes.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import json
import math
import re
import time
import numpy as np
import pandas as pd
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
from rewards import (
    FEN_TO_BEST_EVAL,
    chess_reward,
    extract_move,
    is_move_legal,
    is_position_checkmate,
    shutdown_engine,
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

MODE = "train"                  # "smoke" -> wiring checks only. "measure" -> price
                                 # steps + eval. "train" -> run.

BASE_MODEL     = "unsloth/qwen3-14b-bnb-4bit"
STAGE2_ADAPTER = "rung2_checkpoints/stage_2/final"        
EVALS_CSV      = "grpo_training_data_evals.csv"           # ONLY valid training source
RUN_DIR        = "rung3_gumbel_checkpoints"                # separate from rung3_checkpoints

# ---- latent config ----
N_THOUGHTS = 4                   # stage 2 = 4 silent thoughts. Do not change.

# ---- Gumbel-Softmax (straight-through) -- Experiment 2's one real change ----
# LOCKED from smoke testing: tau=1 gave 0 variation (model too confident for
# unmodified sampling), tau=8 with no top-k gave garbage (softened ALL ~150k
# candidates at once, swamping the leader). tau=4 + top_k=20 gave real variation
# with semantically plausible picks (e.g. spelled out "[CAPTURE", "</thinking>").
GUMBEL_TAU = 4.0
GUMBEL_TOP_K = 20
STRAIGHT_THROUGH = True          # forward pass uses a REAL word's own embedding (never
                                 # a blurry mixture); gradient flows through the soft
                                 # distribution behind the hard pick.

# ---- GRPO ----
GROUP_SIZE = 4
PROMPTS_PER_STEP = 4
KL_COEF = 0.02
TEMPERATURE = 0.7                # spoken-part sampling temperature
TOP_P = 1.0
MAX_NEW = 320
ADV_EPS = 1e-4

# ---- optimization ----
LR = 2e-6
MAX_GRAD_NORM = 1.0
MAX_STEPS = 150                  # step count exactly --
                                 # measured at 316.5s/step -> ~14h40m total. One
                                 # variable per experiment: mechanism changes, budget
                                 # doesn't, so any difference in outcome is attributable.
WARMUP_STEPS = 10
SEED = 3407

# ---- eval / gate ----
EVAL_EVERY = 50
EVAL_SET_SIZE = 100
SAVE_EVERY = 50

# ---- measure mode ----
MEASURE_STEPS = 8
MEASURE_EVAL_POS = 10

# ---- format ----
BOT_TOKEN = "<|box_start|>"
EOT_TOKEN = "<|box_end|>"

SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""

# ═══════════════════════════════════════════════════════════════════════════


def banner(msg):
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78, flush=True)


def fmt_hms(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def build_latent_prompt(fen: str) -> str:
    """Prefix only, ends at BOT."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nFEN: {fen}<|im_end|>\n"
        f"<|im_start|>assistant\n<thinking>\n{BOT_TOKEN}"
    )


def _fen_from_prompt(prompt):
    m = re.search(r"FEN:\s*(.+)", prompt)
    if not m:
        return ""
    return re.sub(r"<\|.*?\|>", "", m.group(1)).strip()


# ───────────────────────────────────────────────────────────────────────────
# DATA (unchanged from Experiment 1)
# ───────────────────────────────────────────────────────────────────────────

def load_data():
    """seed/split: first 100 rows are the frozen harness, the rest
    are the training pool. Populates FEN_TO_BEST_EVAL."""
    df = pd.read_csv(EVALS_CSV)
    if "best_eval_cp" not in df.columns:
        raise SystemExit(f"{EVALS_CSV} has no best_eval_cp column. Run precompute_evals.py first.")
    n_before = len(df)
    df = df.dropna(subset=["best_eval_cp"]).reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing best_eval_cp.")

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    for _, r in df.iterrows():
        FEN_TO_BEST_EVAL[r["FEN"]] = int(r["best_eval_cp"])
    print(f"  {len(FEN_TO_BEST_EVAL)} FEN -> best_eval_cp mappings loaded.")

    eval_df = df.iloc[:EVAL_SET_SIZE].reset_index(drop=True)
    train_df = df.iloc[EVAL_SET_SIZE:].reset_index(drop=True)
    eval_rows = [(r["FEN"], str(r["Best Move"]).strip()) for _, r in eval_df.iterrows()]
    train_fens = [r["FEN"] for _, r in train_df.iterrows()]
    print(f"  train positions: {len(train_fens)} | held-out harness: {len(eval_rows)}")
    return train_fens, eval_rows


# ───────────────────────────────────────────────────────────────────────────
# MODEL  +  the frozen reference adapter
# ───────────────────────────────────────────────────────────────────────────

def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    tok.eos_token = "<|im_end|>"
    tok.eos_token_id = im_end
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for t in (BOT_TOKEN, EOT_TOKEN):
        if len(tok.encode(t, add_special_tokens=False)) != 1:
            raise SystemExit(f"FATAL: {t} not single-token.")
    return tok, im_end


def load_model():
    """One base, two adapters: 'policy' (trainable, = stage-2) and 'reference' (frozen,
    = stage-2), loaded from the SAME checkpoint twice on purpose. disable_adapter()
    would give raw base Qwen instead of stage-2, anchoring the KL leash to the wrong
    thing -- never use it here."""
    print(f"\nLoading base {BASE_MODEL} in 4-bit...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map="auto",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    print(f"Loading stage-2 as trainable 'policy' from {STAGE2_ADAPTER}...", flush=True)
    model = PeftModel.from_pretrained(base, STAGE2_ADAPTER,
                                      adapter_name="policy", is_trainable=True)
    print("Loading stage-2 again as frozen 'reference'...", flush=True)
    model.load_adapter(STAGE2_ADAPTER, adapter_name="reference")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False

    model.set_adapter("policy")
    print("\nPolicy lora_B norms (must be > 0 = stage-2 loaded, not fresh):")
    shown = 0
    for name, p in model.named_parameters():
        if "lora_B" in name and "policy" in name:
            print(f"  ...{name.split('.')[-4]}.{name.split('.')[-3]}: {p.data.norm().item():.4f}")
            shown += 1
            if shown >= 3:
                break
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (policy only): {n_train:,}")
    return model


# ───────────────────────────────────────────────────────────────────────────
# GUMBEL-SOFTMAX CORE
# ───────────────────────────────────────────────────────────────────────────

def sample_gumbel_noise(shape, device, dtype):
    """One draw of Gumbel(0,1) noise per element: uniform random, then -log(-log(.)).
    Adding this to a set of scores and taking the argmax is mathematically equivalent
    to sampling from those scores' softmax distribution -- a form we can backprop
    through."""
    eps = 1e-10
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_thought_loop(model, tok, embed_matrix, cur_emb, cur_mask, n_thoughts,
                        tau, top_k, straight_through, saved_noises=None,
                        sample_noise=True, verbose=False):
    """Runs n_thoughts silent steps: forward pass, mask out everything except the
    top_k logits, scale by 1/tau, add Gumbel noise, softmax, hard-argmax a real
    vocabulary token, feed its own embedding back in. saved_noises replays an exact
    prior trajectory with gradient attached; sample_noise=False gives a deterministic
    (zero-noise) argmax for evaluation."""
    noises_used = []
    picked_ids = []
    for step in range(n_thoughts):
        out = model(inputs_embeds=cur_emb, attention_mask=cur_mask, use_cache=False)
        logits = out.logits[:, -1, :].float()             # [1, vocab] -- already unembedded

        if top_k is not None and top_k < logits.shape[-1]:
            topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(-1, topk_idx, topk_vals)
            logits = masked                                 # outside top_k can never win

        if verbose:
            top3 = torch.topk(logits[0], k=3)
            words = [tok.decode([i]).strip() for i in top3.indices.tolist()]
            vals = top3.values.tolist()
            gap = vals[0] - vals[1]
            print(f"      thought {step+1}: top3 = "
                  f"{list(zip(words, [f'{v:.2f}' for v in vals]))}  "
                  f"gap(#1-#2) = {gap:.2f}")

        if saved_noises is not None:
            noise = saved_noises[step]
        elif sample_noise:
            noise = sample_gumbel_noise(logits.shape, logits.device, logits.dtype)
        else:
            noise = torch.zeros_like(logits)

        noisy = logits / tau + noise                        # tau scales logits only
        soft = F.softmax(noisy, dim=-1)                     # differentiable
        idx = soft.argmax(dim=-1)                            # [1] hard pick, real vocab token

        if straight_through:
            hard = F.one_hot(idx, num_classes=soft.shape[-1]).to(soft.dtype)
            onehot = hard - soft.detach() + soft              # fwd = hard, bwd = soft's grad
        else:
            onehot = soft

        thought_emb = (onehot.to(embed_matrix.dtype) @ embed_matrix)  # [1, dim]
        thought_emb = thought_emb.unsqueeze(1).to(cur_emb.dtype)        # [1, 1, dim]

        cur_emb = torch.cat([cur_emb, thought_emb], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(cur_mask[:, :1])], dim=1)

        noises_used.append(noise)
        picked_ids.append(idx.item())

    return cur_emb, cur_mask, noises_used, picked_ids


# ───────────────────────────────────────────────────────────────────────────
# ROLLOUT  --  prefix -> Gumbel thoughts (real noise) -> SAMPLED spoken part
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_gumbel(model, tok, im_end, fen, device, verbose=False):
    """One completion. Saves the per-step Gumbel noise so a later grad-connected pass
    can replay the exact same sampled trajectory with gradient attached."""
    prompt = build_latent_prompt(fen)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    mask = torch.ones_like(ids)
    embed = model.get_input_embeddings()
    embed_matrix = embed.weight

    cur_emb = embed(ids)
    cur_mask = mask
    cur_emb, cur_mask, noises_used, picked_ids = gumbel_thought_loop(
        model, tok, embed_matrix, cur_emb, cur_mask, N_THOUGHTS,
        GUMBEL_TAU, GUMBEL_TOP_K, STRAIGHT_THROUGH, saved_noises=None,
        sample_noise=True, verbose=verbose)

    gen = model.generate(
        inputs_embeds=cur_emb, attention_mask=cur_mask, max_new_tokens=MAX_NEW,
        max_length=None, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
        use_cache=True, eos_token_id=im_end, pad_token_id=tok.pad_token_id,
    )
    spoken_ids = gen[0]
    completion = tok.decode(spoken_ids, skip_special_tokens=False)
    return prompt, completion, spoken_ids.detach(), noises_used, picked_ids, ids.detach()


@torch.no_grad()
def generate_deterministic(model, tok, im_end, fen, device):
    """The OLD Coconut path: raw hidden-state feedback, no Gumbel. Used only in the
    smoke test to confirm the reference adapter is completely untouched."""
    prompt = build_latent_prompt(fen)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    mask = torch.ones_like(ids)
    embed = model.get_input_embeddings()
    cur_emb = embed(ids)
    cur_mask = mask
    for _ in range(N_THOUGHTS):
        out = model(inputs_embeds=cur_emb, attention_mask=cur_mask,
                    output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][:, -1:, :].to(cur_emb.dtype)
        cur_emb = torch.cat([cur_emb, h], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(cur_mask[:, :1])], dim=1)
    gen = model.generate(
        inputs_embeds=cur_emb, attention_mask=cur_mask, max_new_tokens=MAX_NEW,
        max_length=None, do_sample=False, use_cache=True, eos_token_id=im_end,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(gen[0], skip_special_tokens=False)


def spoken_logprobs(model, adapter, thought_embeds, thought_mask, spoken_ids):
    """Log-prob the given adapter assigns to spoken_ids, conditioned on the fixed
    prefix+thoughts context. Agnostic to how thought_embeds were produced."""
    model.set_adapter(adapter)
    embed = model.get_input_embeddings()
    spoken_ids = spoken_ids.to(thought_embeds.device)
    spk_emb = embed(spoken_ids.unsqueeze(0))
    full_emb = torch.cat([thought_embeds, spk_emb], dim=1)
    full_mask = torch.cat(
        [thought_mask, torch.ones((1, spk_emb.shape[1]), device=thought_mask.device,
                                  dtype=thought_mask.dtype)], dim=1)
    out = model(inputs_embeds=full_emb, attention_mask=full_mask, use_cache=False)

    T = thought_embeds.shape[1]
    S = spoken_ids.shape[0]
    logits = out.logits[:, T - 1: T - 1 + S, :]
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp[0, torch.arange(S), spoken_ids]


def policy_and_reference_logprobs(model, tok, prefix_ids, noises_used, spoken_ids):
    """Grad-connected replay of the Gumbel thought loop using rollout's exact saved
    noise, then spoken-token log-probs under both adapters. Reference reuses policy's
    realized thought embeddings as fixed conditioning; a cross-mechanism thought-level 
    KL isn't well-defined, so the leash stays on spoken words only."""
    embed = model.get_input_embeddings()
    embed_matrix = embed.weight

    model.set_adapter("policy")
    cur_emb = embed(prefix_ids)
    cur_mask = torch.ones_like(prefix_ids)
    cur_emb, cur_mask, _, _ = gumbel_thought_loop(
        model, tok, embed_matrix, cur_emb, cur_mask, N_THOUGHTS,
        GUMBEL_TAU, GUMBEL_TOP_K, STRAIGHT_THROUGH, saved_noises=noises_used)

    pol_lp = spoken_logprobs(model, "policy", cur_emb, cur_mask, spoken_ids)
    with torch.no_grad():
        ref_lp = spoken_logprobs(model, "reference", cur_emb.detach(), cur_mask, spoken_ids)
    model.set_adapter("policy")
    return pol_lp, ref_lp


# ───────────────────────────────────────────────────────────────────────────
# ONE GRPO STEP
# ───────────────────────────────────────────────────────────────────────────

def grpo_step(model, tok, im_end, optim, sched, fens, device, show_bars=True):
    all_prompts, all_completions, group_of, cached = [], [], [], []

    model.set_adapter("policy")
    model.eval()
    total = len(fens) * GROUP_SIZE
    bar = tqdm(total=total, desc="  rollouts", unit="gen", leave=False) if show_bars else None
    for gi, fen in enumerate(fens):
        for _ in range(GROUP_SIZE):
            p, c, sp, noises, _picked, prefix_ids = rollout_gumbel(model, tok, im_end, fen, device)
            all_prompts.append(p); all_completions.append(c)
            group_of.append(gi)
            cached.append((sp, noises, prefix_ids))
            if bar: bar.update(1)
    if bar: bar.close()

    rewards = torch.tensor(chess_reward(all_completions, prompts=all_prompts),
                           dtype=torch.float32)

    group_of_t = torch.tensor(group_of)
    adv = torch.zeros_like(rewards)
    for gi in range(len(fens)):
        m = group_of_t == gi
        g = rewards[m]
        adv[m] = (g - g.mean()) / (g.std(unbiased=False) + ADV_EPS)

    model.train()
    optim.zero_grad(set_to_none=True)
    total_pg = total_kl = 0.0
    n = len(all_completions)
    upd = tqdm(range(n), desc="  policy update", unit="gen", leave=False) if show_bars else range(n)
    for i in upd:
        sp, noises, prefix_ids = cached[i]
        a = adv[i].item()
        pol_lp, ref_lp = policy_and_reference_logprobs(model, tok, prefix_ids, noises, sp)
        log_ratio = ref_lp - pol_lp
        kl = (torch.exp(log_ratio) - 1 - log_ratio).mean()          # k3, >= 0
        pg = -(a * pol_lp.mean())
        loss = (pg + KL_COEF * kl) / n
        loss.backward()
        total_pg += pg.item(); total_kl += kl.item()

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)
    optim.step(); sched.step()

    false_mate = sum(
        1 for c, p in zip(all_completions, all_prompts)
        if extract_move(c) == "CHECKMATE" and not is_position_checkmate(_fen_from_prompt(p)))
    return {
        "reward_mean": float(rewards.mean()), "reward_max": float(rewards.max()),
        "legal_frac": float((rewards > 0).float().mean()), "false_mate": false_mate,
        "pg": total_pg / n, "kl": total_kl / n, "lr": sched.get_last_lr()[0],
    }


# ───────────────────────────────────────────────────────────────────────────
# HARNESS  --  frozen 100 positions, batch-1, ZERO-noise Gumbel argmax
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tok, im_end, eval_rows, device, desc="harness"):
    """Deterministic eval: thoughts via zero-noise Gumbel argmax, spoken part greedy.
    Reproducible run-to-run and comparable in kind to Experiment 1's frozen-harness
    numbers."""
    model.set_adapter("policy")
    was = model.training
    model.eval()
    embed = model.get_input_embeddings()
    embed_matrix = embed.weight
    fmt = legal = correct = false_mate = 0

    for fen, best in tqdm(eval_rows, desc=desc, unit="pos", leave=False):
        prompt = build_latent_prompt(fen)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        cur_emb = embed(ids)
        cur_mask = torch.ones_like(ids)
        cur_emb, cur_mask, _, _ = gumbel_thought_loop(
            model, tok, embed_matrix, cur_emb, cur_mask, N_THOUGHTS,
            GUMBEL_TAU, GUMBEL_TOP_K, STRAIGHT_THROUGH, saved_noises=None, sample_noise=False)
        gen = model.generate(
            inputs_embeds=cur_emb, attention_mask=cur_mask, max_new_tokens=MAX_NEW,
            max_length=None, do_sample=False, use_cache=True, eos_token_id=im_end,
            pad_token_id=tok.pad_token_id)
        comp = tok.decode(gen[0], skip_special_tokens=False)

        move = extract_move(comp)
        if "</thinking>" in comp and "<output>" in comp and move is not None:
            fmt += 1
        if move == "CHECKMATE":
            if is_position_checkmate(fen):
                legal += 1
                if best.upper() == "CHECKMATE":
                    correct += 1
            else:
                false_mate += 1
        elif move is not None and is_move_legal(fen, move):
            legal += 1
            if move == best:
                correct += 1

    if was:
        model.train()
    n = len(eval_rows)
    return {"format": 100*fmt/n, "legal": 100*legal/n,
            "accuracy": 100*correct/n, "false_mate": false_mate}


# ───────────────────────────────────────────────────────────────────────────
# SMOKE TEST  --  4 wiring checks, no training
# ───────────────────────────────────────────────────────────────────────────

def run_smoke_test(model, tok, im_end, eval_rows, device):
    banner("SMOKE TEST")
    fen = eval_rows[0][0]
    print(f"  test position (held-out #0): {fen}")
    print(f"  GUMBEL_TAU = {GUMBEL_TAU}  GUMBEL_TOP_K = {GUMBEL_TOP_K}\n")

    print("  [1/4] Randomness check -- same board, 3 rollouts, thought picks:")
    model.set_adapter("policy")
    model.eval()
    for run_i in range(3):
        print(f"    run {run_i + 1}:")
        _, _completion, _sp, _noises, picked_ids, _prefix = rollout_gumbel(
            model, tok, im_end, fen, device, verbose=(run_i == 0))
        words = [tok.decode([pid]).strip() for pid in picked_ids]
        print(f"      picked: {words}")
    print("    PASS if the three 'picked' rows are NOT all identical, AND the words")
    print("    look like plausible near-misses (not CJK/code-fragment garbage).\n")

    print("  [2/4] Gradient-flow check -- one backward pass through the Gumbel path:")
    _prompt, _completion, spoken_ids, noises_used, _picked, prefix_ids = rollout_gumbel(
        model, tok, im_end, fen, device)
    model.train()
    pol_lp, _ref_lp = policy_and_reference_logprobs(model, tok, prefix_ids, noises_used, spoken_ids)
    loss = -pol_lp.mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    lora_grads = [(n, p.grad) for n, p in model.named_parameters()
                 if "lora_B" in n and "policy" in n and p.grad is not None]
    if not lora_grads:
        print("    FAIL: no policy LoRA parameter received a gradient at all.")
    else:
        total_norm = sum(g.norm().item() ** 2 for _, g in lora_grads) ** 0.5
        finite = math.isfinite(total_norm)
        ok = finite and total_norm > 0
        print(f"    grad norm across {len(lora_grads)} policy lora_B tensors: {total_norm:.6f}")
        print(f"    {'PASS' if ok else 'FAIL'}: finite and nonzero = the thought loop is connected.")
    model.zero_grad(set_to_none=True)
    model.eval()
    print()

    print("  [3/4] Reference determinism check -- old raw-hidden-state path, run twice:")
    model.set_adapter("reference")
    c1 = generate_deterministic(model, tok, im_end, fen, device)
    c2 = generate_deterministic(model, tok, im_end, fen, device)
    same = (c1 == c2)
    print(f"    identical across two runs: {same}")
    print(f"    {'PASS' if same else 'FAIL'}: reference must be untouched and deterministic.")
    model.set_adapter("policy")
    print()

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  [4/4] Peak GPU memory so far: {peak:.1f} GB\n")

    banner("SMOKE TEST COMPLETE -- review all 4 checks above before MEASURE/TRAIN")


# ───────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA.")

    banner(f"train_rung3_gumbel.py  --  MODE = {MODE}  (Rung 3, Experiment 2: Gumbel thoughts)")
    print(f"launchpad : {STAGE2_ADAPTER}")
    print(f"data      : {EVALS_CSV}")
    print(f"G={GROUP_SIZE}  prompts/step={PROMPTS_PER_STEP}  "
          f"gens/step={GROUP_SIZE*PROMPTS_PER_STEP}  thoughts={N_THOUGHTS}")
    print(f"kl_coef={KL_COEF}  temp={TEMPERATURE}  lr={LR}  max_new={MAX_NEW}")
    print(f"gumbel_tau={GUMBEL_TAU}  gumbel_top_k={GUMBEL_TOP_K}  straight_through={STRAIGHT_THROUGH}")

    tok, im_end = build_tokenizer()
    train_fens, eval_rows = load_data()
    model = load_model()

    if MODE == "smoke":
        run_smoke_test(model, tok, im_end, eval_rows, model.device)
        shutdown_engine()
        return

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=LR)
    sched = get_cosine_schedule_with_warmup(optim, WARMUP_STEPS, max(MAX_STEPS, 1))
    rng = np.random.default_rng(SEED)

    if MODE == "measure":
        banner("MEASURE: pricing steps")
        t0 = time.time()
        for s in range(MEASURE_STEPS):
            fens = [train_fens[i] for i in rng.integers(0, len(train_fens), PROMPTS_PER_STEP)]
            st = time.time()
            m = grpo_step(model, tok, im_end, optim, sched, fens, model.device)
            peak = torch.cuda.max_memory_allocated()/1e9
            print(f"  step {s+1}/{MEASURE_STEPS}: {time.time()-st:.1f}s | "
                  f"reward {m['reward_mean']:+.3f} | legal {m['legal_frac']:.0%} | "
                  f"fmate {m['false_mate']} | kl {m['kl']:.5f} | peak {peak:.1f}GB", flush=True)
        per_step = (time.time() - t0) / MEASURE_STEPS

        print(f"\n  Timing one harness eval on {MEASURE_EVAL_POS} positions...", flush=True)
        te = time.time()
        _ = evaluate(model, tok, im_end, eval_rows[:MEASURE_EVAL_POS], model.device, desc="  mini-eval")
        eval_full = (time.time() - te) / MEASURE_EVAL_POS * EVAL_SET_SIZE

        banner("MEASURE COMPLETE  --  wall-clock projections")
        print(f"  measured: {per_step:.1f}s/step, one full harness eval ~{fmt_hms(eval_full)}\n")
        print(f"  {'steps':>6}{'train':>10}{'evals':>9}{'total':>10}   under 20h?")
        print("  " + "-" * 48)
        for N in (100, 150, 200, 250, 300, 400):
            train_s = N * per_step
            evals_s = math.ceil(N / EVAL_EVERY) * eval_full
            total_h = (train_s + evals_s) / 3600
            flag = "yes" if total_h < 20 else "NO"
            print(f"  {N:>6}{fmt_hms(train_s):>10}{fmt_hms(evals_s):>9}"
                  f"{fmt_hms(train_s+evals_s):>10}   {flag}")
        print("\n  Pick the largest 'yes' row that fits, set MAX_STEPS, flip MODE='train'.")
        shutdown_engine()
        return

    if MODE != "train":
        raise SystemExit(f"MODE must be smoke|measure|train, got {MODE!r}")

    os.makedirs(RUN_DIR, exist_ok=True)
    banner("BASELINE (stage-2 policy under the Gumbel mechanism, before any training)")
    base_m = evaluate(model, tok, im_end, eval_rows, model.device, desc="baseline")
    print(f"  gumbel-init: legal {base_m['legal']:.1f}%  acc {base_m['accuracy']:.1f}%  "
          f"fmate {base_m['false_mate']}")
    print(f"  reference points: stage-2 raw-hidden-state was 48/10/19, "
          f"grpo_v2 (explicit+RL) 52/10/0, rung-3 exp1 (spoken-only random) 61/9/0")
    history = [{"step": 0, **base_m}]

    banner(f"TRAINING  --  {MAX_STEPS} steps")
    t_train = time.time()
    try:
        for step in range(1, MAX_STEPS + 1):
            fens = [train_fens[i] for i in rng.integers(0, len(train_fens), PROMPTS_PER_STEP)]
            m = grpo_step(model, tok, im_end, optim, sched, fens, model.device)
            peak = torch.cuda.max_memory_allocated()/1e9
            elapsed = time.time() - t_train
            eta = (elapsed / step) * (MAX_STEPS - step)
            print(f"step {step}/{MAX_STEPS} ({100*step/MAX_STEPS:.0f}%, eta {fmt_hms(eta)}) | "
                  f"reward {m['reward_mean']:+.3f} (max {m['reward_max']:+.2f}) | "
                  f"legal {m['legal_frac']:.0%} | fmate {m['false_mate']} | "
                  f"kl {m['kl']:.4f} | pg {m['pg']:+.3f} | lr {m['lr']:.1e} | "
                  f"peak {peak:.1f}GB", flush=True)

            if step % EVAL_EVERY == 0:
                ev = evaluate(model, tok, im_end, eval_rows, model.device)
                ev["step"] = step
                history.append(ev)
                print(f"  >> HARNESS @ {step}: legal {ev['legal']:.1f}% "
                      f"acc {ev['accuracy']:.1f}% fmate {ev['false_mate']}  "
                      f"(gumbel-init {base_m['legal']:.0f}/{base_m['false_mate']}, "
                      f"rung-3 exp1 61/0)", flush=True)
                with open(os.path.join(RUN_DIR, "history.json"), "w") as f:
                    json.dump(history, f, indent=2)

            if step % SAVE_EVERY == 0:
                d = os.path.join(RUN_DIR, "final_model_lora")
                model.set_adapter("policy")
                model.save_pretrained(d, selected_adapters=["policy"])
                print(f"  [ckpt] policy adapter -> {d}", flush=True)
    finally:
        shutdown_engine()

    d = os.path.join(RUN_DIR, "final_model_lora")
    model.set_adapter("policy")
    model.save_pretrained(d, selected_adapters=["policy"])
    banner("RUNG 3 EXPERIMENT 2 (GUMBEL) COMPLETE")
    for h in history:
        print(f"  step {h['step']:>4}: legal {h['legal']:.1f}%  acc {h['accuracy']:.1f}%  "
              f"fmate {h['false_mate']}")
    print(f"\nFinal policy adapter: {d}")
    print("Compare against rung-3 exp1 (61/9/0), stage-2 (48/10/19), grpo_v2 (52/10/0).")


if __name__ == "__main__":
    main()