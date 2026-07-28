"""
train_rung3.py  --  Rung 3: GRPO on the latent (stage-2) policy.

Hand-written GRPO loop. TRL 0.15.2 exposes no generation-override seam, so I
drive the silent-thinking rollout myself.

EXPERIMENT 1 of 2: randomness in the SPOKEN part only. The 4 silent thoughts are
deterministic; spoken tokens are sampled (temperature > 0) so the G completions
per position differ. Gumbel thought-perturbation is experiment 2.

Launchpad:  rung2_checkpoints/stage_2/final  (4 thoughts; frozen harness 48% legal,
            10% acc, 19 false mates).
Reward:     rewards.py chess_reward, UNCHANGED. Gated legality + dense cp.
            False CHECKMATE -> -1.0, the habit we want crushed.
Data:       grpo_training_data_evals.csv ONLY (positions without best_eval_cp get
            a flat 0.1 dead signal, so we train only where the signal is live).
Win:        legal% climbs toward/past grpo_v2's 52 AND false_mate drops toward 0.

Two adapters off one base: 'policy' (trainable, = stage-2) and 'reference'
(frozen, = stage-2). The KL leash anchors to reference. I do NOT use
disable_adapter() for reference: that yields the BASE model, not stage-2.

Model loading is byte-identical to train_grpo.py for harness comparability.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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

MODE = "train"                 # "measure" -> price steps + eval + exit. "train" -> run.

BASE_MODEL     = "unsloth/qwen3-14b-bnb-4bit"
STAGE2_ADAPTER = "rung2_checkpoints/stage_2/final"     # the launchpad
EVALS_CSV      = "grpo_training_data_evals.csv"        # ONLY valid training source
RUN_DIR        = "rung3_checkpoints"

# ---- latent config, LOCKED to the stage-2 launchpad ----
N_THOUGHTS = 4                   # stage 2 = 4 silent thoughts. Do not change.

# ---- GRPO ----
GROUP_SIZE = 4                   # G completions per position (standard group size)
PROMPTS_PER_STEP = 4             # distinct FENs per step; gens/step = 4*4 = 16
KL_COEF = 0.02                   # leash strength (matches Rung 1's beta)
TEMPERATURE = 0.7                # spoken-part sampling. 0.7 (was 0.9): more legal moves
                                 # per group -> more informative groups -> more signal/step.
                                 # Matches Rung 1's working temperature.
TOP_P = 1.0
MAX_NEW = 320                    # covers stage-2 p100 (264 tok) + headroom, caps runaways
ADV_EPS = 1e-4                   # std floor in advantage normalization

# ---- optimization ----
LR = 2e-6                        # RL is touchy; gentler than Rung 1's 5e-6 on the latent policy
MAX_GRAD_NORM = 1.0
MAX_STEPS = 150                  # SET from the measure table. 150 @ 0.7 is the current plan.
WARMUP_STEPS = 10
SEED = 3407

# ---- eval / gate ----
EVAL_EVERY = 50                  # steps between frozen-harness evals
EVAL_SET_SIZE = 100
SAVE_EVERY = 50

# ---- measure mode ----
MEASURE_STEPS = 8                # pricing steps
MEASURE_EVAL_POS = 10            # mini-eval size, extrapolated to EVAL_SET_SIZE

# ---- format (LOCKED, must match SFT + stage-2) ----
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
    """Prefix only, ends at BOT. Identical to eval_latent / the training gate.
    The reward's extract_fen reads 'FEN: {fen}' back out of THIS string."""
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
# DATA
# ───────────────────────────────────────────────────────────────────────────

def load_data():
    """Same seed/split as eval_models.py: first 100 rows are the frozen harness
    (held out), the rest are the Rung 3 training pool. Populates FEN_TO_BEST_EVAL."""
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
    """One base, TWO adapters: 'policy' (trainable, = stage-2) and 'reference'
    (frozen, = stage-2). Both start identical; the leash anchors to reference.
    Loading stage-2 twice is deliberate: disable_adapter() would give BASE, not
    stage-2, and anchor the leash to raw Qwen."""
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
# ROLLOUT  --  prefix -> 4 silent thoughts -> SAMPLED spoken part
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout(model, tok, im_end, fen, device):
    """One completion. Deterministic thoughts, sampled spoken tokens.
    Returns (prompt_text, completion_text, spoken_ids, thought_embeds, thought_mask).
    thought caches let the log-prob passes skip recomputing the silent loop."""
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
        max_length=None, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
        use_cache=True, eos_token_id=im_end, pad_token_id=tok.pad_token_id,
    )
    spoken_ids = gen[0]                                  # inputs_embeds -> new tokens only
    completion = tok.decode(spoken_ids, skip_special_tokens=False)
    return prompt, completion, spoken_ids.detach(), cur_emb.detach(), cur_mask.detach()


def spoken_logprobs(model, adapter, thought_embeds, thought_mask, spoken_ids):
    """Log-prob the given adapter assigns to spoken_ids, conditioned on the fixed
    prefix+thoughts. Policy: grad ON. Reference: called under no_grad. The silent
    loop is not recomputed; we reuse cached thought_embeds."""
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
    logits = out.logits[:, T - 1: T - 1 + S, :]          # position T-1+i predicts spoken[i]
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp[0, torch.arange(S), spoken_ids]          # [S]


# ───────────────────────────────────────────────────────────────────────────
# ONE GRPO STEP
# ───────────────────────────────────────────────────────────────────────────

def grpo_step(model, tok, im_end, optim, sched, fens, device, show_bars=True):
    all_prompts, all_completions, group_of, cached = [], [], [], []

    # 1-2. rollouts
    model.set_adapter("policy")
    model.eval()
    it = range(len(fens) * GROUP_SIZE)
    bar = tqdm(total=len(it), desc="  rollouts", unit="gen", leave=False) if show_bars else None
    for gi, fen in enumerate(fens):
        for _ in range(GROUP_SIZE):
            p, c, sp, temb, tmask = rollout(model, tok, im_end, fen, device)
            all_prompts.append(p); all_completions.append(c)
            group_of.append(gi); cached.append((sp, temb, tmask))
            if bar: bar.update(1)
    if bar: bar.close()

    # 3. reward
    rewards = torch.tensor(chess_reward(all_completions, prompts=all_prompts),
                           dtype=torch.float32)

    # 4. group-relative advantage
    group_of_t = torch.tensor(group_of)
    adv = torch.zeros_like(rewards)
    for gi in range(len(fens)):
        m = group_of_t == gi
        g = rewards[m]
        adv[m] = (g - g.mean()) / (g.std(unbiased=False) + ADV_EPS)

    # 5-7. policy update, one completion at a time (memory-bounded)
    model.train()
    optim.zero_grad(set_to_none=True)
    total_pg = total_kl = 0.0
    n = len(all_completions)
    upd = tqdm(range(n), desc="  policy update", unit="gen", leave=False) if show_bars else range(n)
    for i in upd:
        sp, temb, tmask = cached[i]
        a = adv[i].item()
        pol_lp = spoken_logprobs(model, "policy", temb, tmask, sp)          # grad ON
        with torch.no_grad():
            ref_lp = spoken_logprobs(model, "reference", temb, tmask, sp)
        model.set_adapter("policy")
        log_ratio = ref_lp - pol_lp
        kl = (torch.exp(log_ratio) - 1 - log_ratio).mean()                  # k3, >= 0
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
# HARNESS  --  frozen 100 positions, batch-1 greedy
# ───────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tok, im_end, eval_rows, device, desc="harness"):
    model.set_adapter("policy")
    was = model.training
    model.eval()
    embed = model.get_input_embeddings()
    fmt = legal = correct = false_mate = 0

    for fen, best in tqdm(eval_rows, desc=desc, unit="pos", leave=False):
        prompt = build_latent_prompt(fen)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        cur_emb = embed(ids)
        cur_mask = torch.ones_like(ids)
        for _ in range(N_THOUGHTS):
            out = model(inputs_embeds=cur_emb, attention_mask=cur_mask,
                        output_hidden_states=True, use_cache=False)
            h = out.hidden_states[-1][:, -1:, :].to(cur_emb.dtype)
            cur_emb = torch.cat([cur_emb, h], dim=1)
            cur_mask = torch.cat([cur_mask, torch.ones_like(cur_mask[:, :1])], dim=1)
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

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA.")

    banner(f"train_rung3.py  --  MODE = {MODE}")
    print(f"launchpad : {STAGE2_ADAPTER}")
    print(f"data      : {EVALS_CSV}")
    print(f"G={GROUP_SIZE}  prompts/step={PROMPTS_PER_STEP}  "
          f"gens/step={GROUP_SIZE*PROMPTS_PER_STEP}  thoughts={N_THOUGHTS}")
    print(f"kl_coef={KL_COEF}  temp={TEMPERATURE}  lr={LR}  max_new={MAX_NEW}")

    tok, im_end = build_tokenizer()
    train_fens, eval_rows = load_data()
    model = load_model()

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=LR)
    sched = get_cosine_schedule_with_warmup(optim, WARMUP_STEPS, max(MAX_STEPS, 1))
    rng = np.random.default_rng(SEED)

    # ─────────────────────────────── MEASURE ────────────────────────────────
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
        print("  Watch: does reward come UP vs the 0.9 run (was ~-0.5)? Higher reward")
        print("  = more legal moves per group = more informative GRPO groups.")
        print("  KL is ~0 now (policy==reference at start); it should lift once training moves.")
        shutdown_engine()
        return

    if MODE != "train":
        raise SystemExit(f"MODE must be measure|train, got {MODE!r}")

    # ──────────────────────────────── TRAIN ─────────────────────────────────
    os.makedirs(RUN_DIR, exist_ok=True)
    banner("BASELINE (stage-2 policy, before any GRPO)")
    base_m = evaluate(model, tok, im_end, eval_rows, model.device, desc="baseline")
    print(f"  stage-2: legal {base_m['legal']:.1f}%  acc {base_m['accuracy']:.1f}%  "
          f"fmate {base_m['false_mate']}")
    print("  targets: legal -> past 52, fmate -> toward 0")
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
                      f"(stage-2 {base_m['legal']:.0f}/{base_m['false_mate']}, "
                      f"grpo_v2 52/0)", flush=True)
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
    banner("RUNG 3 COMPLETE")
    for h in history:
        print(f"  step {h['step']:>4}: legal {h['legal']:.1f}%  acc {h['accuracy']:.1f}%  "
              f"fmate {h['false_mate']}")
    print(f"\nFinal policy adapter: {d}")
    print("Compare against grpo_v2 (52/10/0) and stage-2 (48/10/19).")


if __name__ == "__main__":
    main()