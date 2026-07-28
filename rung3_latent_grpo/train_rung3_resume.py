"""
train_rung3_resume.py  --  Resume Rung 3 from the step-100 checkpoint, +50 steps.

The original run died ~step 145 (broken pipe) with only the step-100 checkpoint
saved (SAVE_EVERY=50). This continues from that checkpoint for 50 more steps to
reach the intended 150.

Differences from train_rung3.py:
  - policy loads from the SAVED step-100 adapter, NOT stage-2.
  - reference still loads stage-2 (the leash anchor is unchanged).
  - fresh optimizer (old Adam state died with the process; negligible at 2e-6).
  - CONSTANT lr 2e-6 (no fresh cosine, which would spike the rate).
  - history is appended to a NEW file so the original step 0/50/100 is preserved.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
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

BASE_MODEL     = "unsloth/qwen3-14b-bnb-4bit"
RESUME_ADAPTER = "rung3_checkpoints/final_model_lora/policy"   # the SAVED step-100 policy
REF_ADAPTER    = "rung2_checkpoints/stage_2/final"             # leash anchor, unchanged
EVALS_CSV      = "grpo_training_data_evals.csv"
RUN_DIR        = "rung3_checkpoints"
HISTORY_OUT    = os.path.join(RUN_DIR, "history_resume.json")   # new file, preserves original

# ---- resume window ----
RESUME_FROM_STEP = 100           # for display/logging only
EXTRA_STEPS      = 50            # 100 -> 150

# ---- latent config, LOCKED ----
N_THOUGHTS = 4

# ---- GRPO (identical to the run that produced 46->54->56) ----
GROUP_SIZE = 4
PROMPTS_PER_STEP = 4
KL_COEF = 0.02
TEMPERATURE = 0.7
TOP_P = 1.0
MAX_NEW = 320
ADV_EPS = 1e-4

# ---- optimization ----
LR = 2e-6                        # CONSTANT. No scheduler on a 50-step continuation.
MAX_GRAD_NORM = 1.0
SEED = 3407 + 100                # offset so we don't replay the exact same FEN draws as steps 1-100

# ---- eval ----
EVAL_EVERY = 25                  # eval at +25 and +50 (i.e. step 125 and 150)
EVAL_SET_SIZE = 100
SAVE_EVERY = 25

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


def load_data():
    df = pd.read_csv(EVALS_CSV)
    if "best_eval_cp" not in df.columns:
        raise SystemExit(f"{EVALS_CSV} has no best_eval_cp column.")
    df = df.dropna(subset=["best_eval_cp"]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=3407).reset_index(drop=True)   # SAME split as original
    for _, r in df.iterrows():
        FEN_TO_BEST_EVAL[r["FEN"]] = int(r["best_eval_cp"])
    print(f"  {len(FEN_TO_BEST_EVAL)} FEN -> best_eval_cp mappings loaded.")
    eval_df = df.iloc[:EVAL_SET_SIZE].reset_index(drop=True)
    train_df = df.iloc[EVAL_SET_SIZE:].reset_index(drop=True)
    eval_rows = [(r["FEN"], str(r["Best Move"]).strip()) for _, r in eval_df.iterrows()]
    train_fens = [r["FEN"] for _, r in train_df.iterrows()]
    print(f"  train positions: {len(train_fens)} | held-out harness: {len(eval_rows)}")
    return train_fens, eval_rows


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
    """policy = SAVED step-100 adapter (trainable). reference = stage-2 (frozen)."""
    print(f"\nLoading base {BASE_MODEL} in 4-bit...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map="auto")
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    print(f"Loading step-100 policy (TRAINABLE) from {RESUME_ADAPTER}...", flush=True)
    model = PeftModel.from_pretrained(base, RESUME_ADAPTER,
                                      adapter_name="policy", is_trainable=True)
    print(f"Loading stage-2 as frozen 'reference' from {REF_ADAPTER}...", flush=True)
    model.load_adapter(REF_ADAPTER, adapter_name="reference")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False

    model.set_adapter("policy")
    print("\nPolicy lora_B norms (should reflect trained step-100 weights):")
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


@torch.no_grad()
def rollout(model, tok, im_end, fen, device):
    prompt = build_latent_prompt(fen)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    mask = torch.ones_like(ids)
    embed = model.get_input_embeddings()
    cur_emb = embed(ids); cur_mask = mask
    for _ in range(N_THOUGHTS):
        out = model(inputs_embeds=cur_emb, attention_mask=cur_mask,
                    output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][:, -1:, :].to(cur_emb.dtype)
        cur_emb = torch.cat([cur_emb, h], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(cur_mask[:, :1])], dim=1)
    gen = model.generate(
        inputs_embeds=cur_emb, attention_mask=cur_mask, max_new_tokens=MAX_NEW,
        max_length=None, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
        use_cache=True, eos_token_id=im_end, pad_token_id=tok.pad_token_id)
    spoken_ids = gen[0]
    completion = tok.decode(spoken_ids, skip_special_tokens=False)
    return prompt, completion, spoken_ids.detach(), cur_emb.detach(), cur_mask.detach()


def spoken_logprobs(model, adapter, thought_embeds, thought_mask, spoken_ids):
    model.set_adapter(adapter)
    embed = model.get_input_embeddings()
    spoken_ids = spoken_ids.to(thought_embeds.device)
    spk_emb = embed(spoken_ids.unsqueeze(0))
    full_emb = torch.cat([thought_embeds, spk_emb], dim=1)
    full_mask = torch.cat(
        [thought_mask, torch.ones((1, spk_emb.shape[1]), device=thought_mask.device,
                                  dtype=thought_mask.dtype)], dim=1)
    out = model(inputs_embeds=full_emb, attention_mask=full_mask, use_cache=False)
    T = thought_embeds.shape[1]; S = spoken_ids.shape[0]
    logits = out.logits[:, T - 1: T - 1 + S, :]
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp[0, torch.arange(S), spoken_ids]


def grpo_step(model, tok, im_end, optim, fens, device, show_bars=True):
    all_prompts, all_completions, group_of, cached = [], [], [], []
    model.set_adapter("policy"); model.eval()
    bar = tqdm(total=len(fens)*GROUP_SIZE, desc="  rollouts", unit="gen",
               leave=False) if show_bars else None
    for gi, fen in enumerate(fens):
        for _ in range(GROUP_SIZE):
            p, c, sp, temb, tmask = rollout(model, tok, im_end, fen, device)
            all_prompts.append(p); all_completions.append(c)
            group_of.append(gi); cached.append((sp, temb, tmask))
            if bar: bar.update(1)
    if bar: bar.close()

    rewards = torch.tensor(chess_reward(all_completions, prompts=all_prompts),
                           dtype=torch.float32)
    group_of_t = torch.tensor(group_of)
    adv = torch.zeros_like(rewards)
    for gi in range(len(fens)):
        mm = group_of_t == gi
        g = rewards[mm]
        adv[mm] = (g - g.mean()) / (g.std(unbiased=False) + ADV_EPS)

    model.train(); optim.zero_grad(set_to_none=True)
    total_pg = total_kl = 0.0; n = len(all_completions)
    upd = tqdm(range(n), desc="  policy update", unit="gen", leave=False) if show_bars else range(n)
    for i in upd:
        sp, temb, tmask = cached[i]; a = adv[i].item()
        pol_lp = spoken_logprobs(model, "policy", temb, tmask, sp)
        with torch.no_grad():
            ref_lp = spoken_logprobs(model, "reference", temb, tmask, sp)
        model.set_adapter("policy")
        log_ratio = ref_lp - pol_lp
        kl = (torch.exp(log_ratio) - 1 - log_ratio).mean()
        pg = -(a * pol_lp.mean())
        ((pg + KL_COEF * kl) / n).backward()
        total_pg += pg.item(); total_kl += kl.item()

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)
    optim.step()

    false_mate = sum(
        1 for c, p in zip(all_completions, all_prompts)
        if extract_move(c) == "CHECKMATE" and not is_position_checkmate(_fen_from_prompt(p)))
    return {
        "reward_mean": float(rewards.mean()), "reward_max": float(rewards.max()),
        "legal_frac": float((rewards > 0).float().mean()), "false_mate": false_mate,
        "pg": total_pg / n, "kl": total_kl / n,
    }


@torch.no_grad()
def evaluate(model, tok, im_end, eval_rows, device, desc="harness"):
    model.set_adapter("policy"); was = model.training; model.eval()
    embed = model.get_input_embeddings()
    fmt = legal = correct = false_mate = 0
    for fen, best in tqdm(eval_rows, desc=desc, unit="pos", leave=False):
        prompt = build_latent_prompt(fen)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        cur_emb = embed(ids); cur_mask = torch.ones_like(ids)
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
    if was: model.train()
    n = len(eval_rows)
    return {"format": 100*fmt/n, "legal": 100*legal/n,
            "accuracy": 100*correct/n, "false_mate": false_mate}


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA.")

    banner(f"train_rung3_resume.py  --  step {RESUME_FROM_STEP} -> "
           f"{RESUME_FROM_STEP + EXTRA_STEPS}")
    print(f"policy    : {RESUME_ADAPTER}")
    print(f"reference : {REF_ADAPTER}")
    print(f"lr={LR} (constant)  temp={TEMPERATURE}  kl_coef={KL_COEF}")

    tok, im_end = build_tokenizer()
    train_fens, eval_rows = load_data()
    model = load_model()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=LR)
    rng = np.random.default_rng(SEED)

    os.makedirs(RUN_DIR, exist_ok=True)
    banner("VERIFY: harness on the loaded step-100 checkpoint")
    print("Should reproduce ~56% legal / ~2 fmate. If it doesn't, the wrong")
    print("adapter loaded, STOP and check RESUME_ADAPTER.", flush=True)
    base_m = evaluate(model, tok, im_end, eval_rows, model.device, desc="verify")
    print(f"  loaded checkpoint: legal {base_m['legal']:.1f}%  "
          f"acc {base_m['accuracy']:.1f}%  fmate {base_m['false_mate']}")
    history = [{"step": RESUME_FROM_STEP, **base_m}]

    banner(f"TRAINING  --  {EXTRA_STEPS} more steps ({RESUME_FROM_STEP} -> "
           f"{RESUME_FROM_STEP + EXTRA_STEPS})")
    t0 = time.time()
    try:
        for i in range(1, EXTRA_STEPS + 1):
            step = RESUME_FROM_STEP + i
            fens = [train_fens[j] for j in rng.integers(0, len(train_fens), PROMPTS_PER_STEP)]
            m = grpo_step(model, tok, im_end, optim, fens, model.device)
            peak = torch.cuda.max_memory_allocated()/1e9
            eta = (time.time()-t0)/i * (EXTRA_STEPS - i)
            print(f"step {step}/{RESUME_FROM_STEP+EXTRA_STEPS} "
                  f"({100*i/EXTRA_STEPS:.0f}% of resume, eta {fmt_hms(eta)}) | "
                  f"reward {m['reward_mean']:+.3f} (max {m['reward_max']:+.2f}) | "
                  f"legal {m['legal_frac']:.0%} | fmate {m['false_mate']} | "
                  f"kl {m['kl']:.4f} | pg {m['pg']:+.3f} | peak {peak:.1f}GB", flush=True)

            if i % EVAL_EVERY == 0:
                ev = evaluate(model, tok, im_end, eval_rows, model.device)
                ev["step"] = step
                history.append(ev)
                print(f"  >> HARNESS @ {step}: legal {ev['legal']:.1f}% "
                      f"acc {ev['accuracy']:.1f}% fmate {ev['false_mate']}  "
                      f"(step-100 was {base_m['legal']:.0f}/{base_m['false_mate']}, "
                      f"grpo_v2 52/0)", flush=True)
                with open(HISTORY_OUT, "w") as f:
                    json.dump(history, f, indent=2)

            if i % SAVE_EVERY == 0:
                d = os.path.join(RUN_DIR, "final_model_lora_150")
                model.set_adapter("policy")
                model.save_pretrained(d, selected_adapters=["policy"])
                print(f"  [ckpt] policy -> {d}", flush=True)
    finally:
        shutdown_engine()

    d = os.path.join(RUN_DIR, "final_model_lora_150")
    model.set_adapter("policy")
    model.save_pretrained(d, selected_adapters=["policy"])
    banner("RESUME COMPLETE  --  step 150 reached")
    for h in history:
        print(f"  step {h['step']:>4}: legal {h['legal']:.1f}%  "
              f"acc {h['accuracy']:.1f}%  fmate {h['false_mate']}")
    print(f"\nFinal (step-150) policy adapter: {d}")
    print("Compare vs grpo_v2 (52/10/0), stage-2 (48/10/19), rung3@100 (~56/9/2).")


if __name__ == "__main__":
    main()
