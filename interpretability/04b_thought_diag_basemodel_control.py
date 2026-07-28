"""
04b_thought_diag_basemodel_control.py -- Does board-invariance survive without LoRA?

Direct control for 04_thought_diag.py's headline finding (cross-board cosine ~0.99
on Rung-3's raw thought vectors, measured on n=4 boards). Runs the identical
capture mechanism on the RAW base model (no adapter at all) to test whether the
collapse is something the LoRA training induced, or something the base
architecture already does on its own before any training touches it.

Runs BOTH models in one script, sequentially, so nothing needs reloading twice:
  1. base model, no adapter
  2. rung-3 adapter (reproduces 04_thought_diag.py's original number as a
     built-in check that this mirror is faithful before trusting the base-model
     comparison against it)

Boards: same data source and seed as the original script (grpo_training_data.csv,
seed=3407), but N_EXTENDED=100 instead of 4 -- the first 4 of those 100 are
byte-identical to the original run's boards, so n=4 results here should
reproduce ~0.99 for rung3_adapter. n=100 gives a far more robust estimate for
both models (435 pairs per slot instead of 6).
"""

import logging
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import time

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL   = "unsloth/qwen3-14b-bnb-4bit"
ADAPTER_PATH = os.path.expanduser("~/g6e_prep/rung3_final_150/final_model_lora_150/policy")
DATASET      = os.path.expanduser("~/g6e_prep/grpo_training_data.csv")
OUT_DIR      = os.path.expanduser("~/g6e_prep/results_basemodel_control")

N_THOUGHTS  = 4
N_SMALL     = 4      # exact replication of 04_thought_diag.py
N_EXTENDED  = 100     # same pool, larger sample, more robust estimate
SEED        = 3407
TOPK        = 8

BOT_TOKEN = "<|box_start|>"
SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""


def build_latent_prompt(fen):
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nFEN: {fen}<|im_end|>\n"
            f"<|im_start|>assistant\n<thinking>\n{BOT_TOKEN}")


def load_model(use_adapter):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    if use_adapter:
        peft_model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        peft_model.eval()
        inner = peft_model.base_model.model
        wrapper = peft_model
    else:
        base.eval()
        inner = base
        wrapper = base

    device = next(inner.parameters()).device
    _norm, _head = inner.model.norm, inner.get_output_embeddings()

    def unembed(h):
        return _head(_norm(h.to(_norm.weight.dtype)))[0]

    return wrapper, inner, tok, device, unembed


def capture_thoughts(inner, tok, device, fens, desc):
    embed = inner.get_input_embeddings()
    thoughts = []
    for fen in tqdm(fens, desc=desc, unit="board"):
        pids = tok(build_latent_prompt(fen), return_tensors="pt",
                    add_special_tokens=False).input_ids.to(device)
        cur = embed(pids)
        mask = torch.ones(1, pids.shape[1], device=device)
        vecs = []
        for _ in range(N_THOUGHTS):
            with torch.no_grad():
                o = inner(inputs_embeds=cur, attention_mask=mask,
                          output_hidden_states=True, use_cache=False)
            h = o.hidden_states[-1][:, -1:, :].to(cur.dtype)
            vecs.append(h[0, 0].clone().cpu())   # off GPU immediately, keep memory clean
            cur = torch.cat([cur, h], dim=1)
            mask = torch.cat([mask, torch.ones(1, 1, device=device)], dim=1)
        thoughts.append(vecs)
    return thoughts


def cross_board_cosine(thoughts, n):
    result = {}
    for t in range(N_THOUGHTS):
        sims = [F.cosine_similarity(thoughts[i][t].float(), thoughts[j][t].float(), dim=0).item()
                for i in range(n) for j in range(i + 1, n)]
        result[t] = sum(sims) / len(sims)
    return result


def within_board_matrix(thoughts, board_idx=0):
    mat = []
    for a in range(N_THOUGHTS):
        row = [F.cosine_similarity(thoughts[board_idx][a].float(), thoughts[board_idx][b].float(), dim=0).item()
               for b in range(N_THOUGHTS)]
        mat.append(row)
    return mat


def direct_read(unembed_fn, tok, thoughts, fens, n, device, topk=TOPK):
    lines = []
    for pos in range(n):
        lines.append(f"  position {pos}  FEN: {fens[pos]}")
        for t in range(N_THOUGHTS):
            vec = thoughts[pos][t].unsqueeze(0).to(device)
            words = [tok.decode([i]) for i in unembed_fn(vec).topk(topk).indices]
            lines.append(f"    thought {t}: {words}")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(OUT_DIR, "basemodel_control.log")),
                  logging.StreamHandler()])
    log = logging.getLogger("diag")
    t_start = time.time()

    df = pd.read_csv(DATASET).sample(frac=1, random_state=SEED).reset_index(drop=True)
    fens_full = [df.iloc[i]["FEN"] for i in range(N_EXTENDED)]

    log.info("=" * 70)
    log.info("BASE-MODEL J-LENS CONTROL: does board-invariance survive without LoRA?")
    log.info("N_SMALL=%d (exact replication of 04_thought_diag.py), N_EXTENDED=%d", N_SMALL, N_EXTENDED)
    log.info("=" * 70)

    all_rows = []
    raw_vectors = {}

    for label, use_adapter in [("base_no_adapter", False), ("rung3_adapter", True)]:
        log.info("\n" + "-" * 70)
        log.info("Loading model: %s", label)
        t0 = time.time()
        wrapper, inner, tok, device, unembed_fn = load_model(use_adapter)
        log.info("  loaded in %.1fs", time.time() - t0)

        t0 = time.time()
        thoughts = capture_thoughts(inner, tok, device, fens_full, desc=f"{label} capture")
        log.info("  captured %d boards x %d thoughts in %.1fs", N_EXTENDED, N_THOUGHTS, time.time() - t0)

        raw_vectors[label] = thoughts

        log.info("\nCHECK 1 (%s): direct read of thought vectors, top-%d tokens, first %d boards",
                  label, TOPK, N_SMALL)
        log.info(direct_read(unembed_fn, tok, thoughts, fens_full, N_SMALL, device))

        cos_small = cross_board_cosine(thoughts, N_SMALL)
        cos_full = cross_board_cosine(thoughts, N_EXTENDED)
        log.info("\nCHECK 2 (%s): cross-board cosine similarity per thought slot", label)
        for t in range(N_THOUGHTS):
            log.info("  thought %d: n=%-3d -> %.4f   |   n=%-3d -> %.4f",
                      t, N_SMALL, cos_small[t], N_EXTENDED, cos_full[t])
            all_rows.append({"model": label, "n_boards": N_SMALL, "thought_slot": t, "mean_cosine": cos_small[t]})
            all_rows.append({"model": label, "n_boards": N_EXTENDED, "thought_slot": t, "mean_cosine": cos_full[t]})

        mat = within_board_matrix(thoughts, board_idx=0)
        log.info("\n  within board 0, thought-to-thought cosine (%s):", label)
        for a, row in enumerate(mat):
            log.info("    t%d: %s", a, [f"{v:.2f}" for v in row])

        del wrapper, inner, unembed_fn
        gc.collect()
        torch.cuda.empty_cache()
        log.info("  freed GPU memory")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(os.path.join(OUT_DIR, "basemodel_control_summary.csv"), index=False)
    torch.save(raw_vectors, os.path.join(OUT_DIR, "basemodel_control_raw_thoughts.pt"))

    rung3_small_mean = sum(r["mean_cosine"] for r in all_rows
                            if r["model"] == "rung3_adapter" and r["n_boards"] == N_SMALL) / N_THOUGHTS
    log.info("\n" + "=" * 70)
    log.info("REPLICATION CHECK: rung3_adapter at n=%d should be ~0.99 (the paper's reported figure)", N_SMALL)
    log.info("  measured: %.4f  -->  %s", rung3_small_mean,
              "PASSED" if rung3_small_mean > 0.9 else "MISMATCH -- investigate before trusting anything below")
    log.info("=" * 70)

    log.info("\nFINAL COMPARISON (mean cosine across all 4 thought slots):")
    for n in (N_SMALL, N_EXTENDED):
        base_mean = sum(r["mean_cosine"] for r in all_rows if r["model"] == "base_no_adapter" and r["n_boards"] == n) / N_THOUGHTS
        rung3_mean = sum(r["mean_cosine"] for r in all_rows if r["model"] == "rung3_adapter" and r["n_boards"] == n) / N_THOUGHTS
        log.info("  n=%-4d   base (no adapter): %.4f   |   rung3 (adapter): %.4f   |   delta: %+.4f",
                  n, base_mean, rung3_mean, rung3_mean - base_mean)

    log.info("\ntotal runtime: %.1f min", (time.time() - t_start) / 60)
    log.info("outputs saved to: %s", OUT_DIR)
    log.info("diag complete.")


if __name__ == "__main__":
    main()

