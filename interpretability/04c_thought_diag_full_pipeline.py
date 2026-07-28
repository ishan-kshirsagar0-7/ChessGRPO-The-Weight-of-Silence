"""
04c_thought_diag_full_pipeline.py -- Where along the pipeline does board-invariance
appear: architecture, chess training, the latent curriculum, or RL?

Extends 04b's base-vs-rung3 control to the full four-point training ladder:
  1. base_no_adapter   -- raw model, no training at all (off-distribution probe)
  2. sft                -- chess-trained, explicit reasoning, no latent mechanism
                           ever trained (off-distribution probe, same caveat as base)
  3. stage2             -- latent curriculum, imitation only, no RL
                           (FIRST checkpoint actually trained to do this operation)
  4. rung3              -- latent curriculum + RL (reproduces 04b's number as a
                           built-in check)

Same board source/seed/mechanism as 04_thought_diag.py and 04b throughout, so
n=100 numbers are directly comparable across all four checkpoints and across runs.
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

BASE_MODEL = "unsloth/qwen3-14b-bnb-4bit"
DATASET    = os.path.expanduser("~/g6e_prep/grpo_training_data.csv")
OUT_DIR    = os.path.expanduser("~/g6e_prep/results_full_pipeline_control")

# label -> adapter path, or None for no adapter
CHECKPOINTS = [
    ("base_no_adapter", None),
    ("sft",              os.path.expanduser("~/qwen3_14b_sft_checkpoints/final_model_lora")),
    ("stage2",           os.path.expanduser("~/g6e_prep/rung2_checkpoints/stage_2/final")),
    ("rung3_adapter",    os.path.expanduser("~/g6e_prep/rung3_final_150/final_model_lora_150/policy")),
]

N_THOUGHTS  = 4
N_SMALL     = 4       # exact replication size of the original 04_thought_diag.py
N_EXTENDED  = 100
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


def load_model(adapter_path):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    if adapter_path is not None:
        peft_model = PeftModel.from_pretrained(base, adapter_path)
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
            vecs.append(h[0, 0].clone().cpu())
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(OUT_DIR, "full_pipeline_control.log")),
                  logging.StreamHandler()])
    log = logging.getLogger("diag")
    t_start = time.time()

    for label, path in CHECKPOINTS:
        if path is not None and not os.path.exists(path):
            raise FileNotFoundError(f"Adapter path for '{label}' does not exist: {path}\n"
                                     f"Fix CHECKPOINTS before running -- verify, don't assume.")

    df = pd.read_csv(DATASET).sample(frac=1, random_state=SEED).reset_index(drop=True)
    fens_full = [df.iloc[i]["FEN"] for i in range(N_EXTENDED)]

    log.info("=" * 70)
    log.info("FULL PIPELINE CONTROL: base -> sft -> stage2 -> rung3")
    log.info("N_SMALL=%d (exact replication), N_EXTENDED=%d", N_SMALL, N_EXTENDED)
    log.info("=" * 70)

    all_rows = []
    raw_vectors = {}

    for i, (label, path) in enumerate(CHECKPOINTS, 1):
        log.info("\n" + "-" * 70)
        log.info("[%d/%d] Loading model: %s (adapter=%s)", i, len(CHECKPOINTS), label, path or "none")
        t0 = time.time()
        wrapper, inner, tok, device, unembed_fn = load_model(path)
        log.info("  loaded in %.1fs", time.time() - t0)

        t0 = time.time()
        thoughts = capture_thoughts(inner, tok, device, fens_full, desc=f"{label} capture")
        log.info("  captured %d boards x %d thoughts in %.1fs", N_EXTENDED, N_THOUGHTS, time.time() - t0)

        raw_vectors[label] = thoughts

        cos_small = cross_board_cosine(thoughts, N_SMALL)
        cos_full = cross_board_cosine(thoughts, N_EXTENDED)
        log.info("CHECK 2 (%s): cross-board cosine similarity per thought slot", label)
        for t in range(N_THOUGHTS):
            log.info("  thought %d: n=%-3d -> %.4f   |   n=%-3d -> %.4f",
                      t, N_SMALL, cos_small[t], N_EXTENDED, cos_full[t])
            all_rows.append({"model": label, "n_boards": N_SMALL, "thought_slot": t, "mean_cosine": cos_small[t]})
            all_rows.append({"model": label, "n_boards": N_EXTENDED, "thought_slot": t, "mean_cosine": cos_full[t]})

        mat = within_board_matrix(thoughts, board_idx=0)
        log.info("  within board 0, thought-to-thought cosine (%s):", label)
        for a, row in enumerate(mat):
            log.info("    t%d: %s", a, [f"{v:.2f}" for v in row])

        del wrapper, inner, unembed_fn
        gc.collect()
        torch.cuda.empty_cache()
        log.info("  freed GPU memory")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(os.path.join(OUT_DIR, "full_pipeline_summary.csv"), index=False)
    torch.save(raw_vectors, os.path.join(OUT_DIR, "full_pipeline_raw_thoughts.pt"))

    rung3_small_mean = sum(r["mean_cosine"] for r in all_rows
                            if r["model"] == "rung3_adapter" and r["n_boards"] == N_SMALL) / N_THOUGHTS
    log.info("\n" + "=" * 70)
    log.info("REPLICATION CHECK: rung3_adapter at n=%d should be ~0.99", N_SMALL)
    log.info("  measured: %.4f  -->  %s", rung3_small_mean,
              "PASSED" if rung3_small_mean > 0.9 else "MISMATCH -- investigate")
    log.info("=" * 70)

    log.info("\nFULL PIPELINE (mean cosine across all 4 thought slots, n=%d):", N_EXTENDED)
    for label, _ in CHECKPOINTS:
        mean = sum(r["mean_cosine"] for r in all_rows if r["model"] == label and r["n_boards"] == N_EXTENDED) / N_THOUGHTS
        log.info("  %-18s %.4f", label, mean)

    log.info("\ntotal runtime: %.1f min", (time.time() - t_start) / 60)
    log.info("outputs saved to: %s", OUT_DIR)
    log.info("diag complete.")


if __name__ == "__main__":
    main()