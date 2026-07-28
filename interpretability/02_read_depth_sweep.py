"""
02_read_depth_sweep.py  --  Map which layers the lens reads chess pieces at.

Calibrates the lens's READ DEPTH on ground-truth chess text: at every position
right before a single-token piece word (King/Queen/Knight/Bishop/Pawn; Rook is
multi-token and excluded), it checks which layers rank that piece high. Runs both
the J-lens and the logit-lens control (use_jacobian=False) so I can see whether
J-lens surfaces piece content earlier/better. Output is a per-layer legibility
table + the best-reading band, which the thought-position experiments will target.

Read-only, forward passes only (no fitting).
"""

import json
import logging
import os
import random

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import jlens
from jlens.lens import JacobianLens

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_MODEL   = "unsloth/qwen3-14b-bnb-4bit"
ADAPTER_PATH = os.path.expanduser("~/g6e_prep/rung3_final_150/final_model_lora_150/policy")
DATASET      = os.path.expanduser("~/g6e_prep/parsed_dataset.jsonl")
LENS_PATH    = os.path.expanduser("~/g6e_prep/rung3_jlens.pt")

N_TEXTS      = 40
SEED         = 3407
MAX_SEQ      = 96          # short texts so positions align cleanly with apply
SKIP_FIRST   = 16          # ignore early attention-sink positions
TOPK         = 10          # "hit" = true piece token in the layer's top-K

# Piece words as they appear in the reasoning (space-prefixed). The script keeps
# only the ones that are genuinely single-token in THIS tokenizer (drops Rook).
PIECE_CANDIDATES = [
    " King", " Queen", " Knight", " Bishop", " Pawn", " Rook",
    " king", " queen", " knight", " bishop", " pawn", " rook",
]

LOG_FILE = os.path.expanduser("~/g6e_prep/read_depth_sweep.log")


def build_logger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    for h in (logging.FileHandler(LOG_FILE), logging.StreamHandler()):
        h.setFormatter(fmt)
        root.addHandler(h)
    return logging.getLogger("sweep")


def reconstruct(row):
    try:
        sections = json.loads(row.get("section_texts") or "[]")
    except (json.JSONDecodeError, TypeError):
        sections = []
    body = (row.get("preamble") or "") + "\n".join(sections) + (row.get("conclusion") or "")
    return f"FEN: {row['fen']}\n{body}\nMove: {row['best_move']}"


def rank_of(logits_row, token_id):
    """0-based rank of token_id in a [vocab] logit row (0 = top)."""
    return int((logits_row > logits_row[token_id]).sum().item())


def main():
    log = build_logger()

    log.info("loading model + adapter + lens")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    peft_model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    peft_model.eval()
    jm = jlens.from_hf(peft_model.base_model.model, tok)
    lens = JacobianLens.load(LENS_PATH)

    # keep only single-token piece ids (drops multi-token ones like Rook)
    piece_names = {}
    for w in PIECE_CANDIDATES:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            piece_names[ids[0]] = w
    log.info("single-token pieces kept: %s", sorted(set(piece_names.values())))
    log.info("dropped (multi-token): %s",
             [w for w in PIECE_CANDIDATES if len(tok.encode(w, add_special_tokens=False)) != 1])
    piece_id_set = set(piece_names)

    with open(DATASET) as f:
        rows = [json.loads(line) for line in f]
    random.seed(SEED)
    texts = [reconstruct(r) for r in random.sample(rows, min(N_TEXTS, len(rows)))]

    layers = sorted(lens.jacobians.keys())
    jl_hit = {L: 0 for L in layers}
    lg_hit = {L: 0 for L in layers}
    total  = {L: 0 for L in layers}

    shown = 0
    for text in tqdm(texts, desc="sweeping", unit="text"):
        ids = jm.encode(text, max_length=MAX_SEQ)[0].tolist()
        positions, targets = [], []
        for i in range(SKIP_FIRST, len(ids) - 1):
            if ids[i + 1] in piece_id_set:
                positions.append(i)
                targets.append(ids[i + 1])
        if not positions:
            continue

        if shown < 2:   # eyeball that position i+1 really is a piece (alignment)
            ex = [(tok.decode([ids[p]]), tok.decode([ids[p + 1]])) for p in positions[:4]]
            tqdm.write(f"  align check (tok@i -> tok@i+1): {ex}")
            shown += 1

        jl_logits, _, _ = lens.apply(jm, text, positions=positions)
        lg_logits, _, _ = lens.apply(jm, text, positions=positions, use_jacobian=False)

        for L in layers:
            jl_L, lg_L = jl_logits[L], lg_logits[L]
            for j, t in enumerate(targets):
                total[L] += 1
                if rank_of(jl_L[j], t) < TOPK:
                    jl_hit[L] += 1
                if rank_of(lg_L[j], t) < TOPK:
                    lg_hit[L] += 1

    n_pos = total[layers[0]] if layers else 0
    log.info("=" * 58)
    log.info("READ-DEPTH SWEEP  hit@%d of true piece, over %d piece positions", TOPK, n_pos)
    log.info("%5s  %11s  %11s", "layer", "Jlens hit%", "logit hit%")
    ranking = []
    for L in layers:
        n = total[L] or 1
        jlp, lgp = 100.0 * jl_hit[L] / n, 100.0 * lg_hit[L] / n
        ranking.append((jlp, L))
        log.info("%5d  %10.1f%%  %10.1f%%", L, jlp, lgp)

    ranking.sort(reverse=True)
    band = sorted(L for _, L in ranking[:5])
    log.info("-" * 58)
    log.info("best-reading band (top 5 layers by Jlens hit%%): %s", band)
    log.info("this band is where the thought-position experiments will read.")


if __name__ == "__main__":
    main()