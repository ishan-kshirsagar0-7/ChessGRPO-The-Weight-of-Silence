"""
04_thought_diag.py  --  Is there board-specific signal in the thoughts at all?

The mid-layer read came back as position-invariant junk. Before assuming the
thoughts are empty, run two checks on the RAW thought vectors (the last-layer
hidden states Coconut feeds back):

  CHECK 1 (direct read): unembed each thought vector at its native last layer
    ("what was the model about to say"). May be blank even if signal exists,
    because the thoughts were trained to be non-verbal.
  CHECK 2 (decider, needs no readout): cosine similarity of the thought vectors
    ACROSS boards. ~1.0 => the model collapsed thoughts to a near-constant (they
    carry no board info). Low => board-specific signal is present and we simply
    read it in the wrong place.
"""

import logging
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL   = "unsloth/qwen3-14b-bnb-4bit"
ADAPTER_PATH = os.path.expanduser("~/g6e_prep/rung3_final_150/final_model_lora_150/policy")
DATASET      = os.path.expanduser("~/g6e_prep/grpo_training_data.csv")

N_THOUGHTS   = 4
N_POSITIONS  = 4
SEED         = 3407
TOPK         = 8

BOT_TOKEN = "<|box_start|>"
SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""


def build_latent_prompt(fen):
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nFEN: {fen}<|im_end|>\n"
            f"<|im_start|>assistant\n<thinking>\n{BOT_TOKEN}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("diag")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    peft_model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    peft_model.eval()
    inner = peft_model.base_model.model
    device = next(inner.parameters()).device
    _norm, _head = inner.model.norm, inner.get_output_embeddings()

    def unembed(h):                       # h [1,d] -> logits [vocab]
        return _head(_norm(h.to(_norm.weight.dtype)))[0]

    df = pd.read_csv(DATASET).sample(frac=1, random_state=SEED).reset_index(drop=True)
    fens = [df.iloc[i]["FEN"] for i in range(N_POSITIONS)]
    embed = inner.get_input_embeddings()

    thoughts = []                          # thoughts[pos] = [h_0..h_{n-1}], each [d]
    for fen in fens:
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
            vecs.append(h[0, 0].clone())
            cur = torch.cat([cur, h], dim=1)
            mask = torch.cat([mask, torch.ones(1, 1, device=device)], dim=1)
        thoughts.append(vecs)

    log.info("=" * 62)
    log.info("CHECK 1: direct read of each thought vector (native last layer)")
    log.info("=" * 62)
    for pos, fen in enumerate(fens):
        log.info("\nposition %d  FEN: %s", pos, fen)
        for t in range(N_THOUGHTS):
            words = [tok.decode([i]) for i in unembed(thoughts[pos][t].unsqueeze(0)).topk(TOPK).indices]
            log.info("  thought %d: %s", t, words)

    log.info("\n" + "=" * 62)
    log.info("CHECK 2 (decider): cosine of thought vectors ACROSS boards")
    log.info("  ~1.0 => near-constant thoughts (no board info); low => board-specific")
    log.info("=" * 62)
    for t in range(N_THOUGHTS):
        sims = [F.cosine_similarity(thoughts[i][t].float(), thoughts[j][t].float(), dim=0).item()
                for i in range(len(fens)) for j in range(i + 1, len(fens))]
        log.info("  thought %d: mean cross-board cosine = %.3f   pairs %s",
                 t, sum(sims) / len(sims), [f"{s:.2f}" for s in sims])

    log.info("\n  within board 0, thought-to-thought cosine (are the 4 distinct?):")
    for a in range(N_THOUGHTS):
        row = [f"{F.cosine_similarity(thoughts[0][a].float(), thoughts[0][b].float(), dim=0).item():.2f}"
               for b in range(N_THOUGHTS)]
        log.info("    t%d: %s", a, row)

    log.info("\ndiag complete.")


if __name__ == "__main__":
    main()