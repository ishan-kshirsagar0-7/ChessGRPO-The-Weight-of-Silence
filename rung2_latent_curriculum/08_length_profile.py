"""
08_length_profile.py

Suffix and prefix token lengths per curriculum stage, from the real 14B
tokenizer on the real data. CPU only. No model weights.

Sizes two config values that are currently guesses:
  MAX_TOTAL_TOKENS     -> where to clip the long tail
  GATE_MAX_NEW_TOKENS  -> how much room the gate needs to finish a generation
"""

import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

BASE_MODEL = "unsloth/qwen3-14b-bnb-4bit"
PARSED_JSONL = "parsed_dataset.jsonl"
STAGES = [1, 2, 3, 4, 5]
THOUGHTS_PER_CHUNK = 2
BATCH = 4
PCTS = [50, 90, 95, 99, 99.9, 100]

BOT_TOKEN = "<|box_start|>"
EOT_TOKEN = "<|box_end|>"

SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""


def get_chunks(row):
    s = row["section_texts"]
    return [row["preamble"] + s[0], s[1], s[2], s[3], s[4] + row["conclusion"]]


def build_example(row, stage):
    chunks = get_chunks(row)
    spoken = "".join(chunks[stage:])
    shared_prefix = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nFEN: {row['fen']}<|im_end|>\n"
        f"<|im_start|>assistant\n<thinking>\n"
    )
    closing = f"\n</thinking>\n<output>\n{row['best_move']}\n</output><|im_end|>"
    suffix = f"{EOT_TOKEN}\n{spoken}{closing}" if spoken else f"{EOT_TOKEN}{closing}"
    return {"prefix": shared_prefix + BOT_TOKEN,
            "n_thoughts": THOUGHTS_PER_CHUNK * stage,
            "suffix": suffix}


def report(name, arr):
    vals = " ".join(f"{np.percentile(arr, p):>8.0f}" for p in PCTS)
    print(f"  {name:<10}{vals}")


def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    with open(PARSED_JSONL, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print(f"{len(rows)} rows\n")

    n = lambda s: len(tok(s, add_special_tokens=False).input_ids)

    prefix_lens = np.array([n(build_example(r, 1)["prefix"])
                            for r in tqdm(rows, desc="prefix", unit="row")])

    header = "  " + f"{'':<10}" + " ".join(f"{('p'+str(p)):>8}" for p in PCTS)

    print("\n" + "=" * 78)
    print("PREFIX TOKENS (identical across stages)")
    print("=" * 78)
    print(header)
    report("prefix", prefix_lens)

    print("\n" + "=" * 78)
    print("SUFFIX TOKENS PER STAGE  (== what the gate must generate)")
    print("=" * 78)
    print(header)
    suffix_by_stage = {}
    for stage in STAGES:
        lens = np.array([n(build_example(r, stage)["suffix"])
                         for r in tqdm(rows, desc=f"suffix s{stage}",
                                       unit="row", leave=False)])
        suffix_by_stage[stage] = lens
        report(f"stage {stage}", lens)

    print("\n" + "=" * 78)
    print(f"TOTAL SEQUENCE (prefix + thoughts + suffix)")
    print("=" * 78)
    print(header)
    for stage in STAGES:
        total = prefix_lens + THOUGHTS_PER_CHUNK * stage + suffix_by_stage[stage]
        report(f"stage {stage}", total)

    print("\n" + "=" * 78)
    print(f"WORST-CASE PADDED BATCH OF {BATCH}")
    print("  (batch cost is BATCH x longest-in-batch, not BATCH x mean)")
    print("=" * 78)
    for stage in STAGES:
        total = prefix_lens + THOUGHTS_PER_CHUNK * stage + suffix_by_stage[stage]
        mean_batch = BATCH * total.mean()
        p999_batch = BATCH * np.percentile(total, 99.9)
        max_batch = BATCH * total.max()
        print(f"  stage {stage}:  mean {mean_batch:>7.0f} tok  "
              f"p99.9 {p999_batch:>7.0f} tok  max {max_batch:>7.0f} tok  "
              f"(ratio max/mean {max_batch/mean_batch:.2f}x)")

    print("\n" + "=" * 78)
    print("RECOMMENDATIONS")
    print("=" * 78)
    s1 = suffix_by_stage[1]
    gate_need = int(np.percentile(s1, 99.9) * 1.15)
    print(f"  GATE_MAX_NEW_TOKENS  >= {gate_need}   "
          f"(stage-1 suffix p99.9 x 1.15; current value 512)")
    if gate_need > 512:
        print("  ^^ CURRENT VALUE OF 512 WOULD TRUNCATE THE STAGE-1 GATE. Fix before training.")

    all_tot = prefix_lens + THOUGHTS_PER_CHUNK * 1 + s1
    clip = int(np.percentile(all_tot, 99))
    drop = int((all_tot > clip).sum())
    print(f"  MAX_TOTAL_TOKENS     ~= {clip}   "
          f"(p99 of stage-1 totals; would drop {drop} rows, {100*drop/len(rows):.1f}%)")
    print(f"\n  max/mean ratio above tells you the OOM headroom you actually have.")
    print("  Paste this whole output.")


if __name__ == "__main__":
    main()