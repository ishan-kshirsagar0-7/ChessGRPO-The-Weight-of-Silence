# 01_parse_dataset.py
# D3 Call 1, revised after discovery pass findings:
#   - normalize reasoning EXACTLY like prep_sft_data.py did (SFT saw normalized text)
#   - preserve each row's ORIGINAL header strings (CAPTURES & TRADES vs CAPTURES)
#   - chunk into: preamble | 5 sections | conclusion
#   - PROOF: rebuild reasoning from chunks, assert byte-identical to normalized
#     original for every row. Chunking must be provably lossless.

import csv
import json
import os
import re
import sys
from collections import Counter

from tqdm import tqdm

# ============ CONFIG ============
INPUT_CSV = r"D:\ChessGRPO_v2\Step_1_SFT\chess_reasoning_v2_dataset.csv"
OUTPUT_JSONL = r"D:\ChessGRPO_v2\Step_2_GRPO\Rung2\parsed_dataset.jsonl"
REJECTS_LOG = r"D:\ChessGRPO_v2\Step_2_GRPO\Rung2\parse_rejects.jsonl"

CANONICAL_ORDER = ["KING SAFETY", "CHECKS", "CAPTURES", "THREATS", "IMPROVEMENT"]
HEADER_ALIASES = {
    "CAPTURES & TRADES": "CAPTURES",
    "CAPTURES AND TRADES": "CAPTURES",
}
# ================================

HEADER_RE = re.compile(r"\[([A-Z][A-Z &]*?)\]\s*:")
CONCLUSION_RE = re.compile(r"(?:^|\n)\s*Conclusion\s*:", re.IGNORECASE)

csv.field_size_limit(10_000_000)


def normalize_reasoning(text):
    """Byte-for-byte copy of prep_sft_data.py's normalization.
    The SFT model trained on text that went through exactly this."""
    if not isinstance(text, str):
        return ""
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def canonical(name: str) -> str:
    name = name.strip()
    return HEADER_ALIASES.get(name, name)


def parse_reasoning(text: str):
    """Chunk normalized reasoning. Returns (parsed_dict, error). Exactly one is None.

    Chunks are stored as EXACT substrings of the normalized text:
      preamble         = everything before the first header
      section_texts[i] = from the start of header i to the start of header i+1
                         (header string itself INCLUDED, so reconstruction is
                         trivial and original header variants are preserved)
      conclusion       = the 'Conclusion:' tail split out of the last section
    Invariant: preamble + sections + conclusion, concatenated, == original text.
    """
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return None, "no section headers found"

    names = [canonical(m.group(1)) for m in matches]

    if names != CANONICAL_ORDER:
        if sorted(set(names)) != sorted(set(CANONICAL_ORDER)):
            missing = set(CANONICAL_ORDER) - set(names)
            extra = set(names) - set(CANONICAL_ORDER)
            return None, f"header mismatch, missing={sorted(missing)}, extra={sorted(extra)}"
        if len(names) != len(CANONICAL_ORDER):
            return None, f"duplicate headers: {names}"
        return None, f"headers out of order: {names}"

    preamble = text[: matches[0].start()]

    # Each section chunk = header included, running to the next header's start
    section_texts = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_texts.append(text[start:end])

    # Split conclusion out of the last section
    last = section_texts[-1]
    conclusion = ""
    cm = CONCLUSION_RE.search(last)
    if cm:
        # +1 offset if the match starts at a newline: keep that newline with
        # the conclusion chunk so concatenation stays byte-exact
        conclusion = last[cm.start():]
        section_texts[-1] = last[: cm.start()]

    if any(not s.strip() for s in section_texts):
        return None, "empty section after split"

    return {
        "preamble": preamble,
        "section_texts": section_texts,
        "conclusion": conclusion,
        "original_headers": [m.group(0) for m in matches],  # e.g. '[CAPTURES & TRADES]:'
    }, None


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Input not found: {INPUT_CSV}")
        sys.exit(1)
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    with open(INPUT_CSV, "r", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {INPUT_CSV}")

    clean, rejects = [], []
    no_conclusion = 0
    rebuild_failures = 0
    header_variant_rows = Counter()

    for r in tqdm(rows, desc="Parsing"):
        normalized = normalize_reasoning(r["Reasoning"])
        parsed, err = parse_reasoning(normalized)
        if err:
            rejects.append({"id": r["id"], "error": err})
            continue

        # THE PROOF: chunks must reassemble into the normalized original exactly
        rebuilt = parsed["preamble"] + "".join(parsed["section_texts"]) + parsed["conclusion"]
        if rebuilt != normalized:
            rebuild_failures += 1
            rejects.append({"id": r["id"], "error": "REBUILD MISMATCH (lossy chunking)"})
            continue

        if not parsed["conclusion"]:
            no_conclusion += 1
        header_variant_rows[parsed["original_headers"][2]] += 1  # captures-slot variant

        clean.append({
            "id": r["id"],
            "fen": r["FEN"],
            "best_move": r["Best Move"].strip(),
            "preamble": parsed["preamble"],
            "section_texts": parsed["section_texts"],
            "conclusion": parsed["conclusion"],
        })

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in clean:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(REJECTS_LOG, "w", encoding="utf-8") as f:
        for row in rejects:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Clean rows:            {len(clean)}  -> {OUTPUT_JSONL}")
    print(f"Rejected:              {len(rejects)}  -> {REJECTS_LOG}")
    print(f"REBUILD FAILURES:      {rebuild_failures}   <-- must be 0")
    print(f"Rows with no Conclusion line: {no_conclusion}")
    print("Captures-header variants preserved per row:")
    for h, c in header_variant_rows.most_common():
        print(f"  {h}  : {c} rows")
    print("=" * 60)
    print("REBUILD FAILURES must be 0. If 0, chunking is provably lossless")
    print("and D3 parsing is done. Paste this output back.")


if __name__ == "__main__":
    main()