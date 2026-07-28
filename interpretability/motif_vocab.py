"""
07_motif_vocab.py

Builds the per-chunk motif vocabulary for the J-lens analysis arm.

Runs two gates before any J-lens work is scheduled:
  G1 distinctiveness: do the 5 chunks have distinct vocabularies?
                      If not, the 5x5 diagonal cannot exist. Arm dies here, free.
  G2 readability:     what fraction of each chunk's chunk-specific vocabulary
                      is single-token, i.e. addressable by J-lens?

Chunk map:
  1 = preamble + section_texts[0]   ([KING SAFETY])
  2 = section_texts[1]              ([CHECKS])
  3 = section_texts[2]              ([CAPTURES & TRADES])
  4 = section_texts[3]              ([THREATS])
  5 = section_texts[4] + conclusion ([IMPROVEMENT] + Conclusion)
"""

import json
import re
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer

# -----------------------------config---------------------------------
PARSED_PATH = r"D:\ChessGRPO_v2\Step_2_GRPO\Rung2\parsed_dataset.jsonl"
OUT_PATH    = r"D:\ChessGRPO_v2\Step_2_GRPO\Rung2\motif_vocab.json"
TOKENIZER   = "Qwen/Qwen3-0.6B"

MIN_WORD_LEN     = 3      # kills coordinate debris like the h/g of "h6g6"
MIN_TOTAL_FREQ   = 200    # a word must appear this often across the whole corpus
SPECIFICITY_MIN  = 0.45   # freq_in_chunk_k / freq_across_all_chunks
TOP_N_REPORT     = 25
JACCARD_TOP_N    = 60     # top-N per chunk used for the G1 overlap matrix

PIECE_WORDS = ["king", "queen", "rook", "bishop", "knight", "pawn"]

STOPWORDS = {
    "the", "and", "for", "are", "was", "this", "that", "with", "from", "there",
    "only", "have", "has", "not", "but", "its", "it", "is", "to", "of", "in",
    "on", "at", "by", "as", "an", "a", "be", "can", "will", "which", "any",
    "all", "one", "two", "three", "more", "than", "then", "also", "into",
    "would", "could", "does", "did", "if", "no", "yes", "so", "such", "these",
    "those", "their", "they", "them", "his", "her", "he", "she", "we", "you",
    "move", "moves", "best", "position", "board", "side", "player",
}

HEADER_RE = re.compile(r"^\s*\[([^\]]+)\]\s*:?")
WORD_RE   = re.compile(rf"[A-Za-z]{{{MIN_WORD_LEN},}}")
# ----------------------------------------------------------------------


def single_token_forms(tok, word):
    """Return the surface variants of `word` that encode to exactly one token."""
    forms = [word, " " + word]
    return [f for f in forms if len(tok.encode(f, add_special_tokens=False)) == 1]


def build_chunks(row):
    """Assemble the 5 chunk texts from a parsed row. Returns list[5] of str."""
    sections = row["section_texts"]
    if len(sections) != 5:
        raise ValueError(f"row {row.get('id')} has {len(sections)} sections, expected 5")

    preamble   = row.get("preamble", "") or ""
    conclusion = row.get("conclusion", "") or ""

    return [
        (preamble + "\n" + sections[0]).strip(),
        sections[1].strip(),
        sections[2].strip(),
        sections[3].strip(),
        (sections[4] + "\n" + conclusion).strip(),
    ]


def strip_header(text):
    """Pull the leading [SECTION NAME]: off a chunk. Returns (header_or_None, body)."""
    m = HEADER_RE.match(text)
    if not m:
        return None, text
    return m.group(1), text[m.end():]


def main():
    print(f"Loading tokenizer: {TOKENIZER}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    with open(PARSED_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print(f"Loaded {len(rows)} rows from {PARSED_PATH}\n")

    chunk_counts  = [Counter() for _ in range(5)]   # surface form -> freq, per chunk
    header_counts = [Counter() for _ in range(5)]
    piece_surfaces = Counter()                      # observed casings of piece words

    for row in tqdm(rows, desc="counting", unit="row"):
        for k, chunk_text in enumerate(build_chunks(row)):
            header, body = strip_header(chunk_text)
            if header is not None:
                header_counts[k][header] += 1

            for w in WORD_RE.findall(body):
                if w.lower() in STOPWORDS:
                    continue
                if w.isupper() and len(w) > 3:
                    continue                        # all-caps boilerplate, e.g. PRIORITY
                chunk_counts[k][w] += 1
                if w.lower() in PIECE_WORDS:
                    piece_surfaces[w] += 1

    total_counts = Counter()
    for c in chunk_counts:
        total_counts.update(c)

    print("\n" + "=" * 72)
    print("SECTION HEADERS OBSERVED (per chunk)")
    print("=" * 72)
    for k in range(5):
        top = header_counts[k].most_common(3)
        print(f"  chunk {k+1}: {top if top else 'no bracketed header'}")

    print("\n" + "=" * 72)
    print("PIECE-NAME SURFACE FORMS AS THEY APPEAR IN THE CORPUS")
    print("(this is the casing the latent thought is trained to reproduce)")
    print("=" * 72)
    print(f"{'surface':<12} {'freq':>8}   single-token forms")
    print("-" * 72)
    for surf, freq in piece_surfaces.most_common():
        forms = single_token_forms(tok, surf)
        flag = "OK " if forms else "DEAD"
        print(f"{surf:<12} {freq:>8}   {flag} {forms if forms else ''}")

    print("\n" + "=" * 72)
    print(f"G1  DISTINCTIVENESS: Jaccard overlap of top-{JACCARD_TOP_N} vocab per chunk")
    print("    diagonal = 1.00 by construction. Off-diagonal should be LOW.")
    print("    If off-diagonal is high everywhere, the 5x5 experiment is dead.")
    print("=" * 72)

    top_sets = [set(w for w, _ in c.most_common(JACCARD_TOP_N)) for c in chunk_counts]
    print("        " + "".join(f"{'c'+str(j+1):>8}" for j in range(5)))
    for i in range(5):
        cells = []
        for j in range(5):
            inter = len(top_sets[i] & top_sets[j])
            union = len(top_sets[i] | top_sets[j])
            cells.append(f"{inter/union:>8.2f}")
        print(f"  c{i+1}   " + "".join(cells))

    motif = {}
    print("\n" + "=" * 72)
    print("MOTIF SETS  (total_freq >= "
          f"{MIN_TOTAL_FREQ}, specificity >= {SPECIFICITY_MIN})")
    print("=" * 72)

    for k in range(5):
        entries = []
        for w, f_k in chunk_counts[k].items():
            f_all = total_counts[w]
            if f_all < MIN_TOTAL_FREQ:
                continue
            spec = f_k / f_all
            if spec < SPECIFICITY_MIN:
                continue
            entries.append((w, f_k, spec, single_token_forms(tok, w)))

        entries.sort(key=lambda e: -e[1])

        readable_mass = sum(e[1] for e in entries if e[3])
        total_mass    = sum(e[1] for e in entries)
        coverage = readable_mass / total_mass if total_mass else 0.0

        print(f"\n--- chunk {k+1} --- {len(entries)} motif words, "
              f"G2 readable mass = {coverage:.1%}")
        print(f"{'word':<16} {'freq':>7} {'spec':>7}   single-token forms")
        print("-" * 72)
        for w, f_k, spec, forms in entries[:TOP_N_REPORT]:
            flag = "OK " if forms else "DEAD"
            shown = ", ".join(forms) if forms else ""
            print(f"{w:<16} {f_k:>7} {spec:>7.2f}   {flag} {shown}")

        motif[f"chunk_{k+1}"] = {
            "coverage_readable_mass": coverage,
            "n_motif_words": len(entries),
            "words": [
                {"word": w, "freq": f_k, "specificity": round(spec, 4),
                 "single_token_forms": forms}
                for w, f_k, spec, forms in entries
            ],
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(motif, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("G1 passes if the off-diagonal Jaccard cells are clearly below the diagonal.")
    print("G2 passes if readable mass is comfortably above ~50% for most chunks.")
    print(f"\nWritten: {OUT_PATH}")
    print("Paste the full stdout back.")


if __name__ == "__main__":
    main()