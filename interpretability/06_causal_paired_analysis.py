"""
06_causal_paired_analysis.py -- Paired, per-position comparison of the three
causal-test conditions (baseline / substitute / ablate).

The aggregate legal% (58/58/57) can't rule out cancelling flips: e.g. 3
positions going legal->illegal in ablate while 2 different ones go
illegal->legal would still net out to "~57%" while something real happened
underneath. This pairs each condition's result BY FEN (all three ran the same
100 positions, same order) and reports:

  - confusion matrix (both-legal / both-illegal / flipped) for each of the
    3 condition pairs
  - exact-move-match rate: how many positions picked the IDENTICAL move
    string across conditions (stronger evidence than just legal/illegal match)
  - the specific FENs where legality flipped, if any, for manual inspection

Reads causal_completions.csv from the real 100-position run. Does not touch
the model or GPU at all -- pure pandas, runs in seconds.
"""

import logging
import os

import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────────────
COMPLETIONS_IN = os.path.expanduser("~/g6e_prep/causal_completions.csv")
OUT_REPORT     = os.path.expanduser("~/g6e_prep/06_paired_analysis_report.txt")

PAIRS = [("baseline", "substitute"), ("baseline", "ablate"), ("substitute", "ablate")]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(OUT_REPORT, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger("paired")


def main():
    log = setup_logging()
    df = pd.read_csv(COMPLETIONS_IN)

    conditions = df["condition"].unique().tolist()
    log.info("Loaded %d rows, conditions found: %s", len(df), conditions)

    # pivot to one row per FEN, one column per condition, for both legal-bool
    # and the parsed move string
    legal_pivot = df.pivot(index="fen", columns="condition", values="legal")
    move_pivot  = df.pivot(index="fen", columns="condition", values="parsed_move")

    n = len(legal_pivot)
    log.info("Positions common to all conditions: %d\n", n)

    for a, b in PAIRS:
        if a not in legal_pivot.columns or b not in legal_pivot.columns:
            log.info("Skipping %s vs %s (condition missing from data)\n", a, b)
            continue

        both_legal   = int(((legal_pivot[a] == True)  & (legal_pivot[b] == True)).sum())
        both_illegal = int(((legal_pivot[a] == False) & (legal_pivot[b] == False)).sum())
        a_only       = int(((legal_pivot[a] == True)  & (legal_pivot[b] == False)).sum())
        b_only       = int(((legal_pivot[a] == False) & (legal_pivot[b] == True)).sum())
        flipped      = a_only + b_only

        exact_match = int((move_pivot[a] == move_pivot[b]).sum())

        log.info("=" * 60)
        log.info("%s vs %s", a, b)
        log.info("=" * 60)
        log.info("  both legal:       %3d", both_legal)
        log.info("  both illegal:     %3d", both_illegal)
        log.info("  %-10s only legal: %3d", a, a_only)
        log.info("  %-10s only legal: %3d", b, b_only)
        log.info("  total flipped:    %3d / %d", flipped, n)
        log.info("  exact same move:  %3d / %d (%.1f%%)", exact_match, n, 100 * exact_match / n)

        if flipped > 0:
            log.info("\n  flipped positions:")
            flip_mask = (legal_pivot[a] != legal_pivot[b])
            for fen in legal_pivot[flip_mask].index:
                log.info("    FEN: %s", fen)
                log.info("      %-10s legal=%-5s move=%s", a, legal_pivot.loc[fen, a], move_pivot.loc[fen, a])
                log.info("      %-10s legal=%-5s move=%s", b, legal_pivot.loc[fen, b], move_pivot.loc[fen, b])
        log.info("")

    log.info("Report written to %s", OUT_REPORT)


if __name__ == "__main__":
    main()