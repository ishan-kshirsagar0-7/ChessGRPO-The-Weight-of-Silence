"""
mcnemar_n1000_full.py -- Paired McNemar test for EVERY pairwise combination
of the six n=1000 rung-3 causal conditions (15 pairs total).
"""

import os
from itertools import combinations
import pandas as pd
from scipy.stats import binomtest

ROOT_DIR = "/mnt/d/ChessGRPO_v2"
COMPLETIONS_PATH = os.path.join(ROOT_DIR, "06_results_and_eval", "interpretability", "causal_completions_rung3.csv")
OUT_PATH = os.path.join(ROOT_DIR, "06_results_and_eval", "interpretability", "mcnemar_n1000_full_pairwise.csv")

ALL_CONDITIONS = ["baseline", "substitute", "ablate", "zero", "noise", "lenmatch_ablate"]


def to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"


def exact_mcnemar(b, c):
    """Exact binomial McNemar test on discordant pair counts b, c.
    Under H0 (marginal symmetry), min(b,c) ~ Binomial(n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    return binomtest(k, n, 0.5, alternative="two-sided").pvalue


def main():
    df = pd.read_csv(COMPLETIONS_PATH)
    df["legal"] = df["legal"].apply(to_bool)

    dupes = df.duplicated(subset=["position_idx", "condition"], keep=False)
    if dupes.any():
        print(f"WARNING: {dupes.sum()} duplicate (position_idx, condition) rows found -- "
              f"keeping first occurrence of each.")
        df = df.drop_duplicates(subset=["position_idx", "condition"], keep="first")

    pivot = df.pivot_table(index="position_idx", columns="condition", values="legal", aggfunc="first")

    missing_conditions = [c for c in ALL_CONDITIONS if c not in pivot.columns]
    if missing_conditions:
        raise SystemExit(f"FATAL: conditions missing from data entirely: {missing_conditions}")

    incomplete = pivot[ALL_CONDITIONS].isna().any(axis=1)
    if incomplete.any():
        print(f"WARNING: {incomplete.sum()} positions missing a result for at least one "
              f"condition -- excluding from ALL pairwise comparisons so every pair uses "
              f"the identical position set.")
        pivot = pivot[~incomplete]

    n_positions = len(pivot)
    print(f"Paired McNemar test, all {len(ALL_CONDITIONS)} conditions, "
          f"n={n_positions} positions with complete data across all conditions.\n")

    rows = []
    print(f"{'pair':<32s} {'legal% a':>9s} {'legal% b':>9s} {'b':>5s} {'c':>5s} {'n_disc':>7s}  {'McNemar p':>12s}")
    print("-" * 90)
    for cond_a, cond_b in combinations(ALL_CONDITIONS, 2):
        a, b_series = pivot[cond_a], pivot[cond_b]

        b = int(((a == True) & (b_series == False)).sum())   # a legal, b illegal
        c = int(((a == False) & (b_series == True)).sum())   # a illegal, b legal
        both_legal = int(((a == True) & (b_series == True)).sum())
        both_illegal = int(((a == False) & (b_series == False)).sum())

        p = exact_mcnemar(b, c)

        rows.append({
            "condition_a": cond_a, "condition_b": cond_b, "n": n_positions,
            "legal_pct_a": round(100.0 * a.mean(), 1),
            "legal_pct_b": round(100.0 * b_series.mean(), 1),
            "both_legal": both_legal, "both_illegal": both_illegal,
            "a_legal_b_illegal": b, "a_illegal_b_legal": c,
            "total_discordant": b + c,
            "mcnemar_p": p,
        })

        pair_label = f"{cond_a} vs {cond_b}"
        print(f"{pair_label:<32s} {a.mean()*100:9.1f} {b_series.mean()*100:9.1f} "
              f"{b:5d} {c:5d} {b+c:7d}  {p:12.4g}")

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")

    print("\n" + "=" * 78)
    print("Highlighted: Ablate vs Lenmatch-Ablate (the remaining z-test in Appendix A.8)")
    print("=" * 78)
    hi = next(r for r in rows if {r["condition_a"], r["condition_b"]} == {"ablate", "lenmatch_ablate"})
    print(f"  Ablate: {hi['legal_pct_a' if hi['condition_a']=='ablate' else 'legal_pct_b']}% legal")
    print(f"  Lenmatch-Ablate: {hi['legal_pct_b' if hi['condition_a']=='ablate' else 'legal_pct_a']}% legal")
    print(f"  Discordant pairs: {hi['a_legal_b_illegal']}, {hi['a_illegal_b_legal']} "
          f"(total {hi['total_discordant']})")
    print(f"  McNemar p = {hi['mcnemar_p']:.4g}")


if __name__ == "__main__":
    main()