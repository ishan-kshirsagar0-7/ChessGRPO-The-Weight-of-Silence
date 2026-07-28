"""
audit_legal.py  --  Decompose the Rung 2 legal-move rate.

Answers, per stage:
  - Of the LEGAL count, how many were real board moves vs correct CHECKMATE claims?
  - How many CHECKMATE claims did the model make total, and how many were false?
  - If the checkmate crutch were removed, what is the "real-move legal rate"?

The point: decide whether stage 2's legal% is a solid launchpad for Rung 3 or
whether it is inflated by lucky checkmate declarations.
"""

import pandas as pd

COMPLETIONS = "eval_completions_rung2.csv"


def as_bool(series):
    """CSV round-trips bools to strings sometimes. Normalize."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def main():
    df = pd.read_csv(COMPLETIONS)
    for col in ("format_ok", "legal", "correct", "false_mate"):
        df[col] = as_bool(df[col])

    df["parsed_move"] = df["parsed_move"].astype(str)
    df["is_mate_claim"] = df["parsed_move"].str.upper() == "CHECKMATE"

    print("=" * 74)
    print("LEGAL-RATE DECOMPOSITION PER STAGE")
    print("=" * 74)
    print(f"{'stage':>5}{'n':>5}{'legal':>7}{'  = real':>9}{' + mate':>8}"
          f"{'  real-move legal%':>19}")
    print("-" * 74)

    for stage in sorted(df["stage"].unique()):
        s = df[df["stage"] == stage]
        n = len(s)

        legal = s["legal"].sum()
        # a legal that was a checkmate claim = correct mate; a legal that was a
        # normal move = real board move
        legal_mate = s[s["legal"] & s["is_mate_claim"]].shape[0]
        legal_real = s[s["legal"] & ~s["is_mate_claim"]].shape[0]

        real_move_legal_pct = 100.0 * legal_real / n

        print(f"{stage:>5}{n:>5}{legal:>7}{legal_real:>9}{legal_mate:>8}"
              f"{real_move_legal_pct:>18.1f}%")

    print("\n" + "=" * 74)
    print("CHECKMATE-CLAIM AUDIT PER STAGE")
    print("  (every time the model said CHECKMATE, was it right?)")
    print("=" * 74)
    print(f"{'stage':>5}{'mate_claims':>13}{'  correct':>10}{'  false':>8}"
          f"{'  false-mate rate':>18}")
    print("-" * 74)

    for stage in sorted(df["stage"].unique()):
        s = df[df["stage"] == stage]
        n = len(s)
        claims = s["is_mate_claim"].sum()
        correct_mate = s[s["is_mate_claim"] & s["legal"]].shape[0]
        false_mate = s["false_mate"].sum()
        print(f"{stage:>5}{claims:>13}{correct_mate:>10}{false_mate:>8}"
              f"{100.0*false_mate/n:>16.1f}%")

    print("\n" + "=" * 74)
    print("READING")
    print("=" * 74)
    print("real-move legal% = legality with the checkmate crutch removed entirely.")
    print("If a stage's real-move legal% is close to its headline legal%, the")
    print("number is solid. If real-move legal% is much lower, the headline was")
    print("propped up by checkmate claims that happened to land.")
    print()
    print("For the Rung 3 launchpad decision: we want the stage with the highest")
    print("REAL-MOVE legal%, not the highest headline legal%. Those may differ.")


if __name__ == "__main__":
    main()