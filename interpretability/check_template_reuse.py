import difflib
import itertools
import random
import pandas as pd
random.seed(3407)

df = pd.read_csv('causal_stage2_results/causal_completions_stage2.csv')
df = df[df['condition'] != 'zero'].copy()   # zero rows are empty, not comparable

df['false_mate'] = df['false_mate'].astype(str).str.strip() == 'True'


def reasoning_text(completion):
    if not isinstance(completion, str):
        return ""
    return completion.split("</thinking>")[0].strip()


df['reasoning'] = df['completion'].apply(reasoning_text)

checkmate_df = df[df['parsed_move'] == 'CHECKMATE'].copy()
move_df = df[df['parsed_move'].notna() & (df['parsed_move'] != 'CHECKMATE')].copy()

print(f"CHECKMATE completions (all conditions except zero): {len(checkmate_df)}")
print(f"  real mate calls (false_mate=False): {(~checkmate_df['false_mate']).sum()}")
print(f"  FALSE checkmate hallucinations (false_mate=True): {checkmate_df['false_mate'].sum()}")
print(f"Real-move completions (control group): {len(move_df)}")
print()

# is there even a real mate to find in this eval set? check best_move directly.
n_real_mates_in_eval = (df['best_move'].astype(str).str.upper() == 'CHECKMATE').sum()
print(f"Positions in this run where best_move == CHECKMATE (ground truth): {n_real_mates_in_eval}")
print("(if this is 0, the eval set simply contains no real mates -- 'the model never")
print(" correctly calls checkmate' would then mean 'never had the chance', not a miss.)")
print()


def cross_board_pairs(sub_df, max_pairs=2000):
    """Only pairs where fen differs -- same-board duplicates across conditions are
    excluded, so a high similarity score can only come from genuinely different
    boards converging on similar wording."""
    rows = list(sub_df[['fen', 'reasoning']].itertuples(index=False, name=None))
    rows = [(f, r) for f, r in rows if r]
    all_pairs = list(itertools.combinations(rows, 2))
    cross = [(a, b) for (fa, a), (fb, b) in all_pairs if fa != fb]
    if len(cross) > max_pairs:
        cross = random.sample(cross, max_pairs)
    sims = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in cross]
    return sims, cross


def report(name, sims, pairs):
    if not sims:
        print(f"--- {name}: no cross-board pairs to compare ---\n")
        return
    sims_sorted = sorted(sims)
    n = len(sims)
    print(f"--- {name} (CROSS-BOARD ONLY) ---")
    print(f"  pairs compared = {n}")
    print(f"  mean similarity  = {sum(sims)/n:.3f}")
    print(f"  median similarity = {sims_sorted[n//2]:.3f}")
    print(f"  min / max         = {sims_sorted[0]:.3f} / {sims_sorted[-1]:.3f}")
    above70 = sum(1 for s in sims if s > 0.70)
    above85 = sum(1 for s in sims if s > 0.85)
    print(f"  pairs above 0.70 similarity: {above70}/{n} ({100*above70/n:.1f}%)")
    print(f"  pairs above 0.85 similarity: {above85}/{n} ({100*above85/n:.1f}%)")
    print()


print("=" * 90)
print("Q1 (FIXED): Do FALSE checkmate hallucinations reuse near-identical wording")
print("            across DIFFERENT boards specifically (same-board dupes excluded)?")
print("=" * 90)
false_sub = checkmate_df[checkmate_df['false_mate']]
sims_false, pairs_false = cross_board_pairs(false_sub)
report("FALSE checkmate hallucinations", sims_false, pairs_false)

print("=" * 90)
print("Q2 (FIXED): Same cross-board-only treatment for the real-move control group")
print("            (mostly moot here since real moves rarely repeat verbatim anyway,")
print("            included for a clean apples-to-apples comparison).")
print("=" * 90)
move_sub = move_df
if len(move_sub) > 120:
    move_sub = move_sub.sample(120, random_state=3407)
sims_move, pairs_move = cross_board_pairs(move_sub)
report("Real-move completions (control)", sims_move, pairs_move)

print("=" * 90)
print("Highest / lowest similarity CROSS-BOARD false-checkmate pair:")
print("=" * 90)
if pairs_false:
    ranked = sorted(zip(sims_false, pairs_false), key=lambda x: x[0])
    worst_sim, worst_pair = ranked[0]
    best_sim, best_pair = ranked[-1]
    print(f"HIGHEST cross-board similarity = {best_sim:.3f}")
    print("--- TEXT A ---"); print(best_pair[0])
    print("--- TEXT B ---"); print(best_pair[1])
    print()
    print(f"LOWEST cross-board similarity = {worst_sim:.3f}")
    print("--- TEXT A ---"); print(worst_pair[0])
    print("--- TEXT B ---"); print(worst_pair[1])