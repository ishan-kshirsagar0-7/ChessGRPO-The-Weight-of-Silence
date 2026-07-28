"""
delta_w_svd.py -- Computes the RL-induced weight change (stage-2 -> rung-3) per
LoRA module, using exact SVD without ever materializing full dense weight
matrices.

KEY IDEA: a LoRA adapter's effective weight update is scaling * B @ A, where
B is (out_dim x r) and A is (r x in_dim), r << out_dim/in_dim. The quantity
we want is:

    delta_W = update_rung3 - update_stage2
            = scaling3 * B3 @ A3 - scaling2 * B2 @ A2
            = [scaling3*B3 | -scaling2*B2] @ [A3; A2]

This is a product of an (out_dim x 2r) matrix and a (2r x in_dim) matrix, so
delta_W has rank <= 2r (e.g. <= 32 for r=16 LoRA) REGARDLESS of out_dim/in_dim
being in the thousands. Rather than forming the huge dense delta_W matrix and
running full SVD on it (tens of seconds to minutes PER matrix, ~hours across
all ~280 matrices in this model, and wasteful since every singular value past
the first 2r is exactly/numerically zero anyway), we get the SVD directly
from the skinny factors via QR + a tiny SVD:

    [s3*B3 | -s2*B2] = Q1 R1     (Q1: out_dim x k orthonormal, R1: k x k)
    [A3; A2]         = R2^T Q2^T (Q2: in_dim  x k orthonormal, R2: k x k)
    delta_W = Q1 (R1 R2^T) Q2^T
    R1 R2^T = U' S V'^T          (tiny k x k SVD, milliseconds)
    => true singular values of delta_W = S  (exact, not approximate)
    => true singular vectors = Q1 @ U', Q2 @ V'

Top-K modules by Frobenius norm also get their actual U/V singular vectors
saved to disk (delta_w_top_singular_vectors.pt) -- these are the ready-to-use
patch/steering directions for the causal follow-up experiment named in the
paper's Future Work section.
"""

import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

# ── CONFIG ──────────────────────────────────────────────────────────────────
ROOT_DIR = "/mnt/d/ChessGRPO_v2"
STAGE2_ADAPTER_DIR = os.path.join(ROOT_DIR, "03_checkpoints_local_only", "rung2_stage2_launchpad")
RUNG3_ADAPTER_DIR  = os.path.join(ROOT_DIR, "03_checkpoints_local_only", "rung3_latent_grpo_final", "policy")

OUT_DIR = os.path.join(ROOT_DIR, "06_results_and_eval", "interpretability")
PER_MODULE_OUT = os.path.join(OUT_DIR, "delta_w_per_module.csv")
BY_TYPE_OUT    = os.path.join(OUT_DIR, "delta_w_by_module_type.csv")
SINGVALS_OUT   = os.path.join(OUT_DIR, "delta_w_singular_values.csv")
VECTORS_OUT    = os.path.join(OUT_DIR, "delta_w_top_singular_vectors.pt")

COMPUTE_SINGULAR_VECTORS = True   # saves U/V for the TOP_K_TO_SAVE biggest movers,
                                    # for a later steering/patching experiment
TOP_K_TO_SAVE = 10                 # only the biggest movers -- keeps output small

# ═══════════════════════════════════════════════════════════════════════════


def find_safetensors(adapter_dir):
    p = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")
    return p


def load_lora_config(adapter_dir):
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    r, alpha = cfg["r"], cfg["lora_alpha"]
    return r, alpha, alpha / r


def group_by_module(state_dict):
    """PEFT LoRA keys end in '.lora_A.weight' / '.lora_B.weight'. Groups into
    {module_key: {"A": tensor, "B": tensor}}, robust to whatever prefix this
    particular PEFT version saved (doesn't assume an exact 'base_model.model.'
    prefix, just matches on the suffix)."""
    modules = defaultdict(dict)
    for key, tensor in state_dict.items():
        m = re.match(r"(.*)\.lora_(A|B)\.weight$", key)
        if not m:
            continue
        modules[m.group(1)][m.group(2)] = tensor.float()
    incomplete = [k for k, v in modules.items() if "A" not in v or "B" not in v]
    if incomplete:
        raise RuntimeError(f"Modules missing A or B side: {incomplete}")
    return dict(modules)


def classify_module(module_key: str):
    """'...layers.14.mlp.gate_proj' -> ('mlp.gate_proj', 14)."""
    m = re.search(r"\.layers\.(\d+)\.(.+)$", module_key)
    if not m:
        return module_key, -1
    return m.group(2), int(m.group(1))


def delta_w_singular_values(A2, B2, scaling2, A3, B3, scaling3, compute_vectors=False):
    """Exact singular values (and optionally vectors) of
    scaling3*B3@A3 - scaling2*B2@A2, via QR + tiny SVD. Never materializes
    the full out_dim x in_dim dense matrix."""
    B2s = B2 * scaling2
    B3s = B3 * scaling3

    M1 = torch.cat([B3s, -B2s], dim=1)          # out_dim x (r3+r2)
    M2 = torch.cat([A3, A2], dim=0)             # (r3+r2) x in_dim

    Q1, R1 = torch.linalg.qr(M1, mode="reduced")
    Q2, R2 = torch.linalg.qr(M2.T, mode="reduced")

    core = R1 @ R2.T
    U_core, S, Vh_core = torch.linalg.svd(core, full_matrices=False)
    singular_values = S.numpy()

    if not compute_vectors:
        return singular_values, None, None
    U = (Q1 @ U_core).numpy()   # output-side directions, shape (out_dim, k)
    V = (Q2 @ Vh_core.T).numpy()  # input-side directions, shape (in_dim, k)
    return singular_values, U, V


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading adapter configs...")
    r2, alpha2, scaling2 = load_lora_config(STAGE2_ADAPTER_DIR)
    r3, alpha3, scaling3 = load_lora_config(RUNG3_ADAPTER_DIR)
    print(f"  stage-2: r={r2} alpha={alpha2} scaling={scaling2:.4f}")
    print(f"  rung-3:  r={r3} alpha={alpha3} scaling={scaling3:.4f}")
    if r2 != r3:
        print(f"  NOTE: ranks differ (stage-2 r={r2}, rung-3 r={r3}). Handled "
              f"correctly by the concatenation approach, just flagging it.")

    print("\nLoading state dicts (CPU, fp32)...")
    modules2 = group_by_module(load_file(find_safetensors(STAGE2_ADAPTER_DIR)))
    modules3 = group_by_module(load_file(find_safetensors(RUNG3_ADAPTER_DIR)))

    keys2, keys3 = set(modules2), set(modules3)
    if keys2 != keys3:
        raise RuntimeError(
            f"Module key mismatch between checkpoints.\n"
            f"  Only in stage-2: {sorted(keys2 - keys3)[:5]}\n"
            f"  Only in rung-3:  {sorted(keys3 - keys2)[:5]}"
        )
    print(f"  {len(modules2)} matched LoRA modules found in both checkpoints.")

    per_module_rows, singval_rows = [], []
    total_energy = 0.0
    energy_by_type = defaultdict(float)

    print("\nComputing per-module delta_W singular values (QR trick, CPU)...")
    for i, module_key in enumerate(sorted(modules2)):
        A2, B2 = modules2[module_key]["A"], modules2[module_key]["B"]
        A3, B3 = modules3[module_key]["A"], modules3[module_key]["B"]

        if A2.shape[1] != A3.shape[1] or B2.shape[0] != B3.shape[0]:
            raise RuntimeError(f"Shape mismatch on {module_key}: "
                                f"stage2 A{tuple(A2.shape)} B{tuple(B2.shape)} vs "
                                f"rung3 A{tuple(A3.shape)} B{tuple(B3.shape)}")

        s_vals, _, _ = delta_w_singular_values(
            A2, B2, scaling2, A3, B3, scaling3, compute_vectors=False
        )
        frob_sq = float(np.sum(s_vals ** 2))
        frob = float(np.sqrt(frob_sq))
        top1_share = float(s_vals[0] ** 2 / frob_sq) if frob_sq > 0 else 0.0
        # Effective rank via participation ratio: 1.0 = all change concentrated
        # in one direction (rank-1-like, coherent adjustment); up to len(s_vals)
        # = spread evenly across every available direction (diffuse). No
        # arbitrary threshold needed -- this falls out of the energy distribution.
        p = (s_vals ** 2) / frob_sq if frob_sq > 0 else np.zeros_like(s_vals)
        effective_rank = float(1.0 / np.sum(p ** 2)) if frob_sq > 0 else 0.0

        module_type, layer_idx = classify_module(module_key)
        per_module_rows.append({
            "module_key": module_key, "module_type": module_type, "layer_idx": layer_idx,
            "frobenius_norm": frob, "top1_singular_value": float(s_vals[0]),
            "top1_energy_share": top1_share, "effective_rank": effective_rank,
            "n_singular_values": len(s_vals),
        })
        for j, sv in enumerate(s_vals):
            singval_rows.append({"module_key": module_key, "module_type": module_type,
                                   "layer_idx": layer_idx, "rank_position": j,
                                   "singular_value": float(sv)})

        total_energy += frob_sq
        energy_by_type[module_type] += frob_sq

        if (i + 1) % 20 == 0 or (i + 1) == len(modules2):
            print(f"  {i + 1}/{len(modules2)} modules done")

    per_module_df = pd.DataFrame(per_module_rows).sort_values("frobenius_norm", ascending=False)
    per_module_df.to_csv(PER_MODULE_OUT, index=False)

    by_type_rows = [
        {"module_type": t, "energy_share_pct": round(100.0 * e / total_energy, 2),
         "n_modules": sum(1 for r in per_module_rows if r["module_type"] == t)}
        for t, e in sorted(energy_by_type.items(), key=lambda x: -x[1])
    ]
    pd.DataFrame(by_type_rows).to_csv(BY_TYPE_OUT, index=False)
    pd.DataFrame(singval_rows).to_csv(SINGVALS_OUT, index=False)

    # ── Save top-K singular vectors for the steering/patching follow-up ─────
    if COMPUTE_SINGULAR_VECTORS:
        top_modules = per_module_df.head(TOP_K_TO_SAVE)["module_key"].tolist()
        print(f"\nComputing full singular vectors for the top {TOP_K_TO_SAVE} "
              f"modules by Frobenius norm...")
        saved_vectors = {}
        for module_key in top_modules:
            A2, B2 = modules2[module_key]["A"], modules2[module_key]["B"]
            A3, B3 = modules3[module_key]["A"], modules3[module_key]["B"]
            s_vals, U, V = delta_w_singular_values(
                A2, B2, scaling2, A3, B3, scaling3, compute_vectors=True
            )
            saved_vectors[module_key] = {
                "singular_values": torch.tensor(s_vals),
                "U": torch.tensor(U),   # output-side directions, shape (out_dim, k)
                "V": torch.tensor(V),   # input-side directions, shape (in_dim, k)
            }
            print(f"  {module_key}: saved {s_vals.shape[0]} singular vector pairs")
        torch.save(saved_vectors, VECTORS_OUT)
        print(f"Saved top singular vectors for {len(top_modules)} modules to {VECTORS_OUT}")

    print("\n" + "=" * 70)
    print("delta_W LOCALIZATION -- % of total RL-induced weight-change energy")
    print("=" * 70)
    for row in by_type_rows:
        print(f"  {row['module_type']:<20s} {row['energy_share_pct']:>6.2f}%   ({row['n_modules']} modules)")
    mlp_share = sum(r["energy_share_pct"] for r in by_type_rows if "mlp" in r["module_type"])
    attn_share = sum(r["energy_share_pct"] for r in by_type_rows if "attn" in r["module_type"])
    print(f"\n  MLP total:  {mlp_share:.1f}%")
    print(f"  Attn total: {attn_share:.1f}%")

    print("\n" + "=" * 70)
    print("Top 10 modules by Frobenius norm (biggest RL-induced changes)")
    print("=" * 70)
    print(per_module_df.head(10)[
        ["module_key", "frobenius_norm", "top1_energy_share", "effective_rank"]
    ].to_string(index=False))

    max_possible_rank = r2 + r3
    print("\n" + "=" * 70)
    print(f"Effective rank distribution (1.0 = fully rank-1/coherent; "
          f"max possible = {max_possible_rank})")
    print("=" * 70)
    print(per_module_df["effective_rank"].describe().to_string())

    print(f"\nWrote:\n  {PER_MODULE_OUT}\n  {BY_TYPE_OUT}\n  {SINGVALS_OUT}")
    if COMPUTE_SINGULAR_VECTORS:
        print(f"  {VECTORS_OUT}")


if __name__ == "__main__":
    main()