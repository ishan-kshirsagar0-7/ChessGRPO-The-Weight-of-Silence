"""
delta_w_analysis.py — Characterizes where reinforcement learning's effect lives in
the LoRA weights, and whether it's structurally similar across two independently
trained RL deltas (Stage-2->Rung-3 through the latent bottleneck, vs SFT->Rung-1-v2
through explicit reasoning). Directly answers the two questions named in this
paper's own Future Work section. No GPU needed -- rank-16 matmuls on CPU.

Verifies key-set identity across every pair before computing anything (this
project's "verify don't assume" convention) -- config JSONs already confirmed
identical r/alpha/target_modules across all four checkpoints, so no rank
normalization is needed here.
"""

import os
import torch
from safetensors import safe_open
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────
# BASE = r"D:\ChessGRPO_v2\04_huggingface_release"
BASE = "/mnt/d/ChessGRPO_v2/04_huggingface_release"
STAGE2_PATH  = os.path.join(BASE, "rung2-stage2-latent-chess", "adapter_model.safetensors")
RUNG3_PATH   = os.path.join(BASE, "rung3-latent-grpo-chess", "adapter_model.safetensors")
SFT_PATH     = os.path.join(BASE, "sft-qwen3-14b-chess", "adapter_model.safetensors")
RUNG1V2_PATH = os.path.join(BASE, "rung1-grpo-v2-explicit-chess", "adapter_model.safetensors")

# OUT_DIR = r"D:\ChessGRPO_v2\06_results_and_eval\interpretability"
OUT_DIR = "/mnt/d/ChessGRPO_v2/06_results_and_eval/interpretability"
OUT_LAYERWISE   = os.path.join(OUT_DIR, "delta_w_layerwise.csv")
OUT_MODULETYPE  = os.path.join(OUT_DIR, "delta_w_by_module_type.csv")
OUT_PERCELL     = os.path.join(OUT_DIR, "delta_w_cross_similarity_percell.csv")
LORA_R = 16
LORA_ALPHA = 16
SCALING = LORA_ALPHA / LORA_R  # = 1.0 here; kept explicit rather than hardcoded to 1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}


def discover_layer_module_pairs(path):
    """Parses every lora_A key in a safetensors file into (layer, module) pairs,
    without assuming a fixed layer count -- reads it from the actual keys."""
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
    pairs = set()
    for k in keys:
        if not k.endswith(".lora_A.weight"):
            continue
        # base_model.model.model.layers.{N}.{mlp|self_attn}.{module}.lora_A.weight
        parts = k.split(".")
        layer_idx = int(parts[parts.index("layers") + 1])
        module = parts[-3]  # the proj name, immediately before "lora_A"/"lora_B"
        pairs.add((layer_idx, module))
    return pairs, keys


def get_module_name_from_key(key):
    # e.g. "...layers.0.mlp.down_proj.lora_A.weight" -> "down_proj"
    return key.split(".")[-3]


def load_lora_pair(handle, layer, module):
    """Finds the actual key path for a given (layer, module) -- handles the
    mlp/self_attn branch difference -- and returns (A, B) as float32 tensors."""
    branch = "mlp" if module in MLP_MODULES else "self_attn"
    prefix = f"base_model.model.model.layers.{layer}.{branch}.{module}"
    A = handle.get_tensor(f"{prefix}.lora_A.weight").float()
    B = handle.get_tensor(f"{prefix}.lora_B.weight").float()
    return A, B


def verify_pair(path_a, path_b, name_a, name_b):
    """Confirms both checkpoints expose the identical set of (layer, module)
    LoRA pairs before anything downstream trusts that assumption."""
    pairs_a, keys_a = discover_layer_module_pairs(path_a)
    pairs_b, keys_b = discover_layer_module_pairs(path_b)
    print(f"{name_a}: {len(keys_a)} tensors, {len(pairs_a)} (layer, module) pairs.")
    print(f"{name_b}: {len(keys_b)} tensors, {len(pairs_b)} (layer, module) pairs.")
    if pairs_a != pairs_b:
        missing_in_b = pairs_a - pairs_b
        missing_in_a = pairs_b - pairs_a
        raise ValueError(
            f"(layer, module) sets differ between {name_a} and {name_b}. "
            f"In {name_a} but not {name_b}: {missing_in_b}. "
            f"In {name_b} but not {name_a}: {missing_in_a}. Stop and inspect."
        )
    print(f"  -> (layer, module) sets identical. Proceeding.\n")
    return sorted(pairs_a)


def compute_delta(handle_new, handle_old, layer, module):
    """Reconstructs the LoRA delta at one (layer, module): scaling * B @ A,
    then the RL-induced change = delta_new - delta_old. Full dense matrix at
    this single (layer, module) is at most a few hundred MB and is discarded
    immediately after -- never accumulated across the loop."""
    A_new, B_new = load_lora_pair(handle_new, layer, module)
    A_old, B_old = load_lora_pair(handle_old, layer, module)
    W_new = SCALING * (B_new @ A_new)
    W_old = SCALING * (B_old @ A_old)
    return W_new - W_old  # the RL-induced weight change at this slot


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 78)
    print("VERIFICATION")
    print("=" * 78)
    pairs_rung3 = verify_pair(STAGE2_PATH, RUNG3_PATH, "Stage-2", "Rung-3")
    pairs_rung1 = verify_pair(SFT_PATH, RUNG1V2_PATH, "SFT", "Rung-1-v2")
    if set(pairs_rung3) != set(pairs_rung1):
        raise ValueError("The two ladders don't share the same (layer, module) slots -- stop and inspect.")
    print(f"Both ladders share the same {len(pairs_rung3)} (layer, module) slots. Proceeding.\n")
    print("=" * 78 + "\n")

    with safe_open(STAGE2_PATH, framework="pt") as h_stage2, \
         safe_open(RUNG3_PATH, framework="pt") as h_rung3, \
         safe_open(SFT_PATH, framework="pt") as h_sft, \
         safe_open(RUNG1V2_PATH, framework="pt") as h_rung1v2:

        rows = []
        with torch.no_grad():
            for layer, module in pairs_rung3:
                dW_rung3 = compute_delta(h_rung3, h_stage2, layer, module)
                dW_rung1 = compute_delta(h_rung1v2, h_sft, layer, module)

                norm_rung3 = torch.norm(dW_rung3, p="fro").item()
                norm_rung1 = torch.norm(dW_rung1, p="fro").item()

                inner = torch.sum(dW_rung3 * dW_rung1).item()
                denom = norm_rung3 * norm_rung1
                cos_sim = inner / denom if denom > 1e-12 else float("nan")

                rows.append({
                    "layer": layer,
                    "module": module,
                    "branch": "mlp" if module in MLP_MODULES else "attn",
                    "norm_rung3_delta": norm_rung3,
                    "norm_rung1_delta": norm_rung1,
                    "cosine_similarity": cos_sim,
                    "inner_product": inner,
                })

                del dW_rung3, dW_rung1  # explicit, since these can be a few hundred MB each

    percell = pd.DataFrame(rows).sort_values(["layer", "module"]).reset_index(drop=True)
    percell.to_csv(OUT_PERCELL, index=False)
    print(f"Wrote per-cell results ({len(percell)} rows) to {OUT_PERCELL}\n")

    # ── Layer-wise aggregation (sum norms across the 7 modules per layer) ──
    layerwise = percell.groupby("layer").agg(
        total_norm_rung3=("norm_rung3_delta", "sum"),
        total_norm_rung1=("norm_rung1_delta", "sum"),
        mean_cosine_sim=("cosine_similarity", "mean"),
    ).reset_index()
    layerwise.to_csv(OUT_LAYERWISE, index=False)
    print("=" * 78)
    print("LAYER-WISE: where does the Rung-3 delta concentrate?")
    print("=" * 78)
    print(layerwise.to_string(index=False))
    print(f"\nWrote {OUT_LAYERWISE}\n")

    # ── Module-type aggregation (sum norms across the 40 layers per module) ──
    moduletype = percell.groupby("module").agg(
        total_norm_rung3=("norm_rung3_delta", "sum"),
        total_norm_rung1=("norm_rung1_delta", "sum"),
        mean_cosine_sim=("cosine_similarity", "mean"),
    ).reset_index()
    moduletype["pct_of_rung3_total"] = 100 * moduletype["total_norm_rung3"] / moduletype["total_norm_rung3"].sum()
    moduletype = moduletype.sort_values("pct_of_rung3_total", ascending=False)
    moduletype.to_csv(OUT_MODULETYPE, index=False)
    print("=" * 78)
    print("MODULE-TYPE: attention (q/k/v/o) vs MLP (gate/up/down) -- Rung-3 delta")
    print("=" * 78)
    print(moduletype.to_string(index=False))
    print(f"\nWrote {OUT_MODULETYPE}\n")

    attn_total = percell[percell["branch"] == "attn"]["norm_rung3_delta"].sum()
    mlp_total = percell[percell["branch"] == "mlp"]["norm_rung3_delta"].sum()
    print(f"Attention total: {attn_total:.2f} ({100*attn_total/(attn_total+mlp_total):.1f}%)")
    print(f"MLP total:       {mlp_total:.2f} ({100*mlp_total/(attn_total+mlp_total):.1f}%)")

    # ── Global structural-similarity number: the single citable figure ──
    total_inner = percell["inner_product"].sum()
    total_norm_rung3 = (percell["norm_rung3_delta"] ** 2).sum() ** 0.5
    total_norm_rung1 = (percell["norm_rung1_delta"] ** 2).sum() ** 0.5
    global_cos_sim = total_inner / (total_norm_rung3 * total_norm_rung1)

    print("\n" + "=" * 78)
    print("GLOBAL STRUCTURAL SIMILARITY: Rung-1 delta vs Rung-3 delta")
    print("=" * 78)
    print(f"Whole-adapter cosine similarity: {global_cos_sim:.4f}")
    print(f"Mean per-cell cosine similarity: {percell['cosine_similarity'].mean():.4f}")
    print(f"Median per-cell cosine similarity: {percell['cosine_similarity'].median():.4f}")
    print(f"Per-cell cosine similarity range: [{percell['cosine_similarity'].min():.4f}, "
          f"{percell['cosine_similarity'].max():.4f}]")
    print("=" * 78)


if __name__ == "__main__":
    main()