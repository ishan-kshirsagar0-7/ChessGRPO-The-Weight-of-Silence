# ChessGRPO: The Weight of Silence

[![arXiv](https://img.shields.io/badge/arXiv-2607.20952-b31b1b.svg)](https://arxiv.org/abs/2607.20952)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21619703-blue.svg)](https://doi.org/10.5281/zenodo.21619703)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code accompanying *"The Weight of Silence: A Causal Case for Weights Over the Scratchpad in Latent Chess Reasoning."*

## Summary

I trained a chess-playing language model through a staged latent-reasoning curriculum, then applied reinforcement learning on top of it. The model's legal-move rate improved substantially, and it stopped fabricating checkmates entirely. I then causally tested whether these gains depended on the model actually using its silent "thoughts" during inference.

They did not. Substituting the thought vectors with a fixed average, replacing them with matched noise, or removing them altogether left performance largely unchanged. Only zeroing the vectors outright caused a collapse, and that reflects an out-of-distribution failure rather than evidence that the content of the thoughts was load-bearing. The improvement from reinforcement learning is encoded in the model's weights, not in computation the model performs at inference time.

This repository contains the code behind that result: the three-stage training ladder, the six-condition causal intervention suite, and the interpretability analysis identifying where in the model the change actually occurred.

## Headline Results

| Model | Legal move rate | Accuracy | False checkmates (out of ~100) |
|---|---|---|---|
| SFT (imitation baseline) | 38% | 9% | 28 |
| Rung 1: explicit chain-of-thought + RL | 52% | 10% | 0 |
| Rung 2: latent curriculum, no RL | 48% | 10% | 19 |
| **Rung 3: latent curriculum + RL** | **61%** | 9% | **0** |

Format compliance is 100% across every model listed above. Accuracy, measured as an exact match to Stockfish's top move, remains near chance throughout, by design. This is a result about legality and confabulation, not about chess strength. That distinction is addressed directly below.

## The Confabulation Cutoff

The SFT baseline hallucinates checkmates on 28 of roughly 100 held-out positions: it plays a move and then reports a win that did not occur. Reinforcement learning eliminates this behavior entirely, independently, across two distinct training recipes (28 to 19 to 0 across imitation, a latent curriculum without RL, and RL on top of that curriculum; separately, 28 to 0 on an explicit chain-of-thought plus RL run). Two unrelated paths converge on the same outcome: whatever objective RL is optimizing, elimination of false-checkmate claims follows as a consequence.

## Templated Confabulation

Before this behavior disappears, the false-checkmate explanations are not random. Cross-board text similarity between hallucinated explanations on unrelated positions measures 0.68, compared to 0.27 for explanations accompanying correct moves. The model is not reasoning its way to an incorrect conclusion on each board individually. It is reusing near-identical phrasing regardless of the position, which indicates the underlying "reasoning" in these failure cases was never board-specific.

## Testing Whether the Thoughts Matter

The central method is a six-condition causal intervention suite, applied to the same model both before and after the RL stage:

- **Baseline**: the model's own generated thought vectors, unmodified.
- **Substitute**: the thoughts replaced with a fixed vector averaged over 30 unrelated reference boards.
- **Noise**: the thoughts replaced with random noise matched in magnitude to the originals.
- **Ablate**: the thought-generation step skipped entirely.
- **Zero**: the thoughts replaced with all-zero vectors.
- **Length-matched ablate**: a pad-token control, ruling out sequence length as a confound.

If the model were genuinely relying on its own thoughts, substituting or ablating them should degrade performance meaningfully. It does not. Only the zero-vector condition collapses performance, and this is attributable to the vectors falling entirely outside the model's training distribution rather than to the content of the thoughts being causally necessary. This result holds at n=100 and is confirmed at a full n=1,000 replication, with Bonferroni-corrected significance testing and paired McNemar tests across every condition pair.

## Where the Gain Actually Lives

Two additional findings support the same conclusion:

- **Representational collapse.** Reading the thought vectors back into words using a Jacobian lens, cross-board cosine similarity rises from 0.874 (base model) to 0.812 (SFT) to 0.992 by the end of the latent curriculum, with reinforcement learning contributing no further increase. The thoughts converge into a fixed, board-invariant scaffold well before the RL stage begins.
- **Weight localization.** Decomposing the RL-induced weight change (ΔW) between the pre-RL and post-RL checkpoints shows that 68.9% of it is concentrated in MLP projections, with `gate_proj` alone accounting for 34.8%. The update is localized to a specific, identifiable region of the network rather than distributed diffusely.

Taken together, these results indicate that reinforcement learning does not teach the model to reason more effectively at inference time. Instead, it encodes an improvement directly into the weights, while the latent "thinking" step that was assumed to be doing the work contributes little to the outcome.

## Scope

This is not a chess-strength paper. Accuracy against Stockfish's top move remains at approximately 9 to 10% across every checkpoint, consistent with prior null results reported for reinforcement learning on latent-reasoning setups elsewhere in the literature. This work does not produce a model that plays strong chess. It examines whether the model produces legal moves and reports the state of the board truthfully, independent of playing strength.

## Repository Structure

```
data_pipeline/            Builds and validates the structured chess-reasoning training data
                           from raw FEN positions and Stockfish evaluations.

sft/                       Supervised fine-tuning pipeline. Produces the imitation-learning
                           baseline on Qwen3-14B with LoRA.

rung1_explicit_grpo/       GRPO reinforcement learning applied to explicit, written-out
                           chain-of-thought reasoning. The explicit-reasoning control arm.

rung2_latent_curriculum/   The staged curriculum that gradually replaces written reasoning
                           with compressed latent thought vectors. No RL at this stage.

rung3_latent_grpo/         GRPO applied on top of the latent curriculum. Produces the
                           headline 61%-legal result, along with a second Gumbel-Softmax
                           variant of the same experiment.

interpretability/          Jacobian-lens fitting, the six-condition causal intervention
                           suite, ΔW weight-localization analysis, the confabulation-template
                           detector, and significance testing (Bonferroni, McNemar).
```

## Reproducing This Work

Requirements include `transformers`, `peft`, `trl`, `bitsandbytes`, `torch`, and `python-chess`, along with a local Stockfish binary for move evaluation. Training was conducted on a single 24GB GPU (A10G); the interpretability scripts are CPU-only where indicated in their file headers.

Complete hyperparameters, prompts, and training methodology are documented in the paper. This repository provides the implementation and is not a substitute for it.

## Models and Data

- **Trained model checkpoints (LoRA adapters):** Hugging Face, link forthcoming.
- **Training dataset (12,002 rows: FEN, best move, structured reasoning):** Kaggle, link forthcoming.

## Citation

```bibtex
@misc{kshirsagar2026weightofsilence,
  title         = {The Weight of Silence: A Causal Case for Weights Over the Scratchpad in Latent Chess Reasoning},
  author        = {Kshirsagar, Ishan S.},
  year          = {2026},
  eprint        = {2607.20952},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.20952}
}
```

A Zenodo-hosted copy of the paper is also available at [doi.org/10.5281/zenodo.21619703](https://doi.org/10.5281/zenodo.21619703).

## License

Released under the [MIT License](LICENSE).

## About

Ishan S. Kshirsagar, independent researcher.
Paper: [arxiv.org/abs/2607.20952](https://arxiv.org/abs/2607.20952)
