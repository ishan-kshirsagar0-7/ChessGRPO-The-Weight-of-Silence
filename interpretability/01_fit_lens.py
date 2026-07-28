"""
01_fit_lens.py  --  Fit a Jacobian lens on the 4-bit + rung-3 model (safe/fast).

Drives jlens's per-prompt Jacobian directly with a memory-safe dim_batch, a
short sequence length, and boilerplate-stripped chess prompts so every prompt
is dense with real chess content. Crash-proof: OOM on one prompt skips it
rather than killing the run. Shows a live tqdm ETA + a memory readout.
"""

import gc
import json
import logging
import os
import random
import time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import jlens
from jlens.fitting import jacobian_for_prompt
from jlens.lens import JacobianLens

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_MODEL   = "unsloth/qwen3-14b-bnb-4bit"
ADAPTER_PATH = os.path.expanduser(
    "~/g6e_prep/rung3_final_150/final_model_lora_150/policy"
)
DATASET      = os.path.expanduser("~/g6e_prep/parsed_dataset.jsonl")

N_CHESS      = 60           # chess prompts (+ generic below). Lower this to go faster.
SEED         = 3407
DIM_BATCH    = 48           # output dims per backward pass. Memory-safe on 44 GB.
                            # If prompt 1 reports lots of free GPU, raise toward 64.
MAX_SEQ_LEN  = 48           # short + boilerplate-stripped = dense chess content
SAVE_EVERY   = 30           # periodic partial-lens save (crash insurance)

LENS_OUT     = os.path.expanduser("~/g6e_prep/rung3_jlens.pt")
LOG_FILE     = os.path.expanduser("~/g6e_prep/fit_lens.log")

GENERIC_PROMPTS = [
    "The history of written language spans thousands of years, from simple "
    "pictographs to the complex alphabets we use every day.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into chemical "
    "energy, forming the base of nearly every food chain on Earth.",
    "Ocean currents carry enormous quantities of heat around the globe, shaping "
    "coastal climates and driving weather systems far from their origin.",
    "The printing press dramatically lowered the cost of books and helped spread "
    "literacy across Europe over the following centuries.",
    "A healthy immune system tells the body's own cells apart from invaders and "
    "mounts targeted responses against bacteria and viruses.",
    "Continental drift slowly rearranges the world's land masses, opening oceans "
    "and raising mountain ranges over millions of years.",
    "Classical music of the eighteenth century prized balance and clarity, built "
    "around clear themes and careful formal development.",
    "The water cycle moves moisture between oceans, atmosphere, and land through "
    "evaporation, condensation, precipitation, and runoff.",
    "Early astronomers tracked the planets across the night sky and slowly built "
    "a model of a solar system centered on the sun.",
    "Markets coordinate the decisions of countless individuals through prices, "
    "which signal scarcity and guide the flow of resources.",
    "The nervous system carries electrical and chemical signals between cells, "
    "letting organisms sense the world and react within milliseconds.",
    "Ancient trade routes linked distant civilizations, carrying goods, ideas, "
    "technologies, languages, and religions across whole continents.",
    "Volcanic eruptions can reshape landscapes in hours, burying the land in ash "
    "and releasing gases that shift the climate for years.",
    "The scientific method forms hypotheses, tests them by careful experiment, "
    "and revises theories in light of the resulting evidence.",
    "Bridges must withstand not only their own weight but also traffic, wind, "
    "temperature swings, and the slow fatigue of repeated stress.",
    "Human memory is reconstructive rather than perfectly recorded, reshaped by "
    "expectation and new information each time it is recalled.",
    "Agriculture let early societies settle in one place, supporting larger "
    "populations and eventually the growth of the first cities.",
    "Light behaves as both a wave and a stream of particles, a duality at the "
    "very heart of modern quantum physics.",
    "Democracies depend on the peaceful transfer of power, the rule of law, and "
    "the protection of minorities against temporary majorities.",
    "The digestive system breaks food into absorbable nutrients using mechanical "
    "grinding and a cascade of specialized enzymes.",
]


def build_logger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    for h in (logging.FileHandler(LOG_FILE), logging.StreamHandler()):
        h.setFormatter(fmt)
        root.addHandler(h)
    return logging.getLogger("fit_lens")


def reconstruct_chess_prompt(row):
    """Dense chess text: FEN + reasoning body + move, no system-prompt boilerplate."""
    try:
        sections = json.loads(row.get("section_texts") or "[]")
    except (json.JSONDecodeError, TypeError):
        sections = []
    body = (row.get("preamble") or "") + "\n".join(sections) + (row.get("conclusion") or "")
    return f"FEN: {row['fen']}\n{body}\nMove: {row['best_move']}"


def main():
    log = build_logger()

    with open(DATASET) as f:
        rows = [json.loads(line) for line in f]
    random.seed(SEED)
    picked = random.sample(rows, min(N_CHESS, len(rows)))

    log.info("loading %s in 4-bit + rung-3 adapter", BASE_MODEL)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, dtype=torch.bfloat16, device_map="auto",
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    peft_model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    peft_model.eval()
    inner = peft_model.base_model.model
    jm = jlens.from_hf(inner, tok)

    chess_prompts = [reconstruct_chess_prompt(r) for r in picked]
    corpus = chess_prompts + GENERIC_PROMPTS
    random.shuffle(corpus)

    n_layers, d_model = jm.n_layers, jm.d_model
    target_layer = n_layers - 1
    source_layers = list(range(target_layer))
    n_passes = -(-d_model // DIM_BATCH)
    log.info("corpus: %d chess + %d generic = %d prompts | dim_batch=%d -> %d "
             "passes/prompt | max_seq_len=%d",
             len(chess_prompts), len(GENERIC_PROMPTS), len(corpus),
             DIM_BATCH, n_passes, MAX_SEQ_LEN)

    jac_sum = {l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in source_layers}
    n_done = 0

    def save_partial():
        mean = {l: jac_sum[l] / n_done for l in source_layers}
        JacobianLens(jacobians=mean, n_prompts=n_done, d_model=d_model).save(LENS_OUT)

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    bar = tqdm(corpus, desc="fitting lens", unit="prompt")
    for i, prompt in enumerate(bar):
        t0 = time.time()
        try:
            per_J, seq_len, n_valid = jacobian_for_prompt(
                jm, prompt, source_layers,
                target_layer=target_layer, dim_batch=DIM_BATCH, max_seq_len=MAX_SEQ_LEN,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            bar.write(f"  OOM on prompt {i}, skipped and freed cache")
            continue
        except ValueError as exc:
            bar.write(f"  skip prompt {i}: {exc}")
            continue

        for l in source_layers:
            jac_sum[l] += per_J[l]
        n_done += 1
        bar.set_postfix(sec=f"{time.time()-t0:.0f}", valid=n_valid)

        if i == 0:
            mem = torch.cuda.max_memory_allocated() / 1e9
            per = time.time() - t0
            bar.write(f"  [prompt 1] {per:.0f}s | GPU {mem:.1f}/44 GB | "
                      f"full-run ETA ~{per * len(corpus) / 60:.0f} min | "
                      f"headroom to raise DIM_BATCH: {'yes' if mem < 34 else 'no'}")
        if n_done and n_done % SAVE_EVERY == 0:
            save_partial()

    if n_done == 0:
        raise SystemExit("no prompts were long enough to fit on")

    save_partial()
    peak = torch.cuda.max_memory_allocated() / 1e9
    log.info("DONE: %d prompts in %.1f min | peak GPU %.1f GB | saved -> %s",
             n_done, (time.time() - t_start) / 60, peak, LENS_OUT)

    lens = JacobianLens.load(LENS_OUT)
    ll, _, _ = lens.apply(jm, chess_prompts[0], positions=[-2])
    layers = sorted(ll.keys())
    for L in layers[:: max(1, len(layers) // 6)]:
        top5 = [tok.decode([t]) for t in ll[L][0].topk(5).indices]
        log.info("  layer %2d top5: %s", L, top5)
    log.info("fit complete.")


if __name__ == "__main__":
    main()