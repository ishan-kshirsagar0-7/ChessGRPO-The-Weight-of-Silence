"""
train_grpo.py — GRPO v2: Rung 1 (gated legality + dense centipawn reward)
==============================================================================
Vanilla HF/PEFT/TRL, no Unsloth. Changes vs v1:
  - Single chess_reward (gate + dense Stockfish cp), replacing the 5-fn stack
  - beta=0.02 (small KL leash, was 0.0)
  - Loads grpo_training_data_evals.csv (has precomputed best_eval_cp column)
  - OUTPUT_DIR=grpo_v2_checkpoints (v1 artifacts in grpo_checkpoints untouched)
Everything else deliberately identical to the last healthy run: one variable
changed (the reward), so when results move we know why.
"""

import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from rewards import FEN_TO_BEST_EVAL, chess_reward, shutdown_engine


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL = "unsloth/qwen3-14b-bnb-4bit"
SFT_ADAPTER_PATH = "qwen3_14b_sft_checkpoints/final_model_lora"
TRAINING_DATA = "grpo_training_data_evals.csv"
OUTPUT_DIR = "grpo_v2_checkpoints"
SEED = 3407
EVAL_SET_SIZE = 100

SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA + SPLIT + POPULATE REWARD LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

print("Loading training data (with precomputed best-move evals)...")
df = pd.read_csv(TRAINING_DATA)

n_before = len(df)
df = df.dropna(subset=["best_eval_cp"]).reset_index(drop=True)
n_dropped = n_before - len(df)
if n_dropped > 0:
    print(f"  Dropped {n_dropped} rows with missing best_eval_cp.")

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

eval_df = df.iloc[:EVAL_SET_SIZE].reset_index(drop=True)
train_df = df.iloc[EVAL_SET_SIZE:].reset_index(drop=True)
print(f"  Train rows: {len(train_df)}")
print(f"  Eval rows:  {len(eval_df)}")

for _, row in df.iterrows():
    FEN_TO_BEST_EVAL[row["FEN"]] = int(row["best_eval_cp"])
print(f"  {len(FEN_TO_BEST_EVAL)} FEN -> best_eval_cp mappings loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD MODEL VANILLA (no Unsloth)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nLoading base model {BASE_MODEL} in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Prepare for k-bit training (enables grad checkpointing, layer norm fp32, etc.)
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Force EOS to <|im_end|> for Qwen3 chat-format generation termination
_im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
if _im_end_id is not None and _im_end_id != tokenizer.unk_token_id:
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = _im_end_id
    print(f"EOS forced to <|im_end|> (id: {_im_end_id})")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"\nLoading SFT LoRA adapter from {SFT_ADAPTER_PATH}...")
model = PeftModel.from_pretrained(
    base_model,
    SFT_ADAPTER_PATH,
    is_trainable=True,
)
print("Base + SFT LoRA loaded (vanilla path). Fresh GRPO starts from SFT, "
      "NOT from the v1 GRPO adapter (v1 damaged move quality).")

# Sync model generation config with tokenizer EOS
if hasattr(model, "generation_config") and _im_end_id is not None:
    model.generation_config.eos_token_id = _im_end_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

_orig_generate = model.generate

def _eval_mode_generate(*args, **kwargs):
    _was_training = model.training
    model.eval()
    kwargs["use_cache"] = True  # safe in eval mode; restores fast cached generation
    try:
        return _orig_generate(*args, **kwargs)
    finally:
        if _was_training:
            model.train()

model.generate = _eval_mode_generate
print("Wrapped generate() to run in eval mode + KV cache (fixes GC-during-generation corruption).")

# Sanity check: lora_B norms should be > 0 if SFT preserved
print("\nLoRA weight sanity check (lora_B norms should be > 0):")
shown = 0
for name, param in model.named_parameters():
    if "lora_B" in name:
        norm = param.data.norm().item()
        print(f"  {name}: norm = {norm:.6f}")
        shown += 1
        if shown >= 4:
            break
print("  (If all norms are 0.0, SFT weights were reset.)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD PROMPTS — match SFT format byte-for-byte
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(fen: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nFEN: {fen}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

print("\nPreparing prompts...")
train_prompts = [{"prompt": build_prompt(r["FEN"])} for _, r in train_df.iterrows()]
eval_prompts  = [{"prompt": build_prompt(r["FEN"])} for _, r in eval_df.iterrows()]
train_dataset = Dataset.from_list(train_prompts)
eval_dataset  = Dataset.from_list(eval_prompts)
print(f"  {len(train_dataset)} train prompts, {len(eval_dataset)} eval prompts.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONFIGURE GRPO
# ─────────────────────────────────────────────────────────────────────────────

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    beta=0.02,                          # was 0.0 — small KL leash as insurance
    reward_weights=[1.0],               # ONE reward. Nothing to balance.
    num_generations=4,
    temperature=0.7,
    max_completion_length=512,
    max_prompt_length=384,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_steps=500,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=250,
    per_device_eval_batch_size=4,
    logging_steps=1,
    save_steps=100,
    save_total_limit=3,
    report_to="tensorboard",
    seed=SEED,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD TRAINER
# ─────────────────────────────────────────────────────────────────────────────

if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    reward_funcs=[chess_reward],
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN
# ─────────────────────────────────────────────────────────────────────────────

# Defensive: keep the trainer's GenerationConfig EOS in sync (harmless, cheap)
if hasattr(trainer, 'generation_config'):
    trainer.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.generation_config.pad_token_id = tokenizer.pad_token_id
    print(f"Patched trainer.generation_config: eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}")

last_ckpt = None
if os.path.isdir(OUTPUT_DIR):
    last_ckpt = get_last_checkpoint(OUTPUT_DIR)
    if last_ckpt is not None:
        print(f"\nResuming from checkpoint: {last_ckpt}")

print(f"\nStarting GRPO v2 Training (Rung 1: gated + dense centipawn reward)...")
print(f"  Beta (KL):           {training_args.beta}")
print(f"  Num generations:     {training_args.num_generations}")
print(f"  Max steps:           {training_args.max_steps}")
print(f"  Learning rate:       {training_args.learning_rate}")
print(f"  Batch size:          {training_args.per_device_train_batch_size}")
print(f"  Eval every:          {training_args.eval_steps} steps")
print(f"  Reward:              chess_reward (gate + exp(-cp_loss/120), depth 12)")
print()

try:
    trainer.train(resume_from_checkpoint=last_ckpt)
finally:
    shutdown_engine()


# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────────────────────────────────────

print("\nSaving final GRPO v2 model...")
final_path = os.path.join(OUTPUT_DIR, "final_model_lora")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"  Saved to: {final_path}")
print("Done!")