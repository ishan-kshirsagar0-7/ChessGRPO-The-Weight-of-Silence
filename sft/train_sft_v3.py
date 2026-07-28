"""
train_sft_v2.py — SFT Training for Chess Reasoning Model (V2)
===============================================================
Trains on the clean V2 dataset with:
  - unsloth/Qwen3-14B-bnb-4bit via Unsloth + LoRA
  - Loss masking on system/user turns (train_on_responses_only)
  - Cosine LR scheduler, gentler hyperparameters than V1

If 14B OOMs, change MODEL_NAME to the 7B variant and BATCH_SIZE to 2.
"""

import torch
import os
from unsloth import FastLanguageModel
from unsloth import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # Files
    DATASET_FILE = "sft_ready_dataset.csv"
    OUTPUT_DIR = "sft_v2_checkpoints"

    # Model — Try 7B first. If OOM, switch to 3B.
    # 14B: "unsloth/Qwen3-14B-bnb-4bit"
    # 7B: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    # 3B: "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    MODEL_NAME = "unsloth/Qwen3-14B-bnb-4bit"
    MAX_SEQ_LENGTH = 2048
    DTYPE = None            # Auto-detect (Float16/Bfloat16)
    LOAD_IN_4BIT = True

    # Training
    EPOCHS = 1              # Start with 1, evaluate, then decide on 2nd
    BATCH_SIZE = 2          # 1 for 7B (VRAM constrained), 2 for 3B
    GRAD_ACCUMULATION = 4   # Effective batch size = 2 * 4 = 8
    LEARNING_RATE = 1e-4    # Half of V1's 2e-4 — gentler
    WARMUP_STEPS = 10       # V1 had 5, this is smoother
    SEED = 3407

config = Config()


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MODEL & TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading {config.MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.MODEL_NAME,
    max_seq_length=config.MAX_SEQ_LENGTH,
    dtype=config.DTYPE,
    load_in_4bit=config.LOAD_IN_4BIT,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=config.SEED,
    use_rslora=False,
    loftq_config=None,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD & PREPARE DATASET
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading dataset from {config.DATASET_FILE}...")
dataset = load_dataset("csv", data_files=config.DATASET_FILE, split="train")
print(f"  {len(dataset)} examples loaded.")

# The 'text' column already contains the full ChatML-formatted string
# from prep_sft_data.py — no further formatting needed.


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD TRAINER
# ─────────────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=config.MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,

    args=TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION,
        warmup_steps=config.WARMUP_STEPS,
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=config.SEED,
        save_strategy="epoch",
        save_total_limit=2,
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. APPLY LOSS MASKING
# ─────────────────────────────────────────────────────────────────────────────

# This zeros out loss on system + user tokens.
# Model only learns from the assistant response (reasoning + move).
# Qwen2.5-Instruct ChatML boundary tokens:
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nStarting SFT Training...")
print(f"  Model: {config.MODEL_NAME}")
print(f"  Epochs: {config.EPOCHS}")
print(f"  Effective batch size: {config.BATCH_SIZE * config.GRAD_ACCUMULATION}")
print(f"  Learning rate: {config.LEARNING_RATE}")
print(f"  Scheduler: cosine")
print(f"  Max seq length: {config.MAX_SEQ_LENGTH}")
print()

trainer_stats = trainer.train()


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE
# ─────────────────────────────────────────────────────────────────────────────

print("\nSaving final model...")
final_path = os.path.join(config.OUTPUT_DIR, "final_model_lora")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

print(f"\nDone!")
print(f"  Model saved to: {final_path}")
print(f"  Training time: {trainer_stats.metrics['train_runtime']:.0f} seconds")
print(f"  Final loss: {trainer_stats.metrics['train_loss']:.4f}")