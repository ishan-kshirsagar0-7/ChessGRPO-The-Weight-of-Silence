"""
diagnostic2.py — isolate why generation corrupts inside the GRPO training loop.
Loads the model, then generates one short completion under three conditions to pinpoint
whether gradient-checkpointing/no-cache, training mode, or the combination is what produces garbage.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

BASE_MODEL = "unsloth/qwen3-14b-bnb-4bit"
SFT_ADAPTER_PATH = "qwen3_14b_sft_checkpoints/final_model_lora"
MAX_NEW = 120

SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""

FEN = "5r1k/6p1/R6p/8/6Q1/3q1P1K/7P/8 w - - 5 37"


def build_prompt(fen: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nFEN: {fen}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


print("Loading exactly like train_grpo.py...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map="auto",
)
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
im_end = tok.convert_tokens_to_ids("<|im_end|>")
tok.eos_token = "<|im_end|>"
tok.eos_token_id = im_end
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH, is_trainable=True)
model.generation_config.eos_token_id = im_end
model.generation_config.pad_token_id = tok.pad_token_id

prompt = build_prompt(FEN)
inputs = tok(prompt, return_tensors="pt").to(model.device)


def run(label: str, training_mode: bool, grad_ckpt: bool):
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    else:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True

    model.train() if training_mode else model.eval()

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=True,
            temperature=0.7,
            eos_token_id=im_end,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(repr(text))


run("A) eval  + GC OFF + cache ON   (diagnostic-like, expect clean)", False, False)
run("B) eval  + GC ON  + cache OFF", False, True)
run("C) train + GC ON  + cache OFF  (matches TRL, expect garbage)", True, True)

print("\nDone.")