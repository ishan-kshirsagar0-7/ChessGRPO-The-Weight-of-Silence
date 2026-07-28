"""
prep_sft_data.py — Prepare the V2 reasoning dataset for SFT training.
=======================================================================
Reads chess_reasoning_v2_dataset.csv and outputs sft_ready_dataset.csv with:
  1. Normalized reasoning (consistent newlines between sections)
  2. Full ChatML-formatted text column ready for SFT
"""

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV = "chess_reasoning_v2_dataset.csv"
OUTPUT_CSV = "sft_ready_dataset.csv"

# This system prompt MUST be identical in: SFT training, GRPO training, and inference.
# Keep it short. The model learns format from examples, not from the prompt.
SYSTEM_PROMPT = """You are a Chess Grandmaster. Given a FEN string, analyze the position and determine the best move.

Format:
1. Reason inside <thinking> tags using [KING SAFETY], [CHECKS], [CAPTURES & TRADES], [THREATS], [IMPROVEMENT] sections.
2. Output the move in UCI format inside <output> tags."""


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_reasoning(text):
    """
    Normalize reasoning text to have consistent formatting:
    - Single newline between each bracketed section
    - No double/triple newlines
    - No leading/trailing whitespace per line
    - No blank lines within sections
    """
    if not isinstance(text, str):
        return ""

    # Split into lines and strip each
    lines = [line.strip() for line in text.split("\n")]

    # Remove empty lines
    lines = [line for line in lines if line]

    # Rejoin with single newlines
    normalized = "\n".join(lines)

    return normalized


def format_chat_example(fen, move, reasoning):
    """
    Build the full ChatML-formatted training example.
    Returns the raw string that tokenizer.apply_chat_template would produce.

    We use Qwen2.5-Instruct's ChatML format:
        <|im_start|>system\n{system}\n<|im_end|>
        <|im_start|>user\n{user}\n<|im_end|>
        <|im_start|>assistant\n{assistant}\n<|im_end|>
    """
    assistant_content = f"<thinking>\n{reasoning}\n</thinking>\n<output>\n{move}\n</output>"

    # I store the messages as a structured format.
    # The actual ChatML formatting will be done by tokenizer.apply_chat_template
    # during training. Here I just store the components.
    return {
        "system": SYSTEM_PROMPT,
        "user": f"FEN: {fen}",
        "assistant": assistant_content,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df)} rows.")

    # Normalize reasoning
    print("Normalizing reasoning text...")
    df["Reasoning"] = df["Reasoning"].apply(normalize_reasoning)

    # Check for any empty reasonings after normalization
    empty = df["Reasoning"].str.len() == 0
    if empty.any():
        print(f"  WARNING: {empty.sum()} rows have empty reasoning after normalization. Dropping them.")
        df = df[~empty]

    # Build ChatML-formatted text using tokenizer-compatible format
    print("Formatting into ChatML structure...")

    texts = []
    for _, row in df.iterrows():
        fen = row["FEN"]
        move = row["Best Move"]
        reasoning = row["Reasoning"]

        assistant_content = f"<thinking>\n{reasoning}\n</thinking>\n<output>\n{move}\n</output>"

        # Qwen2.5 ChatML format — I write it directly so there's no
        # dependency on having the tokenizer installed in this script.
        # train_sft_v2.py will use tokenizer.apply_chat_template instead,
        # but I need this for inspection and verification.
        chatml = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nFEN: {fen}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
        )
        texts.append(chatml)

    df["text"] = texts

    # Stats
    token_estimates = df["text"].str.len() / 4  # rough estimate: 4 chars per token
    print(f"\n  Rows: {len(df)}")
    print(f"  Avg estimated tokens per example: {token_estimates.mean():.0f}")
    print(f"  Max estimated tokens per example: {token_estimates.max():.0f}")
    print(f"  Min estimated tokens per example: {token_estimates.min():.0f}")

    # Verify a sample
    print(f"\n{'='*70}")
    print("SAMPLE OUTPUT (first row):")
    print(f"{'='*70}")
    print(texts[0])

    # Save
    # I only need the 'text' column for SFT, but keep id/FEN/Best Move for reference
    output_df = df[["id", "FEN", "Best Move", "text"]]
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(output_df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()