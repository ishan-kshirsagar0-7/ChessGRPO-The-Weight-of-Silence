"""
rewards.py — Rung 1 reward: gated legality + dense Stockfish centipawn quality
================================================================================
ONE reward function, weight 1.0.

Design:
  GATE (hard fail, -1.0): missing/junk <output> tag, unparseable FEN,
    illegal move, unparseable move, or false CHECKMATE claim.
  Behind the gate: true CHECKMATE claim -> +1.0.
    Legal move -> exp(-cp_loss / K), where
    cp_loss = max(0, best_eval_cp - model_move_eval), both evals produced by
    the IDENTICAL procedure (push move, analyse at depth 12 single-threaded,
    score from mover POV, mate_score 1500). best_eval_cp is precomputed by
    precompute_evals.py; model_move_eval is computed live here.

  Reward range: -1.0 (gate fail) or (0.0, 1.0] (legal, quality-scaled).
  A 50cp inaccuracy earns ~0.66, 200cp mistake ~0.19, 500cp blunder ~0.02.

Engine: one persistent single-threaded Stockfish process, lazily opened,
restarted once on error. If an eval fails twice, the move earns
ENGINE_FALLBACK (small positive: it did pass the legality gate) and the run
continues. Never crash a 12-hour run over one flaky eval.

Test with: python rewards.py  (requires stockfish installed)
"""

import math
import re
import shutil

import chess
import chess.engine


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL LOOKUP — populated at training script startup
# ─────────────────────────────────────────────────────────────────────────────

FEN_TO_BEST_EVAL = {}  # populated by train_grpo.py: { fen_string: best_eval_cp (int) }


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

STOCKFISH_PATH = shutil.which("stockfish") or "/usr/games/stockfish"
DEPTH = 12
MAX_SECONDS_PER_EVAL = 10.0
MATE_SCORE = 1500
ENGINE_THREADS = 1
ENGINE_HASH_MB = 256

K = 120.0              # exp decay constant for cp_loss -> reward
GATE_FAIL = -1.0       # illegal / unparseable / false CHECKMATE
ENGINE_FALLBACK = 0.1  # legal move but Stockfish failed twice


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE LIFECYCLE — one persistent process, restart on failure
# ─────────────────────────────────────────────────────────────────────────────

_engine = None


def _open_engine():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": ENGINE_THREADS, "Hash": ENGINE_HASH_MB})
    return engine


def _get_engine():
    global _engine
    if _engine is None:
        _engine = _open_engine()
    return _engine


def _restart_engine():
    global _engine
    try:
        if _engine is not None:
            _engine.quit()
    except Exception:
        pass
    _engine = _open_engine()
    return _engine


def shutdown_engine():
    """Call once at the end of training."""
    global _engine
    try:
        if _engine is not None:
            _engine.quit()
    except Exception:
        pass
    _engine = None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_move(completion: str) -> str | None:
    """Extract the move string from <output>...</output> tags."""
    match = re.search(r"<output>\s*(.*?)\s*</output>", completion, re.DOTALL)
    if not match:
        return None
    move_text = match.group(1).strip()
    # Anti-hack: reject if contains newlines or multiple words (junk)
    if "\n" in move_text or len(move_text.split()) > 1:
        return None
    if len(move_text) == 0:
        return None
    return move_text


def extract_fen(prompt: str) -> str | None:
    """Extract FEN string from the prompt."""
    match = re.search(r"FEN:\s*(.+)", prompt)
    if not match:
        return None
    fen = match.group(1).strip()
    # Strip any ChatML tokens that got captured
    fen = re.sub(r"<\|.*?\|>", "", fen).strip()
    return fen


def _eval_move_cp(fen: str, move: chess.Move) -> int:
    """
    Eval of the position AFTER pushing `move`, from the mover's POV.
    Raises on engine failure.
    """
    board = chess.Board(fen)
    mover = board.turn
    board.push(move)
    if board.is_checkmate():
        return MATE_SCORE  # the move delivers mate, no engine needed
    limit = chess.engine.Limit(depth=DEPTH, time=MAX_SECONDS_PER_EVAL)
    info = _get_engine().analyse(board, limit)
    return info["score"].pov(mover).score(mate_score=MATE_SCORE)


def _eval_move_cp_safe(fen: str, move: chess.Move) -> int | None:
    """One retry with a fresh engine, then give up (returns None)."""
    try:
        return _eval_move_cp(fen, move)
    except Exception:
        try:
            _restart_engine()
            return _eval_move_cp(fen, move)
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# THE REWARD
# ─────────────────────────────────────────────────────────────────────────────

def chess_reward(completions, prompts=None, **kwargs) -> list[float]:
    """
    Gated legality + dense centipawn quality.
    """
    # ── ONE-TIME DEBUG: sanity-check the first batch of a run ────────────────
    if not getattr(chess_reward, "_dbg", False):
        print("\n" + "=" * 70)
        print("FIRST GENERATION DEBUG (one-time)")
        print("=" * 70)
        print(f"completions in this batch: {len(completions)}")
        if prompts:
            print("\n--- PROMPT[0] (repr) ---")
            print(repr(prompts[0]))
        for i, c in enumerate(completions[:2]):
            print(f"\n--- COMPLETION {i}  (len={len(c)} chars) ---")
            print(repr(c[:3000]))
            print("--- END ---")
        print("=" * 70 + "\n", flush=True)
        chess_reward._dbg = True
    # ──────────────────────────────────────────────────────────────────────────

    scores = []
    for completion, prompt in zip(completions, prompts):
        fen = extract_fen(prompt)
        move_text = extract_move(completion)

        # GATE: parse failures
        if fen is None or move_text is None:
            scores.append(GATE_FAIL)
            continue

        try:
            board = chess.Board(fen)
        except Exception:
            scores.append(GATE_FAIL)
            continue

        # GATE: CHECKMATE claims resolve entirely at the gate
        if move_text == "CHECKMATE":
            scores.append(1.0 if board.is_checkmate() else GATE_FAIL)
            continue

        # GATE: move must parse and be legal
        try:
            move = chess.Move.from_uci(move_text)
        except ValueError:
            scores.append(GATE_FAIL)
            continue
        if move not in board.legal_moves:
            scores.append(GATE_FAIL)
            continue

        # ── through the gate: dense quality signal ──
        best_cp = FEN_TO_BEST_EVAL.get(fen)
        if best_cp is None:
            # FEN missing from precomputed evals; shouldn't happen since the
            # loader drops eval-less rows, but never crash the run over it.
            scores.append(ENGINE_FALLBACK)
            continue

        move_cp = _eval_move_cp_safe(fen, move)
        if move_cp is None:
            print(f"[chess_reward] Stockfish failed twice on {fen} / {move_text}, "
                  f"fallback {ENGINE_FALLBACK}", flush=True)
            scores.append(ENGINE_FALLBACK)
            continue

        cp_loss = max(0, best_cp - move_cp)
        scores.append(math.exp(-cp_loss / K))

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# TEST — gate behavior + dense scaling on known cases (needs stockfish)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    if not os.path.exists(STOCKFISH_PATH):
        print(f"Stockfish not found at {STOCKFISH_PATH}. Run: sudo apt install -y stockfish")
        sys.exit(1)

    test_cases = [
        {
            "prompt": "<|im_start|>user\nFEN: 3r1rk1/1qB1bppp/p3p3/1p1b4/8/1B3P2/PP2Q1PP/2R1R2K w - - 2 22<|im_end|>",
            "completion": "<thinking>\n[CHECKS]: c7d8 delivers check.\n</thinking>\n<output>\nc7d8\n</output>",
            "best_move": "c7d8",
            "description": "Legal + best move -> reward ~1.0",
        },
        {
            "prompt": "<|im_start|>user\nFEN: 6k1/1p3ppp/p7/8/8/1N6/P5PP/3r1R1K b - - 1 32<|im_end|>",
            "completion": "<thinking>\n[KING SAFETY]: In checkmate.\n</thinking>\n<output>\nCHECKMATE\n</output>",
            "best_move": "d1f1",
            "description": "False CHECKMATE -> gate fail -1.0",
        },
        {
            "prompt": "<|im_start|>user\nFEN: r2r4/p4pPp/3k4/1pp1NP2/8/1P6/6R1/2K5 w - - 0 35<|im_end|>",
            "completion": "<thinking>\n[CAPTURES]: g2g6 captures pawn.\n</thinking>\n<output>\ng2g6\n</output>",
            "best_move": "e5f7",
            "description": "Legal but worse move -> reward in (0, 1), scaled by cp_loss",
        },
        {
            "prompt": "<|im_start|>user\nFEN: 1rb1r1k1/pp3ppp/3P1n2/2q4B/8/1Q6/P5PP/5R1K w - - 7 23<|im_end|>",
            "completion": "<thinking>\n[CHECKS]: f5f8 delivers check.\n</thinking>\n<output>\nf5f8\n</output>",
            "best_move": "b3f7",
            "description": "Illegal move -> gate fail -1.0",
        },
        {
            "prompt": "<|im_start|>user\nFEN: 8/5k2/3K3p/1pp5/P4p2/7P/8/8 w - - 0 41<|im_end|>",
            "completion": "I don't know what to do here",
            "best_move": "a4a5",
            "description": "No tags -> gate fail -1.0",
        },
    ]

    # populate FEN_TO_BEST_EVAL live
    print("Precomputing best-move evals for test positions...")
    for tc in test_cases:
        fen = extract_fen(tc["prompt"])
        board = chess.Board(fen)
        try:
            best = chess.Move.from_uci(tc["best_move"])
        except ValueError:
            continue
        if best in board.legal_moves:
            FEN_TO_BEST_EVAL[fen] = _eval_move_cp(fen, best)

    print("\n" + "=" * 70)
    print("REWARD v2 UNIT TESTS (gated legality + dense centipawn)")
    print("=" * 70)

    prompts = [tc["prompt"] for tc in test_cases]
    completions = [tc["completion"] for tc in test_cases]
    chess_reward._dbg = True  # suppress the one-time debug dump in tests
    rewards = chess_reward(completions, prompts=prompts)

    for tc, r in zip(test_cases, rewards):
        print(f"\n{tc['description']}")
        print(f"  reward = {r:+.4f}")

    print("\nExpected shape:")
    print("  Test 1: ~ +1.0        (best move, cp_loss ~ 0)")
    print("  Test 2:   -1.0 exact  (false CHECKMATE, gate)")
    print("  Test 3:   between 0 and 1, well below test 1")
    print("  Test 4:   -1.0 exact  (illegal, gate)")
    print("  Test 5:   -1.0 exact  (no tags, gate)")

    shutdown_engine()

# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_move_legal(fen: str, move_uci: str) -> bool:
    """Check if a UCI move is legal in the given FEN position."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        return move in board.legal_moves
    except Exception:
        return False


def is_position_checkmate(fen: str) -> bool:
    """Check if the position is specifically checkmate (not stalemate or other game-over)."""
    try:
        board = chess.Board(fen)
        return board.is_checkmate()
    except Exception:
        return False
