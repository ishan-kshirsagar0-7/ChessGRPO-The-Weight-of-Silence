"""
better_reasonings.py — Test the new reasoning generation pipeline on 5 FEN positions.
==================================================================================
Combines:
  1. Expanded fact extraction (python-chess)
  2. New airtight zero-shot prompt (KCCTI priority cascade with bracketed headers)
  3. Ollama API call (local or cloud model)
  4. Validation checks imported from validator.py

Configure MODEL_ID below to switch between local and cloud models.
Just open in VS Code and Ctrl+Alt+N to run.

IMPORTANT: validator.py must be in the same directory as this file.
"""

import chess
import re
import requests
from validator import validate_row


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Switch between local and cloud models:
#   Local:  "qwen2.5:7b"
#   Cloud:  "minimax-m2.7:cloud"  or  "gpt-oss:120b-cloud"
MODEL_ID = "qwen2.5:7b"

OLLAMA_URL = "http://localhost:11434/api/generate"
TEMPERATURE = 0.2
NUM_CTX = 4096


# ─────────────────────────────────────────────────────────────────────────────
# 5 TEST CASES — covering all major failure modes from validator results
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "label": "1. IN CHECK — must respond to Qh4 check",
        "fen": "rnb1kbnr/pppp1ppp/8/4p3/4PP1q/8/PPPP2PP/RNBQKBNR w KQkq - 1 3",
        "move": "g2g3",
    },
    {
        "label": "2. CAPTURE — Pawn captures Queen",
        "fen": "6k1/pp6/3p4/2p1p3/2P1P1q1/1P1P2pP/P5P1/5K2 w - - 0 31",
        "move": "h3g4",
    },
    {
        "label": "3. CHECK + CAPTURE — Rook captures Queen with check",
        "fen": "4r1k1/p1p2p2/1p1p2q1/3Q4/5P2/2P1r3/PP5P/2K3R1 w - - 0 27",
        "move": "g1g6",
    },
    {
        "label": "4. QUIET MOVE — Rook repositions, no capture, no check",
        "fen": "5r2/1p3ppk/2q2n1p/p1p2N2/PbP1p3/4Q3/1P3PPP/3R1BK1 w - - 1 25",
        "move": "d1d6",
    },
    {
        "label": "5. CHECKMATE — Terminal position, Black is mated",
        "fen": "r4r1k/pp5R/1b5B/4PB1q/3p4/2P2N2/PP6/2K3R1 b - - 2 29",
        "move": "CHECKMATE",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# PIECE NAME MAPPING & MATERIAL VALUES
# ─────────────────────────────────────────────────────────────────────────────

PIECE_NAMES = {
    chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
    chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King",
}

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9,
}


# ─────────────────────────────────────────────────────────────────────────────
# FACT EXTRACTOR — Squeezes maximum information from python-chess
# ─────────────────────────────────────────────────────────────────────────────

def count_material(board):
    w = sum(len(board.pieces(p, chess.WHITE)) * v for p, v in PIECE_VALUES.items())
    b = sum(len(board.pieces(p, chess.BLACK)) * v for p, v in PIECE_VALUES.items())
    return w, b


def get_piece_list(board, color):
    priority = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    pieces = []
    for pt in priority:
        for sq in board.pieces(pt, color):
            pieces.append(f"{PIECE_NAMES[pt]} on {chess.square_name(sq)}")
    return pieces


def extract_facts(fen, move_uci):
    board = chess.Board(fen)
    is_terminal = move_uci in ("CHECKMATE", "STALEMATE", "DRAW")
    facts = {}

    # --- WHO PLAYS ---
    side = "White" if board.turn == chess.WHITE else "Black"
    opponent = "Black" if board.turn == chess.WHITE else "White"
    facts["side_to_move"] = side
    facts["opponent"] = opponent

    # --- MATERIAL ---
    w_mat, b_mat = count_material(board)
    facts["material_white"] = w_mat
    facts["material_black"] = b_mat
    diff = w_mat - b_mat
    if diff > 0:
        facts["material_balance"] = f"White leads by {diff} points"
    elif diff < 0:
        facts["material_balance"] = f"Black leads by {abs(diff)} points"
    else:
        facts["material_balance"] = "Material is equal"

    # --- PIECE LAYOUTS ---
    facts["white_pieces"] = get_piece_list(board, chess.WHITE)
    facts["black_pieces"] = get_piece_list(board, chess.BLACK)

    # --- KING SAFETY ---
    for color, name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
        king_sq = board.king(color)
        if king_sq is not None:
            facts[f"{name}_king"] = chess.square_name(king_sq)
            kr = board.has_kingside_castling_rights(color)
            qr = board.has_queenside_castling_rights(color)
            if kr and qr:
                facts[f"{name}_castling"] = "can castle Kingside and Queenside"
            elif kr:
                facts[f"{name}_castling"] = "can castle Kingside only"
            elif qr:
                facts[f"{name}_castling"] = "can castle Queenside only"
            else:
                facts[f"{name}_castling"] = "cannot castle"

    # --- CHECK STATUS ---
    facts["in_check"] = board.is_check()
    facts["legal_move_count"] = board.legal_moves.count()

    if board.is_check():
        king_sq = board.king(board.turn)
        attackers = board.attackers(not board.turn, king_sq)
        attacker_strs = []
        for sq in attackers:
            p = board.piece_at(sq)
            attacker_strs.append(f"{PIECE_NAMES[p.piece_type]} on {chess.square_name(sq)}")
        facts["checked_by"] = attacker_strs

    # --- TERMINAL STATE ---
    if is_terminal:
        facts["is_checkmate"] = board.is_checkmate()
        facts["is_stalemate"] = board.is_stalemate()
        if board.is_checkmate():
            facts["winner"] = opponent
        return facts

    # --- MOVE FACTS ---
    move = chess.Move.from_uci(move_uci)
    piece = board.piece_at(move.from_square)

    facts["moving_piece"] = PIECE_NAMES[piece.piece_type]
    facts["from_square"] = chess.square_name(move.from_square)
    facts["to_square"] = chess.square_name(move.to_square)
    facts["move_uci"] = move_uci

    # Capture
    facts["is_capture"] = board.is_capture(move)
    if board.is_capture(move):
        if board.is_en_passant(move):
            facts["captured_piece"] = "Pawn"
            facts["capture_type"] = "en passant"
        else:
            cap = board.piece_at(move.to_square)
            facts["captured_piece"] = PIECE_NAMES[cap.piece_type]
            facts["capture_type"] = "standard"

    # Special moves
    facts["is_castling"] = board.is_castling(move)
    facts["is_en_passant"] = board.is_en_passant(move)
    facts["is_promotion"] = move.promotion is not None
    if move.promotion:
        facts["promotion_piece"] = PIECE_NAMES[move.promotion]

    # Post-move analysis
    board_after = board.copy()
    board_after.push(move)
    facts["gives_check"] = board_after.is_check()
    facts["gives_checkmate"] = board_after.is_checkmate()

    facts["piece_now_attacked"] = board_after.is_attacked_by(not board.turn, move.to_square)

    threats = []
    for sq in board_after.attacks(move.to_square):
        target = board_after.piece_at(sq)
        if target and target.color != board.turn:
            threats.append(f"{PIECE_NAMES[target.piece_type]} on {chess.square_name(sq)}")
    facts["now_threatens"] = threats

    pins = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == (not board.turn):
            if board.is_pinned(not board.turn, sq):
                pins.append(f"{PIECE_NAMES[p.piece_type]} on {chess.square_name(sq)}")
    facts["pinned_enemy_pieces"] = pins

    w_after, b_after = count_material(board_after)
    facts["material_white_after"] = w_after
    facts["material_black_after"] = b_after

    return facts


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER — Airtight, zero-shot, KCCTI cascade with bracketed headers
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(fen, move_uci, facts):
    is_terminal = move_uci in ("CHECKMATE", "STALEMATE", "DRAW")

    # --- BUILD THE FACT SHEET ---
    fact_lines = []
    fact_lines.append(f"SIDE TO MOVE: {facts['side_to_move']}")
    fact_lines.append(f"OPPONENT: {facts['opponent']}")
    fact_lines.append(f"FEN: {fen}")
    fact_lines.append(f"MATERIAL: White {facts['material_white']} pts — Black {facts['material_black']} pts ({facts['material_balance']})")
    fact_lines.append(f"WHITE KING: {facts['White_king']} ({facts['White_castling']})")
    fact_lines.append(f"BLACK KING: {facts['Black_king']} ({facts['Black_castling']})")

    if facts["in_check"]:
        attackers = ", ".join(facts["checked_by"])
        fact_lines.append(f"⚠ {facts['side_to_move']} King is IN CHECK from: {attackers}")
        fact_lines.append(f"LEGAL MOVES AVAILABLE: {facts['legal_move_count']}")

    if is_terminal:
        if facts.get("is_checkmate"):
            fact_lines.append(f"GAME STATE: CHECKMATE — {facts['winner']} wins")
            fact_lines.append(f"{facts['side_to_move']} has 0 legal moves and is in check")
        elif facts.get("is_stalemate"):
            fact_lines.append(f"GAME STATE: STALEMATE — Draw")
            fact_lines.append(f"{facts['side_to_move']} has 0 legal moves but is NOT in check")
    else:
        fact_lines.append(f"THE MOVE: {move_uci}")
        fact_lines.append(f"MOVING PIECE: {facts['moving_piece']} (from {facts['from_square']} to {facts['to_square']})")

        if facts["is_capture"]:
            fact_lines.append(f"THIS MOVE CAPTURES: Yes — takes {facts['opponent']}'s {facts['captured_piece']} on {facts['to_square']}")
            fact_lines.append(f"MATERIAL AFTER: White {facts['material_white_after']} pts — Black {facts['material_black_after']} pts")
        else:
            fact_lines.append(f"THIS MOVE CAPTURES: No")

        if facts["gives_checkmate"]:
            fact_lines.append(f"THIS MOVE DELIVERS: CHECKMATE — game over")
        elif facts["gives_check"]:
            fact_lines.append(f"THIS MOVE DELIVERS: CHECK to the {facts['opponent']} King")
        else:
            fact_lines.append(f"THIS MOVE DELIVERS: No check")

        if facts["is_castling"]:
            fact_lines.append(f"SPECIAL: This is a CASTLING move")
        if facts["is_en_passant"]:
            fact_lines.append(f"SPECIAL: This is an EN PASSANT capture")
        if facts["is_promotion"]:
            fact_lines.append(f"SPECIAL: Pawn PROMOTES to {facts['promotion_piece']}")

        if facts["piece_now_attacked"]:
            fact_lines.append(f"WARNING: After moving, the {facts['moving_piece']} on {facts['to_square']} IS under attack")
        else:
            fact_lines.append(f"SAFETY: After moving, the {facts['moving_piece']} on {facts['to_square']} is NOT under attack")

        if facts["now_threatens"]:
            fact_lines.append(f"NEW THREATS CREATED: {', '.join(facts['now_threatens'])}")

        if facts["pinned_enemy_pieces"]:
            fact_lines.append(f"PINNED ENEMY PIECES: {', '.join(facts['pinned_enemy_pieces'])}")

    fact_sheet = "\n".join(fact_lines)

    # --- TERMINAL PROMPT ---
    if is_terminal:
        state_word = "CHECKMATE" if facts.get("is_checkmate") else "STALEMATE"
        prompt = f"""### ROLE
You are a Chess Grandmaster writing analysis for a training dataset. The game has ended. Your task is to explain WHY the game is over using ONLY the facts provided below. Do NOT invent or assume any information not given.

### FACTS (TRUST THESE COMPLETELY — DO NOT CONTRADICT THEM)
{fact_sheet}

### RULES (CRITICAL — VIOLATING ANY RULE INVALIDATES YOUR RESPONSE)
1. Use ONLY the facts above. Do NOT add your own chess analysis or assumptions.
2. You MUST use the word "{state_word}" in your analysis. This is mandatory.
3. Explain why the position is {state_word.lower()} by referencing the specific pieces and squares from the facts.
4. State that the side to move has 0 legal moves.
5. Use ONLY UCI format for moves. Never use algebraic notation.
6. Keep your response between 80 and 150 words.
7. Write naturally as if you derived this analysis yourself. Do NOT reference "the facts" or say "I was told."
8. Your analysis MUST use the bracketed section headers shown in the format below.

### OUTPUT FORMAT
<thinking>
[KING SAFETY]: Describe the losing King's situation — where it is, why it cannot escape.
[CHECKS]: State who is delivering the check and from where (or state no check for stalemate).
[CAPTURES & TRADES]: State that material is irrelevant because the game is over.
[THREATS]: Describe the mating/stalemate pattern — which pieces create the net.
[IMPROVEMENT]: State that no further moves are possible.
Conclusion: This position is {state_word}. {facts.get('winner', 'Draw')} wins. State this clearly.
</thinking>
<output>
{move_uci}
</output>

Your analysis:"""

    # --- STANDARD MOVE PROMPT ---
    else:
        # Determine KCCTI priority and build section guidance
        if facts["in_check"]:
            section_guidance = f"""[KING SAFETY]: THIS IS THE PRIORITY. {facts['side_to_move']} is in check from {', '.join(facts['checked_by'])}. There are only {facts['legal_move_count']} legal moves. Explain in detail how {move_uci} responds to the check — does it block, move the King, or capture the attacker? Go deep here.
[CHECKS]: State whether {move_uci} itself delivers a check. {"Yes it does." if facts['gives_check'] else "No it does not."}
[CAPTURES & TRADES]: {"State that " + move_uci + " captures " + facts['opponent'] + "'s " + facts['captured_piece'] + "." if facts['is_capture'] else "Not applicable — this move does not capture."}
[THREATS]: Briefly mention any new threats created after the move. {("New threats: " + ", ".join(facts["now_threatens"])) if facts.get("now_threatens") else "No immediate new threats."}
[IMPROVEMENT]: Not applicable — this move is about responding to check.
Conclusion: {move_uci} is the best move because it addresses the check. State this clearly."""

        elif facts["gives_checkmate"]:
            section_guidance = f"""[KING SAFETY]: Briefly state both Kings' positions.
[CHECKS]: THIS IS THE PRIORITY. {move_uci} delivers CHECKMATE. Explain the mating pattern in detail — which pieces coordinate, why the King cannot escape.
[CAPTURES & TRADES]: {"State that " + move_uci + " captures " + facts['opponent'] + "'s " + facts['captured_piece'] + "." if facts['is_capture'] else "State whether this move captures anything."}
[THREATS]: The ultimate threat has been executed — checkmate.
[IMPROVEMENT]: Not applicable — the game is over.
Conclusion: {move_uci} is the best move because it delivers checkmate. State this clearly."""

        elif facts["gives_check"] and facts["is_capture"]:
            section_guidance = f"""[KING SAFETY]: Briefly state both Kings' positions and safety.
[CHECKS]: THIS IS A PRIORITY. {move_uci} delivers check to the {facts['opponent']} King. Explain why the check is forcing and what it means for the opponent.
[CAPTURES & TRADES]: THIS IS ALSO A PRIORITY. {move_uci} captures {facts['opponent']}'s {facts['captured_piece']} on {facts['to_square']}. Material before: White {facts['material_white']} — Black {facts['material_black']}. Material after: White {facts['material_white_after']} — Black {facts['material_black_after']}. Explain the material impact.
[THREATS]: Mention any new threats from the {facts['moving_piece']} on {facts['to_square']}. {("Threatens: " + ", ".join(facts["now_threatens"])) if facts.get("now_threatens") else ""}
[IMPROVEMENT]: Explain how the combination of check + capture improves {facts['side_to_move']}'s position.
Conclusion: {move_uci} is the best move because it captures material AND delivers check simultaneously. State this clearly."""

        elif facts["gives_check"]:
            section_guidance = f"""[KING SAFETY]: Briefly state both Kings' positions and safety.
[CHECKS]: THIS IS THE PRIORITY. {move_uci} delivers check to the {facts['opponent']} King. Explain in detail why delivering check here is strong — does it force the King to a worse square? Does it win a tempo? Does it enable a follow-up?
[CAPTURES & TRADES]: Not applicable — this move does not capture.
[THREATS]: Mention any new threats created. {("Threatens: " + ", ".join(facts["now_threatens"])) if facts.get("now_threatens") else "No new threats."}
[IMPROVEMENT]: Explain the strategic benefit of forcing the opponent to respond to check.
Conclusion: {move_uci} is the best move because it delivers check. State this clearly."""

        elif facts["is_capture"]:
            section_guidance = f"""[KING SAFETY]: Briefly state both Kings' positions and safety.
[CHECKS]: {move_uci} does not deliver check. State this briefly.
[CAPTURES & TRADES]: THIS IS THE PRIORITY. {move_uci} captures {facts['opponent']}'s {facts['captured_piece']} on {facts['to_square']}. Material before: White {facts['material_white']} — Black {facts['material_black']}. Material after: White {facts['material_white_after']} — Black {facts['material_black_after']}. Analyze the material impact in detail — is it free material, a fair trade, or a sacrifice?
[THREATS]: Mention any new threats from {facts['to_square']}. {("Threatens: " + ", ".join(facts["now_threatens"])) if facts.get("now_threatens") else "No new threats."}
[IMPROVEMENT]: Explain how the capture improves {facts['side_to_move']}'s position.
Conclusion: {move_uci} is the best move because of the material gain. State this clearly."""

        else:
            section_guidance = f"""[KING SAFETY]: Briefly state both Kings' positions and safety.
[CHECKS]: {move_uci} does not deliver check. State this briefly.
[CAPTURES & TRADES]: {move_uci} does not capture any piece. State this briefly.
[THREATS]: Mention any new threats the {facts['moving_piece']} creates from {facts['to_square']}. {("Threatens: " + ", ".join(facts["now_threatens"])) if facts.get("now_threatens") else "No immediate threats."}
[IMPROVEMENT]: THIS IS THE PRIORITY. Explain in detail how moving the {facts['moving_piece']} from {facts['from_square']} to {facts['to_square']} improves {facts['side_to_move']}'s position. Does it control key squares? Improve coordination? Prepare an attack?
Conclusion: {move_uci} is the best move because of its positional improvement. State this clearly."""

        prompt = f"""### ROLE
You are a Chess Grandmaster writing analysis for a training dataset. You are given a position and the best move. Your task is to explain WHY it is the best move using ONLY the facts provided below. Do NOT invent or assume any information not given.

### FACTS (TRUST THESE COMPLETELY — DO NOT CONTRADICT THEM)
{fact_sheet}

### RULES (CRITICAL — VIOLATING ANY RULE INVALIDATES YOUR RESPONSE)
1. The moving piece is a {facts['moving_piece']}. You MUST call it a {facts['moving_piece']}. NOT a Pawn, NOT a King, NOT any other piece unless it IS that piece. A {facts['moving_piece']}.
2. {"This move IS a capture. You MUST state that it captures " + facts['opponent'] + "'s " + facts['captured_piece'] + "." if facts['is_capture'] else "This move is NOT a capture. Do NOT claim it captures anything."}
3. {"This move DOES deliver check. You MUST state that it delivers check." if facts['gives_check'] else "This move does NOT deliver check. Do NOT claim it gives check."}
4. {"The side to move IS in check. You MUST address how this move responds to the check." if facts['in_check'] else ""}
5. Use ONLY UCI format for moves (e.g., "{move_uci}"). Never use algebraic notation like "Rd6" or "Bxg6".
6. Keep your response between 80 and 200 words.
7. Write naturally as if you derived this analysis yourself. Do NOT say "according to the facts" or "I was told."
8. Your analysis MUST use the bracketed section headers exactly as shown below.
9. The section marked "THIS IS THE PRIORITY" must be the longest and most detailed section. Other sections should be brief (1-2 sentences max).

### OUTPUT FORMAT — You MUST follow this structure:
<thinking>
{section_guidance}
</thinking>
<output>
{move_uci}
</output>

Your analysis:"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA API CALL
# ─────────────────────────────────────────────────────────────────────────────

def call_ollama(prompt):
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("response", "")
        else:
            return f"[ERROR] Ollama returned status {resp.status_code}: {resp.text}"
    except requests.exceptions.ConnectionError:
        return "[ERROR] Could not connect to Ollama. Is it running?"
    except Exception as e:
        return f"[ERROR] {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run the test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"{'='*70}")
    print(f"CHESS REASONING GENERATOR — TEST RUN")
    print(f"Model: {MODEL_ID}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"{'='*70}\n")

    results = []

    for case in TEST_CASES:
        fen = case["fen"]
        move = case["move"]
        label = case["label"]

        print(f"\n{'─'*70}")
        print(f"TEST: {label}")
        print(f"FEN:  {fen}")
        print(f"Move: {move}")
        print(f"{'─'*70}")

        # 1. Extract facts
        facts = extract_facts(fen, move)

        # 2. Build prompt
        prompt = build_prompt(fen, move, facts)

        # 3. Call model
        print(f"Calling {MODEL_ID}...")
        raw_output = call_ollama(prompt)

        # 4. Extract thinking
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw_output, re.DOTALL)
        output_match = re.search(r"<output>(.*?)</output>", raw_output, re.DOTALL)

        thinking = thinking_match.group(1).strip() if thinking_match else "[NO THINKING TAGS FOUND]"
        output_move = output_match.group(1).strip() if output_match else "[NO OUTPUT TAGS FOUND]"

        print(f"\n📝 REASONING:")
        print(thinking)
        print(f"\n🎯 OUTPUT: {output_move}")
        print(f"{'✅' if output_move == move else '❌'} Output matches expected: {output_move == move}")

        # 5. Validate using the real validator
        report = validate_row(
            row_id=0,
            fen=fen,
            best_move=move,
            reasoning=thinking
        )

        print(f"\n🔍 VALIDATION (via validator.py):")
        check_fields = [
            ("side_to_move", "Side to Move"),
            ("piece_id", "Piece ID"),
            ("is_capture", "Capture"),
            ("captured_piece", "Captured Piece"),
            ("gives_check", "Gives Check"),
            ("in_check", "In Check"),
            ("is_checkmate", "Checkmate"),
            ("is_castling", "Castling"),
            ("is_en_passant", "En Passant"),
            ("is_promotion", "Promotion"),
        ]

        has_fail = False
        for attr, label_name in check_fields:
            status = getattr(report, attr)
            if status == "PASS":
                print(f"  ✅ {label_name}: PASS")
            elif status == "FAIL":
                print(f"  ❌ {label_name}: FAIL")
                has_fail = True
            # Skip printing SKIPs to keep output clean

        print(f"\n  Overall: {'❌ FAIL' if report.overall == 'FAIL' else '✅ PASS'}")
        if report.fail_reasons:
            print(f"  Reasons: {report.fail_reasons}")

        results.append({
            "label": case["label"],
            "output_correct": output_move == move,
            "validation_pass": report.overall == "PASS",
        })

    # Summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    correct_output = sum(1 for r in results if r["output_correct"])
    valid_reasoning = sum(1 for r in results if r["validation_pass"])
    print(f"Output matches expected: {correct_output}/5")
    print(f"Reasoning passes validation: {valid_reasoning}/5")

    for r in results:
        out_sym = "✅" if r["output_correct"] else "❌"
        val_sym = "✅" if r["validation_pass"] else "❌"
        print(f"  {out_sym} Output  {val_sym} Reasoning  — {r['label']}")


if __name__ == "__main__":
    main()