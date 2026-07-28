"""
validator.py — Chess Reasoning Dataset Quality Validator
=========================================================
Runs deterministic python-chess checks against each row's Reasoning text.
Produces a detailed quality report showing pass/fail per check, per row.

Checks performed (all deterministic via python-chess):
    1. SIDE_TO_MOVE    — Does reasoning correctly say "White to play" or "Black to play"?
    2. PIECE_ID        — Does reasoning correctly name the piece being moved?
    3. IS_CAPTURE      — Does reasoning correctly claim capture vs no-capture?
    4. CAPTURED_PIECE  — If capture, does reasoning name the correct captured piece type?
    5. GIVES_CHECK     — Does reasoning correctly claim check vs no-check?
    6. IN_CHECK        — If side to move is in check, does reasoning acknowledge it?
    7. IS_CHECKMATE    — For terminal positions (Best Move = CHECKMATE), does reasoning acknowledge mate?
    8. IS_CASTLING     — If the move is castling, does reasoning acknowledge it?
    9. IS_EN_PASSANT   — If the move is en passant, does reasoning acknowledge it?
    10. IS_PROMOTION   — If the move is a promotion, does reasoning acknowledge it?
"""

import csv
import re
from dataclasses import dataclass, asdict
from collections import Counter
import chess


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Change these paths before running
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV = "chess_reasoning_v2_dataset.csv"
OUTPUT_CSV = "v2_report.csv"


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """Result of a single check: pass, fail, or skipped (not applicable)."""
    status: str  # "PASS", "FAIL", "SKIP"
    detail: str = ""


@dataclass
class RowReport:
    """Full validation report for one row."""
    row_id: int = 0
    fen: str = ""
    best_move: str = ""
    side_to_move: str = ""
    piece_id: str = ""
    is_capture: str = ""
    captured_piece: str = ""
    gives_check: str = ""
    in_check: str = ""
    is_checkmate: str = ""
    is_castling: str = ""
    is_en_passant: str = ""
    is_promotion: str = ""
    overall: str = ""
    fail_reasons: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# PIECE NAME MAPPING
# ─────────────────────────────────────────────────────────────────────────────

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


# ─────────────────────────────────────────────────────────────────────────────
# TEXT SEARCH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def text_contains_any(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_side_to_move(reasoning: str, board: chess.Board) -> CheckResult:
    """Check 1: Does reasoning correctly identify whose turn it is?"""
    if board.turn == chess.WHITE:
        correct_phrases = ["white to play", "white to move", "shows white to play",
                           "shows white to move", "position shows white"]
        wrong_phrases = ["black to play", "black to move", "shows black to play",
                         "shows black to move", "position shows black"]
    else:
        correct_phrases = ["black to play", "black to move", "shows black to play",
                           "shows black to move", "position shows black"]
        wrong_phrases = ["white to play", "white to move", "shows white to play",
                         "shows white to move", "position shows white"]

    has_correct = text_contains_any(reasoning, correct_phrases)
    has_wrong = text_contains_any(reasoning, wrong_phrases)

    if has_correct and not has_wrong:
        return CheckResult("PASS")
    elif has_wrong:
        expected = "White" if board.turn == chess.WHITE else "Black"
        return CheckResult("FAIL", f"Expected '{expected} to play', found opposite")
    else:
        return CheckResult("SKIP", "Could not detect side-to-move claim in text")


def check_piece_id(reasoning: str, board: chess.Board, move: chess.Move) -> CheckResult:
    """Check 2: Does reasoning correctly name the piece being moved?"""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return CheckResult("SKIP", "No piece on from-square (invalid position?)")

    correct_name = PIECE_NAMES[piece.piece_type]
    move_uci = move.uci()
    from_sq = chess.square_name(move.from_square)
    reasoning_lower = reasoning.lower()

    # --- Phase 1: Look for explicit move-action phrases ---
    action_patterns_correct = [
        rf"the {correct_name}\b.*\b{re.escape(move_uci)}",
        rf"\b{re.escape(move_uci)}\b.*\bthe {correct_name}",
        rf"{correct_name} (?:on|from|at) {re.escape(from_sq)}",
        rf"(?:move|moves|moving|advance|advancing|push|pushing) (?:the |a )?{correct_name}",
        rf"{correct_name}.*(?:to|toward|captures|takes)",
        rf"by (?:moving|playing|advancing) (?:the )?{correct_name}",
    ]

    for pattern in action_patterns_correct:
        if re.search(pattern, reasoning_lower):
            return CheckResult("PASS")

    # --- Phase 2: Look for explicit WRONG piece attribution ---
    wrong_piece_names = [name for pt, name in PIECE_NAMES.items()
                         if pt != piece.piece_type]

    for wrong_name in wrong_piece_names:
        wrong_action_patterns = [
            rf"the {wrong_name}\b.*\b{re.escape(move_uci)}",
            rf"\b{re.escape(move_uci)}\b.*\bthe {wrong_name}",
            rf"{wrong_name} (?:on|from|at) {re.escape(from_sq)}",
            rf"(?:move|moves|moving|advance|advancing|push|pushing) (?:the |a )?{wrong_name}\b(?!.*king safety)",
            rf"by (?:moving|playing|advancing) (?:the )?{wrong_name}",
        ]

        for pattern in wrong_action_patterns:
            match = re.search(pattern, reasoning_lower)
            if match:
                context_start = max(0, match.start() - 30)
                context = reasoning_lower[context_start:match.start()]
                if any(skip in context for skip in ["king safety", "is on ", "is tucked",
                                                     "is in the center", "is exposed"]):
                    continue
                return CheckResult("FAIL",
                    f"Move {move_uci}: piece is {correct_name}, "
                    f"but reasoning says '{wrong_name}'")

    # --- Phase 3: Broader check ---
    if re.search(rf"{correct_name}\b.*\b{re.escape(from_sq)}", reasoning_lower):
        return CheckResult("PASS", f"Found '{correct_name}' associated with '{from_sq}'")

    move_context_phrases = [f"the {correct_name}", f"a {correct_name}",
                            f"{correct_name} capture", f"{correct_name} move"]
    if any(phrase in reasoning_lower for phrase in move_context_phrases):
        return CheckResult("PASS", "Correct piece name found in move context")

    return CheckResult("SKIP", "Could not determine piece identification from text")


def check_is_capture(reasoning: str, board: chess.Board, move: chess.Move) -> CheckResult:
    """Check 3: Does reasoning correctly state whether the move captures?"""
    is_capture = board.is_capture(move)

    capture_phrases = ["capture", "takes", "wins material", "wins the",
                       "takes the", "captured"]
    no_capture_phrases = ["does not capture", "no capture", "doesn't capture",
                          "not capture any", "no material exchange",
                          "no immediate capture", "quiet move",
                          "does not immediately capture"]

    claims_capture = text_contains_any(reasoning, capture_phrases)
    claims_no_capture = text_contains_any(reasoning, no_capture_phrases)

    if is_capture:
        if claims_capture and not claims_no_capture:
            return CheckResult("PASS")
        elif claims_no_capture:
            return CheckResult("FAIL", "Move is a capture, but reasoning says no capture")
        else:
            return CheckResult("FAIL", "Move is a capture, but reasoning doesn't mention it")
    else:
        if claims_no_capture:
            return CheckResult("PASS")
        elif claims_capture:
            move_uci = move.uci()
            reasoning_lower = reasoning.lower()
            for phrase in capture_phrases:
                phrase_pos = reasoning_lower.find(phrase.lower())
                if phrase_pos != -1:
                    move_pos = reasoning_lower.find(move_uci)
                    if move_pos != -1 and abs(phrase_pos - move_pos) < 150:
                        return CheckResult("FAIL",
                            "Move is NOT a capture, but reasoning claims it captures")
            return CheckResult("SKIP", "Capture language found but not clearly about this move")
        else:
            return CheckResult("PASS", "Quiet move, no capture claims made")


def check_captured_piece(reasoning: str, board: chess.Board, move: chess.Move) -> CheckResult:
    """Check 4: If it's a capture, does reasoning name the correct captured piece?"""
    if not board.is_capture(move):
        return CheckResult("SKIP", "Not a capture")

    if board.is_en_passant(move):
        captured_name = "pawn"
    else:
        captured_piece = board.piece_at(move.to_square)
        if captured_piece is None:
            return CheckResult("SKIP", "Capture detected but no piece on target square")
        captured_name = PIECE_NAMES[captured_piece.piece_type]

    reasoning_lower = reasoning.lower()

    capture_patterns = [
        f"captures the {captured_name}", f"captures a {captured_name}",
        f"capture the {captured_name}", f"capture a {captured_name}",
        f"captures the opponent's {captured_name}",
        f"captures opponent's {captured_name}",
        f"takes the {captured_name}", f"taking the {captured_name}",
        f"wins the {captured_name}", f"captured {captured_name}",
        f"capturing the {captured_name}",
    ]

    for pattern in capture_patterns:
        if pattern in reasoning_lower:
            return CheckResult("PASS")

    # Also try regex for "captures.*<piece>"
    if re.search(rf"captures?\s+.*{captured_name}", reasoning_lower):
        return CheckResult("PASS")

    wrong_names = [name for pt, name in PIECE_NAMES.items()
                   if name != captured_name and pt != chess.KING]
    for wrong_name in wrong_names:
        wrong_patterns = [
            f"captures the {wrong_name}", f"captures a {wrong_name}",
            f"capture the {wrong_name}", f"takes the {wrong_name}",
            f"wins the {wrong_name}",
        ]
        for pattern in wrong_patterns:
            if pattern in reasoning_lower:
                return CheckResult("FAIL",
                    f"Captured piece is {captured_name}, "
                    f"but reasoning says '{wrong_name}'")

    if text_contains_any(reasoning, ["capture", "takes", "wins"]):
        return CheckResult("SKIP", f"Capture mentioned but {captured_name} not named")

    return CheckResult("FAIL", f"Capture of {captured_name} not mentioned at all")


def check_gives_check(reasoning: str, board: chess.Board, move: chess.Move) -> CheckResult:
    """Check 5: Does reasoning correctly state whether the move gives check?"""
    board_copy = board.copy()
    board_copy.push(move)
    gives_check = board_copy.is_check()

    check_phrases = ["delivers check", "delivers a check", "gives check",
                     "gives a check", "with check", "direct check",
                     "delivers a direct check", "putting the king in check",
                     "check to the", "results in check"]
    no_check_phrases = ["no check", "no immediate check", "does not deliver",
                        "doesn't deliver", "quiet move", "without check",
                        "not a check", "does not give check"]

    claims_check = text_contains_any(reasoning, check_phrases)
    claims_no_check = text_contains_any(reasoning, no_check_phrases)

    if gives_check:
        if claims_check:
            return CheckResult("PASS")
        elif claims_no_check:
            return CheckResult("FAIL", "Move GIVES check, but reasoning denies it")
        else:
            return CheckResult("FAIL", "Move GIVES check, but reasoning doesn't mention it")
    else:
        if claims_no_check or not claims_check:
            return CheckResult("PASS")
        elif claims_check:
            return CheckResult("FAIL", "Move does NOT give check, but reasoning claims it does")

    return CheckResult("SKIP")


def check_in_check(reasoning: str, board: chess.Board) -> CheckResult:
    """Check 6: If the side to move is currently in check, does reasoning note it?"""
    if not board.is_check():
        return CheckResult("SKIP", "Side to move is not in check")

    in_check_phrases = ["in check", "is in check", "under check", "under attack",
                        "king is attacked", "being checked", "currently in check",
                        "respond to check", "escape check", "block the check"]

    if text_contains_any(reasoning, in_check_phrases):
        return CheckResult("PASS")
    else:
        return CheckResult("FAIL", "Side to move is IN CHECK but reasoning doesn't mention it")


def check_is_checkmate(reasoning: str, best_move: str, board: chess.Board) -> CheckResult:
    """Check 7: For terminal positions, does reasoning acknowledge checkmate?"""
    if best_move != "CHECKMATE":
        return CheckResult("SKIP", "Not a checkmate position")

    mate_phrases = ["checkmate", "checkmated", "mating", "mate", "mated",
                    "game is over", "no legal moves"]

    if text_contains_any(reasoning, mate_phrases):
        return CheckResult("PASS")
    else:
        return CheckResult("FAIL", "Position is CHECKMATE but reasoning doesn't say so")


def check_is_castling(reasoning: str, board: chess.Board, move: chess.Move) -> CheckResult:
    """Check 8: If the move is castling, does reasoning acknowledge it?"""
    if not board.is_castling(move):
        return CheckResult("SKIP", "Not a castling move")

    castle_phrases = ["castle", "castles", "castling", "castled"]
    if text_contains_any(reasoning, castle_phrases):
        return CheckResult("PASS")
    else:
        return CheckResult("FAIL", "Move is CASTLING but reasoning doesn't mention it")


def check_is_en_passant(reasoning: str, board: chess.Board, move: chess.Move) -> CheckResult:
    """Check 9: If the move is en passant, does reasoning acknowledge it?"""
    if not board.is_en_passant(move):
        return CheckResult("SKIP", "Not an en passant move")

    ep_phrases = ["en passant", "en-passant", "e.p."]
    if text_contains_any(reasoning, ep_phrases):
        return CheckResult("PASS")
    else:
        return CheckResult("FAIL", "Move is EN PASSANT but reasoning doesn't mention it")


def check_is_promotion(reasoning: str, move: chess.Move) -> CheckResult:
    """Check 10: If the move is a promotion, does reasoning acknowledge it?"""
    if not move.promotion:
        return CheckResult("SKIP", "Not a promotion move")

    promo_phrases = ["promot", "promotion", "promotes", "promoted", "queening",
                     "queens", "underpromotion"]
    if text_contains_any(reasoning, promo_phrases):
        return CheckResult("PASS")
    else:
        promo_piece = PIECE_NAMES.get(move.promotion, "unknown")
        return CheckResult("FAIL",
            f"Move is a PROMOTION to {promo_piece} but reasoning doesn't mention it")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def validate_row(row_id: int, fen: str, best_move: str, reasoning: str) -> RowReport:
    """Run all checks on a single row and return the report."""
    report = RowReport(row_id=row_id, fen=fen, best_move=best_move)
    failures = []

    try:
        board = chess.Board(fen)
    except Exception as e:
        report.overall = "ERROR"
        report.fail_reasons = f"Invalid FEN: {e}"
        return report

    is_terminal = best_move in ("CHECKMATE", "STALEMATE", "DRAW")

    move = None
    if not is_terminal:
        try:
            move = chess.Move.from_uci(best_move)
            if move not in board.legal_moves:
                report.overall = "ERROR"
                report.fail_reasons = f"Move {best_move} is not legal in this position"
                return report
        except Exception as e:
            report.overall = "ERROR"
            report.fail_reasons = f"Invalid UCI move '{best_move}': {e}"
            return report

    # --- Run checks ---

    r = check_side_to_move(reasoning, board)
    report.side_to_move = r.status
    if r.status == "FAIL":
        failures.append(f"SIDE_TO_MOVE: {r.detail}")

    r = check_is_checkmate(reasoning, best_move, board)
    report.is_checkmate = r.status
    if r.status == "FAIL":
        failures.append(f"IS_CHECKMATE: {r.detail}")

    if is_terminal:
        report.piece_id = "SKIP"
        report.is_capture = "SKIP"
        report.captured_piece = "SKIP"
        report.gives_check = "SKIP"
        report.in_check = "SKIP"
        report.is_castling = "SKIP"
        report.is_en_passant = "SKIP"
        report.is_promotion = "SKIP"
    else:
        r = check_piece_id(reasoning, board, move)
        report.piece_id = r.status
        if r.status == "FAIL":
            failures.append(f"PIECE_ID: {r.detail}")

        r = check_is_capture(reasoning, board, move)
        report.is_capture = r.status
        if r.status == "FAIL":
            failures.append(f"IS_CAPTURE: {r.detail}")

        r = check_captured_piece(reasoning, board, move)
        report.captured_piece = r.status
        if r.status == "FAIL":
            failures.append(f"CAPTURED_PIECE: {r.detail}")

        r = check_gives_check(reasoning, board, move)
        report.gives_check = r.status
        if r.status == "FAIL":
            failures.append(f"GIVES_CHECK: {r.detail}")

        r = check_in_check(reasoning, board)
        report.in_check = r.status
        if r.status == "FAIL":
            failures.append(f"IN_CHECK: {r.detail}")

        r = check_is_castling(reasoning, board, move)
        report.is_castling = r.status
        if r.status == "FAIL":
            failures.append(f"IS_CASTLING: {r.detail}")

        r = check_is_en_passant(reasoning, board, move)
        report.is_en_passant = r.status
        if r.status == "FAIL":
            failures.append(f"IS_EN_PASSANT: {r.detail}")

        r = check_is_promotion(reasoning, move)
        report.is_promotion = r.status
        if r.status == "FAIL":
            failures.append(f"IS_PROMOTION: {r.detail}")

    report.fail_reasons = "; ".join(failures)
    report.overall = "FAIL" if failures else "PASS"

    return report


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(reports: list[RowReport], total_rows: int):
    passed = sum(1 for r in reports if r.overall == "PASS")
    failed = sum(1 for r in reports if r.overall == "FAIL")
    errors = sum(1 for r in reports if r.overall == "ERROR")

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nTotal rows processed : {total_rows}")
    print(f"PASS                : {passed} ({passed/total_rows*100:.1f}%)")
    print(f"FAIL                : {failed} ({failed/total_rows*100:.1f}%)")
    print(f"ERROR               : {errors} ({errors/total_rows*100:.1f}%)")

    check_names = [
        ("side_to_move", "Side to Move"),
        ("piece_id", "Piece Identification"),
        ("is_capture", "Capture Detection"),
        ("captured_piece", "Captured Piece Type"),
        ("gives_check", "Gives Check"),
        ("in_check", "Currently In Check"),
        ("is_checkmate", "Checkmate Detection"),
        ("is_castling", "Castling Detection"),
        ("is_en_passant", "En Passant Detection"),
        ("is_promotion", "Promotion Detection"),
    ]

    print("\n" + "-" * 70)
    print("PER-CHECK BREAKDOWN")
    print("-" * 70)
    print(f"{'Check':<25} {'PASS':>8} {'FAIL':>8} {'SKIP':>8} {'Fail%':>8}")
    print("-" * 70)

    for attr, label in check_names:
        p = sum(1 for r in reports if getattr(r, attr) == "PASS")
        f = sum(1 for r in reports if getattr(r, attr) == "FAIL")
        s = sum(1 for r in reports if getattr(r, attr) == "SKIP")
        applicable = p + f
        fail_pct = f"{f/applicable*100:.1f}%" if applicable > 0 else "N/A"
        print(f"{label:<25} {p:>8} {f:>8} {s:>8} {fail_pct:>8}")

    all_failures = []
    for r in reports:
        if r.fail_reasons:
            for reason in r.fail_reasons.split("; "):
                check_name = reason.split(":")[0] if ":" in reason else reason
                all_failures.append(check_name)

    if all_failures:
        failure_counts = Counter(all_failures).most_common(10)
        print("\n" + "-" * 70)
        print("TOP FAILURE CATEGORIES")
        print("-" * 70)
        for reason, count in failure_counts:
            print(f"  {reason:<30} {count:>6} occurrences")

    failed_reports = [r for r in reports if r.overall == "FAIL"]
    if failed_reports:
        print("\n" + "-" * 70)
        print(f"SAMPLE FAILURES (first 5 of {len(failed_reports)})")
        print("-" * 70)
        for r in failed_reports[:5]:
            print(f"\n  ID: {r.row_id}")
            print(f"  FEN: {r.fen}")
            print(f"  Move: {r.best_move}")
            print(f"  Failures: {r.fail_reasons}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — Just run this file. No CLI args needed.
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading {INPUT_CSV}...")
    reports = []
    total_rows = 0

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            row_id = int(row["id"])
            fen = row["FEN"]
            best_move = row["Best Move"]
            reasoning = row["Reasoning"]

            report = validate_row(row_id, fen, best_move, reasoning)
            reports.append(report)

            if total_rows % 1000 == 0:
                print(f"  Processed {total_rows} rows...")

    print(f"  Done. {total_rows} rows processed.")

    print(f"\nWriting report to {OUTPUT_CSV}...")
    fieldnames = [
        "row_id", "fen", "best_move", "overall",
        "side_to_move", "piece_id", "is_capture", "captured_piece",
        "gives_check", "in_check", "is_checkmate",
        "is_castling", "is_en_passant", "is_promotion",
        "fail_reasons"
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reports:
            writer.writerow(asdict(r))

    print_summary(reports, total_rows)


if __name__ == "__main__":
    main()