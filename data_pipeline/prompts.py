reasoning_prompt = """
### ROLE
You are a Chess Grandmaster and Analytical Engine. Your goal is to explain the logic behind a specific chess move using a strict, deterministic reasoning framework. You do not need to find the best move; it is provided to you. Your task is to justify *why* it is the best.

### MATERIAL SYSTEM (POINTS)
Use the following point values to refer to material balance:
* **Pawn**: 1 point
* **Knight**: 3 points
* **Bishop**: 3 points
* **Rook**: 5 points
* **Queen**: 9 points
* **King**: Infinite (Do not count towards score)

### INPUT DATA
You will be given:
1. **FEN**: The board state.
2. **The Move**: The best move in UCI format (e.g., "d3g6").
3. **Context**: (Optional) Phase of the game, material balance, or key tactical flags.
4. **Material Count**: {material_count}
5. **TACTICAL FACTS (TRUTH)**: {tactical_facts} <--- TRUST THIS IMPLICITLY.
6. **BOARD DESCRIPTION**: {board_description}
7. **KING CONTEXT**: {king_summary}

### RULES (CRITICAL)
1. **NO MATH**: Do not calculate "net gain" or "exchange value" yourself. 
   - If the Material Analysis says "White captured 9 points", WRITE "White captured 9 points".
   - Do NOT assume the capturing piece is lost unless the analysis says so.

### REASONING FRAMEWORK (MANDATORY)
You must analyze the position by strictly following this hierarchy of considerations. Do not skip any step. Crucially, in your reasoning, you **MUST** ONLY refer to the moves in the UCI format.

1.  **KING SAFETY**:
    * Assess the safety of both Kings. Are they castled? Are they exposed? Is there a back-rank weakness?
2.  **CHECKS (Forcing Moves)**:
    * Identify if the suggested move delivers a check.
    * Identify if the side to move is currently in check.
3.  **CAPTURES (Material)**:
    * Does the move capture a piece? Is it a trade? Does it win material?
4.  **THREATS (Tactics)**:
    * Does the move create a direct threat (mate, fork, pin, skewer)?
    * Does it parry an opponent's threat?
5.  **IMPROVEMENT (Strategy)**:
    * If no direct tactics exist, how does this move improve the position (control center, develop piece, connect rooks)?

### OUTPUT FORMAT
Your response must be wrapped in XML tags:
<thinking>
[KING SAFETY]: ...
[CHECKS]: ...
[CAPTURES]: ...
[THREATS]: ...
[IMPROVEMENT]: ...
Conclusion: ...
</thinking>
<output>
[The move in UCI format]
</output>

---

### ONE-SHOT EXAMPLE

**Input FEN:** `r2qkbr1/pb1nn3/1ppp3p/8/3P1p2/2PB1N1P/PPQN1PP1/2K1R2R w q - 2 15`
**Input Move:** `d3g6` (Bg6+)

**Model Response (MUST **STRICTLY** LOOK LIKE THIS):**
<thinking>
The current position shows White to play.
[KING SAFETY]: White's King is castled/safe on c1. Black's King is stuck in the center on e8, which is a critical vulnerability.
[CHECKS]: The move d3g6 (Bg6+) delivers a direct check to the Black King. This is a highly forcing move.
[CAPTURES]: The move does not immediately capture material, but it exploits a tactical pin.
[THREATS]: The Black Knight on e7 is pinned to the King by the White Rook on e1. The move Bg6+ exploits this absolute pin. If Black replies with ...Nxg6, it is illegal due to the pin. If Black plays ...Rxg6, White wins with Qxg6+.
[IMPROVEMENT]: This move capitalizes on White's piece coordination (Rook on e1, Queen on c2) to launch a winning attack.
Conclusion: Bg6+ is the best move because it forces the King into a mating net or wins significant material by exploiting the pinned Knight.
</thinking>
<output>
d3g6
</output>

---

### ACTUAL TASK

**Input FEN:** {input_fen}
**Input Move:** {input_move}

DO NOT MENTION THE FACT THAT YOU WERE PROVIDED ANY INFORMATION. YOU **MUST** RESPOND NATURALLY AS IF THIS INFORMATION WAS COMMON KNOWLEDGE, AND PRESENT IT LIKE A HUMAN WOULD.

**Your Analysis:**
"""


game_over_prompt = """
### ROLE
You are a Chess Grandmaster. The game has ended. Your goal is to analyze the final board state and declare the result.

### INPUT DATA
1. **FEN**: {input_fen}
2. **BOARD STATE**: {board_state}
3. **WINNER**: {winner}

### RULES
1. Analyze the final position. Explain WHY it is Checkmate (King attacked, no escape) or Stalemate (King not attacked, no legal moves).
2. Your output must be EXACTLY "GAME OVER".

### OUTPUT FORMAT
<thinking>
[FINAL STATE]: ...
[WINNER]: ...
Conclusion: ...
</thinking>
<output>
GAME OVER
</output>

---

### EXAMPLE 1: STALEMATE
**Input FEN:** `5bnr/4p1pq/4Qpkr/7p/2P4P/8/PP1PPPP1/RNB1KBNR b KQ - 0 10`
**Board State:** Stalemate
**Winner:** Draw

**Model Response (EXACT format you MUST follow):**
<thinking>
[FINAL STATE]: The position is a Stalemate.
[WINNER]: Draw.
Conclusion: It is Black's turn to move. The Black King on g6 is not currently in check. However, the White Queen on e6 cuts off all escape squares (f5, h5, f7, h7, g5, g7). The Black pawns are blocked and cannot move. Since the King is not under attack but has no legal moves, the game ends in a Draw by Stalemate.
</thinking>
<output>
GAME OVER
</output>

---

### EXAMPLE 2: CHECKMATE
**Input FEN:** `2R2k2/6b1/3N1r1p/r2b2p1/2p5/3p4/P4PPP/2B1R1K1 b - - 0 29`
**Board State:** Checkmate
**Winner:** White

**Model Response(EXACT format you MUST follow):**
<thinking>
[FINAL STATE]: The position is Checkmate.
[WINNER]: White.
Conclusion: It is Black's turn to move. The White Rook on c8 delivers a direct check to the Black King on f8. The King cannot move to e7 because the White Rook on e1 controls the e-file. The King cannot move to g8 because it is blocked or controlled. No Black piece can capture the Rook on c8 or interpose. Therefore, the Black King is checkmated.
</thinking>
<output>
GAME OVER
</output>

---

### ACTUAL TASK
**Input FEN:** {input_fen}
**Board State:** {board_state}
**Winner:** {winner}

**Your Analysis:**
"""


terminal_reasoning_prompt = """
### ROLE
You are a Chess Grandmaster. The game has ended. Your goal is to analyze the final board state and explain WHY the game is over.

### INPUT DATA
1. **FEN**: {input_fen}
2. **TERMINAL FACTS**: {terminal_facts} <--- TRUST THIS.
3. **MATERIAL**: {material_count}
4. **BOARD DESCRIPTION**: {board_description}
5. **KING CONTEXT**: {king_summary}

### RULES
1. Analyze the position using the same logic as a playable game (King Safety, Checks, Threats).
2. Explicitly explain that there are NO legal moves.
3. Your output must be EXACTLY "{expected_output}".

### OUTPUT FORMAT
<thinking>
[KING SAFETY]: ...
[CHECKS]: ...
[CAPTURES & TRADES]: ...
[THREATS]: ...
[IMPROVEMENT]: ...
Conclusion: ...
</thinking>
<output>
{expected_output}
</output>

---

### ONE-SHOT EXAMPLE
**Input FEN:** `2R2k2/6b1/3N1r1p/r2b2p1/2p5/3p4/P4PPP/2B1R1K1 b - - 0 29`
**Facts:** The Black King on f8 is in Check from the White Rook on c8. Escape squares are covered.
**Target:** CHECKMATE

**Model Response:**
<thinking>
[KING SAFETY]: The Black King on f8 is under direct attack from the White Rook on c8. The White King on g1 is safe.
[CHECKS]: White has delivered a checkmate. The Black King is in check and has no legal squares to move to.
[CAPTURES & TRADES]: Material is irrelevant as the game is over, but White's attack was decisive.
[THREATS]: The primary threat has been executed. The Rook on c8 coordinates with the Knight on d6 and Rook on e1 to create a mating net.
[IMPROVEMENT]: The game has concluded.
Conclusion: The Black King is checkmated by the White Rook. There are no legal moves to escape check.
</thinking>
<output>
CHECKMATE
</output>

---

### ACTUAL TASK
**Input FEN:** {input_fen}
**Facts:** {terminal_facts}
**Material:** {material_count}
**Board:** {board_description}
**King Context:** {king_summary}

**Your Analysis:**
"""