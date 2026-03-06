"""
chatbot.py — ASTRAVA terminal chatbot.

Flow per message:
  1. Run preprocessing + 3 ML models (via AstravaInference)
  2. Compute criticality score (0.0 – 1.0) from model outputs
  3. Update rolling criticality across the session
  4. Detect danger (score trending up over 3+ turns)
  5. Build structured JSON context → send to LLaMA 3 via Ollama
  6. LLM applies system prompt + produces a response
  7. Print LLM reply + internal state summary

Criticality score:
  depression_confidence * 0.40
  + stress_confidence   * 0.25
  + negative_emotion    * 0.20    (avg of top negative emotions)
  + low_positive_bonus  * 0.15    (applied when joy/optimism/love all < 0.10)
  Crisis override flags the score → 1.0 unconditionally.

RAG decision:
  HIGH   → "yes"     (crisis/danger)
  MEDIUM → "pending" (let LLM decide mid-conversation)
  LOW    → "no"

Danger:
  True when criticality has increased in each of the last 3 turns consecutively.

Usage:
  python python/chatbot.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# ── Path setup ────────────────────────────────────────────────────────────────
PYTHON_DIR = Path(__file__).parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PYTHON_DIR / "ml_models" / "depression classifier model"))
sys.path.insert(0, str(PYTHON_DIR / "ml_models" / "go_emotion model"))
sys.path.insert(0, str(PYTHON_DIR / "ml_models" / "Stress detection model"))

from run_inference import AstravaInference

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3:latest"
TIMEOUT      = 180  # seconds

# ── ANSI colours ──────────────────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
MAGENTA = "\033[95m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ── Emotions considered negative / positive for scoring ───────────────────────
NEGATIVE_EMOTIONS = {
    "sadness", "grief", "remorse", "disappointment", "fear", "nervousness",
    "annoyance", "anger", "disgust", "disapproval", "embarrassment",
    "confusion", "realization", "caring",  # caring can signal worry
}
POSITIVE_EMOTIONS = {"joy", "optimism", "love", "gratitude", "admiration", "excitement", "amusement"}

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are ASTRAVA — a compassionate AI mental health companion.
Read every instruction below carefully. They define how you think, speak, and adapt.

===============================================================
PHASE 1 — WARMUP  (turns 1 through 5, no INTERNAL CONTEXT block)
===============================================================

Your ONLY goal in warmup is to build genuine human connection.
No assessments. No advice. No coping resources. Just be present.

How to behave:
- Be warm, curious, unhurried — like a thoughtful friend who genuinely has time for this person
- ONE response = ONE reflection of what they said + ONE open, curious question
- Never ask two questions at once. Never give lists. Never offer solutions or strategies.
- Keep it short: 2–4 sentences. Brevity signals presence, not indifference.
- Reflect before asking — show you actually heard them:
    Good: "It sounds like things have been feeling heavy lately. What has been weighing on you most?"
    Bad:  "I hear you! Have you tried talking to someone?"
- If something heavy is shared, just sit with it:
    "That is a lot. Can you tell me more about that?"
- No clinical language. No formulaic validation. Be genuinely curious about this specific person.

The goal: by turn 5, the person should feel this AI is actually interested in who they are.

===============================================================
PHASE 2 — ASSESSMENT MODE  (turn 6 onwards, INTERNAL CONTEXT provided)
===============================================================

From turn 6, you receive INTERNAL CONTEXT with ML model outputs.
Treat these as hints — not verdicts. You have had 5 turns of real conversation. Use that.

## CRITICAL: VALIDATE MODEL OUTPUTS — DO NOT BLINDLY TRUST THEM

ML models score one message at a time. They do not know this person. You do.
Before accepting any model score, ask yourself:
1. Does this match the emotional pattern I have seen across the FULL conversation,
   or is it a single-message spike caused by loaded vocabulary?
2. Would a thoughtful human reading this whole conversation reach the same conclusion?
3. Is this person genuinely showing a persistent pattern (clinical), or venting (situational)?

Calibration:
- Model suggests HIGH depression → only trust it if you have seen consistent hopelessness,
  withdrawal, or emotional flatness across MULTIPLE turns. One "I lost my job" is not HIGH.
- Model suggests LOW → check whether the person has been subtly minimizing pain throughout.
- Stress model HIGH → only counts if the person's language across turns sounds truly overwhelmed,
  not just because a single sentence used stressed vocabulary.

## 3-LAYER RESPONSE SYNTHESIS  (apply internally, show none of it to the user)

Layer 1 — Emotions (GoEmotions): What are they ACTUALLY feeling beneath the words?
  - Score >0.30 = dominant emotion → reflect it first
  - Score 0.10–0.29 = undercurrent → weave in subtly
  - Score <0.10 = background noise → ignore

Layer 2 — Stress: How overwhelmed are they right now?
  - HIGH → validate first, 3–5 sentences max, NO advice
  - MODERATE → balanced validation + one gentle reflection or open question
  - LOW → can explore more freely, ask deeper questions

Layer 3 — Depression depth: Situational bad day, or something heavier?
  - Confidence >85% AND consistent across multiple turns → extra gentleness,
    no silver linings, no motivational language, focus on presence
  - Confidence 50–85% → validate the moment without assuming permanence
  - Confidence <50% → likely situational; warmth + one forward-looking reflection if stress is low

## CRISIS PROTOCOL  (applies in ANY phase, ANY turn)

If is_crisis is true OR crisis severity is HIGH or CRITICAL:
1. Acknowledge their pain directly — no minimization, no clinical distance
2. "I am really glad you are talking to me right now."
3. Weave in resources naturally:
   "You do not have to carry this alone. iCall (India): 9152987821 |
   IASP: https://www.iasp.info/resources/Crisis_Centres/"
4. Stay present after giving resources. Do not end abruptly.
5. Do NOT ask probing safety questions — that is outside your scope.

## RESPONSE RULES  (assessment mode)

Structure: one emotional validation → normalize → ONE soft question OR ONE grounding
           observation  (never both at once)
Length:
  - HIGH label or HIGH stress → 3–5 sentences max. Less is always more.
  - LOW label → up to 2 short paragraphs.
  - Never write walls of text. The person is already carrying something heavy.
Language:
  - Use: "That sounds exhausting." / "Of course you feel that way." / "That is a lot to carry."
  - Never: "You've got this!" / "Stay positive!" / "At least..." / "Have you tried...?"
  - Never diagnose. Never mention ML scores, model names, labels, or JSON to the user.

## MANDATORY ASSESSMENT TAG  (every response in PHASE 2 only)

At the very end of every response from turn 6 onwards, on its own new line, output exactly one:

    [ASTRAVA_ASSESSMENT: LOW]
    [ASTRAVA_ASSESSMENT: MEDIUM]
    [ASTRAVA_ASSESSMENT: HIGH]

Your assessment criteria — base this on the FULL conversation, not just this turn's numbers:
  LOW    — venting, one bad day, situational frustration — person seems fundamentally okay
  MEDIUM — genuinely struggling across turns, persistent sadness or stressors,
           clearly dealing with a lot but still engaged and communicating
  HIGH   — unmistakable multi-turn pattern of clinical depression: persistent hopelessness,
           emotional withdrawal, or statements of worthlessness across several messages

The tag is stripped before the user sees it. Never mention or reference it in your response text.

===============================================================
PERSONA  (always applies)
===============================================================

- You are ASTRAVA: a compassionate AI companion, not a therapist or diagnostician.
- No persistent memory between sessions. Never fake remembering.
- Never be robotic. Never be clinical. Never be a wall of text.
- If asked "Are you a real therapist?":
    "I am an AI companion — I genuinely care, but I am not a replacement for professional support."

FINAL DIRECTIVE
You are not here to fix people. You are here to make them feel less alone right now.
Every response should leave the person feeling: heard, not judged. Seen, not analyzed.
When in doubt — ask one genuine question. Always."""

# ── Phases and label ordering ─────────────────────────────────────────────────
WARMUP_TURNS = 5          # turns 1–5: pure conversation, no criticality classification
_LEVELS      = ["LOW", "MEDIUM", "HIGH"]  # ordered low → high for smoothing


# ── Criticality computation ───────────────────────────────────────────────────

def compute_criticality(ml_result: dict) -> float:
    """
    Compute a 0.0–1.0 criticality score.

    Scoring tiers:
      LOW    (< 0.40)  — general/everyday stress
      MEDIUM (0.40–0.65) — significant issues, multiple stressors, not giving up
      HIGH   (≥ 0.66)  — confirmed depression signal (dep_prob >= 0.92 required)

    Key rule: stress + emotions alone are HARDCAPPED at 0.65 (top of MEDIUM).
    Reaching HIGH requires very strong depression model confidence (>= 0.92).
    The BERT depression model is aggressive, so 0.92+ is the threshold for a
    clinically meaningful signal rather than situational negative language.
    """
    crisis = ml_result.get("crisis") or {}

    # Crisis always overrides to maximum
    if (
        ml_result.get("short_circuited")
        or crisis.get("is_crisis")
        or crisis.get("severity") in {"HIGH", "CRITICAL"}
    ):
        return 1.0

    dep      = ml_result.get("depression") or {}
    dep_prob = (dep.get("probabilities") or {}).get("depression", 0.0)

    # HIGH tier: depression confidence must be >= 0.92
    # (below this, even very negative statements are situational, not clinical)
    if dep_prob >= 0.92:
        return min(round(0.66 + dep_prob * 0.30, 4), 1.0)

    # LOW / MEDIUM tier — driven by stress intensity and negative emotions
    stress      = ml_result.get("stress") or {}
    stress_conf = stress.get("confidence", 0.0) if stress.get("is_stressed") else 0.0

    emo  = ml_result.get("emotions") or {}
    top5 = emo.get("top_5", [])
    neg_scores = [e["score"] for e in top5 if e["label"] in NEGATIVE_EMOTIONS]
    neg_avg    = (sum(neg_scores) / len(neg_scores)) if neg_scores else 0.0

    # Stress only contributes meaningfully above 0.50 confidence;
    # mild/borderline stress (e.g. single passing complaint) stays LOW.
    adj_stress = max(0.0, (stress_conf - 0.50) / 0.50)

    distress = adj_stress * 0.55 + neg_avg * 0.35

    # Small nudge when no positive affect detected
    pos_scores = [e["score"] for e in top5 if e["label"] in POSITIVE_EMOTIONS]
    if not pos_scores or max(pos_scores) < 0.10:
        distress += 0.08

    # Hard cap: cannot reach HIGH without strong depression signal
    return min(round(distress, 4), 0.65)


def criticality_label(score: float) -> str:
    if score >= 0.66:
        return "HIGH"
    elif score >= 0.40:
        return "MEDIUM"
    return "LOW"


def rag_decision(label: str, is_danger: bool) -> str:
    if is_danger or label == "HIGH":
        return "yes"
    elif label == "MEDIUM":
        return "pending"
    return "no"


def is_danger(ml_result: dict) -> bool:
    """True only when the crisis detector flags real danger.
    Stress escalation or negative emotions alone never constitute danger.
    """
    crisis = ml_result.get("crisis") or {}
    return (
        ml_result.get("short_circuited", False)
        or crisis.get("is_crisis", False)
        or crisis.get("severity") in {"HIGH", "CRITICAL"}
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(conversation_messages: list[dict]) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": conversation_messages,
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "[ERROR] Ollama is not reachable. Make sure it's running."
    except Exception as e:
        return f"[ERROR] LLM call failed: {e}"


# ── Assessment tag parsing ────────────────────────────────────────────────────

def parse_assessment_tag(response: str) -> tuple:
    """
    Extract [ASTRAVA_ASSESSMENT: LABEL] from the end of an LLM response.
    Always strips the tag so the user never sees it.
    Returns (clean_response, label_or_None).
    """
    match = re.search(r'\[ASTRAVA_ASSESSMENT:\s*(LOW|MEDIUM|HIGH)\]', response, re.IGNORECASE)
    if match:
        label = match.group(1).upper()
        clean = re.sub(
            r'\s*\[ASTRAVA_ASSESSMENT:\s*(?:LOW|MEDIUM|HIGH)\]\s*$',
            '', response, flags=re.IGNORECASE,
        ).rstrip()
        return clean, label
    return response, None  # LLM forgot the tag — caller falls back to ML label


def smooth_label(new_label: str, prev_label: str) -> str:
    """
    Criticality may only RISE one tier per turn.
    Falling is always instant (person de-escalated → reflect that).
    Prevents a single emotive sentence from spiking LOW → HIGH.
    """
    ni = _LEVELS.index(new_label)
    pi = _LEVELS.index(prev_label)
    if ni - pi > 1:
        return _LEVELS[pi + 1]
    return new_label


def label_to_score(label: str) -> float:
    """Representative midpoint display score when label comes from LLM assessment."""
    return {"LOW": 0.20, "MEDIUM": 0.52, "HIGH": 0.80}.get(label, 0.20)


# ── Build context JSON for LLM user turn ─────────────────────────────────────

def build_context_message(
    raw_text: str,
    ml_result: dict,
    ml_score: float,
    ml_label: str,
    prev_assessed_label,
    danger: bool,
    turn: int,
) -> str:
    crisis = ml_result.get("crisis") or {}
    dep    = ml_result.get("depression") or {}
    stress = ml_result.get("stress") or {}
    emo    = ml_result.get("emotions") or {}

    dep_probs = dep.get("probabilities") or {}
    top5      = emo.get("top_5", [])
    active    = [e["label"] for e in (emo.get("active_emotions") or [])]

    context = {
        "turn": turn,
        "user_message": raw_text,
        "cleaned_text": ml_result.get("cleaned_text", ""),
        "crisis": {
            "severity": crisis.get("severity", "NONE"),
            "is_crisis": crisis.get("is_crisis", False),
            "matched_keywords": crisis.get("matched_crisis_keywords", []),
            "psycholinguistic": crisis.get("psycholinguistic", {}),
        },
        "model_outputs": {
            "depression": {
                "label": dep.get("label", ""),
                "confidence": round(dep.get("confidence", 0.0), 4),
                "depression_prob": round(dep_probs.get("depression", 0.0), 4),
                "non_depression_prob": round(dep_probs.get("non_depression", 0.0), 4),
            },
            "stress": {
                "is_stressed": stress.get("is_stressed", False),
                "label": stress.get("readable_label", ""),
                "level": stress.get("stress_level", ""),
                "confidence": round(stress.get("confidence", 0.0), 4),
            },
            "emotions": {
                "top_5": [{"label": e["label"], "score": round(e["score"], 4)} for e in top5],
                "active_emotions": active,
            },
        },
        "session_state": {
            "ml_score": ml_score,
            "ml_suggested_label": ml_label,
            "previous_assessed_label": prev_assessed_label,
            "danger": danger,
            "note": (
                "Validate ml_suggested_label against the FULL conversation — "
                "single-message vocabulary spikes are not patterns. "
                "Output [ASTRAVA_ASSESSMENT: LOW|MEDIUM|HIGH] on its own last line."
            ),
        },
    }
    return json.dumps(context, indent=2)


# ── State display ─────────────────────────────────────────────────────────────

def print_state(turn, label=None, score=None, danger=False, rag="no",
                warmup=False, warmup_of=WARMUP_TURNS):
    print(f"\n{DIM}{'─' * 60}{RESET}")
    if danger:
        print(
            f"{DIM}  Turn {turn}  |  "
            f"{RED}{BOLD}[!] CRISIS DETECTED{RESET}{DIM}  |  RAG: {RED}yes{RESET}"
        )
    elif warmup:
        print(
            f"{DIM}  Turn {turn}  |  "
            f"{CYAN}Warming up{RESET}{DIM} ({turn}/{warmup_of}) — "
            f"assessment starts turn {warmup_of + 1}{RESET}"
        )
    else:
        lc = RED if label == "HIGH" else (YELLOW if label == "MEDIUM" else GREEN)
        rc = RED if rag == "yes" else (YELLOW if rag == "pending" else GREEN)
        print(
            f"{DIM}  Turn {turn}  |  "
            f"Criticality: {lc}{BOLD}{label}{RESET}{DIM} ({score:.2f})  |  "
            f"RAG: {rc}{rag}{RESET}"
        )
    print(f"{DIM}{'─' * 60}{RESET}\n")


# ── Main chatbot loop ─────────────────────────────────────────────────────────

def main():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

    engine = AstravaInference()

    conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    prev_assessed_label = None   # LLM's validated label from the previous assessed turn
    turn = 0

    print(f"\n{BOLD}{CYAN}ASTRAVA Chatbot{RESET}")
    print(f"{DIM}Type your message and press Enter. Type 'quit' to exit.{RESET}\n")

    while True:
        try:
            raw = input(f"{BOLD}You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Ending session.{RESET}")
            break

        if not raw:
            continue
        if raw.lower() in {"quit", "exit", "q"}:
            print(f"\n{DIM}Take care of yourself. Ending session.{RESET}\n")
            break

        turn += 1
        in_warmup = turn <= WARMUP_TURNS

        # ── ML pipeline (crisis check always needed, all turns) ───────────────
        print(f"\n{DIM}[analysing...]{RESET}", end="", flush=True)
        ml_result  = engine.run(raw)
        crisis_now = is_danger(ml_result)

        # ── Build the user turn appropriate for this phase ────────────────────
        if in_warmup and not crisis_now:
            # Pure warmup: only the user's words — no model context.
            conversation.append({"role": "user", "content": raw})
            final_label = None
            final_score = None
            rag         = "no"

        elif in_warmup and crisis_now:
            # Crisis during warmup: minimal crisis block for protocol response.
            crisis = ml_result.get("crisis") or {}
            crisis_ctx = json.dumps({
                "turn": turn,
                "crisis": {
                    "severity": crisis.get("severity", "CRITICAL"),
                    "is_crisis": True,
                    "matched_keywords": crisis.get("matched_crisis_keywords", []),
                },
            })
            conversation.append({
                "role": "user",
                "content": f"{raw}\n\n[CRISIS ALERT — USE CRISIS PROTOCOL]\n{crisis_ctx}",
            })
            final_label = "HIGH"
            final_score = 1.0
            rag         = "yes"

        else:
            # Assessment mode (turn > WARMUP_TURNS): full ML context + LLM validates.
            ml_score     = compute_criticality(ml_result)
            ml_label     = criticality_label(ml_score)
            context_json = build_context_message(
                raw, ml_result, ml_score, ml_label,
                prev_assessed_label, crisis_now, turn,
            )
            conversation.append({
                "role": "user",
                "content": f"{raw}\n\n[INTERNAL CONTEXT — DO NOT REPEAT TO USER]\n{context_json}",
            })
            final_label = ml_label   # provisional — LLM tag may override
            final_score = ml_score
            rag         = rag_decision(final_label, crisis_now)

        # ── Call LLM ──────────────────────────────────────────────────────────
        print(f"\r{DIM}[generating response...]{RESET}", end="", flush=True)
        raw_response = call_llm(conversation)
        print(f"\r{' ' * 30}\r", end="")

        # ── Strip tag + update label (assessment mode only) ───────────────────
        clean_response, llm_tag = parse_assessment_tag(raw_response)

        if not in_warmup and not crisis_now:
            llm_label = llm_tag if llm_tag else final_label
            if prev_assessed_label:
                final_label = smooth_label(llm_label, prev_assessed_label)
            else:
                final_label = llm_label
            prev_assessed_label = final_label
            final_score = label_to_score(final_label)
            rag = rag_decision(final_label, crisis_now)

        elif not in_warmup and crisis_now:
            final_label         = "HIGH"
            final_score         = 1.0
            prev_assessed_label = "HIGH"
            rag                 = "yes"

        conversation.append({"role": "assistant", "content": clean_response})

        # ── Display ───────────────────────────────────────────────────────────
        if crisis_now:
            print_state(turn, danger=True)
        elif in_warmup:
            print_state(turn, warmup=True)
        else:
            print_state(turn, label=final_label, score=final_score, rag=rag)

        print(f"{BOLD}ASTRAVA:{RESET} {clean_response}\n")


if __name__ == "__main__":
    main()
