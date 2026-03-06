"""
backend/main.py — ASTRAVA FastAPI server.

Wraps the chatbot_groq.py pipeline so the React frontend can call it.
Sessions are stored in-memory (per process lifetime).

Run:
  uvicorn backend.main:app --reload --port 8000
  (from repo root)

Endpoints:
  POST /api/session          → create a new chat session, returns { session_id }
  POST /api/chat             → send a message, returns { message, ... }
  DELETE /api/session/{id}   → clean up a session
"""

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────────
PYTHON_DIR = Path(__file__).parent.parent / "python"
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PYTHON_DIR / "ml_models" / "depression classifier model"))
sys.path.insert(0, str(PYTHON_DIR / "ml_models" / "go_emotion model"))
sys.path.insert(0, str(PYTHON_DIR / "ml_models" / "Stress detection model"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PYTHON_DIR / ".env")

from run_inference import AstravaInference
from chatbot_groq import (
    SYSTEM_PROMPT,
    WARMUP_TURNS,
    GROQ_MODEL,
    compute_criticality,
    criticality_label,
    rag_decision,
    is_danger,
    build_context_message,
    parse_assessment_tag,
    smooth_label,
    label_to_score,
)
from groq import Groq

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ASTRAVA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",   # Vite sometimes uses 5174 when 5173 is busy
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global singletons (loaded once at startup) ────────────────────────────────
_engine: Optional[AstravaInference] = None
_groq: Optional[Groq] = None

# in-memory session store: session_id → session dict
# session dict: { conversation, prev_assessed_label, turn }
_sessions: dict = {}


@app.on_event("startup")
async def startup():
    global _engine, _groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found. Add it to python/.env")
    _engine = AstravaInference()
    _groq   = Groq(api_key=api_key)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    message: str
    session_id: str
    turn: int
    in_warmup: bool
    criticality_label: Optional[str] = None
    score: Optional[float] = None
    rag: Optional[str] = None
    danger: bool = False


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/api/session")
def new_session():
    """Create a fresh chat session. Call this when the user opens the chat page."""
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "conversation":         [{"role": "system", "content": SYSTEM_PROMPT}],
        "prev_assessed_label":  None,
        "turn":                 0,
    }
    return {"session_id": sid}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a user message and get ASTRAVA's response."""
    if req.session_id not in _sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. POST /api/session to create one.",
        )

    sess = _sessions[req.session_id]
    sess["turn"] += 1
    turn      = sess["turn"]
    in_warmup = turn <= WARMUP_TURNS

    # ── ML pipeline ───────────────────────────────────────────────────────────
    ml_result  = _engine.run(req.message)
    crisis_now = is_danger(ml_result)

    final_label: Optional[str] = None
    final_score: Optional[float] = None
    rag = "no"

    if in_warmup and not crisis_now:
        # Pure warmup — no ML context sent to LLM
        sess["conversation"].append({"role": "user", "content": req.message})

    elif in_warmup and crisis_now:
        # Crisis during warmup — send crisis block only
        crisis     = ml_result.get("crisis") or {}
        crisis_ctx = json.dumps({
            "turn": turn,
            "crisis": {
                "severity":         crisis.get("severity", "CRITICAL"),
                "is_crisis":        True,
                "matched_keywords": crisis.get("matched_crisis_keywords", []),
            },
        })
        sess["conversation"].append({
            "role":    "user",
            "content": f"{req.message}\n\n[CRISIS ALERT — USE CRISIS PROTOCOL]\n{crisis_ctx}",
        })
        final_label = "HIGH"
        final_score = 1.0
        rag         = "yes"

    else:
        # Assessment mode (turn > WARMUP_TURNS)
        ml_score     = compute_criticality(ml_result)
        ml_label     = criticality_label(ml_score)
        context_json = build_context_message(
            req.message, ml_result, ml_score, ml_label,
            sess["prev_assessed_label"], crisis_now, turn,
        )
        sess["conversation"].append({
            "role":    "user",
            "content": f"{req.message}\n\n[INTERNAL CONTEXT — DO NOT REPEAT TO USER]\n{context_json}",
        })
        final_label = ml_label
        final_score = ml_score
        rag         = rag_decision(final_label, crisis_now)

    # ── Call Groq ─────────────────────────────────────────────────────────────
    try:
        completion = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=sess["conversation"],
            temperature=0.75,
            max_tokens=512,
        )
        raw_response = completion.choices[0].message.content
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Groq error: {exc}")

    # ── Strip tag + finalise label ────────────────────────────────────────────
    clean_response, llm_tag = parse_assessment_tag(raw_response)

    if not in_warmup and not crisis_now:
        llm_label = llm_tag if llm_tag else final_label
        if sess["prev_assessed_label"]:
            final_label = smooth_label(llm_label, sess["prev_assessed_label"])
        else:
            final_label = llm_label
        sess["prev_assessed_label"] = final_label
        final_score = label_to_score(final_label)
        rag         = rag_decision(final_label, crisis_now)

    elif not in_warmup and crisis_now:
        final_label                 = "HIGH"
        final_score                 = 1.0
        sess["prev_assessed_label"] = "HIGH"
        rag                         = "yes"

    sess["conversation"].append({"role": "assistant", "content": clean_response})

    return ChatResponse(
        message=clean_response,
        session_id=req.session_id,
        turn=turn,
        in_warmup=in_warmup,
        criticality_label=final_label,
        score=final_score,
        rag=rag,
        danger=crisis_now,
    )


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    """Free up memory for a session when the user closes the chat."""
    _sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(_sessions)}
