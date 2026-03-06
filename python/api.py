"""
api.py — FastAPI wrapper using local Ollama LLM + shared logic from chatbot.py.

Runs on http://localhost:8080
Frontend calls POST /api/chat  with { session_id, message }

Requires: Ollama running locally (http://localhost:11434) with llama3 pulled.

Start:
  python -m uvicorn python.api:app --host 0.0.0.0 --port 8080 --reload
"""

import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"   # suppress TF info/warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import smtplib
import sys
import time
import uuid
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import requests as http_requests

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pydantic import BaseModel

# ── env + paths ───────────────────────────────────────────────────────────────
ENV_PATH   = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ── Ollama config ─────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
OLLAMA_TIMEOUT = 180  # seconds

# ── Alert email config (optional — alerts silently disabled if unconfigured) ──
ALERT_EMAIL_FROM   = os.getenv("ALERT_EMAIL_FROM", "")
ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")
ALERT_EMAIL_TO     = os.getenv("ALERT_EMAIL_TO", "")
ALERT_SMTP_SERVER  = os.getenv("ALERT_SMTP_SERVER", "smtp.gmail.com")
ALERT_SMTP_PORT    = int(os.getenv("ALERT_SMTP_PORT", "587"))
ALERT_CONFIGURED   = bool(ALERT_EMAIL_FROM and ALERT_EMAIL_PASSWORD and ALERT_EMAIL_TO)

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
_mongo_client: MongoClient = None
_db = None
_chats_col = None

def get_db():
    """Lazy connect; returns the chats collection or None if Mongo is down."""
    global _mongo_client, _db, _chats_col
    if _chats_col is not None:
        return _chats_col
    try:
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        _mongo_client.admin.command("ping")          # verify connection
        _db        = _mongo_client["astrava"]
        _chats_col = _db["chats"]
        print("[ASTRAVA API] MongoDB connected ✔")
    except ConnectionFailure:
        print("[ASTRAVA API] MongoDB unavailable — chat persistence disabled.")
    return _chats_col

PYTHON_DIR = Path(__file__).parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PYTHON_DIR / "ml_models" / "depression classifier model"))
sys.path.insert(0, str(PYTHON_DIR / "ml_models" / "go_emotion model"))
sys.path.insert(0, str(PYTHON_DIR / "ml_models" / "Stress detection model"))

# ── import shared chatbot logic ───────────────────────────────────────────────
# We reuse every function and constant from chatbot — no duplication.

from chatbot import (
    SYSTEM_PROMPT,
    WARMUP_TURNS,
    THERAPIST_OFFER_TURN,
    _LEVELS,
    MAX_WINDOW_PAIRS,
    compute_criticality,
    criticality_label,
    rag_decision,
    is_danger,
    parse_assessment_tag,
    parse_therapist_offer_tag,
    smooth_label,
    label_to_score,
    build_context_message,
    build_llm_payload,
    evict_old_turns,
)
from run_inference import AstravaInference

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="ASTRAVA API", version="1.0")

# NOTE: In FastAPI, middleware added LATER wraps middleware added EARLIER.
# CORS must be the outermost layer so error responses also get CORS headers.
# Therefore: add logger first, then CORS on top.

@app.middleware("http")
async def _log_http(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed = (time.monotonic() - start) * 1000
    print(f"[HTTP] {request.method} {request.url.path}  ->  {response.status_code}  ({elapsed:.0f}ms)")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── global singletons (loaded once at startup) ────────────────────────────────
inference_engine: AstravaInference = None

@app.on_event("startup")
async def startup():
    global inference_engine
    print(f"[ASTRAVA API] Using Ollama  model={OLLAMA_MODEL}  url={OLLAMA_URL}")
    print("[ASTRAVA API] Loading ML pipeline (this takes ~10s)...")
    inference_engine = AstravaInference()
    get_db()          # attempt Mongo connection on startup (non-fatal if down)
    print("[ASTRAVA API] Ready.")

# ── in-memory session store ───────────────────────────────────────────────────
# { session_id: { conversation, prev_assessed_label, turn } }
sessions: dict = {}

def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        anon_uid = f"anon_{uuid.uuid4().hex[:12]}"   # random anonymous user id
        sessions[session_id] = {
            "turns_history":       [],    # completed turns (clean text only)
            "summary":             "",    # rolling summary of evicted older turns
            "prev_assessed_label": None,
            "turn":                0,
            "alert_sent":          False,
            "therapist_offered":   False,
            "anon_uid":            anon_uid,
            "started_at":          datetime.now(timezone.utc),
        }
        # Create / re-attach the Mongo document (upsert handles server restarts)
        col = get_db()
        if col is not None:
            col.update_one(
                {"_id": session_id},
                {"$setOnInsert": {
                    "user_id":    anon_uid,
                    "email":      None,
                    "started_at": sessions[session_id]["started_at"],
                    "messages":   [],
                }},
                upsert=True,
            )
    return sessions[session_id]


# ── Emergency alert ───────────────────────────────────────────────────────────
def send_emergency_alert(
    session_id: str,
    location: dict,
    recent_messages: list[dict],
) -> bool:
    """Send an email to the configured emergency contact.
    Returns True on success, False on any failure."""
    if not ALERT_CONFIGURED:
        print("[ASTRAVA ALERT] Email not configured — skipping alert.")
        return False

    lat = location.get("lat", "?")
    lng = location.get("lng", "?")
    maps_url = f"https://www.google.com/maps?q={lat},{lng}"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    short_id  = session_id[:8]

    # build readable recent conversation (last 6 turns, skip system)
    convo_lines = []
    for t in recent_messages[-6:]:
        convo_lines.append(f"User: {t['user'].strip()}")
        if t.get('assistant'):
            convo_lines.append(f"Solace: {t['assistant'].strip()}")
    convo_text = "\n".join(convo_lines) or "(no recent messages)"

    subject = f"[ASTRAVA EMERGENCY] User in crisis — session {short_id}"
    body = f"""ASTRAVA — Emergency Alert
{'=' * 50}

An ASTRAVA user has triggered a crisis alert.

Time     : {timestamp}
Session  : {short_id}

LOCATION
--------
Coordinates : {lat}, {lng}
Google Maps : {maps_url}

RECENT CONVERSATION
-------------------
{convo_text}

{'=' * 50}
Please reach out or send someone to check on this person as soon as possible.

This message was sent automatically by the ASTRAVA mental health platform.
Do NOT reply to this email.
"""

    try:
        msg = MIMEMultipart()
        msg["From"]    = ALERT_EMAIL_FROM
        msg["To"]      = ALERT_EMAIL_TO
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(ALERT_SMTP_SERVER, ALERT_SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD)
            server.sendmail(ALERT_EMAIL_FROM, ALERT_EMAIL_TO, msg.as_string())

        print(f"[ASTRAVA ALERT] Emergency email sent for session {short_id} → {maps_url}")
        return True
    except Exception as exc:
        print(f"[ASTRAVA ALERT] Failed to send email: {exc}")
        return False

# ── request / response models ─────────────────────────────────────────────────
class LocationPayload(BaseModel):
    lat: float
    lng: float

class ChatRequest(BaseModel):
    session_id: str
    message:    str
    location:   Optional[LocationPayload] = None

class TherapistRequest(BaseModel):
    session_id:   str
    name:         Optional[str] = None
    email:        Optional[str] = None
    preference:   Optional[str] = None   # "online" | "in_person" | None
    migrate_chat: bool = True

class SaveChatRequest(BaseModel):
    session_id: str
    user_id:    Optional[str] = None   # real username / display name
    email:      Optional[str] = None   # real email address

class ChatResponse(BaseModel):
    response:           str
    turn:               int
    in_warmup:          bool
    criticality_label:  Optional[str]
    criticality_score:  Optional[float]
    rag:                Optional[str]
    danger:             bool
    alert_sent:         bool = False
    therapist_offered:  bool = False

# ── main chat endpoint ────────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
  try:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    sess = get_session(req.session_id)
    sess["turn"] += 1
    turn      = sess["turn"]
    in_warmup = turn <= WARMUP_TURNS

    # ── Incoming request log ──────────────────────────────────────────────────
    print(f"[REQUEST] POST /api/chat  session={req.session_id[:8]}  msg={req.message[:80]!r}")
    if req.location:
        print(f"[LOCATION] Got GPS coordinates  lat={req.location.lat:.6f}  lng={req.location.lng:.6f}")

    # ── ML pipeline ───────────────────────────────────────────────────────────
    ml_result  = inference_engine.run(req.message)
    crisis_now = is_danger(ml_result)

    final_label = None
    final_score = None
    rag         = "no"

    # ── Build enriched content for this turn's LLM call ──────────────────────
    # ML context is ONLY in the current turn's payload, never stored in history.
    if in_warmup and not crisis_now:
        enriched_content = req.message

    elif in_warmup and crisis_now:
        crisis = ml_result.get("crisis") or {}
        crisis_ctx = json.dumps({
            "turn": turn,
            "crisis": {
                "severity": crisis.get("severity", "CRITICAL"),
                "is_crisis": True,
                "matched_keywords": crisis.get("matched_crisis_keywords", []),
            },
        })
        enriched_content = f"{req.message}\n\n[CRISIS ALERT — USE CRISIS PROTOCOL]\n{crisis_ctx}"
        final_label = "HIGH"
        final_score = 1.0
        rag         = "yes"

    else:
        ml_score       = compute_criticality(ml_result)
        ml_label       = criticality_label(ml_score)
        offer_eligible = (
            turn >= THERAPIST_OFFER_TURN
            and not crisis_now
            and not sess["therapist_offered"]
            and sess["prev_assessed_label"] in {"MEDIUM", "HIGH"}
        )
        context_json = build_context_message(
            req.message, ml_result, ml_score, ml_label,
            sess["prev_assessed_label"], crisis_now, turn,
            therapist_offer_eligible=offer_eligible,
        )
        enriched_content = f"{req.message}\n\n[INTERNAL CONTEXT — DO NOT REPEAT TO USER]\n{context_json}"
        final_label = ml_label
        final_score = ml_score
        rag         = rag_decision(final_label, crisis_now)

    # ── Build windowed payload + call Ollama ──────────────────────────────────
    llm_messages = build_llm_payload(
        SYSTEM_PROMPT, sess["summary"], sess["turns_history"], enriched_content,
    )
    role_seq = [m["role"] for m in llm_messages]
    print(f"[OLLAMA] Sending {len(role_seq)} messages  roles={role_seq}", flush=True)

    try:
        ollama_payload = {
            "model": OLLAMA_MODEL,
            "messages": llm_messages,
            "stream": False,
            "options": {
                "temperature": 0.75,
                "num_predict": 512,
                "num_ctx": 4096,
            },
        }
        ollama_resp = http_requests.post(
            OLLAMA_URL, json=ollama_payload, timeout=OLLAMA_TIMEOUT
        )
        if ollama_resp.status_code != 200:
            print(f"[OLLAMA ERROR] status={ollama_resp.status_code}  body={ollama_resp.text[:500]}", flush=True)
            sess["turn"] -= 1
        ollama_resp.raise_for_status()
        raw_response = ollama_resp.json()["message"]["content"]
    except http_requests.exceptions.ConnectionError:
        print("[OLLAMA ERROR] Cannot reach Ollama. Is it running?", flush=True)
        sess["turn"] -= 1
        raise HTTPException(status_code=502, detail="Ollama is not reachable. Make sure it is running.")
    except Exception as e:
        import traceback as tb
        print(f"\n[OLLAMA ERROR] {type(e).__name__}: {e}", flush=True)
        tb.print_exc()
        sess["turn"] -= 1
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

    # ── Parse assessment tag + therapist offer tag ─────────────────────────────────
    clean_response, llm_tag            = parse_assessment_tag(raw_response)
    clean_response, therapist_offered_now = parse_therapist_offer_tag(clean_response)
    if therapist_offered_now:
        sess["therapist_offered"] = True

    if not in_warmup and not crisis_now:
        llm_label = llm_tag if llm_tag else final_label
        if sess["prev_assessed_label"]:
            final_label = smooth_label(llm_label, sess["prev_assessed_label"])
        else:
            final_label = llm_label
        sess["prev_assessed_label"] = final_label
        final_score = label_to_score(final_label)
        rag = rag_decision(final_label, crisis_now)

    elif not in_warmup and crisis_now:
        final_label                 = "HIGH"
        final_score                 = 1.0
        sess["prev_assessed_label"] = "HIGH"
        rag                         = "yes"

    # ── Store clean turn + evict old turns into summary ───────────────────────
    sess["turns_history"].append({
        "turn": turn, "user": req.message, "assistant": clean_response,
        "label": final_label, "crisis": crisis_now,
    })
    sess["summary"] = evict_old_turns(sess["turns_history"], sess["summary"])

    # ── Persist turn to MongoDB ───────────────────────────────────────────────
    col = get_db()
    if col is not None:
        print(f"[MONGODB] Persisting turn {turn} for session={req.session_id[:8]}...", flush=True)
        col.update_one(
            {"_id": req.session_id},
            {"$push": {"messages": {
                "turn":              turn,
                "role":              "user",
                "text":              req.message,
                "ts":                datetime.now(timezone.utc),
                "ml": {
                    "depression": {
                        "label":      (ml_result.get("depression") or {}).get("label"),
                        "confidence": round((ml_result.get("depression") or {}).get("confidence", 0), 3),
                    },
                    "stress": {
                        "label":      (ml_result.get("stress") or {}).get("label"),
                        "confidence": round((ml_result.get("stress") or {}).get("confidence", 0), 3),
                    },
                    "emotion": {
                        "top":        ((ml_result.get("emotions") or {}).get("top_5") or [{}])[0].get("label"),
                        "score":      round(((ml_result.get("emotions") or {}).get("top_5") or [{}])[0].get("score", 0), 3),
                    },
                    "crisis":    crisis_now,
                    "criticality_label": final_label,
                    "rag":        rag,
                },
            }}},
        )
        col.update_one(
            {"_id": req.session_id},
            {"$push": {"messages": {
                "turn":  turn,
                "role":  "assistant",
                "text":  clean_response,
                "ts":    datetime.now(timezone.utc),
            }}},
        )
        print(f"[MONGODB] Turn {turn} persisted OK  (user + assistant messages)", flush=True)

    # ── Rich console log (visible in terminal running uvicorn) ───────────────
    dep_r   = ml_result.get("depression") or {}
    str_r   = ml_result.get("stress")     or {}
    emo_top5 = (ml_result.get("emotions") or {}).get("top_5") or []
    top_emo_label = emo_top5[0]["label"] if emo_top5 else "?"
    top_emo_score = emo_top5[0]["score"] if emo_top5 else 0.0
    sep     = "-" * 60
    bar     = "=" * 60
    print(f"\n{bar}")
    print(f"  TURN {turn}  |  session={req.session_id[:8]}  |  {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    print(sep)
    print(f"  USER  : {req.message[:120]}")
    if req.location:
        print(f"  LOCATION  : lat={req.location.lat:.6f}  lng={req.location.lng:.6f}")
    print(sep)
    print(f"  DEPRESSION  : {dep_r.get('label','?'):8s}  conf={dep_r.get('confidence',0):.3f}")
    print(f"  STRESS      : {str_r.get('label','?'):8s}  conf={str_r.get('confidence',0):.3f}")
    print(f"  EMOTION     : {top_emo_label:14s}  score={top_emo_score:.3f}")
    if len(emo_top5) > 1:
        extras = ", ".join(f"{e['label']}({e['score']:.2f})" for e in emo_top5[1:4])
        print(f"              + {extras}")
    print(f"  CRISIS      : {'YES [!]' if crisis_now else 'no'}")
    print(sep)
    print(f"  CRITICALITY : {final_label or '(warmup)'}  |  score={final_score or 0:.2f}  |  RAG={rag}")
    print(sep)
    print(f"  BOT   : {clean_response[:120]}{'...' if len(clean_response) > 120 else ''}")
    print(f"{bar}\n")

    # ── Emergency alert (fires once per session on confirmed crisis with location) ─
    alert_fired = False
    if crisis_now and req.location and not sess["alert_sent"]:
        location_dict = {"lat": req.location.lat, "lng": req.location.lng}
        alert_fired = send_emergency_alert(
            req.session_id,
            location_dict,
            sess["turns_history"],
        )
        if alert_fired:
            sess["alert_sent"] = True

    return ChatResponse(
        response           = clean_response,
        turn               = turn,
        in_warmup          = in_warmup and not crisis_now,
        criticality_label  = final_label,
        criticality_score  = final_score,
        rag                = rag,
        danger             = crisis_now,
        alert_sent         = alert_fired,
        therapist_offered  = therapist_offered_now,
    )
  except HTTPException:
    raise
  except Exception as exc:
    import traceback, sys
    print(f"\n{'!'*60}", flush=True)
    print(f"[CHAT ENDPOINT CRASH] {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    print(f"{'!'*60}\n", flush=True)
    raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.post("/api/save-chat")
async def save_chat(req: SaveChatRequest):
    """
    Attach a real user identity to an anonymous chat document.
    Call this when the user chooses to save their conversation.
    """
    col = get_db()
    if col is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    update = {}
    if req.user_id:
        update["user_id"] = req.user_id
    if req.email:
        update["email"] = req.email
    if not update:
        raise HTTPException(status_code=400, detail="Provide at least user_id or email.")
    result = col.update_one({"_id": req.session_id}, {"$set": update})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"saved": True, "session_id": req.session_id}


@app.post("/api/request-therapist")
async def request_therapist(req: TherapistRequest):
    """
    User has expressed interest in connecting with a real therapist.
    Stores the request on the session document and marks it in memory.
    """
    col = get_db()
    update: dict = {
        "therapist_request": {
            "requested_at": datetime.now(timezone.utc),
            "name":         req.name,
            "email":        req.email,
            "preference":   req.preference,
            "migrate_chat": req.migrate_chat,
        }
    }
    if col is not None:
        col.update_one({"_id": req.session_id}, {"$set": update})
    # also mark in-memory session if still alive
    if req.session_id in sessions:
        sessions[req.session_id]["therapist_request"] = update["therapist_request"]
    print(f"[ASTRAVA] Therapist request for session {req.session_id[:8]} | "
          f"email={req.email} pref={req.preference} migrate={req.migrate_chat}")
    return {"queued": True}


@app.get("/health")
async def health():
    mongo_ok = False
    try:
        col = get_db()
        mongo_ok = col is not None
    except Exception:
        pass
    return {"status": "ok", "model_loaded": inference_engine is not None, "mongo": mongo_ok}
