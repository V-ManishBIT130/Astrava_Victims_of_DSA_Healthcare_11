# ASTRAVA — Continuation Guide

This document is specifically written for any developer or AI model that is picking up this project mid-way. It answers: where exactly are we, what needs to be done next, and in what order?

---

## Project Maturity at Time of This Documentation

The Python ML layer is complete and production-ready. All three models work, the preprocessing pipeline is clean, and the unified inference runner can be demonstrated. This layer is NOT what needs work.

The backend and frontend have not been built — only folder scaffolding exists. The entire remaining work is the Node.js backend, React frontend, LLM integration, RAG system, and wiring everything together.

---

## What to Build Next (In This Order)

### Priority 1: FastAPI Server for the Python Layer

The existing Python logic runs only from the command line. For the backend to use it, it needs to be served as an HTTP API.

Create `python/main.py` as the FastAPI entrypoint. It needs one endpoint: `POST /api/analyze`. This endpoint should instantiate (or reuse a singleton of) the `AstravaInference` class from `python/run_inference.py`, call its `run()` method with the posted text, and return the JSON result.

The server should be started with: `uvicorn main:app --host 0.0.0.0 --port 8000`

The endpoint should accept a JSON body with at minimum:
- `message` (string): the raw user text
- `user_id` (string): the user's identifier
- `session_id` (string): the current session identifier

And return the full result dictionary from `run()`, including preprocessing data, crisis data, depression score, stress score, emotion scores, and total inference time.

See `python/run_inference.py` for the exact shape of the result dictionary. The `AstravaInference` class is already written and tested — it just needs to be wrapped in FastAPI.

---

### Priority 2: Node.js Backend Foundation

Initialize the Node.js project in `backend/` if not already done:
- `package.json` with dependencies: `express`, `socket.io`, `axios`, `mongoose`, `dotenv`, `cors`
- Entry point: `backend/src/server.js`

The server setup should:
- Create an Express app
- Attach Socket.IO to the HTTP server
- Connect to MongoDB Atlas (connection string from `.env`)
- Listen on a port (default 5000)

---

### Priority 3: Chat WebSocket Handler

In `backend/src/socket/chatSocket.js`, set up the Socket.IO event handlers:
- On `user_message` event: receive the message, call the orchestrator
- Emit `typing_indicator` immediately when a message is received
- Emit `bot_response` when the orchestrator returns the LLM reply
- Emit `crisis_alert` (separate event) when risk level is HIGH
- Clear `typing_indicator` after response is emitted

---

### Priority 4: Orchestrator Service

`backend/src/services/orchestrator.js` is the brain of the backend. It receives a message and drives the entire pipeline:

1. Call Python FastAPI `POST /api/analyze` via Axios with the raw message — wait for ML scores and risk level
2. If risk level is MEDIUM: call the RAG retriever to get top-3 passages
3. If risk level is HIGH or CRITICAL (or short-circuited): skip RAG and activate crisis protocol
4. Call Supermemory search to get the user's existing emotional memory
5. Build the full LLM prompt (system prompt + user message + ML scores + RAG passages + memory context + conversation history)
6. Call the LLM client (Ollama first, Groq fallback)
7. Return the LLM reply text and the risk level to the WebSocket handler
8. Asynchronously write to Supermemory and MongoDB after the response is delivered

---

### Priority 5: LLM Client

`backend/src/services/llmClient.js` needs to:
- Attempt a POST to `http://localhost:11434/api/chat` (Ollama) with the prompt and model name `llama3:8b`
- Set a 10-second timeout
- If Ollama fails or times out, fall back to Groq: `POST https://api.groq.com/openai/v1/chat/completions` with the same prompt formatted as OpenAI-style messages
- Groq API key comes from `.env`
- Return the generated text

The system prompt needs to be carefully written. It must define:
- The chatbot's persona (empathetic, warm, non-judgmental, clinically informed but conversational)
- Risk-level-specific behavior rules (LOW: conversational, MEDIUM: use RAG techniques, HIGH: safety-first)
- When to include helpline numbers (HIGH only)
- Not to diagnose or prescribe
- To treat the user with dignity and care

---

### Priority 6: Supermemory Service

`backend/src/services/supermemory.js` needs two functions:

**Search:** Given a user_id and an emotional context query string, POST to `https://api.supermemory.ai/v1/search` with the user_id and query. Return the list of memory objects.

**Store:** After a response is delivered, POST to `https://api.supermemory.ai/v1/memories` with the user_id, a natural language summary of the emotional state, risk level, what techniques were suggested, and whether crisis was escalated. Add a metadata object with session_id, timestamp, and emotion labels.

Supermemory API key comes from `.env`.

---

### Priority 7: RAG System

This is the most self-contained backend component. It involves three phases:

**Phase A — Content curation:** Add text documents to the four data directories in `backend/src/rag/data/`. Each file should be a short paragraph (50–150 words) describing a single technique or resource. Aim for 8–20 files per directory. The content must be plain, accessible, and clinically accurate.

**Phase B — Indexer script (`backend/src/rag/indexer.js`):** Build FAISS indices from the document files. This is a run-once offline script. It reads documents, encodes them with `all-MiniLM-L6-v2`, and saves the resulting FAISS index files to disk.

**Phase C — Retriever (`backend/src/rag/retriever.js`):** At runtime, encodes the query (emotional state + cleaned message), loads the FAISS index files from disk, runs nearest-neighbor search, and returns the top-3 most relevant passages. Only queries the CBT, breathing, and coping indices (not crisis lines) for retrieval to inject into the LLM.

**Note on FAISS in Node.js:** `faiss-node` is the standard Node.js binding. Alternatively, a thin Python sidecar endpoint (`POST /retrieve`) can be added to the FastAPI server and called via Axios — this trades latency for simplicity.

---

### Priority 8: React Frontend

The frontend currently has nothing. Starting from scratch:

1. Scaffold with Vite: `npm create vite@latest frontend -- --template react`
2. Install Tailwind CSS and Socket.IO client
3. Build components:
   - `ChatWindow.jsx` — the main container, lists messages, auto-scrolls
   - `MessageBubble.jsx` — individual message display, different styling for user vs. bot
   - `InputBar.jsx` — text input + send button
   - `CrisisOverlay.jsx` — full-screen overlay, hidden by default, shown on `crisis_alert` event. Shows helplines, a phone number input, and calming design
   - `ResourceCard.jsx` — optional card below MEDIUM-risk responses linking to techniques
4. WebSocket connection via `useSocket` hook
5. Geolocation collection on first session via `useGeolocation` hook
6. Send location to backend on first connect

---

### Priority 9: MongoDB Models and Auth

Define Mongoose schemas in `backend/src/models/`:
- `User.js` — user_id, name, email, stored location (lat/lng), session preferences, created_at
- `Conversation.js` — user_id, session_id, messages array (each with role, text, timestamp, risk_level, ml_scores)

Add basic auth (JWT) so users have persistent identities:
- `POST /api/auth/register` — create user, return JWT
- `POST /api/auth/login` — verify credentials, return JWT
- JWT middleware to protect routes

---

### Priority 10: End-to-End Integration and Testing

Once all components exist, the integration phase:
1. Start Python FastAPI server (`uvicorn main:app --port 8000`)
2. Start Ollama (`ollama run llama3:8b`)
3. Start Node.js backend (`node src/server.js`)
4. Start React dev server (`npm run dev`)
5. Open browser, send a test message
6. Trace the full flow: message → WebSocket → orchestrator → Python scores → RAG → Supermemory → LLM → response → frontend
7. Test the three key scenarios:
   - LOW risk conversation (normal supportive chat)
   - MEDIUM risk (triggers RAG, response references specific techniques)
   - HIGH risk / crisis (triggers overlay, helplines appear, LLM is calming)

---

## Files That Should NOT Be Modified

These files are finished and working. Do not change them without a clear reason:

- `python/preprocessing/pipeline.py`
- `python/preprocessing/cleaner.py`
- `python/preprocessing/crisis_detector.py`
- `python/preprocessing/config.py`
- `python/ml_models/depression classifier model/depression_classifier.py`
- `python/ml_models/go_emotion model/emotion_detector.py`
- `python/ml_models/Stress detection model/stress_detector.py`
- `python/run_inference.py`
- `.gitignore`

---

## Environment Variables Needed

Create `backend/.env` with:
- `MONGODB_URI` — MongoDB Atlas connection string
- `GROQ_API_KEY` — Groq API key for LLM fallback
- `SUPERMEMORY_API_KEY` — Supermemory.ai API key
- `JWT_SECRET` — random secret string for JWT signing
- `PYTHON_API_URL` — URL of the Python FastAPI service (e.g., `http://localhost:8000`)
- `PORT` — backend server port (default 5000)

---

## Demo Scenarios to Prepare

The project needs to demonstrate three scenarios for a convincing health tech demo:

**Scenario A — Normal support conversation:**
A user having a tough but not critical day. System responds with empathy, no overlays, conversational tone. Demonstrates the LOW risk path.

**Scenario B — Sustained distress:**
A user describing persistent stress, burnout, or emotional difficulty over multiple turns. System responds with CBT techniques and breathing exercises referenced by name from the RAG indices. Demonstrates the MEDIUM risk path and RAG in action.

**Scenario C — Crisis escalation:**
A user expressing hopelessness or explicit crisis language. The crisis overlay appears immediately, helplines flash on screen, and the LLM maintains a calming, stabilizing tone. Demonstrates the HIGH risk path and crisis protocol. This scenario is the most important one for the healthcare track.

For Scenario C, the pre-ML short-circuit means the crisis response appears in 3–5ms even before the LLM generates its message. This is the most powerful demo moment — showing that the system can detect a crisis and front-load help without waiting for any ML model.

---

## Quick Reference: Key File Locations

| Component | File |
|---|---|
| Run all three models locally | `python/run_inference.py` |
| Preprocessing pipeline | `python/preprocessing/pipeline.py` |
| Crisis keyword banks | `python/preprocessing/keywords.py` |
| Config (model names, limits) | `python/preprocessing/config.py` |
| Depression model module | `python/ml_models/depression classifier model/depression_classifier.py` |
| Emotion model module | `python/ml_models/go_emotion model/emotion_detector.py` |
| Stress model module | `python/ml_models/Stress detection model/stress_detector.py` |
| RAG data directories | `backend/src/rag/data/` |
| Project documentation | `Project Documentation/` |
| CONTEXT.md (original design doc) | `CONTEXT.md` |
