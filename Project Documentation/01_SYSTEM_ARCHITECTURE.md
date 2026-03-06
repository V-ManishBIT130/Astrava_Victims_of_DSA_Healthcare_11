# ASTRAVA — System Architecture

---

## Overview

ASTRAVA is a multi-layer distributed system. Each layer has a clearly defined job and communicates with adjacent layers through well-defined interfaces. No layer does more than its role requires.

There are five major layers:

1. **Frontend** — React application, chat UI, crisis overlay, geolocation
2. **Backend** — Node.js + Express, orchestration, WebSocket, memory, LLM integration, RAG
3. **Python ML Engine** — FastAPI service, text preprocessing, three ML classifiers, risk scoring
4. **External Services** — Supermemory.ai (user memory), Ollama (local LLM), Groq API (cloud LLM fallback)
5. **Database** — MongoDB Atlas, stores conversation history and user profiles

---

## Layer 1: Frontend (React + Vite)

The frontend is a React 18 application built with Vite. Its job is to be a thin, reactive UI layer — it does no analysis itself.

**Chat UI:** The main interface is a chat window where the user types messages and sees responses. Communication with the backend happens over a WebSocket connection (Socket.IO). REST endpoints are used only for authentication and history fetching, not for the live chat flow.

**Crisis Overlay:** The frontend listens for a specific WebSocket event called `crisis_alert` emitted by the backend. When this event arrives, it renders a full-screen interrupting overlay showing crisis helpline numbers, a prompt asking if the user wants a human callback, and location-based crisis center information if geolocation was previously shared.

**Geolocation:** On the user's first session, the browser requests geolocation permission. The latitude and longitude are sent to the backend and stored in the session. These coordinates are used later to show region-appropriate helplines when the crisis overlay appears.

**Location in folder:** `frontend/`

---

## Layer 2: Backend (Node.js + Express + Socket.IO)

The backend is the orchestration layer. It receives messages from the frontend, routes them through the pipeline, assembles the context bundle, calls the LLM, and streams the response back. It is the only layer that talks to Supermemory.ai and the LLM.

**WebSocket Gateway:** Uses Socket.IO to maintain persistent connections with connected clients. Messages flow in and out through WebSocket events, not HTTP requests. This keeps the chat experience real-time and avoids polling.

**Orchestrator Service:** The core logic of the backend lives in `orchestrator.js`. It is responsible for:
- Receiving the user's message from the WebSocket handler
- Calling the Python FastAPI engine to get preprocessing output and ML scores
- Querying FAISS for relevant resource passages (only when risk is MEDIUM)
- Fetching the user's memory context from Supermemory.ai
- Building the full LLM prompt from all these pieces
- Calling Ollama or Groq to generate the response
- Emitting the response and any crisis events back to the frontend
- Asynchronously writing the session data back to Supermemory and MongoDB

**RAG Module:** FAISS retrieval lives in the backend, not the Python layer. The four FAISS indices (CBT techniques, breathing and grounding, crisis helplines, coping strategies) are queried by the orchestrator. Crucially, RAG is only invoked when the Python layer returns `risk_level = MEDIUM`. For LOW risk there is no retrieval. For HIGH risk the crisis protocol uses hardcoded helplines instead.

**LLM Client:** Tries Ollama first (running locally on port 11434, model llama3:8b). If Ollama is unreachable or takes longer than 10 seconds, it falls back automatically to Groq's cloud API using the same LLaMA 3 8B model.

**Supermemory Service:** A wrapper around the Supermemory.ai API. Before each LLM call it searches the user's stored memories using the current emotional state as a query. After a response is delivered, it asynchronously writes the new emotional state, risk level, any coping tools mentioned, and whether escalation occurred.

**MongoDB:** Stores the raw conversation history (messages and responses with timestamps), user profiles (name, location, preferences), and session metadata. Used for the conversation history view in the frontend.

**Location in folder:** `backend/src/`

---

## Layer 3: Python ML Engine (FastAPI)

The Python layer is a FastAPI service that exposes one primary endpoint: `POST /api/analyze`. It receives a raw text message, runs the full preprocessing and ML inference pipeline, and returns structured scores.

This layer has no knowledge of the LLM, RAG, or Supermemory. It only does text analysis. This clean separation means the Python engine can be tested and improved independently.

**Preprocessing Module:** Cleans the raw text through 15 normalization steps (lowercase, remove URLs, expand contractions, normalize slang, etc.) and then runs crisis detection. The output is a clean text string plus a crisis assessment object with a severity level.

**Three ML Models:** Each model is a HuggingFace Transformer loaded from cache (auto-downloaded on first use). They run sequentially — not in parallel — because running them in parallel causes PyTorch's internal thread pools to compete for CPU resources, which makes total inference slower, not faster. Running them one after another lets each model use 100% of available CPU resources during its pass.

**Score Aggregator:** After all three models run, their outputs are combined into a single risk level (LOW / MEDIUM / HIGH) and returned to the backend.

**Location in folder:** `python/`

The entry point for testing the Python layer in isolation is `python/run_inference.py`, which can accept text directly from the command line and prints all results in a formatted table.

---

## Layer 4: External Services

**Ollama (Primary LLM):** Runs locally on the developer's machine on port 11434. Hosts llama3:8b. The backend calls it via its REST API. No internet required once the model is pulled.

**Groq API (Fallback LLM):** A cloud inference provider. Used when Ollama is unavailable. Also uses llama3:8b (called llama3-8b-8192 on Groq). Fast inference times — typically under 1 second.

**Supermemory.ai:** A third-party persistent memory API. Accessed by the backend using REST calls (search and store). Stores emotional context per user across sessions. This is what allows ASTRAVA to recognize returning users and personalize responses based on their history.

---

## Layer 5: Database (MongoDB Atlas)

MongoDB stores conversation logs, user accounts, and session metadata. The backend connects via Mongoose. MongoDB is cloud-hosted on Atlas, so it is always available without a local database server.

---

## Communication Between Layers

**Frontend → Backend:** Socket.IO WebSocket for all live chat traffic. REST for login, registration, and history retrieval.

**Backend → Python ML Engine:** Internal HTTP call via Axios. The backend sends the raw message and receives JSON scores back. This is synchronous from the backend's perspective — it waits for the Python response before proceeding.

**Backend → Supermemory:** HTTPS REST calls. The read call happens before the LLM prompt is built. The write call happens asynchronously after the response is delivered.

**Backend → Ollama/Groq:** HTTP REST calls. Tried in sequence (Ollama first, Groq if Ollama fails).

**Backend → MongoDB:** Mongoose ODM. All writes are asynchronous and do not block the response path.

---

## Key Architectural Decisions and Reasons

**Why is RAG in the backend, not the Python layer?**
RAG retrieval is closely coupled with LLM prompt construction — the retrieved passages are injected directly into the prompt. Keeping RAG and the LLM client in the same service (the backend) avoids an extra network hop and keeps the context bundle assembly in one place.

**Why does RAG only run for MEDIUM risk?**
For LOW risk, the LLM's own conversational ability is sufficient — no clinical resources needed for light distress. For HIGH risk, the situation is too urgent to take time retrieving passages; hardcoded helplines are faster and more reliable. MEDIUM is the one tier where clinical evidence-based resources meaningfully improve the response quality.

**Why does the Python layer not do LLM inference?**
The Python layer is purpose-built for ML scoring — fast, stateless, and independently testable. Adding LLM inference would mix two very different concerns. The Node.js backend is better positioned for orchestration because it already manages sessions, memory, WebSocket state, and user identity.

**Why sequential ML inference instead of parallel?**
PyTorch uses OpenMP thread pools internally for matrix operations. Running three models in Python threads simultaneously causes those thread pools to fight over the same CPU cores. Measured wall-clock time for sequential inference (Depression → Stress → Emotion) is faster than parallel because each model gets undivided CPU resources. Total inference time is under 700ms on a modern CPU.

**Why BERT/RoBERTa/DistilBERT family specifically?**
These encoder-only transformer architectures are purpose-built for classification tasks on short-to-medium text (which is what user chat messages are). They are significantly smaller than generative LLMs, load in under 6 seconds, and run inference in under 500ms. They do not need a GPU. Using a powerful generative LLM like LLaMA for classification would be slower and less accurate for these specific binary/multi-label tasks.
