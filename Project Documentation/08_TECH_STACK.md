# ASTRAVA — Technology Stack

This document lists every technology used or planned for use in ASTRAVA, organized by layer, with a clear explanation of what each one does, why it was chosen, and where it appears in the codebase.

---

## Python Layer

### Python 3.11
The runtime for all ML and NLP work. Python 3.11 is the current stable version with meaningful performance improvements over 3.10 (faster function calls, better error messages). The entire `python/` folder runs on Python 3.11.

### PyTorch 2.6.0 (CUDA 11.8)
The deep learning framework that backs all three HuggingFace models. Models are loaded as PyTorch modules. Inference runs on CUDA when a compatible GPU is available, otherwise falls back to CPU automatically. CUDA 11.8 corresponds to the NVIDIA driver installed on the development machine.

### HuggingFace Transformers 4.57.0
The library that manages model loading, tokenization, and inference for all three classifiers. It provides the `pipeline()` API and the `AutoTokenizer` / `AutoModelForSequenceClassification` classes. Models are downloaded from HuggingFace Hub on first use and cached locally.

### FastAPI (planned for `python/main.py`)
The async Python web framework that will expose the preprocessing + inference pipeline as an HTTP API. FastAPI was chosen over Flask because of its native async support, automatic OpenAPI documentation generation, built-in Pydantic input validation, and significantly faster throughput. The single endpoint `POST /api/analyze` will accept raw text and return structured ML scores.

### Uvicorn (planned)
The ASGI server that runs the FastAPI app. Uvicorn is the standard production server for FastAPI. It will be started with `uvicorn main:app --host 0.0.0.0 --port 8000`.

---

## Model Checkpoints (HuggingFace Hub)

### `poudel/Depression_and_Non-Depression_Classifier`
Binary depression classifier. BERT-base-uncased fine-tuned on cleaned Reddit depression posts. ~438MB download. Cached at `C:\Users\V Manish\.cache\huggingface\hub\` after first run.

### `jnyx74/stress-prediction`
Binary stress classifier. DistilBERT fine-tuned on the Dreaddit social media stress dataset. ~265MB download. 

### `SamLowe/roberta-base-go_emotions`
Multi-label emotion classifier (28 labels). RoBERTa-base fine-tuned on Google's GoEmotions dataset. ~475MB download.

---

## Backend Layer

### Node.js
The JavaScript runtime for the backend server. Node.js is event-driven and non-blocking, which is ideal for an application that needs to juggle WebSocket connections, HTTP calls to Python, and async database writes simultaneously.

### Express
The web framework for Node.js. Used to set up the REST API routes (auth, history). Minimal and well-documented. REST endpoints in Express are used alongside the Socket.IO WebSocket server on the same port.

### Socket.IO
The WebSocket library that handles real-time bidirectional communication between the frontend and backend. Used for all live chat traffic: emitting `user_message` from the client, `bot_response` back to the client, `crisis_alert` events when escalation is triggered, and `typing_indicator` events. Socket.IO is chosen over raw WebSockets because of its built-in reconnection handling, room management, and cross-browser compatibility.

### Axios
The HTTP client used by the backend to make calls to the Python FastAPI service. When the orchestrator needs ML scores, it sends a POST request via Axios to `http://localhost:8000/api/analyze` and awaits the JSON response.

### Mongoose
The MongoDB object document mapper for Node.js. Mongoose provides schema definitions, validation, and query helpers for MongoDB. Two main schemas are planned: `User` (profile, location, session preferences) and `Conversation` (message history, risk levels, timestamps).

### FAISS (planned, via faiss-node or Python sidecar)
The vector similarity search library. Used to build and query the four knowledge index files for RAG. FAISS stands for Facebook AI Similarity Search. It is designed for fast nearest-neighbor lookup on large sets of dense vectors. In the backend, it will be called via `faiss-node` (the Node.js binding) or via a thin Python sidecar, depending on which approach proves more stable. The four FAISS index files are built once (offline) by the indexer script and loaded into memory at server startup.

### sentence-transformers / all-MiniLM-L6-v2 (planned)
The embedding model used to encode documents and queries for FAISS. Produces 384-dimensional vectors from text. This is the standard lightweight sentence embedding model — fast enough to run synchronously, good enough quality for retrieval. The same model must be used for both building the index and querying it at runtime.

### Ollama (planned, primary LLM)
A tool that lets you run large language models locally. The backend will call Ollama's local REST API at `http://localhost:11434/api/chat` using the `llama3:8b` model. Since Ollama runs on the same machine as the backend, latency is minimal after the initial model load. Inference for an average chat message typically takes 3–10 seconds on a modern CPU/GPU.

### Groq API (planned, LLM fallback)
A cloud inference provider with very fast LLM inference. Used as the fallback when Ollama is unavailable or times out. Uses the same LLaMA 3 8B model (called `llama3-8b-8192` on Groq). The Groq API is OpenAI-compatible, so the same request format works. Requires a Groq API key.

### Supermemory.ai API (planned)
A third-party API for persistent AI memory. Used to store and retrieve per-user emotional context across sessions. Supports two main operations: storing memories (POST) with a user ID and content, and searching memories (POST) with a query. The backend calls it synchronously before LLM prompt assembly and asynchronously after response delivery.

---

## Frontend Layer

### React 18
The UI library. Used to build the chat interface as a single-page application. React's component model is well-suited for a chat application where individual message bubbles, the input bar, and overlay components all need to be independently re-renderable.

### Vite
The build tool and development server for the React app. Vite is significantly faster than Create React App for development — hot module replacement is near-instant. Production builds are also faster. Used as the standard modern React scaffolding tool.

### Tailwind CSS (planned)
Utility-first CSS framework for rapid UI development. Avoids writing custom CSS files by composing styles from utility classes directly in JSX. Good for hackathon pace.

### Socket.IO Client
The browser-side Socket.IO library. Pairs with the Socket.IO server on the backend. Handles connection, reconnection, and event listening for the real-time chat.

### Browser Geolocation API
The standard browser API for requesting the user's latitude and longitude. Used on first session — if the user grants permission, coordinates are sent to the backend and stored. Used later to filter region-appropriate crisis helplines if the crisis overlay is triggered. This is a browser-native API, no library needed.

---

## Database

### MongoDB Atlas
Cloud-hosted MongoDB. Stores user profiles, conversation history, and session data. MongoDB's flexible document model is suitable for conversation storage where message objects have variable schemas (different fields for different risk levels, different sets of ML scores per message). MongoDB Atlas provides a free tier sufficient for a hackathon demo.

---

## Infrastructure and Tooling

### Git + GitHub
Version control and remote backup. Repository: `V-ManishBIT130/Astrava_Victims_of_DSA_Healthcare_11`. The `.gitignore` explicitly excludes all ML model weight file types (`.pkl`, `.safetensors`, `.bin`, `.pth`, `.onnx`) to prevent accidental large-file commits.

### dotenv (.env files)
Environment variable management. API keys (Groq, Supermemory, MongoDB Atlas connection string) are stored in `.env` files and never committed to the repository. The backend reads them at startup via the `dotenv` Node.js library.

### PowerShell (development environment)
The default terminal on Windows 11 where development is happening. Important note: PowerShell uses the system's default code page for output rendering, which may not support UTF-8 box-drawing characters or emoji. The `run_inference.py` output was adjusted to use only ASCII characters for compatibility. When running Python scripts, set `$env:PYTHONIOENCODING="utf-8"` to ensure consistent encoding.
