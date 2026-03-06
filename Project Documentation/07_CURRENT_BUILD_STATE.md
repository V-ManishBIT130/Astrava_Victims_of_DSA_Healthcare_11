# ASTRAVA — Current Build State (as of March 6, 2026)

This document is the definitive reference for what has been built, what has been tested, what bugs have been fixed, and what remains to be built. Anyone picking up this project should read this file first.

---

## Summary: What Works Right Now

The Python ML layer is fully functional and production-ready. Every preprocessing step, crisis detection, and all three ML models work correctly and have been tested with real inputs. The unified inference runner can be executed from the command line.

The backend and frontend are scaffolded (folders and placeholder files exist) but have NO implemented logic. No actual backend routes handle messages, no WebSocket connections are set up, no LLM integration exists, and the frontend has no working components. The RAG system, Supermemory integration, and MongoDB integration are planned but not built.

---

## FULLY BUILT AND TESTED

### 1. Preprocessing Pipeline (`python/preprocessing/`)

**Status:** Complete and working.

All seven modules are implemented:
- `cleaner.py` — 15-step text normalization (fully working)
- `crisis_detector.py` — keyword + psycholinguistic safety system (fully working, word-boundary bug fixed)
- `stopwords.py` — emotional stopword filter (fully working)
- `keywords.py` — crisis keyword banks (fully populated)
- `config.py` — configuration constants (fixed: model name corrected, MAX_SEQ_LENGTH updated to 512)
- `pipeline.py` — orchestrator that calls cleaner → crisis detector → stopword filter (refactored: all dead model-loading code removed)

**Known bugs fixed:**
- `crisis_detector.py` previously matched single-word keywords using plain substring search, which caused false positives. Example: the word "void" matched inside "avoid", "numb" inside "number", "sad" inside "sadly", "pain" inside "explain". Fixed by using word-boundary regex (`\b`) for all single-word keyword lookups.
- `config.py` had the wrong emotion model name (`mental/mental-roberta-base` instead of `SamLowe/roberta-base-go_emotions`). Fixed.
- `config.py` had `MAX_SEQ_LENGTH = 128`, which was silently truncating roughly 60% of real conversational messages at the tokenizer level. Fixed to 512.
- `pipeline.py` previously loaded three large embedding models and ran three extra forward passes during preprocessing. These outputs were never used by any downstream code. That entire dead code block (96 lines) was removed.

---

### 2. Depression Classifier (`python/ml_models/depression classifier model/`)

**Status:** Complete, tested, working.

Files:
- `depression_classifier.py` — the main module with `DepressionClassifier` class
- `testing.py` — test script to verify the model works
- `export_model.py` — optional script to save model weights locally
- `MODEL_SETUP_GUIDE.md` — full documentation

Model: `poudel/Depression_and_Non-Depression_Classifier` (HuggingFace, auto-downloads ~438MB on first use)
Tested: Yes — confirmed working with real inputs.

---

### 3. Stress Detector (`python/ml_models/Stress detection model/`)

**Status:** Complete, tested, working.

Files:
- `stress_detector.py` — main module with `detect_stress()`, `detect_stress_batch()`, `get_stress_level()`, `should_trigger_alert()`
- `SETUP_GUIDE.md` — documentation
- `test_model.py` — test script
- `save_model.py` — optional local save script

Model: `jnyx74/stress-prediction` (HuggingFace, auto-downloads, ~265MB on first use)
Tested: Yes — confirmed working with real inputs.

**Bug fixed:** Added thread-safe singleton loading using `threading.Lock` with double-checked locking pattern, to prevent race conditions in FastAPI multi-worker environments.

---

### 4. Emotion Detector (`python/ml_models/go_emotion model/`)

**Status:** Complete, tested, working.

Files:
- `emotion_detector.py` — main module with `detect_emotions()`, `get_top_emotion()`, `get_all_scores()`
- `SETUP_GUIDE.md` — documentation
- `test_model.py` — test script
- `save_model.py` — optional local save script
- `go_emotions_saved/` — saved model files (config.json, model.safetensors, tokenizer files)

Model: `SamLowe/roberta-base-go_emotions` (HuggingFace, auto-downloads ~475MB on first use; also partially saved locally in `go_emotions_saved/`)
Tested: Yes — confirmed working with real inputs.

**Bug fixed:** Added thread-safe singleton loading using `threading.Lock` with double-checked locking pattern (same as stress_detector).

---

### 5. Unified Inference Runner (`python/run_inference.py`)

**Status:** Complete, tested, working.

This is the command-line interface for testing the full Python pipeline. It loads all three models at startup, accepts text input either via `--text "..."` flag or in interactive mode, runs the complete preprocessing → crisis check → inference pipeline, and prints structured results with color-coded output.

**Usage:**
- `python python/run_inference.py --text "your message here"` — single run
- `python python/run_inference.py` — interactive mode (type messages, `quit` to exit)

**Recommended launch command on Windows PowerShell:**
```
$env:TF_ENABLE_ONEDNN_OPTS="0"; $env:PYTHONIOENCODING="utf-8"; python python/run_inference.py
```
The `TF_ENABLE_ONEDNN_OPTS=0` suppresses TensorFlow deprecation warnings that appear because the Keras library is installed alongside PyTorch. The `PYTHONIOENCODING=utf-8` ensures correct character encoding in the terminal.

**Bug fixed:** An earlier version used Unicode box-drawing characters (`─`) for separator lines and the emoji `⚠️` in output. Windows PowerShell cannot render these characters. Both were replaced with ASCII equivalents (`-` and `[!]`).

**Confirmed test results:**
All four of the following test cases produce correct output.
1. Hopeless/sleepless input → Depression 99.99%, Stress HIGH 96.78%, emotions: disappointment + annoyance, 645ms total
2. Positive/promotion input → Non-Depression 99.74%, No Stress LOW 98.41%, Joy 86.66%, 343ms total
3. Crisis input → HIGH severity, SHORT-CIRCUITED in 3ms, zero model inference
4. Burnout input → Depression 99.97% (known overclassification), Stress HIGH 97.90%, annoyance + disappointment, 489ms total

---

### 6. Repository Configuration

**Status:** Complete.

- `.gitignore` created at repo root: excludes `*.pkl`, `*.safetensors`, `*.bin`, `*.pth`, `*.onnx`
- All large model weight files removed from git history (via `git reset --soft origin/main` after they were accidentally committed)
- Current repo size is well under 1MB
- GitHub repo: `V-ManishBIT130/Astrava_Victims_of_DSA_Healthcare_11`

---

## NOT YET BUILT

### 7. FastAPI Entrypoint (`python/main.py`)

The FastAPI server that would wrap the preprocessing + ML inference and expose it as an HTTP endpoint (`POST /api/analyze`) does not exist yet. Currently all Python inference is run directly from the command line via `run_inference.py`. For the full system to work, a FastAPI app needs to be created that wraps the `AstravaInference` class and exposes it over HTTP.

---

### 8. Backend (Node.js) — `backend/src/`

**Status:** Scaffolded only. No logic implemented.

The folder structure exists with placeholder `.gitkeep` files. None of the following have been built:
- Express server setup
- Socket.IO setup
- Chat WebSocket event handlers
- Orchestrator service (the pipeline coordinator)
- Python HTTP client (Axios calls to FastAPI)
- LLM client (Ollama + Groq fallback)
- Supermemory service wrapper
- RAG indexer, embedder, retriever
- MongoDB models (User, Conversation)
- Auth routes and middleware

---

### 9. RAG Data Content

The four data directories exist but are empty (they contain only `.gitkeep` files):
- `backend/src/rag/data/cbt_techniques/`
- `backend/src/rag/data/breathing_grounding/`
- `backend/src/rag/data/crisis_lines/`
- `backend/src/rag/data/coping_strategies/`

Documents need to be manually curated and added to these folders before FAISS indexing can happen. Target: 8–20 short text passages per folder, each 50–150 words, each focused on a single technique or resource.

---

### 10. Frontend (React) — `frontend/`

**Status:** Scaffolded only. No components built.

The folder exists with nothing inside except `.gitkeep`. None of the following have been built:
- React app setup (Vite)
- Chat interface components
- Socket.IO client connection
- Crisis overlay component
- Geolocation module
- Conversation history view
- Auth pages

---

## Bugs and Issues Previously Fixed (Do Not Re-Introduce)

| # | Issue | Where | Fix Applied |
|---|---|---|---|
| 1 | Wrong emotion model name in config | `config.py` | Changed to `SamLowe/roberta-base-go_emotions` |
| 2 | MAX_SEQ_LENGTH=128 truncating messages | `config.py` | Changed to 512 |
| 3 | Dead tokenizer/embedder model loading in pipeline | `pipeline.py` | Entire dead block removed (96 lines) |
| 4 | No input length guard — 10k char pastes ran full cleaner | `pipeline.py` | MAX_RAW_CHARS=4000 guard added |
| 5 | Single-word crisis keywords matching inside other words | `crisis_detector.py` | Word-boundary `\b` regex applied to single-token keywords |
| 6 | Race condition in emotion model singleton loading | `emotion_detector.py` | Double-checked locking with threading.Lock |
| 7 | Race condition in stress model singleton loading | `stress_detector.py` | Double-checked locking with threading.Lock |
| 8 | Windows PowerShell cannot render box-drawing chars and emoji | `run_inference.py` | Replaced `─` with `-`, `⚠️` with `[!]` |
| 9 | Large model weight files committed to git (1.4GB) | `.gitignore` + git history | gitignore created, history rewritten |
| 10 | Large files still in git history after gitignore | git | `git reset --soft origin/main` → clean recommit |

---

## Python Environment Details

- Python: 3.11
- PyTorch: 2.6.0+cu118 (CUDA 11.8)
- Transformers: 4.57.0
- OS: Windows 11
- Device: CUDA GPU available (models automatically use CUDA when available)
- HuggingFace cache: `C:\Users\V Manish\.cache\huggingface\hub\` (all 3 models now cached)
