# ASTRAVA — AI-Powered Mental Health Chatbot

> **Hackathon Project** · Team: Victims of DSA · Healthcare Track (HC-11)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-19.2.0-61dafb.svg)](https://reactjs.org/)
[![Node.js](https://img.shields.io/badge/Node.js-Backend-green.svg)](https://nodejs.org/)

---

## 🌟 Overview

ASTRAVA is an intelligent conversational AI agent designed to **detect signs of depression, anxiety, and stress** from text conversations. Unlike traditional chatbots, ASTRAVA analyzes every message through a sophisticated machine learning pipeline before generating responses, ensuring empathetic, evidence-based support while escalating to human counselors when users are in crisis.

### The Problem We're Solving

Many people suffering from emotional distress, depression, or crisis-level mental health challenges never seek help on their own. ASTRAVA meets users where they are—inside a casual chat interface—and responds with intelligence, empathy, and appropriate clinical support based on real-time emotional analysis.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🎭 Advanced Emotion Analysis** | Classifies text into 28 fine-grained emotional categories using Google's GoEmotions dataset and RoBERTa |
| **😰 Stress Detection** | Binary stress classification with confidence scores using DistilBERT trained on social media data |
| **💔 Depression Detection** | Identifies depressive language patterns using BERT trained on cleaned Reddit posts |
| **⚠️ Risk Stratification** | Aggregates all ML outputs into Low/Medium/High risk tiers that drive response strategy |
| **🚨 Crisis Detection** | Pre-ML keyword system detects suicidal ideation and self-harm before model inference |
| **📚 RAG-Powered Guidance** | Retrieves CBT techniques, breathing exercises, and coping strategies via FAISS vector search |
| **🧠 Persistent Memory** | Tracks emotional patterns and triggers across sessions using Supermemory.ai |
| **🆘 Crisis Escalation** | Displays location-based helpline numbers and human callback options for high-risk situations |



### How It Works

1. **User sends a message** → Text is cleaned and normalized
2. **Crisis detection runs first** → Pre-ML keyword system checks for immediate danger
3. **Three ML models analyze the text** → Depression, stress, and emotion classifiers run in parallel
4. **Risk level is calculated** → Outputs aggregate into LOW/MEDIUM/HIGH tiers
5. **For MEDIUM risk** → RAG system retrieves relevant CBT techniques
6. **LLM generates response** → Context includes ML scores, RAG passages, and user memory
7. **For HIGH risk** → Crisis overlay displays with helplines and callback options

---

## 🛠️ Technology Stack

### Python ML Layer
- **Python 3.11** — Core runtime
- **PyTorch 2.6.0** — Deep learning framework (CUDA 11.8)
- **HuggingFace Transformers 4.57.0** — Model loading and inference
- **FastAPI** — Async web framework for ML API (planned)
- **FAISS** — Vector similarity search for RAG

### Machine Learning Models
| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| `SamLowe/roberta-base-go_emotions` | 28-label emotion classification | 475MB | HuggingFace |
| `jnyx74/stress-prediction` | Binary stress detection | 265MB | HuggingFace |
| `poudel/Depression_and_Non-Depression_Classifier` | Binary depression detection | 438MB | HuggingFace |

### Backend
- **Node.js** — JavaScript runtime
- **Express** — REST API framework
- **Socket.IO** — Real-time WebSocket communication
- **Mongoose** — MongoDB object document mapper
- **Axios** — HTTP client for Python API calls

### Frontend
- **React 19.2.0** — UI library
- **Vite 7.3.1** — Build tool and dev server
- **Tailwind CSS 4.2.1** — Utility-first CSS framework
- **Framer Motion** — Animation library
- **Lucide React** — Icon library
- **Radix UI** — Headless UI components

### External Services
- **MongoDB Atlas** — Cloud database for conversation history
- **Supermemory.ai** — Persistent user emotional memory
- **Ollama / Groq** — LLM inference (local + cloud fallback)

### Datasets
- **GoEmotions** — 58k Reddit comments with 27 emotion labels
- **Dreaddit** — Social media posts labeled for stress
- **Depression Reddit (Cleaned)** — Depression signal dataset

---

## 📁 Project Structure

```
Astrava_Victims_of_DSA_Healthcare_11/
│
├── python/                              # Python ML Engine
│   ├── preprocessing/                   # ✅ Text preprocessing pipeline
│   │   ├── cleaner.py                  # 15-step text normalization
│   │   ├── crisis_detector.py          # Keyword-based crisis detection
│   │   ├── stopwords.py                # Emotional stopword filtering
│   │   ├── keywords.py                 # Crisis keyword banks
│   │   ├── pipeline.py                 # Preprocessing orchestrator
│   │   └── config.py                   # Configuration constants
│   │
│   ├── ml_models/                       # ✅ Machine Learning Models
│   │   ├── depression classifier model/
│   │   │   └── depression_classifier.py
│   │   ├── go_emotion model/
│   │   │   └── emotion_detector.py
│   │   └── Stress detection model/
│   │       └── stress_detector.py
│   │
│   ├── rag/                             # 🚧 RAG System (planned)
│   │   ├── build_index.py              # FAISS index builder
│   │   ├── retriever.py                # Vector similarity retriever
│   │   └── documents/                  # Knowledge base files
│   │
│   ├── run_inference.py                 # ✅ Unified ML inference runner
│   └── main.py                          # 🚧 FastAPI server (planned)
│
├── backend/                             # 🚧 Node.js Backend (planned)
│   ├── main.py                         # Backend entry point
│   └── src/
│       ├── rag/
│       │   └── data/                   # RAG knowledge base
│       │       ├── breathing_grounding/
│       │       ├── cbt_techniques/
│       │       ├── coping_strategies/
│       │       └── crisis_lines/
│
├── frontend/                            # 🚧 React Frontend (scaffolded)
│   ├── src/
│   │   ├── App.jsx                     # Main application component
│   │   ├── main.jsx                    # Entry point
│   │   ├── translations.js             # i18n translations
│   │   └── components/
│   │       └── ui/                     # Shadcn UI components
│   ├── package.json
│   └── vite.config.js
│
├── embeddings/                          # Vector embeddings storage
├── Project Documentation/               # 📖 Comprehensive documentation
│   ├── 00_PROJECT_OVERVIEW.md
│   ├── 01_SYSTEM_ARCHITECTURE.md
│   ├── 02_MESSAGE_LIFECYCLE.md
│   ├── 03_ML_MODELS.md
│   ├── 04_PREPROCESSING_PIPELINE.md
│   ├── 05_RAG_SYSTEM.md
│   ├── 06_RISK_SCORING.md
│   ├── 07_CURRENT_BUILD_STATE.md
│   ├── 08_TECH_STACK.md
│   └── 09_CONTINUATION_GUIDE.md
│
├── CONTEXT.md                           # Project context overview
└── README.md                            # This file
```

**Legend:**
- ✅ **Complete and tested**
- 🚧 **Planned or partially implemented**

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **CUDA 11.8** (optional, for GPU acceleration)
- **Git**
- **MongoDB Atlas account** (for database)
- **Groq API key** (optional, for LLM fallback)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/V-ManishBIT130/Astrava_Victims_of_DSA_Healthcare_11.git
cd Astrava_Victims_of_DSA_Healthcare_11
```

#### 2. Set Up Python Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers fastapi uvicorn sentence-transformers faiss-cpu
```

#### 3. Set Up Frontend (Optional)

```bash
cd frontend
npm install
cd ..
```

#### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# MongoDB
MONGODB_URI=your_mongodb_atlas_connection_string

# Supermemory.ai
SUPERMEMORY_API_KEY=your_supermemory_api_key

# Groq (LLM fallback)
GROQ_API_KEY=your_groq_api_key

# Server Configuration
PORT=5000
PYTHON_API_URL=http://localhost:8000
```

---

## 💻 Usage

### Running the ML Pipeline (Current State)

The Python ML layer is fully functional and can be run standalone:

```powershell
# Suppress TensorFlow warnings and set proper encoding
$env:TF_ENABLE_ONEDNN_OPTS="0"; $env:PYTHONIOENCODING="utf-8"

# Run inference on a single message
python python/run_inference.py --text "I feel completely hopeless and can't sleep"

# Interactive mode
python python/run_inference.py
```

**Sample Output:**
```
================================================================================
ASTRAVA INFERENCE RESULTS
================================================================================

Input Text: I feel completely hopeless and can't sleep
Processed Text: feel completely hopeless cant sleep
Crisis Level: MEDIUM
Crisis Detected: True
Helpline Recommended: True

Depression Analysis:
  Prediction: Depression
  Score: 99.99%

Stress Analysis:
  Prediction: High Stress
  Stress Level: HIGH
  Score: 96.78%

Emotion Analysis:
  Top Emotion: disappointment (92.34%)
  All Emotions: disappointment, annoyance, sadness

Total Inference Time: 645.23ms
================================================================================
```

### Running the Full Application (Future)

Once the backend and frontend are implemented:

```bash
# Terminal 1: Start Python ML API
cd python
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Node.js Backend
cd backend
npm start

# Terminal 3: Start Frontend
cd frontend
npm run dev
```

---

## 📊 Current Build Status (as of March 7, 2026)

### ✅ Complete and Production-Ready

- **Python ML Layer** (100%)
  - ✅ Text preprocessing pipeline (15 cleaning steps)
  - ✅ Crisis detection system (keyword + pattern matching)
  - ✅ Depression classifier (BERT-based)
  - ✅ Stress detector (DistilBERT-based)
  - ✅ Emotion analyzer (RoBERTa-based, 28 labels)
  - ✅ Unified inference runner
  - ✅ Thread-safe model loading
  - ✅ All bugs fixed and tested

### 🚧 Planned / In Progress

- **Backend** (0%)
  - ⏳ FastAPI server for Python ML layer
  - ⏳ Node.js/Express REST API
  - ⏳ Socket.IO WebSocket handlers
  - ⏳ Orchestrator service
  - ⏳ LLM integration (Ollama + Groq)
  - ⏳ Supermemory.ai integration
  - ⏳ MongoDB models and connections

- **RAG System** (0%)
  - ⏳ FAISS index builder
  - ⏳ Document embeddings
  - ⏳ Retrieval logic
  - ✅ Knowledge base documents (CBT, breathing, coping)

- **Frontend** (10%)
  - ✅ Project scaffolded with Vite
  - ✅ UI components library (Shadcn/Radix)
  - ⏳ Chat interface
  - ⏳ Crisis overlay
  - ⏳ WebSocket integration
  - ⏳ Geolocation handling

---

## 🔧 Known Issues & Fixed Bugs

### Fixed Issues ✅

1. **Crisis Detector False Positives** — Word-boundary regex now prevents "sad" matching "sadly", "void" matching "avoid"
2. **Model Name Error** — Corrected emotion model reference from `mental/mental-roberta-base` to `SamLowe/roberta-base-go_emotions`
3. **Text Truncation** — Increased `MAX_SEQ_LENGTH` from 128 to 512 tokens (60% of messages were being truncated)
4. **Dead Code in Pipeline** — Removed 96 lines of unused embedding model calls
5. **Windows Terminal Encoding** — Replaced Unicode box characters with ASCII equivalents
6. **Thread Safety** — Added singleton pattern with double-checked locking for all model loaders

### Current Limitations

- Backend and frontend are not yet implemented
- RAG system exists as documentation only
- No real-time chat capability yet
- No database integration yet
- LLM integration pending

---

## 🗺️ Roadmap

### Phase 1: Complete ML API (Next Priority)
- [ ] Create FastAPI server (`python/main.py`)
- [ ] Expose `POST /api/analyze` endpoint
- [ ] Add health check and model loading endpoints
- [ ] Deploy Python API locally

### Phase 2: Backend Development
- [ ] Initialize Node.js project
- [ ] Set up Express + Socket.IO
- [ ] Implement orchestrator service
- [ ] Integrate LLM (Ollama primary, Groq fallback)
- [ ] Connect MongoDB Atlas
- [ ] Implement Supermemory.ai service

### Phase 3: RAG System
- [ ] Build FAISS indices from knowledge documents
- [ ] Implement embedding generation
- [ ] Create retrieval service
- [ ] Test RAG pipeline end-to-end

### Phase 4: Frontend Implementation
- [ ] Build chat interface
- [ ] Implement WebSocket client
- [ ] Create crisis overlay component
- [ ] Add geolocation handling
- [ ] Implement conversation history
- [ ] Build user settings panel

### Phase 5: Integration & Testing
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Security hardening
- [ ] User acceptance testing

---

## 📚 Documentation

Comprehensive documentation is available in the [`Project Documentation/`](Project%20Documentation/) folder:

- **[00_PROJECT_OVERVIEW.md](Project%20Documentation/00_PROJECT_OVERVIEW.md)** — High-level project description
- **[01_SYSTEM_ARCHITECTURE.md](Project%20Documentation/01_SYSTEM_ARCHITECTURE.md)** — Detailed architecture diagrams
- **[02_MESSAGE_LIFECYCLE.md](Project%20Documentation/02_MESSAGE_LIFECYCLE.md)** — Message processing flow
- **[03_ML_MODELS.md](Project%20Documentation/03_ML_MODELS.md)** — Model documentation and training
- **[04_PREPROCESSING_PIPELINE.md](Project%20Documentation/04_PREPROCESSING_PIPELINE.md)** — Text preprocessing details
- **[05_RAG_SYSTEM.md](Project%20Documentation/05_RAG_SYSTEM.md)** — RAG implementation guide
- **[06_RISK_SCORING.md](Project%20Documentation/06_RISK_SCORING.md)** — Risk stratification logic
- **[07_CURRENT_BUILD_STATE.md](Project%20Documentation/07_CURRENT_BUILD_STATE.md)** — What's built vs. what's pending
- **[08_TECH_STACK.md](Project%20Documentation/08_TECH_STACK.md)** — Complete technology reference
- **[09_CONTINUATION_GUIDE.md](Project%20Documentation/09_CONTINUATION_GUIDE.md)** — Next steps for developers

---

## 🤝 Contributing

This is a hackathon project built in a 7-hour sprint. Contributions are welcome!

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- **Python**: Follow PEP 8, use type hints
- **JavaScript**: Follow ESLint configuration
- **Documentation**: Update relevant `.md` files

---

## ⚠️ Disclaimer

**ASTRAVA is a demonstration project and should NOT be used as a replacement for professional mental health care.** This chatbot is designed to provide supportive conversation and direct users to appropriate resources, but it is not a substitute for therapy, counseling, or emergency services.

If you or someone you know is in crisis:
- **US**: National Suicide Prevention Lifeline: 988
- **UK**: Samaritans: 116 123
- **International**: [Find a Helpline](https://findahelpline.com)

---

## 👥 Team

**Victims of DSA** — Healthcare Track (HC-11)

Built with ❤️ during a 7-hour hackathon sprint.

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **HuggingFace** for hosting pre-trained models
- **Google Research** for the GoEmotions dataset
- **Dreaddit creators** for the stress detection dataset
- **Supermemory.ai** for persistent memory infrastructure
- **Ollama** for local LLM inference
- **MongoDB** for database services
- **Vite** and **React** teams for excellent developer tools

---

## 📞 Contact

For questions, suggestions, or collaboration:
- GitHub: [@V-ManishBIT130](https://github.com/V-ManishBIT130)
- Repository: [Astrava_Victims_of_DSA_Healthcare_11](https://github.com/V-ManishBIT130/Astrava_Victims_of_DSA_Healthcare_11)

---

<div align="center">

**Built for those who need someone to listen.**

*ASTRAVA — Empathetic AI for Mental Wellness*

</div>
