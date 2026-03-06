# ASTRAVA — Project Overview

---

## What Is ASTRAVA?

ASTRAVA is an AI-powered mental health chatbot built as a hackathon project under the Healthcare track (HC-11). It was created by a small team called "Victims of DSA." The name ASTRAVA reflects the system's aim to serve as a trustworthy conversational companion for people experiencing emotional distress.

The central problem ASTRAVA addresses is the gap between people who are silently struggling with depression, stress, or crisis-level emotional pain, and the professional help or supportive intervention they need. Most people will not seek help on their own, especially in an acute moment of distress. ASTRAVA is designed to meet users where they are — inside a casual chat interface — and respond with intelligence, empathy, and appropriate clinical support.

---

## What ASTRAVA Actually Does

When a user types a message, ASTRAVA does not simply pass it to a large language model and return a response. Instead, it runs the message through a layered analytical pipeline before the LLM ever sees it:

1. The text is cleaned and normalized so slang, contractions, typos, and informal language are handled correctly.
2. A crisis detection system checks the message for suicidal ideation, self-harm signals, or hopelessness patterns — before any machine learning model runs. If a crisis is detected, the system short-circuits and responds immediately with human-readable crisis resources.
3. If no crisis is found, three specialized ML classifiers each analyze the cleaned text independently: one for depression signals, one for stress levels, and one for fine-grained emotion categories.
4. Their outputs are combined into a single risk level: LOW, MEDIUM, or HIGH.
5. Based on that risk level, the system decides what additional context to inject into the LLM prompt — clinical resources for MEDIUM, hardcoded crisis helplines for HIGH, or nothing extra for LOW.
6. The LLM generates a response shaped by all of this context, not just the user's words.

The result is a chatbot where every response is grounded in actual analysis of the user's mental state rather than a generic conversational reply.

---

## Core Capabilities

**Emotion Analysis:** Classifies text into 28 fine-grained emotional categories, including sadness, grief, anxiety, joy, optimism, and 23 others, using Google's GoEmotions dataset and a RoBERTa-based model.

**Stress Detection:** Detects stress signals using a DistilBERT model trained on the Dreaddit dataset (social media posts labeled for stress vs. no stress). Returns a confidence score and a high/medium/low stress level.

**Depression Detection:** Identifies language patterns associated with depression using a BERT-based model trained on cleaned Reddit posts labeled as depressive or non-depressive. Returns a confidence score.

**Risk Stratification:** Aggregates all three model outputs into a single risk tier for the current message (LOW / MEDIUM / HIGH). This tier drives the entire response strategy.

**Crisis Detection (pre-ML):** A keyword and pattern matching system that can identify suicidal ideation, self-harm language, and severe hopelessness before ML inference even runs. This is a hard safety measure — it operates independently of ML confidence scores.

**RAG-Powered Guidance (MEDIUM risk only):** When risk is assessed as MEDIUM, a retrieval system queries a FAISS vector index containing CBT techniques, breathing exercises, grounding exercises, and general coping strategies. The top-3 most relevant passages are injected into the LLM prompt so its response is clinically grounded.

**Persistent Memory:** Supermemory.ai is used as a per-user emotional memory store. Over time it builds a profile of the user's emotional patterns, recurring triggers, and which coping tools they've responded to before. This allows the LLM to personalize its responses across sessions.

**Crisis Escalation:** When HIGH risk or a crisis keyword is detected, the frontend is instructed through a WebSocket event to display a full-screen crisis overlay with localized helpline numbers, a call-to-action for a human callback, and location-based crisis center information.

---

## Who Built It and When

- **Team:** Victims of DSA (2–3 members)
- **Track:** Healthcare (HC-11)
- **Time Budget:** 7 hours (hackathon sprint format)
- **Demo Target:** Local machine with all services running locally

---

## What Makes This Different from a Normal Chatbot

Most chatbots pass user text directly to an LLM and let the model figure out tone and content. ASTRAVA's approach is deliberately different:

- The LLM never "guesses" whether someone is stressed or depressed — it is told explicitly via structured ML output.
- Crisis detection runs as a rule-based system that cannot be fooled by unusual phrasing the ML model might miss.
- RAG ensures that when clinical resources are recommended, they come from curated, evidence-based content — not the LLM's parametric memory.
- The risk-level state machine enforces distinct response protocols for each tier, so a person in crisis never gets treated the same as someone having a mildly bad day.

---

## Project Status at Time of Documentation (March 6, 2026)

The Python ML layer is fully built and tested. All three models are working, the preprocessing pipeline is clean, and a unified inference runner can take any text input and return structured results from all three models in under 700ms. The backend (Node.js) and frontend (React) are scaffolded but their internal logic has not yet been implemented. The RAG data files exist but the FAISS indexing and retrieval system has not been built yet. The LLM integration (Ollama/Groq) has not been implemented yet.

A detailed account of exactly what is built vs. what is pending is in the file `07_CURRENT_BUILD_STATE.md`.
