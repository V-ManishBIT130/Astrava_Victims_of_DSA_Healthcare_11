# ASTRAVA — Message Lifecycle (Full Workflow)

This document traces exactly what happens from the moment a user types a message to the moment they receive a response. Every step is described in the order it actually executes.

---

## Phase 1: User Input Capture

The user types a message into the React chat interface and presses send.

If this is the user's first session and they haven't shared their location yet, the browser's Geolocation API might prompt for permission at this point (or earlier on page load). If the user grants it, the browser's latitude and longitude are sent to the backend and stored in the session. This is used later if a crisis is detected and region-specific helplines need to be displayed.

The message is emitted to the backend over the existing Socket.IO WebSocket connection. The event is called `user_message` and carries the message text, the user's ID, and the session ID.

The backend immediately emits a `typing_indicator` event back to the frontend so the user sees a "..." indicator while the pipeline runs.

---

## Phase 2: NLP Preprocessing (Python Layer)

The backend calls the Python FastAPI service at `POST /api/analyze` with the raw message text. Inside the Python layer, the message enters the preprocessing pipeline.

**Text Cleaning (15 steps):**
The message is processed through a 15-step TextCleaner. The intent of these steps is to normalize the text so that ML models trained on cleaner text can correctly process real-world informal messages. The steps handle:
- Stripping emoji (they carry signal but the specific unicode characters disrupt tokenizers)
- Unicode normalization (converting lookalike characters, smart quotes, etc. to ASCII equivalents)
- Tagging punctuation patterns before removal (e.g., three exclamation marks get replaced with a special token `__EXCLAMATION__` so the emotional signal is preserved even after punctuation removal)
- Lowercasing everything
- Removing URLs, email addresses, phone numbers, usernames, and hashtags
- Removing special characters and symbols
- Expanding contractions ("I'm" → "I am", "can't" → "cannot") — this is critical because negations must be preserved for emotional analysis
- Normalizing internet slang ("rn" → "right now", "idk" → "I do not know", "ngl" → "not going to lie")
- Final whitespace normalization

There is also a hard input cap at 4000 characters. If the raw text exceeds this, it is silently truncated before cleaning begins. 4000 characters is roughly 1000 tokens — well above the 512-token limit of the models. This prevents the cleaner from wasting time processing extremely long pastes.

**Crisis Detection:**
Running concurrently with the cleaning output, the CrisisDetector analyzes the cleaned text for safety signals. This is not an ML model — it is a pattern and keyword matching system. It is designed to be completely reliable and never miss a hard signal.

The detector looks for:
- Explicit suicidal ideation phrases ("want to end it all", "no reason to live", "want to die")
- Self-harm language ("cut myself", "hurt myself", "self-harm")
- Hopelessness clusters ("nobody cares and I can't go on", "better off without me")
- Farewell patterns combined with distress language
- Psycholinguistic signals: an abnormally high ratio of first-person pronouns ("I", "me", "my") which research associates with depression; a high count of absolutist words ("always", "never", "nothing", "everyone"); and negation counts

Each of these dimensions contributes to a severity rating. The severity levels are: NONE, LOW, MEDIUM (called ELEVATED in some parts of the codebase for MEDIUM-HIGH), HIGH, and CRITICAL.

**Output:**
The preprocessing module returns a structured result containing:
- `cleaned_text`: the processed text, ready for the ML classifiers
- `filtered_text`: a further stopword-filtered version used only by RAG/FAISS (not for the models — transformer models need grammatically full sentences)
- `crisis_result`: the full crisis assessment including severity, matched keyword categories, and psycholinguistic feature values
- `was_truncated`: a boolean indicating whether the input was cut off

---

## Phase 3: Crisis Short-Circuit Check

Immediately after preprocessing, the system checks the crisis severity. This check happens BEFORE running any ML model.

If the severity is CRITICAL or HIGH, the system short-circuits. It does not run the depression classifier, stress detector, or emotion model. Instead it immediately returns the crisis result to the backend.

The reason for this is both speed and principle. Speed: if someone is in crisis, they need a response in milliseconds, not after waiting for three transformer models to run three forward passes. Principle: a rule-based safety system that matches explicit crisis language is more reliable for life-safety decisions than a probabilistic ML model.

When the backend receives a crisis result with HIGH or CRITICAL severity, it triggers the crisis escalation protocol — it emits the `crisis_alert` WebSocket event to the frontend and prepares the LLM with special HIGH-risk instructions.

---

## Phase 4: ML Model Inference

If the crisis check does not short-circuit, the cleaned text is passed to the three ML models in sequence.

**Model 1 — Depression Classifier (BERT-base-uncased):**
Uses the HuggingFace checkpoint `poudel/Depression_and_Non-Depression_Classifier`. This is a binary classifier (Depression / Non-Depression) fine-tuned on cleaned Reddit posts. It returns a confidence score for each class. Inference time is approximately 400–600ms on CPU.

**Model 2 — Stress Detector (DistilBERT):**
Uses the HuggingFace checkpoint `jnyx74/stress-prediction`. Fine-tuned on the Dreaddit dataset (real social media posts labeled for stress vs. no stress). Returns a binary label (Stressed / No Stress) plus a probability. Also converts the probability to a three-tier stress level (LOW below 40%, MEDIUM between 40% and 70%, HIGH above 70%). Inference time is approximately 20–30ms on CPU (DistilBERT is much lighter than full BERT).

**Model 3 — Emotion Classifier (RoBERTa-base):**
Uses the HuggingFace checkpoint `SamLowe/roberta-base-go_emotions`. This is a multi-label classification model with 28 emotional categories from Google's GoEmotions dataset. Unlike the other two models which give one label, this one can return multiple active emotions simultaneously (because a person can genuinely feel sad AND annoyed at the same time). It returns a probability score for all 28 labels. Labels above a per-label threshold are considered "active." Inference time is approximately 50–70ms on CPU.

---

## Phase 5: Risk Score Aggregation

After all three models run, their outputs flow into the score aggregator.

The aggregator computes a composite risk score using weighted contributions:
- Depression score contributes 40% (strongest clinical indicator)
- Stress score contributes 25% (amplifies other signals but is less specific)
- Negative emotion intensity (average of the highest-scoring negative emotion labels) contributes 20%
- Absence of positive emotions (joy, optimism, love all below 0.1) adds a 15% risk bonus

The composite score is mapped to three risk levels:
- **LOW:** 0.0 to 0.35
- **MEDIUM:** 0.36 to 0.65
- **HIGH:** 0.66 to 1.0

If the crisis detector flagged a keyword match at any severity, that can also force the risk level to HIGH regardless of model scores.

---

## Phase 6: RAG Retrieval (MEDIUM Risk Only)

At this point the backend has received the ML scores and risk level back from the Python layer.

If and only if `risk_level == MEDIUM`, the orchestrator calls the RAG retriever. The retriever queries the FAISS vector indices using the user's cleaned text combined with their top-emotion labels as the search query.

The four FAISS indices are:
- CBT Techniques — cognitive behavioral therapy exercises and thought-reframing prompts
- Breathing and Grounding — guided breathing scripts and sensory grounding exercises
- Coping Strategies — journaling prompts, social connection suggestions, sleep hygiene, mindfulness
- Crisis Lines — country and region-specific helpline numbers (this index IS queried for HIGH risk but is used differently — directly populated into the crisis overlay, not injected into the LLM prompt)

The retriever returns the top-3 most relevant passages across the applicable indices. Each passage has a source label, its text content, and a similarity score.

For LOW risk: no RAG retrieval happens. The LLM handles a light supportive conversation on its own.
For HIGH risk: no RAG retrieval for clinical content. Helplines are hardcoded into the crisis protocol.

---

## Phase 7: Memory Retrieval (Supermemory.ai)

Before building the LLM prompt, the orchestrator queries Supermemory for the current user's emotional history.

Supermemory is queried by sending the user's ID and the current emotional state (top emotions plus the cleaned message) as the search query. It returns stored memories from previous sessions: past emotional patterns (e.g., "user has expressed sadness in 4 of last 7 sessions"), recurring triggers (e.g., "work deadlines appear frequently"), preferred coping tools (e.g., "user responded positively to breathing exercises"), and escalation history.

If this is the user's first message ever, Supermemory returns nothing. The LLM handles this cold-start gracefully — it simply doesn't reference any past patterns.

---

## Phase 8: LLM Response Generation

The orchestrator assembles the full context bundle and sends it to the LLM.

**What the LLM receives:**
- A system prompt defining the chatbot persona, tone rules, and risk-level-specific behavioral guidelines
- The current user message (cleaned text)
- The ML scores (depression score, stress score, top emotions, composite risk level)
- RAG passages (only if MEDIUM risk; empty strings if LOW or HIGH)
- Memory context from Supermemory (the user's emotional history)
- The last 6–10 turns of conversation for continuity

**How the LLM behaves based on risk level:**

For LOW risk, the LLM engages warmly and conversationally. It acknowledges the user's feelings and gently explores what's on their mind without being clinical or alarming. No RAG content is referenced.

For MEDIUM risk, the LLM gives a more structured and supportive response. It references the retrieved RAG passages explicitly — suggesting specific CBT techniques, breathing exercises, or coping strategies by name. It asks about support systems and checks on basic needs like sleep, food, and movement.

For HIGH risk, the LLM's primary goal is stabilization. It expresses genuine care, directly but gently checks in on the user's safety, and includes the crisis helpline numbers in its response text. The crisis overlay on the frontend is activated by a separate WebSocket event before or alongside the response text.

**LLM Selection:**
The backend first tries Ollama on localhost:11434 (LLaMA 3 8B running locally). If the request fails or takes over 10 seconds, it automatically switches to Groq's API using the same model. The reason for running LLaMA locally as the primary is privacy — user mental health conversations are sensitive, and keeping inference on-device is best practice.

---

## Phase 9: Response Delivery

The LLM response is received by the backend orchestrator.

The response text is emitted to the frontend over WebSocket as a `bot_response` event. The typing indicator is cleared.

If the risk level was HIGH, a separate `crisis_alert` event is also emitted with the helpline data filtered by the user's stored location. The frontend renders this as the full-screen crisis overlay.

The frontend renders the response as a chat bubble. For MEDIUM risk responses, a resource card may be shown below the bubble linking to the referenced techniques. For HIGH risk, the full crisis overlay takes over the screen.

---

## Phase 10: Asynchronous Cleanup

After the response is delivered to the user, the orchestrator performs two non-blocking writes:

**Supermemory write:** The current emotional state, risk level, coping strategies that were suggested, and whether a crisis was escalated are written back to Supermemory. This updates the user's longitudinal profile for future sessions.

**MongoDB write:** The full conversation turn (user message, bot response, timestamp, risk level, ML scores) is stored in the database. This is what populates the conversation history view in the frontend.

Both writes happen after the response is already delivered — they never block the user's experience.
