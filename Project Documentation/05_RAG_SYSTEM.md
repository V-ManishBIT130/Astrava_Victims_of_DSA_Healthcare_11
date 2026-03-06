# ASTRAVA — RAG System (Retrieval-Augmented Generation)

---

## What RAG Means in This Project

RAG stands for Retrieval-Augmented Generation. The idea is that instead of relying purely on an LLM's training data (its "parametric knowledge"), you first retrieve relevant documents from an external knowledge base, then inject those documents into the LLM's prompt. The LLM's response is then grounded in that external content.

In ASTRAVA's case, the knowledge base is a curated collection of evidence-based mental health resources: CBT exercises, breathing and grounding techniques, coping strategies, and crisis helplines. When a user's message is classified as MEDIUM risk, the system retrieves the 3 most relevant passages from this knowledge base and adds them to the LLM prompt, so the chatbot's response can recommend specific, real techniques rather than generic advice.

---

## The Critical Rule: RAG Only Runs for MEDIUM Risk

This is probably the most important thing to understand about ASTRAVA's RAG system.

**LOW risk:** No RAG retrieval. The LLM generates a warm, conversational, empathetic response on its own. At this risk level, pulling clinical resources would feel out of place and potentially alarming. A light conversation about a mildly bad day does not need a CBT worksheet.

**MEDIUM risk:** RAG is activated. The retrieved passages are injected into the LLM prompt, and the LLM is instructed to explicitly reference them in its response — naming the technique, explaining how to use it, framing it with clinical context.

**HIGH risk:** No RAG retrieval for clinical content. The crisis protocol uses hardcoded helpline data instead. In a crisis, there is no time for semantic search and passage retrieval from clinical indexes. The user needs immediate, reliable human contact information, not a breathing exercise. The crisis helplines data (the fourth index) IS referenced during HIGH risk but through a hardcoded lookup by region, not through semantic search.

---

## Where RAG Lives

The RAG module lives in the Node.js backend, not in the Python layer. Specifically it is planned to be in `backend/src/rag/`. This is intentional — RAG retrieval is directly coupled with LLM prompt assembly. Since the LLM client also lives in the backend, keeping them together avoids making an extra network call from the backend to Python just to get passages.

The four data source directories already exist:
- `backend/src/rag/data/cbt_techniques/`
- `backend/src/rag/data/breathing_grounding/`
- `backend/src/rag/data/crisis_lines/`
- `backend/src/rag/data/coping_strategies/`

These directories contain the raw text documents that are encoded into FAISS indices. The indexing script and retrieval module have not been implemented yet (as of March 2026).

---

## The Four Knowledge Indexes

**Index 1 — CBT Techniques**

Contains descriptions of Cognitive Behavioral Therapy exercises and techniques. CBT is one of the most evidence-backed therapeutic approaches for depression, anxiety, and stress. The documents in this index describe practical exercises: thought records (identifying automatic negative thoughts and challenging them with evidence), behavioral activation (scheduling pleasurable activities to break the cycle of withdrawal), cognitive restructuring, problem-solving frameworks, and graded exposure approaches for anxiety.

The content should come from open-access CBT manuals and psychoeducation materials. The passages should be written in plain, accessible language — not academic jargon — because they get injected directly into a conversational response.

**Index 2 — Breathing and Grounding Exercises**

Contains clinically validated relaxation and grounding techniques with step-by-step instructions. The most important ones are:
- Box breathing (4 counts in, hold 4, out 4, hold 4) — used for acute anxiety and stress
- 4-7-8 breathing (inhale 4, hold 7, exhale 8) — activates the parasympathetic nervous system, good for calming before sleep
- Diaphragmatic breathing — belly breathing for chronic stress
- 5-4-3-2-1 grounding (name 5 things you can see, 4 you can hear, 3 you can touch, 2 you can smell, 1 you can taste) — for dissociation or overwhelming anxiety
- Progressive Muscle Relaxation — for physical tension that accompanies stress

These are short, instructional, and action-oriented. They are most useful at MEDIUM risk when the user is distressed but not in crisis.

**Index 3 — Coping Strategies**

A broader set of general mental health coping tools that do not fit neatly into CBT or breathing categories. This includes:
- Journaling prompts designed to externalize difficult emotions
- Social connection strategies (reaching out, identifying support systems)
- Sleep hygiene guidelines
- Physical movement and exercise recommendations (research on exercise and depression is robust)
- Mindfulness techniques
- Psychoeducation about the stress-response cycle and how to interrupt it

**Index 4 — Crisis Lines**

A directory of crisis helpline phone numbers and text lines, organized by country and region. For India specifically (the primary target market): iCall (9152987821), Vandrevala Foundation (18602662345), AASRA, iCall. For international coverage: international crisis text line numbers, Befrienders Worldwide.

This index is special. For MEDIUM risk, the retriever skips this index — helpline numbers are not appropriate to inject into a MEDIUM-risk response. For HIGH risk, the system does not semantically search this index either — instead, the user's stored geolocation coordinates are used to directly look up the appropriate region entries.

---

## How the Retrieval Works (Planned Architecture)

The FAISS retrieval system works through three components, each planned as a separate module:

**Embedder (`backend/src/rag/embedder.js`):**
Wraps the `sentence-transformers/all-MiniLM-L6-v2` model. This is a small, fast sentence embedding model that encodes text into 384-dimensional dense vectors. The same model must be used to encode both the knowledge base documents and the user's query — otherwise the vector space won't be comparable.

`all-MiniLM-L6-v2` was chosen because it is the standard choice for FAISS-based retrieval: fast (runs on CPU in milliseconds), good quality for English text, well-maintained, and widely documented.

**Indexer (`backend/src/rag/indexer.js`):**
A build-time script (not called at runtime). Reads all the text documents from the four data directories, encodes each document using the embedder, and stores the resulting vectors in FAISS index files saved to disk. Each index file corresponds to one knowledge category. This script only needs to be re-run when the knowledge base content changes.

**Retriever (`backend/src/rag/retriever.js`):**
Called at runtime by the orchestrator. Takes the user's emotional state (top emotions + cleaned text) as the query, encodes it using the same embedder, and uses FAISS to find the top K vectors by cosine similarity from the appropriate indices. Returns the top-3 matching passages with their source category and similarity score.

---

## What Gets Injected Into the LLM Prompt

When risk is MEDIUM, the top-3 retrieved passages are formatted into a section of the LLM's system or user message context. Something like:

"Based on the user's current emotional state, the following evidence-based techniques are relevant: [passage 1 from CBT index] [passage 2 from breathing index] [passage 3 from coping index]. Please naturally weave these techniques into your response, explaining them clearly and empathetically."

The LLM is then able to produce a response like: "It sounds like you're carrying a lot right now. One thing that might help is a technique called behavioral activation — it's a CBT approach where you schedule small, meaningful activities even when you don't feel like it, because action often precedes mood change rather than the other way around..."

The key point is that the LLM is not inventing or hallucinating the technique. It is accurately presenting something from the knowledge base that was specifically selected for relevance to this user's state.

---

## What Is Currently Built

The RAG data source directories exist and can have documents added to them. The FAISS indexing script, the retriever, and the embedder wrapper have NOT yet been built as of March 2026. This is one of the major pending tasks in the backend. For the project to work end-to-end, all three RAG components need to be implemented in `backend/src/rag/`.

The documents that go in the data directories also need to be curated and written. These should be short paragraphs (not full articles — FAISS retrieval works best when each chunk is a focused, self-contained description of a single technique or resource). A good target is 50–150 words per document, with 8–20 documents per index.
