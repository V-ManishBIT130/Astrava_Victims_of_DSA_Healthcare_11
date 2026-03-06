# ASTRAVA — Preprocessing Pipeline

All of ASTRAVA's ML inference happens on text that has been prepared by this pipeline first. This document explains every component, every decision, and exactly what the pipeline does and does not do.

---

## Where This Lives

The entire preprocessing layer is in `python/preprocessing/`. It consists of seven source files, each with a specific role:

- `pipeline.py` — the orchestrator; the only file external code needs to call
- `cleaner.py` — the 15-step text normalization logic
- `crisis_detector.py` — keyword and pattern safety system
- `stopwords.py` — the emotionally-aware stopword filter
- `keywords.py` — the crisis keyword banks
- `config.py` — shared configuration constants
- `embedder.py` and `tokenizer.py` — present for legacy reasons but currently not used by the pipeline

---

## What the Pipeline Does NOT Do

This is equally important: the pipeline does not tokenize text for the ML models and does not generate embeddings. Those steps are handled inside each individual model module.

The reason this matters is history. An earlier version of the pipeline loaded all three model tokenizers and ran all three forward passes inside the pipeline itself — resulting in every inference operation running twice (once in the pipeline and once again in the model module), with the pipeline's outputs never actually being used. That dead code was removed. The pipeline now does only what it is designed to do: clean, crisis-check, and stopword-filter.

---

## The Three Outputs of the Pipeline

Every call to `pipeline.process_text_only(raw_text)` returns a `PreprocessingResult` object with four fields:

**cleaned_text:** The product of the 15-step TextCleaner. This is what goes to the ML models. It is lowercase, contraction-expanded, slang-normalized, and stripped of noise. Critically, it is still a grammatically normal sentence. Transformer models were pretrained on full English sentences and expect that kind of input.

**filtered_text:** The cleaned text after additional stopword filtering. Function words and common neutral words are removed, leaving mostly content words and emotion words. This is used exclusively for RAG/FAISS keyword-based search. It is NOT appropriate for the ML models because removing words like "not" or "never" would destroy the negation signals those models rely on.

**crisis_result:** The CrisisDetector's full output — a severity level, a list of matched keyword categories, a list of specific matched keywords, and psycholinguistic measurements.

**was_truncated:** A boolean. True if the raw input was longer than 4000 characters and was cut down before cleaning.

---

## The TextCleaner: 15 Steps

The 15 steps happen in this specific order. Order matters because some steps depend on what previous steps have done.

**Step 1 — Emoji Stripping:**
All Unicode emoji characters are removed. Emojis carry emotional signal (😭 very different from 😊) but they interfere with tokenizers and the models weren't trained on them. The emoji count and types are logged separately for context before removal.

**Step 2 — Unicode Normalization:**
Converts the text to Unicode normal form NFKC. This collapses characters that look the same but have different code points — for example, "ＨＥＬＬＯ" (fullwidth Latin letters) becomes "HELLO", curly quotes become straight quotes, ligatures like "ﬁ" become "fi".

**Step 3 — Emotional Punctuation Tagging:**
Before any punctuation is removed, patterns of repeated punctuation are replaced with special tokens. Three or more exclamation marks become `__EXCLAMATION__`, three or more question marks become `__QUESTION__`. This preserves the emotional signal even after the punctuation itself is cleaned away. These tokens survive into the cleaned text and the models can see them.

**Step 4 — Lowercase:**
The entire text is lowercased. This was originally a step that could be skipped for proper nouns, but for this conversational use case uniformity is more important than preserving case distinctions.

**Step 5 — URL Removal:**
All HTTP/HTTPS URLs are removed. They carry no emotional content and would confuse the tokenizer.

**Step 6 — Email Address Removal:**
Email addresses are removed.

**Step 7 — Phone Number Removal:**
Phone number patterns are removed.

**Step 8 — Username and Mention Removal:**
Social media @mentions and u/ usernames are removed.

**Step 9 — Hashtag Removal:**
Hashtag symbols are removed but the word they attached is kept (e.g., "#anxious" becomes "anxious").

**Step 10 — Special Character Removal:**
Most non-alphabetic characters are removed. The special tokens from Step 3 (`__EXCLAMATION__` etc.) are protected from this step.

**Step 11 — Contraction Expansion:**
All common English contractions are expanded. This step is critical for the ML models. "I'm" must become "I am" so the pronoun "I" is correctly counted. "Can't" must become "cannot" so negation words are preserved. "Won't", "didn't", "shouldn't", "I've", "they're" — all handled. A comprehensive dictionary of contractions is maintained in `cleaner.py`.

**Step 12 — Internet Slang Normalization:**
Informal internet abbreviations are expanded to full phrases. This is a large dictionary. Examples: "rn" → "right now", "idk" → "I do not know", "ngl" → "not going to lie", "brb" → "be right back", "tbh" → "to be honest". Without this step, slang would reduce emotion signals that the model can only detect if the words are spelled out.

**Step 13 — Redundant Whitespace Normalization:**
Multiple consecutive spaces are collapsed to one space.

**Step 14 — Leading and Trailing Whitespace Removal:**
The string is stripped.

**Step 15 — Empty Output Check:**
If the entire cleaning process resulted in an empty string (e.g., the input was just emojis or symbols), a safe fallback is returned.

---

## The Crisis Detector

The CrisisDetector runs on the cleaned text and produces a structured assessment of its safety risk. It is a rule-based system, not a machine learning model.

### Why Rule-Based?

ML models assign probabilities, which means they can miss things with low confidence. A rule-based keyword system, by contrast, is binary — if the phrase is there, the flag is raised. For mental health crisis detection, missing a crisis signal is far worse than a false positive. Rule-based systems are also deterministic, auditable, and much faster than ML inference.

### What It Checks

The detector operates across several dimensions simultaneously:

**Suicidal Ideation Keywords:**
A manually curated list of phrases that explicitly indicate suicidal intent. These include multi-word phrases like "want to end it all", "no reason to live", "want to die", "better off dead", as well as single words like "suicide" and "suicidal". Single-word keywords are matched with word-boundary logic, so "suicide" is found but the word "homocide" would not accidentally match.

**Self-Harm Language:**
Phrases indicating self-harm: "cut myself", "hurt myself", "self-harm", "cutting", "burning myself", "overdose".

**Hopelessness Patterns:**
Compound phrases that signal severe hopelessness: "nobody cares", "better off without me", "can't take it anymore", "can't go on", "want to disappear", "everyone would be better without me".

**Farewell Patterns:**
Words like "goodbye" or "farewell" appearing alongside distress signals. Goodbye alone is not a crisis signal. Goodbye combined with other flags raises the severity.

**Psycholinguistic Features:**
Three numeric signals extracted from the text:
- I-ratio: the proportion of first-person singular pronouns ("I", "me", "my", "myself", "mine") out of total words. Research in computational psycholinguistics (Pennebaker's work) has found that elevated first-person singular usage correlates with depression and emotional distress.
- Absolutist word ratio: the proportion of words that are absolutist ("always", "never", "nothing", "everything", "everyone", "nobody", "completely", "totally", "perfectly", "all"). Research has found that absolutist thinking language is a reliable signal of depression and anxiety.
- Negation count: raw count of negation words ("not", "never", "no", "nobody", "nothing", "nowhere", "none", "isn't", "can't", "won't", etc.). High negation counts amplify the interpretation of other signals.

### Severity Levels

The detector maps its findings to a five-level severity scale:

**NONE:** No signals found. Text is benign from a safety perspective.

**LOW:** Light signals present — mild negative language or slightly elevated I-ratio, but no explicit crisis content. Normal conversation about having a bad day.

**MEDIUM (ELEVATED):** Moderate signals — hopelessness language, multiple negations, high absolutist ratio, or mildly concerning keywords. Worth monitoring, but not yet a crisis.

**HIGH:** Clear crisis signals — at least one keyword from the suicidal ideation or self-harm banks matched, or a combination of high psycholinguistic scores with hopelessness language.

**CRITICAL:** Explicit, unambiguous statements of imminent intent to harm. These are the top-level phrases in the keyword bank.

### The Short-Circuit Rule

When severity is HIGH or CRITICAL, the inference pipeline does not run the ML models. This is the most important safety decision in the entire architecture. An explicit crisis statement must get an immediate response — not a response that waits 600ms for BERT to process the text.

Severities of NONE, LOW, and MEDIUM/ELEVATED allow normal model inference to proceed. The crisis result is still passed along for context.

---

## The Emotional Stopword Filter

After crisis detection, the cleaned text also passes through the `EmotionalStopwordFilter`, which produces the `filtered_text` output.

This filter is NOT a standard stopword filter. Most stopword lists remove common English function words (the, a, is, of, to, etc.). This is fine for information retrieval but terrible for emotional language — "not happy" is completely different from "happy", and removing "not" destroys the meaning.

The emotional stopword filter explicitly protects:
- All negation words: not, never, no, nobody, nothing, nowhere, none, nor
- Intensifiers: very, really, so, extremely, incredibly, absolutely
- Emotional transitional words: but, however, although, yet, still — these often flip the sentiment of a sentence

The output is used for RAG search, where keyword overlap matters more than full grammatical fidelity.

---

## The Input Cap (4000 Characters)

Before the cleaning even begins, the pipeline checks the raw input length. If it exceeds 4000 characters, it is cut to exactly 4000 characters. The `was_truncated` field in the result is set to True.

4000 characters is approximately 1000 tokens. The ML models all have a maximum sequence length of 512 tokens. The cleaner itself is not the bottleneck here — the concern is that cleaning a 10,000-character paste would waste several hundred milliseconds and produce output that the model will silently truncate anyway. Better to be explicit about the truncation at the input stage.

The `MAX_SEQ_LENGTH` in `config.py` is set to 512 — matching the actual limit of the transformer models. An earlier version had this set to 128, which was silently truncating roughly 60% of real conversational messages. That bug was identified and fixed.

---

## config.py Values That Matter

- `EMOTION_MODEL_NAME`: `SamLowe/roberta-base-go_emotions` — the correct GoEmotions checkpoint. An earlier version had this pointing to `mental/mental-roberta-base`, a different model. That was fixed.
- `MAX_SEQ_LENGTH`: 512 — was previously 128, now correct.
- `MAX_RAW_CHARS`: 4000 — the input truncation cap.
