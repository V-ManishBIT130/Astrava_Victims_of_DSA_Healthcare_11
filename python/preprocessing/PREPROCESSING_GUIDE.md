# ASTRAVA — Preprocessing Pipeline

> **Component:** `python/preprocessing/`  
> **Purpose:** Transforms raw user input text into clean, crisis-assessed, model-ready output  
> **Language:** Python 3.11+  
> **Dependencies:** `transformers`, `torch` *(only for full pipeline mode)*

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Pipeline Modes](#4-pipeline-modes)
5. [Step-by-Step Pipeline Walkthrough](#5-step-by-step-pipeline-walkthrough)
   - [Step 1 — Emoji Stripping](#step-1--emoji-stripping)
   - [Step 2 — Unicode Normalization](#step-2--unicode-normalization)
   - [Step 3 — Punctuation Tags](#step-3--punctuation-tags)
   - [Step 4 — Lowercase](#step-4--lowercase)
   - [Steps 5–10 — Noise Removal](#steps-510--noise-removal)
   - [Step 11 — Contraction Expansion](#step-11--contraction-expansion)
   - [Step 12 — Slang Normalization](#step-12--slang-normalization)
   - [Steps 13–15 — Final Normalization](#steps-1315--final-normalization)
   - [Crisis Detection](#crisis-detection)
   - [Stopword Filtering](#stopword-filtering)
6. [The PreprocessingResult Object](#6-the-preprocessingresult-object)
7. [Crisis Detection Deep Dive](#7-crisis-detection-deep-dive)
   - [Severity Levels](#severity-levels)
   - [Detection Categories](#detection-categories)
   - [Psycholinguistic Features](#psycholinguistic-features)
8. [Keyword Banks](#8-keyword-banks)
9. [Full Pipeline Mode (Models)](#9-full-pipeline-mode-models)
10. [Individual Component Usage](#10-individual-component-usage)
11. [Interactive Runner](#11-interactive-runner)
12. [File Reference](#12-file-reference)
13. [Extending the Pipeline](#13-extending-the-pipeline)
14. [Design Decisions](#14-design-decisions)

---

## 1. Overview

The ASTRAVA preprocessing pipeline takes **raw user input** — including slang, emojis, typos, contractions, and informal internet language — and produces a **clean, semantically enriched text** ready for downstream ML classifiers (Emotion, Stress, Depression models).

It also performs a **parallel crisis safety check** that runs *before* any ML model inference, so immediate escalation can be triggered independently of model predictions.

```
Raw user input
    │
    ▼
┌─────────────────────────────────────────┐
│           TextCleaner  (15 steps)       │
│  emoji strip → unicode → punct tags →  │
│  lowercase → noise removal →           │
│  contractions → slang → normalize      │
└────────────────────┬────────────────────┘
                     │  cleaned_text
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌────────────────────────┐
│  CrisisDetector  │   │  EmotionalStopword     │
│  (regex + phrase │   │  Filter                │
│   bank + psycho- │   │  (removes noise,       │
│   linguistic)    │   │   keeps emotion words) │
└──────────────────┘   └────────────────────────┘
          │                     │
          ▼                     ▼
    CrisisResult           filtered_text  ← FINAL OUTPUT
    (severity,                            (ready for ML models)
     categories,
     psycholinguistic)
```

---

## 2. Architecture

The module lives entirely in `python/preprocessing/` and consists of **7 source files**:

| File | Role |
|---|---|
| `config.py` | All regex patterns, model names, `CONTRACTIONS_MAP`, `SLANG_MAP`, `PUNCT_TAGS` |
| `keywords.py` | All keyword/phrase banks: crisis, depression, anxiety, stress, negations, intensifiers, first-person, absolute language, rumination, dissociation |
| `cleaner.py` | `TextCleaner` — 15-step text cleaning pipeline |
| `crisis_detector.py` | `CrisisDetector` + `CrisisResult` — regex + phrase-bank + psycholinguistic detection |
| `stopwords.py` | `EmotionalStopwordFilter` — emotionally-aware stopword removal |
| `tokenizer.py` | `ModelTokenizer` — per-model HuggingFace tokenizer wrapper |
| `embedder.py` | `ModelEmbedder` — per-model CLS-token embedding generator |
| `pipeline.py` | `PreprocessingPipeline` + `PreprocessingResult` — orchestrates everything |

---

## 3. Quick Start

### Installation requirements

```bash
pip install transformers torch
```

> **Note:** `transformers` and `torch` are only needed if you call `pipeline.process()` (full mode).  
> `pipeline.process_text_only()` has **zero ML dependencies** — pure Python.

### Minimal usage

```python
import sys
sys.path.insert(0, "path/to/python/")   # or run from the python/ directory

from preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline()

result = pipeline.process_text_only("I cant do this anymore tbh im so done")

print(result.cleaned_text)
# → "i cannot do this anymore to be honest i am so done"

print(result.filtered_text)
# → "i cannot anymore honest i so done"   (stopwords removed, emotional words kept)

print(result.crisis_result.severity)
# → "ELEVATED"

print(result.crisis_result.psycholinguistic)
# → {'i_ratio': 0.1538, 'absolute_ratio': 0.0, 'negation_count': 1, ...}
```

### Run the interactive terminal runner

```bash
cd python/
python run_pipeline.py          # text-only mode (no ML models)
python run_pipeline.py --full   # full mode (loads tokenizers + embedders)
```

---

## 4. Pipeline Modes

The pipeline has **two modes**. Use the right one for your situation.

### Mode 1 — Text-Only (recommended for real-time use)

```python
result = pipeline.process_text_only("your text here")
```

**What runs:** TextCleaner → CrisisDetector → EmotionalStopwordFilter  
**What does NOT run:** No tokenizers, no ML models downloaded  
**Speed:** Instant (milliseconds)  
**Use when:** You need cleaned text + crisis assessment and will call models separately, or you need a quick safety check.

### Mode 2 — Full Pipeline

```python
result = pipeline.process("your text here")
```

**What runs:** Everything in Mode 1 + per-model tokenization + per-model embeddings  
**Models loaded:** All 3 (emotion, stress, depression) — downloaded from HuggingFace on first run  
**Speed:** Slower on first call (model download), fast after caching  
**Use when:** You need `result.tokens` and `result.embeddings` for direct model inference.

### Mode 3 — Single Model

```python
result = pipeline.process_for_model("your text here", model_key="emotion")
# model_key options: "emotion", "stress", "depression"
```

**Use when:** You only need one model's tokens/embeddings to save memory.

---

## 5. Step-by-Step Pipeline Walkthrough

The `TextCleaner.clean()` method runs **15 ordered steps**. The order is intentional — changing it breaks things.

### Step 1 — Emoji Stripping

```
Input:  "I feel so sad 😭💔 everything is falling apart"
Output: "I feel so sad   everything is falling apart"
```

**Why first:** Emojis must be removed before any other step. If they survive to `remove_special_characters` (Step 13), some emoji bytes can corrupt surrounding tokens.  
**What's removed:** All Unicode emoji ranges — emoticons, pictographs, transport symbols, dingbats, supplemental symbols, ZWJ sequences, variation selectors.  
**Config:** `EMOJI_PATTERN` in `config.py`

---

### Step 2 — Unicode Normalization

```
Input:  "I don\u2019t feel okay"   (curly apostrophe)
Output: "I don't feel okay"        (straight apostrophe)
```

**Why second:** Must run *before* contraction expansion (Step 11). Typed or pasted text often contains curly/smart quotes (`'` U+2019) instead of straight apostrophes (`'`). If not normalized here, `don't` would not match the `don't` key in `CONTRACTIONS_MAP` and would pass through unexpanded.  
**Also removes:** Zero-width spaces, BOM characters, soft hyphens, control characters.  
**Method:** `unicodedata.normalize("NFKC", text)` + `UNICODE_NOISE_PATTERN`

---

### Step 3 — Punctuation Tags

**Why third:** Must run *before* lowercase (Step 4) so ALL_CAPS detection works on original casing.

Patterns that carry emotional meaning beyond what words alone say are converted to semantic tokens:

| Pattern | Tag inserted | Clinical meaning |
|---|---|---|
| `WORD` (3+ all-caps) | `[HIGH_AROUSAL]` | Shouting, extreme emotional activation |
| `!!!` (2+ `!`) | `[HIGH_INTENSITY]` | Heightened emotional urgency |
| `???` (2+ `?`) | `[CONFUSION]` | Cognitive overwhelm, inability to make sense |
| `?!` (mixed) | `[MIXED_INTENSITY]` | Conflicted emotional state |
| `...` or `…` | `[HESITATION]` | Trailing off, difficulty finding words |
| `sooooo` (3+ same char) | `[ELONGATION]` | Emotional emphasis via elongation |
| ` — ` (lone dash) | `[INTERRUPTION]` | Self-interruption, disorganized thought |
| `(not really)` | `[NEGATED_POSITIVITY]` | Masking true feelings behind false positive |
| `(just kidding)` / `(jk)` | `[POSSIBLE_DEFLECTION]` | Humor used to deflect emotional disclosure |

**Example:**
```
Input:  "I JUST CANT TAKE THIS ANYMORE!!!"
Output: "I  [HIGH_AROUSAL] JUST CANT TAKE THIS ANYMORE [HIGH_INTENSITY] "
```

---

### Step 4 — Lowercase

```
Input:  "I  [HIGH_AROUSAL] JUST CANT..."
Output: "i  [high_arousal] just cant..."
```

Standard lowercasing. Runs *after* punct tags so the ALL_CAPS pattern already fired.

---

### Steps 5–10 — Noise Removal

These steps strip content that carries no clinical meaning:

| Step | What's removed | Example |
|---|---|---|
| 5 — HTML tags | `<b>`, `<p>`, all tags | `<b>sad</b>` → `sad` |
| 6 — URLs | `http://`, `https://`, `www.` | `check https://example.com` → `check` |
| 7 — Emails | email addresses | `me@gmail.com` → ` ` |
| 8 — Mentions | `@username` | `@therapist help me` → `help me` |
| 9 — Hashtags | `#` symbol, keeps word | `#depression` → `depression` |
| 10 — Phone numbers | numeric phone patterns | `call 555-1234` → `call` |

---

### Step 11 — Contraction Expansion

```
Input:  "i can't do this, i don't want to, i won't try"
Output: "i cannot do this, i do not want to, i will not try"
```

**Why critical:** Without this, `can't` and `cannot` are treated as completely different tokens by ML models. More importantly, `not` is a negation word that *must* be preserved — but `can't` as a single token loses that negation during stopword filtering.

**150+ entries** in `CONTRACTIONS_MAP` cover:
- Standard contractions: `I'm`, `you've`, `they'd`
- Negative contractions: `can't` → `cannot`, `won't` → `will not`, `don't` → `do not`
- Informal/slang: `gonna` → `going to`, `wanna` → `want to`, `kinda` → `kind of`
- Double contractions: `wouldn't've` → `would not have`
- Chat abbreviations: `idk` → `i do not know`, `tbh` → `to be honest`

---

### Step 12 — Slang Normalization

```
Input:  "ngl im lowkey stressed af rn, kms lol"
Output: "not going to lie i am somewhat stressed af right now, [DARK_HUMOR_POSSIBLE] i said kill myself joking or serious laughing or deflecting emotion"
```

**What it does:** Replaces 100+ slang terms with their full-form clinical equivalents so downstream models see real language.

**Organized by category:**

| Category | Examples |
|---|---|
| Mental state | `tbh` → `to be honest`, `ngl` → `not going to lie`, `idc` → `i do not care` |
| Emotional state | `fml` → `my life is terrible`, `ugh` → `feeling frustrated`, `meh` → `feeling indifferent` |
| Existential/hopelessness | `cant even` → `i cannot cope`, `over it` → `i am exhausted and giving up`, `no point` → `there is no purpose` |
| Abbreviations (no apostrophe) | `cant` → `cannot`, `wont` → `will not`, `im` → `i am` |
| Time/situation | `rn` → `right now`, `atm` → `at the moment`, `tmr` → `tomorrow` |
| Intensifier slang | `lowkey` → `somewhat`, `highkey` → `very much`, `deadass` → `completely serious` |
| Physical symptoms | `brain fog` → `i cannot think clearly`, `burnt out` → `i am completely burned out`, `zoned out` → `i am dissociating` |
| Dark humor (flagged) | `kms` → `[DARK_HUMOR_POSSIBLE] i said kill myself joking or serious` |

> **Dark humor rule:** Terms like `kms` and `ded` are **never silently dropped**. They are flagged with `[DARK_HUMOR_POSSIBLE]` so the crisis detector can see them and set `needs_review=True` without auto-triggering a crisis response. This prevents both false positives (jarring crisis response to a joke) and false negatives (missing genuine distress).

---

### Steps 13–15 — Final Normalization

| Step | What it does | Example |
|---|---|---|
| 13 — Special characters | Remove everything except `a-z 0-9 ? ! . '` | `#@%` → ` ` |
| 14 — Repeated characters | Collapse 3+ same chars to 2 | `sooooo` → `soo`, `!!!` → `!!` |
| 15 — Whitespace | Collapse multiple spaces, strip edges | `"  word   word  "` → `"word word"` |

---

### Crisis Detection

Runs on the `cleaned_text` output. See [Section 7](#7-crisis-detection-deep-dive) for full details.

---

### Stopword Filtering

Runs on `cleaned_text` after crisis detection to produce `filtered_text`.

**Standard NLP stopword removal is NOT used.** The `EmotionalStopwordFilter` removes only words that carry no emotional meaning. The following are **always preserved**:

- **Negation words**: `not`, `never`, `cannot`, `barely`, `hardly`, `hopeless`, `empty`, `void` — removing these inverts meaning (`"I do not feel happy"` → `"feel happy"`)
- **Intensifiers**: `very`, `extremely`, `utterly`, `just`, `kinda`, `somewhat` — removing these loses severity
- **First-person pronouns**: `i`, `me`, `my`, `mine`, `myself` — Pennebaker (2003) depression marker
- **Absolute language**: `always`, `never`, `everyone`, `nothing`, `forever` — CBT black-and-white thinking marker
- **Emotion-bearing function words**: `feel`, `want`, `hate`, `hurt`, `hope`, `fear`, `cry`, `die`, `pain`, and 100+ more

---

## 6. The PreprocessingResult Object

Every pipeline call returns a `PreprocessingResult` dataclass:

```python
@dataclass
class PreprocessingResult:
    original_text:  str                      # The raw input as given
    cleaned_text:   str                      # After TextCleaner (15 steps)
    filtered_text:  str                      # After EmotionalStopwordFilter — FINAL OUTPUT
    crisis_result:  Optional[CrisisResult]   # Full crisis assessment

    # Only populated in full pipeline mode (process() / process_for_model())
    tokens:     Dict[str, Any]   # {"emotion": {...}, "stress": {...}, "depression": {...}}
    embeddings: Dict[str, Any]   # {"emotion": tensor(768), "stress": tensor(768), ...}
```

### Accessing results

```python
result = pipeline.process_text_only("I feel completely hopeless")

# The text your models should receive
print(result.filtered_text)

# Safety check
if result.crisis_result.is_crisis:
    print("ESCALATE IMMEDIATELY")
elif result.crisis_result.needs_review:
    print("PROBE FOR CONTEXT")

# Severity routing
severity = result.crisis_result.severity
# "NONE" | "LOW" | "MEDIUM" | "HIGH" | "ELEVATED" | "CRITICAL"

# Psycholinguistic features for your risk aggregator
psych = result.crisis_result.psycholinguistic
# psych["i_ratio"]          → depression nudge
# psych["absolute_ratio"]   → hopelessness nudge
# psych["has_rumination"]   → anxiety proxy
# psych["has_dissociation"] → change RAG retrieval to grounding exercises

# JSON serialization
data = result.to_dict()   # fully serializable, tensors → lists
```

---

## 7. Crisis Detection Deep Dive

The `CrisisDetector` runs **two parallel detection layers** and one **feature extraction layer** on every call.

### Layer 1 — Regex Pattern Categories (`CRISIS_PATTERNS`)

Six categories, each with compiled regex patterns:

| Category | Triggers | Severity |
|---|---|---|
| `PLANNING` | Having a plan, giving away possessions, goodbye letters, pill-taking, overdose | **CRITICAL** |
| `DIRECT_IDEATION` | `suicid*`, `kill myself`, `want to die`, `better off dead`, `no reason to live` | **HIGH** |
| `SELF_HARM` | `self-harm*`, `cutting myself`, `hurt myself`, `burn myself` | **HIGH** |
| `HOPELESSNESS` | `no future`, `trapped`, `never get better`, `tired of living`, `can't go on` | **ELEVATED** |
| `ISOLATION` | `i'm a burden`, `nobody cares`, `completely alone`, `everyone better off without me` | **ELEVATED** |
| `DARK_HUMOR_FLAG` | `kms`, `kys`, `i'm so dead`, `[DARK_HUMOR_POSSIBLE]` (injected by slang map) | **ELEVATED** |

### Layer 2 — Phrase-Bank Keyword Matching (`CRISIS_KEYWORDS` + domain sets)

Substring matching against four phrase banks:
- `CRISIS_KEYWORDS` — 130+ explicit crisis phrases
- `DEPRESSION_KEYWORDS` — 120+ depressive language indicators  
- `ANXIETY_KEYWORDS` — 100+ anxiety and panic indicators
- `STRESS_KEYWORDS` — 90+ stress and burnout indicators

### Severity Levels

```
CRITICAL   → PLANNING matched — ideation has moved to action
             Response: Immediate unconditional escalation

HIGH       → DIRECT_IDEATION or SELF_HARM matched, or crisis keyword phrases found
             Response: Escalate before model inference

ELEVATED   → HOPELESSNESS, ISOLATION, or DARK_HUMOR_FLAG matched
             needs_review = True
             Response: Gently probe for context — do NOT auto-trigger crisis response

MEDIUM     → 3+ distress signals from depression/anxiety/stress banks,
             OR 1+ signals with intensifiers present
             Response: Monitor, run full model inference

LOW        → 1–2 distress signals present
             Response: Continue with model inference

NONE       → No signals detected
```

### Psycholinguistic Features

Extracted from every input regardless of severity level. Fed into your risk aggregator alongside model scores.

```python
{
    "i_ratio":           0.1538,  # fraction of tokens that are i/me/my/mine/myself
    "absolute_ratio":    0.0769,  # fraction of tokens that are absolute language
    "negation_count":    3,       # raw count of negation tokens
    "intensifier_count": 2,       # raw count of intensifier tokens
    "has_rumination":    True,    # "keep thinking", "what if", "should have", etc.
    "has_dissociation":  False,   # "zoned out", "feel empty", "going through motions", etc.
}
```

**How to use these downstream:**

| Feature | Downstream effect |
|---|---|
| High `i_ratio` | Nudge depression score up |
| High `absolute_ratio` | Lower the MEDIUM risk threshold |
| `has_rumination = True` | Feed into anxiety proxy score |
| `has_dissociation = True` | Change RAG retrieval target to grounding exercises |

---

## 8. Keyword Banks

All keyword sets live in `keywords.py` and are exported from `__init__.py`.

| Export | Type | Description |
|---|---|---|
| `CRISIS_KEYWORDS` | `frozenset` | 130+ explicit crisis phrases |
| `CRISIS_PATTERNS` | `dict[str, list[re.Pattern]]` | 6-category compiled regex patterns |
| `DEPRESSION_KEYWORDS` | `frozenset` | 120+ depressive language phrases |
| `ANXIETY_KEYWORDS` | `frozenset` | 100+ anxiety/panic phrases |
| `STRESS_KEYWORDS` | `frozenset` | 90+ stress/burnout phrases |
| `NEGATION_WORDS` | `frozenset` | Sentiment-flipping words — always preserved |
| `EMOTION_INTENSIFIERS` | `frozenset` | Severity amplifiers + minimizers |
| `FIRST_PERSON` | `frozenset` | `i`, `me`, `my`, `mine`, `myself` |
| `ABSOLUTE_LANGUAGE` | `frozenset` | `always`, `never`, `forever`, `completely`, etc. |
| `RUMINATION_MARKERS` | `frozenset` | Cognitive loop, regret loop, worry loop phrases |
| `DISSOCIATION_MARKERS` | `frozenset` | Dissociation / depersonalization phrases |
| `EMOTIONAL_STOPWORDS_PRESERVE` | `frozenset` | Master preserve set for stopword filter |

### Importing keyword banks directly

```python
from preprocessing import (
    CRISIS_KEYWORDS,
    CRISIS_PATTERNS,
    DEPRESSION_KEYWORDS,
    NEGATION_WORDS,
    FIRST_PERSON,
    RUMINATION_MARKERS,
    DISSOCIATION_MARKERS,
)
```

---

## 9. Full Pipeline Mode (Models)

Only needed when you want `tokens` and `embeddings` from the pipeline.

```python
# Load all 3 models eagerly at startup
pipeline = PreprocessingPipeline(load_models=True)

# Or load lazily on first call (default)
pipeline = PreprocessingPipeline()

# Full inference
result = pipeline.process("I feel completely hopeless and alone")

# Tokens per model — ready for model.forward()
emotion_tokens = result.tokens["emotion"]
# {"input_ids": tensor([[101, ...]]), "attention_mask": tensor([[1, ...]])}

# Embeddings per model — 768-dim CLS token vectors
emotion_embedding = result.embeddings["emotion"]
# tensor([ 0.123, -0.456, ... ])  shape: torch.Size([768])

# Run only one model to save memory
result = pipeline.process_for_model("I feel hopeless", model_key="depression")
```

### Models used

| Key | HuggingFace model | Architecture |
|---|---|---|
| `emotion` | `mental/mental-roberta-base` | RoBERTa (BPE tokenizer) |
| `stress` | `jnyx74/stress-prediction` | DistilBERT (WordPiece) |
| `depression` | `poudel/Depression_and_Non-Depression_Classifier` | BERT-base-uncased (WordPiece) |

All embeddings are **768-dimensional** CLS token vectors from the last hidden layer.  
Max sequence length: **128 tokens** (configurable via `MAX_SEQ_LENGTH` in `config.py`).

---

## 10. Individual Component Usage

You can use any component standalone without the full pipeline.

### TextCleaner only

```python
from preprocessing import TextCleaner

cleaner = TextCleaner()

# Full 15-step clean
cleaned = cleaner.clean("I can't do this anymore!!! tbh kms lol")
# → "i cannot do this anymore [HIGH_INTENSITY] to be honest [DARK_HUMOR_POSSIBLE] ..."

# Individual steps
text = cleaner.strip_emojis("hello 😭 world")         # → "hello   world"
text = cleaner.apply_punct_tags("WHY?? I just...")    # → " [HIGH_AROUSAL] WHY [CONFUSION]  I just [HESITATION] "
text = cleaner.expand_contractions("I can't do this") # → "I cannot do this"
text = cleaner.normalize_slang("ngl im so done rn")   # → "not going to lie i am so done right now"
```

### CrisisDetector only

```python
from preprocessing import CrisisDetector

detector = CrisisDetector()

result = detector.detect("i want to end it all i have a plan")

print(result.severity)                    # "CRITICAL"
print(result.is_crisis)                   # True
print(result.needs_review)                # False
print(result.matched_pattern_categories)  # ["PLANNING", "DIRECT_IDEATION"]
print(result.matched_crisis_keywords)     # ["want to end it", "end it all", "i have a plan"]
print(result.psycholinguistic)
# {"i_ratio": 0.2, "absolute_ratio": 0.0, "negation_count": 0, ...}

# Quick boolean check
if detector.is_crisis("I want to kill myself"):
    print("ESCALATE")
```

### EmotionalStopwordFilter only

```python
from preprocessing import EmotionalStopwordFilter

f = EmotionalStopwordFilter()

# Filter a list of tokens
tokens = ["i", "do", "not", "feel", "happy", "at", "all"]
print(f.filter(tokens))
# → ["i", "do", "not", "feel", "happy", "all"]   ("at" removed, rest preserved)

# Filter a string directly
print(f.filter_text("i cannot stop crying i feel so hopeless everything falling apart"))
# → "i cannot stop crying i feel so hopeless everything falling apart"
#   (all words are emotionally significant — nothing removed)

# Inspect what's preserved vs. what's a stopword
print("not" in f.preserved_words)    # True
print("the" in f.stopwords)          # True
print("feel" in f.preserved_words)   # True
print("at" in f.stopwords)           # True
```

### ModelTokenizer only

```python
from preprocessing.tokenizer import ModelTokenizer

tokenizer = ModelTokenizer("mental/mental-roberta-base")
result = tokenizer.tokenize("I feel really sad today")
# result = {"input_ids": tensor([[0, 100, ...]]), "attention_mask": tensor([[1, 1, ...]])}

# Token count
ids = result["input_ids"]
print(f"Token count: {ids.shape[1]}")
```

### ModelEmbedder only

```python
from preprocessing.embedder import ModelEmbedder

embedder = ModelEmbedder("mental/mental-roberta-base")
embedding = embedder.generate_embedding("I feel really sad today")
# embedding.shape → torch.Size([768])
```

---

## 11. Interactive Runner

The `run_pipeline.py` script in `python/` lets you interactively test the pipeline from the terminal.

### Run it

```bash
cd python/

# Text-only mode (no ML models needed — instant)
python run_pipeline.py

# Full mode (downloads + loads all 3 models on first run)
python run_pipeline.py --full
```

### What it shows for every input

```
────────────────────────────────────────────────────────────
📥  ORIGINAL TEXT
────────────────────────────────────────────────────────────
I can't do this anymore tbh im so done with everything

────────────────────────────────────────────────────────────
🧹  STEP 1 — Cleaned Text  (TextCleaner)
────────────────────────────────────────────────────────────
i cannot do this anymore to be honest i am so done with everything

────────────────────────────────────────────────────────────
🔍  STEP 2 — Crisis Detection  (CrisisDetector)
────────────────────────────────────────────────────────────
  Status           : [SAFE | NEEDS REVIEW]
  Severity         : ELEVATED
  Has Intensifiers : True
  Has Rumination   : False
  Has Dissociation : False
  Pattern Cats     : HOPELESSNESS
  Psycholinguistic :
    i_ratio          = 0.1538   (Pennebaker depression marker)
    absolute_ratio   = 0.0769   (black-and-white thinking)
    negation_count   = 1
    intensifier_count= 1
  Note             : Hopelessness or isolation signals detected — monitor and assess context.

────────────────────────────────────────────────────────────
🗑️   STEP 3 — Filtered Text  (EmotionalStopwordFilter)
────────────────────────────────────────────────────────────
i cannot anymore honest i so done everything

────────────────────────────────────────────────────────────
✅  FINAL PREPROCESSED TEXT  (ready for ML models)
────────────────────────────────────────────────────────────
i cannot anymore honest i so done everything
────────────────────────────────────────────────────────────
```

Type `quit` or `exit` to stop. Press `Ctrl+C` to force quit.

---

## 12. File Reference

```
python/
├── run_pipeline.py              ← Interactive terminal runner
└── preprocessing/
    ├── __init__.py              ← Public exports for the whole module
    ├── config.py                ← Regex patterns, model names, SLANG_MAP, PUNCT_TAGS, CONTRACTIONS_MAP
    ├── keywords.py              ← All keyword/phrase banks + CRISIS_PATTERNS regex dict
    ├── cleaner.py               ← TextCleaner (15-step pipeline)
    ├── crisis_detector.py       ← CrisisDetector + CrisisResult dataclass
    ├── stopwords.py             ← EmotionalStopwordFilter
    ├── tokenizer.py             ← ModelTokenizer (HuggingFace AutoTokenizer wrapper)
    ├── embedder.py              ← ModelEmbedder (768-dim CLS token embeddings)
    └── pipeline.py              ← PreprocessingPipeline + PreprocessingResult (orchestrator)
```

### What `__init__.py` exports

```python
# Pipeline (main entry point)
from preprocessing import PreprocessingPipeline, PreprocessingResult

# Components
from preprocessing import TextCleaner
from preprocessing import CrisisDetector, CrisisResult
from preprocessing import EmotionalStopwordFilter
from preprocessing import ModelEmbedder, ModelTokenizer

# Config
from preprocessing import ALL_MODEL_NAMES, MAX_SEQ_LENGTH, SLANG_MAP, PUNCT_TAGS

# Keyword banks
from preprocessing import (
    CRISIS_KEYWORDS, CRISIS_PATTERNS,
    DEPRESSION_KEYWORDS, ANXIETY_KEYWORDS, STRESS_KEYWORDS,
    NEGATION_WORDS, EMOTION_INTENSIFIERS,
    FIRST_PERSON, ABSOLUTE_LANGUAGE,
    RUMINATION_MARKERS, DISSOCIATION_MARKERS,
)
```

---

## 13. Extending the Pipeline

### Add new crisis phrases

```python
# keywords.py — CRISIS_KEYWORDS frozenset
# Just add to the set. Phrase-bank matching is substring-based, longest first.
"my new crisis phrase",
```

### Add new slang terms

```python
# config.py — SLANG_MAP dict
"new slang": "its full form meaning",

# For dark humor terms that need flagging but not auto-crisis:
"sus term": "[DARK_HUMOR_POSSIBLE] what this term signals",
```

### Add new regex crisis patterns

```python
# keywords.py — CRISIS_PATTERNS dict
# Add to the appropriate category list:
"DIRECT_IDEATION": [
    ...existing patterns...,
    re.compile(r'\byour\s+new\s+pattern\b', re.I),
],
```

### Add a new crisis category

```python
# keywords.py — CRISIS_PATTERNS dict
"NEW_CATEGORY": [
    re.compile(r'\bpattern1\b', re.I),
    re.compile(r'\bpattern2\b', re.I),
],

# crisis_detector.py — _determine_severity()
# Add your new category to the severity routing logic:
if "NEW_CATEGORY" in pattern_categories:
    return "HIGH"   # or whatever severity it warrants
```

### Add words that must never be removed as stopwords

```python
# keywords.py — EMOTIONAL_STOPWORDS_PRESERVE
# Add individual words to the inner set at the bottom of the file
"newword",

# Or add a whole new frozenset and union it in:
MY_NEW_PRESERVE_SET = frozenset(["word1", "word2"])

EMOTIONAL_STOPWORDS_PRESERVE = frozenset(
    NEGATION_WORDS
    | EMOTION_INTENSIFIERS
    | FIRST_PERSON
    | ABSOLUTE_LANGUAGE
    | MY_NEW_PRESERVE_SET   # ← add here
    | { ...existing words... }
)
```

### Use a custom contraction map

```python
from preprocessing import TextCleaner

cleaner = TextCleaner(custom_contractions={
    "ur": "your",
    "u": "you",
    "y'all": "you all",
})
```

### Inject custom crisis keywords at runtime

```python
from preprocessing import CrisisDetector

detector = CrisisDetector(
    custom_crisis_keywords={"my custom phrase", "another phrase"},
)
```

---

## 14. Design Decisions

### Why not use spaCy or NLTK?

The pipeline is intentionally **dependency-minimal** for the text-only path. spaCy/NLTK add significant install overhead and are overkill for what is effectively a deterministic rule-based pipeline. The full pipeline already pulls in `transformers` + `torch` — adding more heavy NLP libraries would make deployment harder without clinical benefit.

### Why are emojis stripped instead of converted to text?

Emoji-to-text conversion (e.g., `😭` → `crying face`) introduces model vocabulary noise — the resulting text tokens (`crying`, `face`) could skew keyword detection and sentiment scores in unpredictable ways. The clinical signal of an emoji (sadness, distress) is already fully captured by the surrounding text in most real inputs. A clean strip is more reliable than a noisy conversion.

### Why does `normalize_unicode` run before `apply_punct_tags`?

Smart quotes (`'` U+2019) from mobile keyboards and pasted text must be converted to straight apostrophes (`'`) before contraction expansion, which happens at Step 11. But punct tags must also fire before lowercase to catch ALL_CAPS. The solution: unicode normalization first (Step 2), then punct tags (Step 3), then lowercase (Step 4).

### Why does `DARK_HUMOR_FLAG` not auto-trigger crisis?

`kms` and `im dead` are used constantly in non-distressed social media contexts. Auto-triggering a crisis response to casual slang would feel jarring, break user trust, and reduce engagement with the chatbot. Instead, `DARK_HUMOR_FLAG` sets `needs_review=True` and severity `ELEVATED` — which tells the LLM to gently probe for context before deciding whether to escalate.

### Why are minimizing intensifiers (`just`, `only`, `kinda`) preserved?

Minimization is a clinically recognized behavior in depression and anxiety — patients frequently downplay their symptoms: *"it's just a bit sad"*, *"i'm only a little worried"*. These words are as clinically significant as amplifying intensifiers. Stripping them would erase that signal.

### Why is `i_ratio` a depression marker?

Pennebaker (2003) and subsequent research established that depressed individuals use significantly more first-person singular pronouns compared to non-depressed individuals. A high `i/me/my` ratio across the conversation window is a feature that nudges the depression classifier score upward in the risk aggregator — not a diagnosis in itself.

---

*ASTRAVA Preprocessing Pipeline — built for the AI-Powered Mental Health Chatbot*  
*Component version: v2.0 | Python 3.11+ | March 2026*
