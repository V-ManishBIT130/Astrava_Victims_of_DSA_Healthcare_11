# ASTRAVA — Risk Scoring and Crisis Escalation

---

## The Purpose of Risk Scoring

ASTRAVA's three ML models each detect one signal in isolation: one measures depression indicators, one measures stress, one measures emotional state. None of them alone is sufficient to determine how the system should respond.

A person can score high on stress but be emotionally positive. A person can score high on sadness without being clinically depressed. The risk scoring layer exists to combine all of these signals into a single, actionable decision: how serious is this person's current state, and what kind of response do they need right now?

The output of risk scoring is one of three tiers: LOW, MEDIUM, or HIGH. This tier drives every downstream decision — what RAG content is retrieved, how the LLM prompt is structured, whether the crisis overlay appears on the frontend.

---

## Layer 1: Pre-ML Crisis Detection (Always Runs First)

Before any risk scoring, the CrisisDetector in the Python preprocessing pipeline runs on the cleaned text. If it finds HIGH or CRITICAL signals, the system short-circuits and never reaches the ML models.

This is not part of the risk scoring formula below — it is a hard override that works independently of model confidence scores. The reasoning is that explicit language indicating suicidal ideation or imminent self-harm should trigger an escalation response regardless of what the ML models would predict.

Crisis keywords that cause a SHORT-CIRCUIT:
- Any match in the suicidal ideation category ("want to end it all", "want to die", "no reason to live", "kill myself", "better off dead", and similar)
- Any match in the self-harm category ("cut myself", "hurt myself", "self-harm", "overdose")
- CRITICAL severity farewell patterns combined with distress context

When the system short-circuits, the crisis result is sent back to the backend immediately, the crisis protocol is activated, and the total response time is under 5ms (no model inference required).

---

## Layer 2: ML-Based Risk Score Aggregation

If the pre-ML crisis check passes (severity is NONE, LOW, or MEDIUM/ELEVATED), the three models run and produce their outputs. These outputs are fed into the score aggregator, which calculates a composite score using weighted contributions.

**Composite Score Formula:**

The composite score is a number between 0.0 and 1.0 calculated as follows:

| Component | Weight | Source |
|---|---|---|
| Depression probability | 40% | `poudel/Depression_and_Non-Depression_Classifier` output |
| Stress probability | 25% | `jnyx74/stress-prediction` output |
| Negative emotion intensity | 20% | Average of top-3 negative emotion scores from GoEmotions |
| Absence of protective emotions | 15% bonus | Applied when joy + optimism + love all score below 0.1 |

**Why these weights:**
Depression carries the highest weight because it is the strongest clinical predictor of serious mental health risk among the three signals. Stress is real but less specific — someone can be highly stressed and functionally fine. Negative emotion intensity adds breadth coverage: even if the depression model is uncertain, a high concentration of sadness, grief, and fear is a meaningful signal. The absence of positive emotion is a well-documented risk amplifier in clinical research — people who report zero positive affect are at significantly higher risk than those who hold even a small degree of hope or joy.

**Risk Tier Thresholds:**

| Tier | Composite Score |
|---|---|
| LOW | 0.0 to 0.35 |
| MEDIUM | 0.36 to 0.65 |
| HIGH | 0.66 to 1.0 |

There is also a crisis keyword flag from the preprocessing stage. Even if the composite ML score falls below 0.66, a keyword flag can force elevation to HIGH. This reflects the principle that language-based red flags should not be overridden by model probabilities.

---

## What Each Risk Tier Means and Triggers

### LOW Risk

The user is either in a neutral state, mildly venting, or having a normal difficult day. There are no significant signals of clinical concern.

**Response protocol:**
- The LLM responds warmly and conversationally
- It acknowledges feelings without being clinical or alarming
- It asks open-ended questions to understand what's on the user's mind
- No RAG retrieval occurs
- No crisis UI elements are shown on the frontend
- Supermemory notes a low-risk turn

**Example inputs that produce LOW risk:**
"Today was rough at work but I think I'll be okay", "I'm a bit tired today", "feeling okay but kinda bored"

### MEDIUM Risk

The user is experiencing meaningful distress. At least one model score is moderately elevated, or negative emotions are clearly present, or the composite calculation is in the middle range. This warrants a more structured, supportive response with real tools.

**Response protocol:**
- RAG retrieval is activated — the top-3 relevant passages are retrieved from the CBT, breathing, and coping indices
- The LLM is instructed to explicitly reference the retrieved techniques by name
- The LLM mentions scientific backing where appropriate
- It checks on basics: sleep, eating, hydration, movement
- It asks whether the user has people they can talk to
- A resource card may be shown below the chat bubble in the frontend
- Supermemory notes the emotional state, risk level, and what techniques were recommended

**Example inputs that produce MEDIUM risk:**
"I've been feeling really down and overwhelmed lately, nothing seems worth it", "I'm so stressed all the time and I can't sleep anymore"

### HIGH Risk

The user is in serious distress. Either the composite ML score is above 0.66, or the preprocessing crisis detector flagged keywords even though it didn't short-circuit (ELEVATED/MEDIUM severity can still result in HIGH composite), or a direct crisis keyword short-circuit occurred.

**Response protocol:**
- Crisis escalation is triggered
- Backend emits a `crisis_alert` WebSocket event to the frontend
- Frontend renders the crisis overlay with helpline numbers (filtered by user's stored location)
- Frontend shows a prompt asking whether the user wants a human callback
- The LLM response is safety-focused: expressing care, gently assessing safety, providing helplines in the response text itself
- The LLM does NOT use RAG clinical content at this level — the priority is stabilization and connection to human help
- Supermemory logs the escalation event with full context for future sessions

**In the case of a pre-ML short-circuit (CRITICAL/HIGH from the crisis detector):**
The LLM does not even run before the crisis alert is sent. The user receives the crisis overlay response in under 5ms. The LLM then generates a supporting message in parallel.

---

## The Crisis Keyword Banks

The keywords are maintained in `python/preprocessing/keywords.py` and organized by category. The technical implementation uses word-boundary regex matching for single-word keywords to prevent false positives from word containment (for example, the word "void" should not match inside "avoid", and "numb" should not match inside "number").

Multi-word phrases are matched as plain substrings after cleaning (since contractions are expanded and slang normalized, phrases like "want to end it all" will reliably appear in the cleaned text).

The keyword categories that contribute to HIGH or CRITICAL severity:
- Suicidal ideation: explicit phrases about wanting to die or end one's life
- Self-harm: explicit phrases about harming oneself
- Hopelessness compound clusters: phrases combining hopelessness with self-negation or irreversibility
- Farewell combined with distress context

The following categories contribute only lower severity (ELEVATED/MEDIUM):
- General hopelessness language without explicit intent
- High negative psycholinguistic ratios without explicit crisis keywords
- References to being a burden, feeling trapped

---

## Session Risk State and Elevation Persistence

After a HIGH risk event, the system should not immediately return to normal processing for the next message. The architecture describes a concept called "elevation persistence": after a crisis event, the risk threshold is lowered for the next 3 turns. This means messages that would normally score LOW or MEDIUM might be classified one tier higher, keeping the system in a heightened monitoring state.

This has not been implemented yet in the Python layer. It would require the backend to pass a session risk state flag to the Python endpoint so the score aggregator can adjust thresholds accordingly.

---

## Calibration Notes

The current thresholds (0.35/0.65 boundary points) are not derived from validated clinical calibration — they are reasonable starting values for a hackathon prototype. In a production system, these thresholds would need to be validated against labeled data and adjusted to optimize the precision-recall tradeoff relevant to the use case (in mental health, recall — not missing real crises — is more important than precision).

The depression model weight (40%) is also deliberately high. This is a known bias in the design: when in doubt, err toward flagging depression. The cost of a false positive (an empathetic MEDIUM-risk response to someone who is fine) is much lower than the cost of a false negative (a LOW-risk response to someone in serious distress).
