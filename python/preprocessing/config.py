"""
config.py — Centralized configuration for the ASTRAVA preprocessing module.

Contains model identifiers, regex patterns, contraction mappings,
slang normalization maps, punctuation/pattern tags, and tokenizer/embedding
settings used across the preprocessing pipeline.
"""

import re

# =============================================================================
# MODEL IDENTIFIERS (HuggingFace Hub)
# =============================================================================

EMOTION_MODEL_NAME = "SamLowe/roberta-base-go_emotions"  # RoBERTa-base, 28-label multi-label
STRESS_MODEL_NAME = "jnyx74/stress-prediction"           # DistilBERT-based (Dreaddit dataset)
DEPRESSION_MODEL_NAME = "poudel/Depression_and_Non-Depression_Classifier"  # BERT-base-uncased

ALL_MODEL_NAMES = {
    "emotion": EMOTION_MODEL_NAME,
    "stress": STRESS_MODEL_NAME,
    "depression": DEPRESSION_MODEL_NAME,
}

# =============================================================================
# TOKENIZER / EMBEDDING SETTINGS
# =============================================================================

MAX_SEQ_LENGTH = 512          # Max tokens per input — all 3 models support 512
EMBEDDING_DIM = 768           # Hidden size for all three base models
BATCH_SIZE = 16               # Default inference batch size
RETURN_TENSORS = "pt"         # PyTorch tensors

# =============================================================================
# REGEX PATTERNS — compiled once, reused everywhere
# =============================================================================

# URLs (http, https, ftp, www)
URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|"
    r"(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.\S+",
    re.IGNORECASE,
)

# Email addresses
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Social media mentions (@username)
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")

# Hashtags (#topic) — keep the word, drop the hash
HASHTAG_PATTERN = re.compile(r"#(\w+)")

# HTML tags
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Phone numbers (international & local formats)
PHONE_PATTERN = re.compile(
    r"(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}"
)

# Special characters — keep letters, digits, spaces, ?, !, ', and periods
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s?!.\']")

# Emoji removal — strip all Unicode emoji / pictograph ranges
# Covers Emoticons, Misc Symbols, Dingbats, Supplemental Symbols,
# Transport/Map, Enclosed Alphanumeric Supplement, Mahjong, etc.
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # Emoticons
    "\U0001F300-\U0001F5FF"   # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"   # Transport and Map
    "\U0001F700-\U0001F77F"   # Alchemical Symbols
    "\U0001F780-\U0001F7FF"   # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"   # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"   # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"   # Chess Symbols
    "\U0001FA70-\U0001FAFF"   # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"   # Dingbats
    "\U000024C2-\U0001F251"   # Enclosed characters
    "\U0000200D"              # Zero-width joiner (used in compound emojis)
    "\U0000FE0F"              # Variation Selector-16 (emoji presentation)
    "\U000020E3"              # Combining Enclosing Keycap
    "]+",
    flags=re.UNICODE,
)

# Repeated characters (3+ of the same char → 2)
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")

# Multiple whitespace
MULTI_WHITESPACE_PATTERN = re.compile(r"\s+")

# Unicode control / non-printable characters
UNICODE_NOISE_PATTERN = re.compile(
    r"[\u200b\u200c\u200d\ufeff\u00ad\u2028\u2029\u0000-\u001f]"
)

# =============================================================================
# CONTRACTION EXPANSION MAP (150+ entries)
# =============================================================================

CONTRACTIONS_MAP = {
    # --- Standard contractions ---
    "i'm": "i am",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "he's": "he is",
    "he'll": "he will",
    "he'd": "he would",
    "she's": "she is",
    "she'll": "she will",
    "she'd": "she would",
    "it's": "it is",
    "it'll": "it will",
    "it'd": "it would",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'd": "they would",
    "that's": "that is",
    "that'll": "that will",
    "that'd": "that would",
    "who's": "who is",
    "who'll": "who will",
    "who'd": "who would",
    "what's": "what is",
    "what'll": "what will",
    "what'd": "what did",
    "where's": "where is",
    "where'll": "where will",
    "where'd": "where did",
    "when's": "when is",
    "when'll": "when will",
    "when'd": "when did",
    "why's": "why is",
    "why'll": "why will",
    "why'd": "why did",
    "how's": "how is",
    "how'll": "how will",
    "how'd": "how did",
    "there's": "there is",
    "there'll": "there will",
    "there'd": "there would",
    "here's": "here is",

    # --- Negative contractions ---
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "can't": "cannot",
    "couldn't": "could not",
    "mustn't": "must not",
    "mightn't": "might not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "daren't": "dare not",

    # --- Informal / slang contractions ---
    "ain't": "am not",
    "gonna": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "lemme": "let me",
    "gimme": "give me",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "do not know",
    "y'all": "you all",
    "ma'am": "madam",
    "o'clock": "of the clock",
    "c'mon": "come on",
    "e'er": "ever",
    "ne'er": "never",
    "'twas": "it was",
    "'tis": "it is",
    "let's": "let us",

    # --- Double contractions ---
    "wouldn't've": "would not have",
    "shouldn't've": "should not have",
    "couldn't've": "could not have",
    "mightn't've": "might not have",
    "mustn't've": "must not have",
    "would've": "would have",
    "could've": "could have",
    "should've": "should have",
    "might've": "might have",
    "must've": "must have",
    "who've": "who have",
    "it'd've": "it would have",
    "they'd've": "they would have",
    "we'd've": "we would have",

    # --- Emotionally significant contractions ---
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "doesn't": "does not",
    "didn't": "did not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "mustn't": "must not",
    "needn't": "need not",

    # --- Chat / texting abbreviations ---
    "idk": "i do not know",
    "imo": "in my opinion",
    "tbh": "to be honest",
    "ngl": "not going to lie",
    "smh": "shaking my head",
    "irl": "in real life",
    "fwiw": "for what it is worth",
    "brb": "be right back",
    "omg": "oh my god",
    "btw": "by the way",
    "rn": "right now",
    "bf": "boyfriend",
    "gf": "girlfriend",
    "ppl": "people",
    "bc": "because",
    "w/o": "without",
    "w/": "with",
    "b4": "before",
    "ur": "your",
    "u": "you",
    "r": "are",
    "2": "to",
    "4": "for",
    "n": "and",
    "thx": "thanks",
    "plz": "please",
    "pls": "please",
    "sry": "sorry",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing",
}

# =============================================================================
# SLANG NORMALIZATION MAP
# Organized by category. Applied AFTER contraction expansion, BEFORE stopword
# filtering. Longest phrases matched first (handled in cleaner.py).
# =============================================================================

SLANG_MAP = {

    # ── MENTAL STATE SLANG ──────────────────────────────────────────────────
    "idk":        "i do not know",
    "idek":       "i do not even know",
    "idgaf":      "i do not care at all",
    "idc":        "i do not care",
    "imo":        "in my opinion",
    "imho":       "in my honest opinion",
    "ngl":        "not going to lie",
    "ngl tho":    "not going to lie though",
    "tbh":        "to be honest",
    "tbf":        "to be fair",
    "tbqh":       "to be quite honest",
    "irl":        "in real life",
    "atm":        "at the moment",
    "afaik":      "as far as i know",
    "afaict":     "as far as i can tell",

    # ── FEELING / EMOTIONAL STATE SLANG ────────────────────────────────────
    "ugh":        "feeling frustrated",
    "meh":        "feeling indifferent",
    "bleh":       "feeling unwell",
    "blah":       "feeling dull and empty",
    "smh":        "shaking my head in disappointment",
    "fml":        "my life is terrible",
    "ffs":        "extremely frustrated",
    "wtf":        "extremely shocked and upset",
    "omg":        "extremely surprised",
    "omfg":       "extremely overwhelmed",
    "lmao":       "laughing but possibly distressed",
    "lmfao":      "laughing but possibly distressed",
    "lol":        "laughing or deflecting emotion",
    "lolol":      "deflecting with humor",
    "rofl":       "laughing or deflecting",
    "haha":       "laughing or deflecting emotion",
    "hahaha":     "laughing or deflecting emotion",
    "heh":        "uneasy laugh",
    "istg":       "i swear this is true",
    "istfg":      "i swear this is true",
    "frfr":       "this is very real to me",

    # ── EXISTENTIAL / HOPELESSNESS SLANG ───────────────────────────────────
    "idc anymore":      "i do not care about anything anymore",
    "cant even":        "i cannot cope",
    "done with it":     "i am done with this",
    "over it":          "i am exhausted and giving up",
    "cant deal":        "i cannot handle this",
    "cant handle":      "i cannot handle this",
    "at my limit":      "i have reached my breaking point",
    "at my wits end":   "i have reached my breaking point",
    "giving up":        "i am giving up",
    "whats the point":  "i see no purpose",
    "what's the point": "i see no purpose",
    "no point":         "there is no purpose",
    "nothing matters":  "i feel hopeless",
    "nobody cares":     "i feel completely alone",
    "no one cares":     "i feel completely alone",

    # ── ABBREVIATIONS (no apostrophe — post-contraction-expansion) ─────────
    "cant":     "cannot",
    "wont":     "will not",
    "dont":     "do not",
    "doesnt":   "does not",
    "didnt":    "did not",
    "isnt":     "is not",
    "arent":    "are not",
    "wasnt":    "was not",
    "werent":   "were not",
    "hasnt":    "has not",
    "havent":   "have not",
    "hadnt":    "had not",
    "wouldnt":  "would not",
    "couldnt":  "could not",
    "shouldnt": "should not",
    "mustnt":   "must not",
    "mightnt":  "might not",
    "neednt":   "need not",
    "ive":      "i have",
    "id":       "i would",
    "im":       "i am",
    "ill":      "i will",
    "youre":    "you are",
    "theyre":   "they are",
    "hes":      "he is",
    "shes":     "she is",
    "thats":    "that is",
    "theres":   "there is",
    "whos":     "who is",
    "whats":    "what is",
    "hows":     "how is",

    # ── TIME / SITUATION SLANG ──────────────────────────────────────────────
    "rn":       "right now",
    "tmr":      "tomorrow",
    "tmrw":     "tomorrow",
    "yday":     "yesterday",
    "tonite":   "tonight",
    "2day":     "today",
    "2moro":    "tomorrow",
    "2nite":    "tonight",
    "4ever":    "forever",
    "nvm":      "never mind",
    "nvr":      "never",

    # ── SOCIAL / RELATIONSHIP SLANG ────────────────────────────────────────
    "bff":      "best friend",
    "bestie":   "best friend",
    "fam":      "family",
    "bro":      "close friend or brother",
    "sis":      "close friend or sister",
    "peeps":    "people",
    "ppl":      "people",

    # ── INTENSIFIER SLANG ──────────────────────────────────────────────────
    "lowkey":       "somewhat",
    "highkey":      "very much",
    "deadass":      "completely serious",
    "fr":           "for real",
    "no cap tho":   "seriously though",
    "no cap":       "seriously",
    "cap":          "lie",
    "mid":          "mediocre and disappointing",
    "sus":          "suspicious",
    "sketchy":      "suspicious and unsafe feeling",
    "shook":        "shocked and disturbed",
    "triggered":    "emotionally distressed",
    "pressed":      "upset and anxious",
    "salty":        "bitter and resentful",
    "vibe":         "feeling or atmosphere",
    "vibing":       "feeling okay",
    "not vibing":   "not feeling okay",
    "off vibes":    "feeling wrong or upset",

    # ── DARK HUMOR / DEFLECTION SLANG (flagged but not removed) ───────────
    "kms":              "[DARK_HUMOR_POSSIBLE] i said kill myself joking or serious",
    "kys":              "[DARK_HUMOR_POSSIBLE] harmful phrase detected",
    "im dead":          "[DARK_HUMOR_POSSIBLE] extremely surprised or exhausted",
    "ded":              "[DARK_HUMOR_POSSIBLE] exhausted or using dark humor",
    "lmao im dead":     "[DARK_HUMOR_POSSIBLE] deflecting with dark humor",
    "send help":        "i need help",
    "help me":          "i need help",
    "save me":          "i am struggling and need support",
    "rip me":           "[DARK_HUMOR_POSSIBLE] feeling hopeless or joking",
    "i cant":           "i cannot cope",
    "i literally cant": "i am completely overwhelmed",

    # ── PHYSICAL SYMPTOM SLANG ─────────────────────────────────────────────
    "no sleep":         "i am not sleeping",
    "cant sleep":       "i cannot sleep",
    "havent slept":     "i have not slept",
    "running on empty": "i am exhausted and depleted",
    "dead tired":       "i am completely exhausted",
    "burnt out":        "i am completely burned out",
    "burned out":       "i am completely burned out",
    "brain fog":        "i cannot think clearly",
    "zoned out":        "i am dissociating",
    "spacing out":      "i am dissociating",
    "numb":             "i feel emotionally numb",
    "checked out":      "i am emotionally disengaged",
}

# =============================================================================
# PUNCTUATION & PATTERN TAGS
# Applied BEFORE lowercasing so ALL_CAPS detection works correctly.
# No emojis — emoji stripping happens as a dedicated step before these tags.
# Tuples of (regex_pattern_string, replacement_string).
# =============================================================================

PUNCT_TAGS = [

    # ── HESITATION / TRAILING OFF ──────────────────────────────────────────
    # "I don't know... it just feels like..."
    (r'\.{3,}',              ' [HESITATION] '),
    (r'\u2026',              ' [HESITATION] '),   # Unicode ellipsis char

    # ── EMOTIONAL INTENSITY (2+ exclamation marks) ─────────────────────────
    # "I can't do this!!!"
    (r'!{2,}',               ' [HIGH_INTENSITY] '),

    # ── CONFUSION / OVERWHELM (2+ question marks) ──────────────────────────
    # "why?? why does this happen??"
    (r'\?{2,}',              ' [CONFUSION] '),

    # ── MIXED EMOTION (question + exclamation interleaved) ─────────────────
    # "why is this happening?!"
    (r'[?!]{2,}',            ' [MIXED_INTENSITY] '),

    # ── ALL CAPS — shouting / extreme emotion ──────────────────────────────
    # "I JUST CANT TAKE THIS ANYMORE"
    # Must run BEFORE lowercasing.
    (r'\b[A-Z]{3,}\b',       ' [HIGH_AROUSAL] '),

    # ── REPEATED LETTERS — elongation = emotional emphasis ─────────────────
    # "I'm sooooo tired", "whyyyyy"
    (r'(.)\1{2,}',           r'\1\1 [ELONGATION] '),

    # ── LONE DASH / INTERRUPTION ───────────────────────────────────────────
    # "I just — I don't know"
    (r'\s[\u2014\u2013\-]{1,2}\s', ' [INTERRUPTION] '),

    # ── PARENTHETICAL SELF-CORRECTION ─────────────────────────────────────
    # "I'm fine (not really)"
    (r'\(not really\)',      ' [NEGATED_POSITIVITY] '),
    (r'\(just kidding\)',    ' [POSSIBLE_DEFLECTION] '),
    (r'\(jk\)',              ' [POSSIBLE_DEFLECTION] '),
]
