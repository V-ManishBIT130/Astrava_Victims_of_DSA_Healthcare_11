"""
crisis_detector.py — Crisis keyword detection for the ASTRAVA preprocessing module.

Scans cleaned user text against:
  1. Regex-based CRISIS_PATTERNS (category-based severity routing)
  2. Phrase-bank CRISIS_KEYWORDS  (substring matching, shared with downstream)
  3. Psycholinguistic feature extraction (i_ratio, absolute_ratio, etc.)

Severity routing:
    PLANNING          → CRITICAL  (ideation has moved to action — immediate escalation)
    DIRECT_IDEATION,
    SELF_HARM         → HIGH      (confirmed crisis — escalate before model inference)
    HOPELESSNESS,
    ISOLATION         → ELEVATED  (monitor — conversational context needed)
    DARK_HUMOR_FLAG   → ELEVATED  (needs_review=True — do NOT auto-trigger crisis)
    (nothing matched) → NONE / LOW / MEDIUM (from phrase-bank keyword counts)

The detector runs on CLEANED text (after TextCleaner pipeline).
Emojis are fully stripped before this point — no emoji patterns here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .keywords import (
    ABSOLUTE_LANGUAGE,
    ANXIETY_KEYWORDS,
    CRISIS_KEYWORDS,
    CRISIS_PATTERNS,
    DEPRESSION_KEYWORDS,
    DISSOCIATION_MARKERS,
    EMOTION_INTENSIFIERS,
    FIRST_PERSON,
    NEGATION_WORDS,
    RUMINATION_MARKERS,
    STRESS_KEYWORDS,
)


@dataclass
class CrisisResult:
    """Result of a crisis detection scan."""

    is_crisis: bool = False
    needs_review: bool = False                    # ELEVATED — needs conversational context
    severity: str = "NONE"                        # NONE, LOW, MEDIUM, HIGH, ELEVATED, CRITICAL
    matched_crisis_keywords: List[str] = field(default_factory=list)
    matched_depression_keywords: List[str] = field(default_factory=list)
    matched_anxiety_keywords: List[str] = field(default_factory=list)
    matched_stress_keywords: List[str] = field(default_factory=list)
    matched_pattern_categories: List[str] = field(default_factory=list)   # from CRISIS_PATTERNS
    has_intensifiers: bool = False
    has_rumination: bool = False
    has_dissociation: bool = False
    psycholinguistic: Dict[str, float] = field(default_factory=dict)
    confidence_note: str = ""

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "is_crisis": self.is_crisis,
            "needs_review": self.needs_review,
            "severity": self.severity,
            "matched_crisis_keywords": self.matched_crisis_keywords,
            "matched_depression_keywords": self.matched_depression_keywords,
            "matched_anxiety_keywords": self.matched_anxiety_keywords,
            "matched_stress_keywords": self.matched_stress_keywords,
            "matched_pattern_categories": self.matched_pattern_categories,
            "has_intensifiers": self.has_intensifiers,
            "has_rumination": self.has_rumination,
            "has_dissociation": self.has_dissociation,
            "psycholinguistic": self.psycholinguistic,
            "confidence_note": self.confidence_note,
        }


class CrisisDetector:
    """
    Scans cleaned text for crisis indicators and mental health keyword matches.

    Two detection layers run in parallel:
      Layer 1 — CRISIS_PATTERNS (regex, category-based severity routing)
      Layer 2 — phrase-bank keyword matching (CRISIS_KEYWORDS + domain sets)

    Plus psycholinguistic feature extraction:
      i_ratio, absolute_ratio, negation_count, intensifier_count,
      has_rumination, has_dissociation.

    Emojis are fully stripped upstream — no emoji handling here.

    Usage:
        detector = CrisisDetector()
        result = detector.detect("i want to end it all nothing matters")
        if result.is_crisis:
            print(f"CRISIS: {result.severity} — {result.matched_pattern_categories}")
    """

    def __init__(
        self,
        custom_crisis_keywords: Optional[set] = None,
        custom_critical_phrases: Optional[set] = None,
    ):
        self._crisis_keywords = set(CRISIS_KEYWORDS)
        self._depression_keywords = set(DEPRESSION_KEYWORDS)
        self._anxiety_keywords = set(ANXIETY_KEYWORDS)
        self._stress_keywords = set(STRESS_KEYWORDS)
        self._intensifiers = set(EMOTION_INTENSIFIERS)

        if custom_crisis_keywords:
            self._crisis_keywords |= custom_crisis_keywords

    # ── Layer 1: Regex category matching ────────────────────────────────────

    def _run_pattern_categories(self, text: str) -> List[str]:
        """
        Run CRISIS_PATTERNS against text and return list of matched category names.
        """
        matched = []
        for category, patterns in CRISIS_PATTERNS.items():
            if any(p.search(text) for p in patterns):
                matched.append(category)
        return matched

    # ── Layer 2: Phrase-bank matching ───────────────────────────────────────

    def _find_matches(self, text: str, keyword_set: set) -> List[str]:
        """Find all keyword/phrase matches (longest first to avoid partial overlaps)."""
        text_lower = text.lower()
        matches = []
        for keyword in sorted(keyword_set, key=len, reverse=True):
            if keyword in text_lower:
                matches.append(keyword)
        return matches

    def _check_intensifiers(self, text: str) -> bool:
        """Check if the text contains emotion intensifiers."""
        words = set(text.lower().split())
        return bool(words & self._intensifiers)

    # ── Psycholinguistic features ────────────────────────────────────────────

    def _extract_psycholinguistic(self, text: str) -> dict:
        """
        Extract psycholinguistic features from cleaned text.

        Features:
          i_ratio         — first-person singular pronoun density (Pennebaker depression marker)
          absolute_ratio  — absolute language density (CBT black-and-white thinking)
          negation_count  — raw count of negation tokens
          intensifier_count — raw count of intensifier tokens
          has_rumination  — presence of rumination language patterns
          has_dissociation — presence of dissociation language patterns
        """
        tokens = text.lower().split()
        total = max(len(tokens), 1)
        token_set = set(tokens)

        i_count = sum(1 for t in tokens if t in FIRST_PERSON)
        abs_count = sum(1 for t in tokens if t in ABSOLUTE_LANGUAGE)
        neg_count = sum(1 for t in tokens if t in {w.lower() for w in NEGATION_WORDS})
        int_count = sum(1 for t in tokens if t in self._intensifiers)

        text_lower = text.lower()
        has_rumination = any(m in text_lower for m in RUMINATION_MARKERS)
        has_dissociation = any(m in text_lower for m in DISSOCIATION_MARKERS)

        return {
            "i_ratio":           round(i_count / total, 4),
            "absolute_ratio":    round(abs_count / total, 4),
            "negation_count":    neg_count,
            "intensifier_count": int_count,
            "has_rumination":    has_rumination,
            "has_dissociation":  has_dissociation,
        }

    # ── Severity determination ───────────────────────────────────────────────

    def _determine_severity(
        self,
        pattern_categories: List[str],
        crisis_matches: List[str],
        depression_matches: List[str],
        anxiety_matches: List[str],
        stress_matches: List[str],
        has_intensifiers: bool,
    ) -> str:
        """
        Determine severity level.

        Priority order (highest wins):
          CRITICAL  — PLANNING patterns matched
          HIGH      — DIRECT_IDEATION or SELF_HARM patterns, or crisis keyword phrases
          ELEVATED  — HOPELESSNESS, ISOLATION, or DARK_HUMOR_FLAG patterns
          MEDIUM    — 3+ distress signals, or 1+ with intensifiers
          LOW       — 1–2 distress signals
          NONE      — nothing detected
        """
        if "PLANNING" in pattern_categories:
            return "CRITICAL"

        if "DIRECT_IDEATION" in pattern_categories or "SELF_HARM" in pattern_categories:
            return "HIGH"

        if crisis_matches:
            return "HIGH"

        elevated_cats = {"HOPELESSNESS", "ISOLATION", "DARK_HUMOR_FLAG"}
        if elevated_cats & set(pattern_categories):
            return "ELEVATED"

        total_distress = len(depression_matches) + len(anxiety_matches) + len(stress_matches)
        if total_distress >= 3 or (total_distress >= 1 and has_intensifiers):
            return "MEDIUM"

        if total_distress >= 1:
            return "LOW"

        return "NONE"

    # ── Public API ───────────────────────────────────────────────────────────

    def detect(self, text: str) -> CrisisResult:
        """
        Perform full crisis and mental health keyword detection.

        Args:
            text: Cleaned text (after TextCleaner pipeline — no emojis).

        Returns:
            CrisisResult with all detection fields populated.
        """
        if not text or not isinstance(text, str):
            return CrisisResult()

        # Layer 1: regex pattern categories
        pattern_categories = self._run_pattern_categories(text)

        # Layer 2: phrase-bank matching
        crisis_matches = self._find_matches(text, self._crisis_keywords)
        depression_matches = self._find_matches(text, self._depression_keywords)
        anxiety_matches = self._find_matches(text, self._anxiety_keywords)
        stress_matches = self._find_matches(text, self._stress_keywords)

        has_intensifiers = self._check_intensifiers(text)

        # Psycholinguistic features
        psych = self._extract_psycholinguistic(text)
        has_rumination = psych["has_rumination"]
        has_dissociation = psych["has_dissociation"]

        # Severity
        severity = self._determine_severity(
            pattern_categories,
            crisis_matches,
            depression_matches,
            anxiety_matches,
            stress_matches,
            has_intensifiers,
        )

        is_crisis = severity in ("HIGH", "CRITICAL")
        needs_review = severity == "ELEVATED" or "DARK_HUMOR_FLAG" in pattern_categories

        # Confidence note
        if severity == "CRITICAL":
            confidence_note = "IMMEDIATE ESCALATION REQUIRED — explicit lethal intent or concrete plan detected."
        elif severity == "HIGH":
            confidence_note = "Crisis language detected — escalate before model inference."
        elif severity == "ELEVATED":
            if "DARK_HUMOR_FLAG" in pattern_categories:
                confidence_note = "Dark humor / deflection detected — probe gently for context before escalating."
            else:
                confidence_note = "Hopelessness or isolation signals detected — monitor and assess context."
        elif severity == "MEDIUM":
            confidence_note = "Elevated emotional distress signals — monitor closely."
        elif severity == "LOW":
            confidence_note = "Some distress indicators present — continue with model inference."
        else:
            confidence_note = ""

        return CrisisResult(
            is_crisis=is_crisis,
            needs_review=needs_review,
            severity=severity,
            matched_crisis_keywords=crisis_matches,
            matched_depression_keywords=depression_matches,
            matched_anxiety_keywords=anxiety_matches,
            matched_stress_keywords=stress_matches,
            matched_pattern_categories=pattern_categories,
            has_intensifiers=has_intensifiers,
            has_rumination=has_rumination,
            has_dissociation=has_dissociation,
            psycholinguistic=psych,
            confidence_note=confidence_note,
        )

    def is_crisis(self, text: str) -> bool:
        """Quick boolean check — crisis detected?"""
        return self.detect(text).is_crisis
