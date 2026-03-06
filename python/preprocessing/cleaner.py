"""
cleaner.py — Text cleaning pipeline for the ASTRAVA preprocessing module.

Provides a sequential, configurable text cleaning pipeline that normalizes
raw user input into a clean format suitable for downstream NLP models.
All three models (Emotion, Stress, Depression) share the same cleaned text.

Pipeline order:
    1.  Strip emojis                 (new — dedicated step, no emoji leakage)
    2.  Normalize unicode            (NFKC — must be before punct tags & contractions)
    3.  Apply punctuation tags       (HIGH_AROUSAL before lowercase, HESITATION, etc.)
    4.  Lowercase
    5.  Remove HTML tags
    6.  Remove URLs
    7.  Remove emails
    8.  Remove mentions
    9.  Remove hashtags (keep word)
    10. Remove phone numbers
    11. Expand contractions
    12. Normalize slang              (new — SLANG_MAP applied after contractions)
    13. Remove special characters
    14. Collapse repeated characters
    15. Normalize whitespace
"""

import re
import unicodedata
from typing import Optional

from .config import (
    CONTRACTIONS_MAP,
    EMAIL_PATTERN,
    EMOJI_PATTERN,
    HASHTAG_PATTERN,
    HTML_TAG_PATTERN,
    MENTION_PATTERN,
    MULTI_WHITESPACE_PATTERN,
    PHONE_PATTERN,
    PUNCT_TAGS,
    REPEATED_CHAR_PATTERN,
    SLANG_MAP,
    SPECIAL_CHAR_PATTERN,
    UNICODE_NOISE_PATTERN,
    URL_PATTERN,
)


class TextCleaner:
    """
    Sequential text cleaning pipeline for mental health text.

    Each step is available individually and also runs in sequence via `clean()`.
    The pipeline:
      - Strips ALL emojis before any other processing (no emoji leakage)
      - Tags high-arousal ALL_CAPS before lowercasing
      - Tags hesitation, intensity, confusion punctuation patterns
      - Preserves emotionally significant punctuation (? !)
      - Ensures negation words survive contraction expansion (e.g. "can't" → "cannot")
      - Normalizes slang terms to their full-form equivalents

    Usage:
        cleaner = TextCleaner()
        cleaned = cleaner.clean("I can't do this anymore!!! Visit https://example.com")
        # → "i cannot do this anymore [HIGH_INTENSITY]"
    """

    def __init__(self, custom_contractions: Optional[dict] = None):
        """
        Initialize the TextCleaner.

        Args:
            custom_contractions: Optional dict of additional contraction mappings
                                 to merge with the default CONTRACTIONS_MAP.
        """
        self.contractions_map = dict(CONTRACTIONS_MAP)
        if custom_contractions:
            self.contractions_map.update(custom_contractions)

        # Build a compiled regex for contraction matching (whole-word, case-insensitive)
        escaped_keys = [re.escape(k) for k in sorted(self.contractions_map.keys(), key=len, reverse=True)]
        self._contraction_pattern = re.compile(
            r"\b(" + "|".join(escaped_keys) + r")\b",
            re.IGNORECASE,
        )

        # Build slang regex: match longest phrases first (sorted by length desc)
        # Uses word-boundary anchors; phrases with spaces handled via \s+
        sorted_slang = sorted(SLANG_MAP.keys(), key=len, reverse=True)
        escaped_slang = [re.escape(k) for k in sorted_slang]
        self._slang_pattern = re.compile(
            r"(?<!\w)(" + "|".join(escaped_slang) + r")(?!\w)",
            re.IGNORECASE,
        )
        self._slang_map_lower = {k.lower(): v for k, v in SLANG_MAP.items()}

    # -------------------------------------------------------------------------
    # Individual cleaning steps
    # -------------------------------------------------------------------------

    @staticmethod
    def strip_emojis(text: str) -> str:
        """
        Remove ALL emojis and pictographic Unicode characters from the text.

        This is the very first step — emojis must be stripped before any other
        processing so they cannot leak into cleaned output or keyword detection.
        """
        return EMOJI_PATTERN.sub(" ", text)

    @staticmethod
    def apply_punct_tags(text: str) -> str:
        """
        Apply punctuation / pattern tags that carry emotional meaning.

        Must run BEFORE lowercasing so ALL_CAPS detection is accurate.
        Tags inserted:
          [HIGH_AROUSAL]        — 3+ consecutive ALL_CAPS word
          [HIGH_INTENSITY]      — 2+ consecutive exclamation marks
          [CONFUSION]           — 2+ consecutive question marks
          [MIXED_INTENSITY]     — mixed ?! sequences
          [HESITATION]          — ellipsis (... or unicode)
          [ELONGATION]          — sooooo / whyyyyy
          [INTERRUPTION]        — lone dash (— or -)
          [NEGATED_POSITIVITY]  — (not really)
          [POSSIBLE_DEFLECTION] — (just kidding) / (jk)
        """
        for pattern_str, replacement in PUNCT_TAGS:
            text = re.sub(pattern_str, replacement, text)
        return text

    @staticmethod
    def to_lowercase(text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    @staticmethod
    def remove_urls(text: str) -> str:
        """Strip all URLs (http, https, www)."""
        return URL_PATTERN.sub(" ", text)

    @staticmethod
    def remove_emails(text: str) -> str:
        """Strip email addresses."""
        return EMAIL_PATTERN.sub(" ", text)

    @staticmethod
    def remove_mentions(text: str) -> str:
        """Strip @mentions."""
        return MENTION_PATTERN.sub(" ", text)

    @staticmethod
    def remove_hashtags(text: str) -> str:
        """Strip hash symbol from hashtags but keep the word. #anxiety → anxiety."""
        return HASHTAG_PATTERN.sub(r"\1", text)

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Strip HTML tags."""
        return HTML_TAG_PATTERN.sub(" ", text)

    @staticmethod
    def remove_phone_numbers(text: str) -> str:
        """Strip phone numbers from text body."""
        return PHONE_PATTERN.sub(" ", text)

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters and remove invisible/control characters."""
        # Normalize to NFKD form (decompose), then recompose to NFC
        text = unicodedata.normalize("NFKC", text)
        # Remove zero-width chars, BOM, soft hyphens, etc.
        text = UNICODE_NOISE_PATTERN.sub("", text)
        return text

    @staticmethod
    def remove_special_characters(text: str) -> str:
        """Remove special characters, keeping letters, digits, spaces, ?, !, ' and periods."""
        return SPECIAL_CHAR_PATTERN.sub(" ", text)

    @staticmethod
    def collapse_repeated_characters(text: str) -> str:
        """Collapse 3+ repeated characters to 2. 'sooooo' → 'soo', '!!!' → '!!'."""
        return REPEATED_CHAR_PATTERN.sub(r"\1\1", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple whitespace to single space and strip edges."""
        return MULTI_WHITESPACE_PATTERN.sub(" ", text).strip()

    def expand_contractions(self, text: str) -> str:
        """
        Expand contractions and informal abbreviations.

        This is critical for mental health text since it converts:
            "I can't" → "I cannot"
            "don't" → "do not"
            "won't" → "will not"
        preserving the negation that is essential for sentiment analysis.
        """
        def _replace_match(match):
            word = match.group(0)
            # Try exact lowercase match first
            expanded = self.contractions_map.get(word.lower())
            if expanded:
                # Preserve original capitalization of first letter
                if word[0].isupper():
                    return expanded.capitalize()
                return expanded
            return word

        return self._contraction_pattern.sub(_replace_match, text)

    def normalize_slang(self, text: str) -> str:
        """
        Normalize slang terms to their full-form equivalents.

        Applied AFTER contraction expansion.
        Longest phrases are matched first to prevent partial overlaps.
        Dark humor terms are flagged with [DARK_HUMOR_POSSIBLE] rather than
        removed, so downstream context analysis can assess them.

        Examples:
            "tbh i cant even" → "to be honest i cannot cope"
            "kms lol"         → "[DARK_HUMOR_POSSIBLE] i said kill myself joking or serious laughing or deflecting emotion"
        """
        def _replace_slang(match):
            return self._slang_map_lower.get(match.group(0).lower(), match.group(0))

        return self._slang_pattern.sub(_replace_slang, text)

    # -------------------------------------------------------------------------
    # Full cleaning pipeline
    # -------------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """
        Run the full text cleaning pipeline in the correct order.

        Pipeline order:
            1.  Strip emojis              (dedicated step — no emoji leakage)
            2.  Normalize unicode         (NFKC — curly apostrophes → straight before contractions)
            3.  Apply punctuation tags    (before lowercase — captures ALL_CAPS)
            4.  Lowercase
            5.  Remove HTML tags
            6.  Remove URLs
            7.  Remove emails
            8.  Remove mentions
            9.  Remove hashtags (keep word)
            10. Remove phone numbers
            11. Expand contractions
            12. Normalize slang
            13. Remove special characters
            14. Collapse repeated characters
            15. Normalize whitespace

        Args:
            text: Raw user input text.

        Returns:
            Cleaned text string, ready for tokenization.
        """
        if not text or not isinstance(text, str):
            return ""

        text = self.strip_emojis(text)            # Step 1  — emoji removal (must be first)
        text = self.normalize_unicode(text)        # Step 2  — NFKC: curly ' → straight ', etc.
        text = self.apply_punct_tags(text)         # Step 3  — tag patterns (needs original case)
        text = self.to_lowercase(text)             # Step 4
        text = self.remove_html_tags(text)         # Step 5
        text = self.remove_urls(text)              # Step 6
        text = self.remove_emails(text)            # Step 7
        text = self.remove_mentions(text)          # Step 8
        text = self.remove_hashtags(text)          # Step 9
        text = self.remove_phone_numbers(text)     # Step 10
        text = self.expand_contractions(text)      # Step 11
        text = self.normalize_slang(text)          # Step 12
        text = self.remove_special_characters(text)        # Step 13
        text = self.collapse_repeated_characters(text)     # Step 14
        text = self.normalize_whitespace(text)             # Step 15

        return text
