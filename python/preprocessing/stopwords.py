"""
stopwords.py — Emotionally-aware stopword filter for the ASTRAVA preprocessing module.

Unlike standard NLP stopword removal, this filter preserves words that carry
emotional significance — negations ("not", "never"), intensifiers ("extremely",
"very"), and emotion-bearing function words ("feel", "hurt", "hate").

Blind stopword removal would destroy critical signals:
    "I do not feel happy" → "feel happy"  (meaning inverted!)
    "I am extremely sad" → "sad"          (intensity lost!)
"""

from typing import List, Optional, Set

from .keywords import EMOTIONAL_STOPWORDS_PRESERVE


# Standard English stopwords (sourced from NLTK's default list)
_STANDARD_STOPWORDS = frozenset([
    "a", "about", "above", "after", "again", "against", "ain", "am", "an",
    "and", "any", "are", "aren", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "by", "d", "did",
    "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "ll", "m", "ma", "me",
    "mightn", "more", "most", "mustn", "my", "myself", "needn", "no",
    "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s",
    "same", "shan", "she", "should", "shouldn", "so", "some", "such", "t",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "ve", "very", "was", "wasn", "we", "were",
    "weren", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "won", "wouldn", "y", "you", "your", "yours",
    "yourself", "yourselves",
])


class EmotionalStopwordFilter:
    """
    Stopword filter that preserves emotionally significant words.

    Given a list of tokens, removes only the stopwords that do NOT carry
    emotional meaning. Words in the EMOTIONAL_STOPWORDS_PRESERVE set
    are always kept, even if they appear in the standard stopword list.

    Usage:
        filter = EmotionalStopwordFilter()
        tokens = ["i", "do", "not", "feel", "happy", "at", "all"]
        filtered = filter.filter(tokens)
        # → ["i", "do", "not", "feel", "happy", "all"]
        # "at" removed; "not", "feel", "all" preserved
    """

    def __init__(
        self,
        preserve_words: Optional[Set[str]] = None,
        additional_stopwords: Optional[Set[str]] = None,
    ):
        """
        Initialize the emotional stopword filter.

        Args:
            preserve_words: Custom set of words to always preserve.
                            Defaults to EMOTIONAL_STOPWORDS_PRESERVE.
            additional_stopwords: Extra words to add to the stopword list.
        """
        self._preserve = preserve_words or EMOTIONAL_STOPWORDS_PRESERVE

        # Build the effective stopword set: standard minus preserved
        base_stopwords = set(_STANDARD_STOPWORDS)
        if additional_stopwords:
            base_stopwords |= additional_stopwords

        # Remove emotionally significant words from the stopword set
        self._stopwords = base_stopwords - self._preserve

    @property
    def stopwords(self) -> frozenset:
        """Return the effective stopword set (after emotional preservation)."""
        return frozenset(self._stopwords)

    @property
    def preserved_words(self) -> frozenset:
        """Return the set of words being preserved."""
        return frozenset(self._preserve)

    def is_stopword(self, word: str) -> bool:
        """
        Check if a word is a removable stopword.

        Args:
            word: A single token (lowercase).

        Returns:
            True if the word should be REMOVED, False if it should be KEPT.
        """
        return word.lower() in self._stopwords

    def filter(self, tokens: List[str]) -> List[str]:
        """
        Filter stopwords from a list of tokens, preserving emotional words.

        Args:
            tokens: List of string tokens (typically from simple whitespace split
                    or tokenizer output).

        Returns:
            Filtered list with emotionally insignificant stopwords removed.
        """
        if not tokens:
            return []

        return [
            token for token in tokens
            if token.lower() not in self._stopwords
        ]

    def filter_text(self, text: str) -> str:
        """
        Convenience method: split text on whitespace, filter, rejoin.

        Args:
            text: A cleaned text string.

        Returns:
            Text with emotionally insignificant stopwords removed.
        """
        if not text:
            return ""

        tokens = text.split()
        filtered = self.filter(tokens)
        return " ".join(filtered)
