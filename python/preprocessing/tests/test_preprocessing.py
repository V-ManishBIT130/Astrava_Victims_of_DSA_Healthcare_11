"""
test_preprocessing.py — Unit tests for the ASTRAVA preprocessing module.

Tests all components:
  - TextCleaner: text cleaning pipeline
  - EmotionalStopwordFilter: emotionally-aware stopword removal
  - CrisisDetector: crisis keyword detection and severity classification
  - PreprocessingPipeline: text-only mode (no model loading needed)

Note: Tests for ModelTokenizer and ModelEmbedder require `transformers` and
      `torch` installed. They are marked with pytest.mark.slow and can be
      skipped with `pytest -m "not slow"`.
"""

import pytest
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from preprocessing.cleaner import TextCleaner
from preprocessing.crisis_detector import CrisisDetector
from preprocessing.stopwords import EmotionalStopwordFilter
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.keywords import (
    CRISIS_KEYWORDS,
    DEPRESSION_KEYWORDS,
    ANXIETY_KEYWORDS,
    STRESS_KEYWORDS,
    NEGATION_WORDS,
    EMOTION_INTENSIFIERS,
)
from preprocessing.config import (
    ALL_MODEL_NAMES,
    EMOTION_MODEL_NAME,
    STRESS_MODEL_NAME,
    DEPRESSION_MODEL_NAME,
)


# =============================================================================
# TextCleaner Tests
# =============================================================================

class TestTextCleaner:
    """Tests for the TextCleaner class."""

    def setup_method(self):
        self.cleaner = TextCleaner()

    # --- Lowercase ---
    def test_to_lowercase(self):
        assert self.cleaner.to_lowercase("HELLO World") == "hello world"
        assert self.cleaner.to_lowercase("I Feel SAD") == "i feel sad"

    # --- URL removal ---
    def test_remove_urls(self):
        text = "check this https://example.com and http://foo.bar"
        result = self.cleaner.remove_urls(text)
        assert "https://example.com" not in result
        assert "http://foo.bar" not in result

    def test_remove_www_urls(self):
        text = "visit www.example.com for help"
        result = self.cleaner.remove_urls(text)
        assert "www.example.com" not in result

    # --- Email removal ---
    def test_remove_emails(self):
        text = "email me at user@example.com please"
        result = self.cleaner.remove_emails(text)
        assert "user@example.com" not in result

    # --- Mention removal ---
    def test_remove_mentions(self):
        text = "hey @username check this out"
        result = self.cleaner.remove_mentions(text)
        assert "@username" not in result

    # --- Hashtag handling ---
    def test_remove_hashtags_keeps_word(self):
        text = "feeling #depressed and #anxious"
        result = self.cleaner.remove_hashtags(text)
        assert "#" not in result
        assert "depressed" in result
        assert "anxious" in result

    # --- HTML tag removal ---
    def test_remove_html_tags(self):
        text = "<p>I feel <b>terrible</b></p>"
        result = self.cleaner.remove_html_tags(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "terrible" in result

    # --- Contraction expansion ---
    def test_expand_contractions_basic(self):
        assert "cannot" in self.cleaner.expand_contractions("I can't do this")
        assert "do not" in self.cleaner.expand_contractions("I don't care")
        assert "will not" in self.cleaner.expand_contractions("I won't stop")

    def test_expand_contractions_preserves_negation(self):
        result = self.cleaner.expand_contractions("I can't sleep and don't want to eat")
        assert "not" in result  # Negation preserved after expansion

    def test_expand_contractions_informal(self):
        result = self.cleaner.expand_contractions("gonna wanna kinda")
        assert "going to" in result
        assert "want to" in result
        assert "kind of" in result

    # --- Special character removal ---
    def test_remove_special_characters_keeps_question_exclamation(self):
        text = "why?! am i feeling this way???"
        result = self.cleaner.remove_special_characters(text)
        assert "?" in result
        assert "!" in result

    def test_remove_special_characters_removes_others(self):
        text = "hello @#$ world & stuff"
        result = self.cleaner.remove_special_characters(text)
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "&" not in result

    # --- Repeated character collapsing ---
    def test_collapse_repeated_characters(self):
        assert self.cleaner.collapse_repeated_characters("sooooo") == "soo"
        assert self.cleaner.collapse_repeated_characters("!!!") == "!!"
        assert self.cleaner.collapse_repeated_characters("hello") == "hello"

    # --- Whitespace normalization ---
    def test_normalize_whitespace(self):
        assert self.cleaner.normalize_whitespace("  hello   world  ") == "hello world"
        assert self.cleaner.normalize_whitespace("a  b  c") == "a b c"

    # --- Full pipeline ---
    def test_clean_full_pipeline(self):
        raw = "I CAN'T do this @user https://help.com <b>anymore</b>!!!"
        result = self.cleaner.clean(raw)
        assert result == result.lower()  # Should be lowercase
        assert "https" not in result
        assert "@user" not in result
        assert "<b>" not in result
        assert "cannot" in result  # Contraction expanded

    def test_clean_empty_input(self):
        assert self.cleaner.clean("") == ""
        assert self.cleaner.clean(None) == ""

    def test_clean_preserves_emotional_content(self):
        raw = "I feel extremely sad and hopeless"
        result = self.cleaner.clean(raw)
        assert "extremely" in result
        assert "sad" in result
        assert "hopeless" in result

    def test_clean_mental_health_text(self):
        raw = "I don't feel happy anymore, I've lost interest in everything"
        result = self.cleaner.clean(raw)
        assert "do not" in result
        assert "feel" in result
        assert "lost interest" in result


# =============================================================================
# EmotionalStopwordFilter Tests
# =============================================================================

class TestEmotionalStopwordFilter:
    """Tests for the EmotionalStopwordFilter class."""

    def setup_method(self):
        self.filter = EmotionalStopwordFilter()

    def test_preserves_negation_words(self):
        tokens = ["i", "do", "not", "feel", "happy"]
        result = self.filter.filter(tokens)
        assert "not" in result
        assert "feel" in result

    def test_preserves_intensifiers(self):
        tokens = ["i", "am", "extremely", "sad"]
        result = self.filter.filter(tokens)
        assert "extremely" in result
        assert "sad" in result

    def test_preserves_never(self):
        tokens = ["i", "will", "never", "be", "happy"]
        result = self.filter.filter(tokens)
        assert "never" in result

    def test_removes_insignificant_stopwords(self):
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        result = self.filter.filter(tokens)
        assert "the" not in result
        assert "on" not in result
        assert "cat" in result

    def test_preserves_emotional_words(self):
        tokens = ["feel", "hurt", "alone", "pain", "suffering"]
        result = self.filter.filter(tokens)
        # All of these should be preserved
        assert all(token in result for token in tokens)

    def test_filter_text_method(self):
        text = "i do not feel happy at the moment"
        result = self.filter.filter_text(text)
        assert "not" in result
        assert "feel" in result
        assert "happy" in result

    def test_empty_input(self):
        assert self.filter.filter([]) == []
        assert self.filter.filter_text("") == ""

    def test_is_stopword(self):
        assert self.filter.is_stopword("the")
        assert not self.filter.is_stopword("not")
        assert not self.filter.is_stopword("feel")
        assert not self.filter.is_stopword("never")


# =============================================================================
# CrisisDetector Tests
# =============================================================================

class TestCrisisDetector:
    """Tests for the CrisisDetector class."""

    def setup_method(self):
        self.detector = CrisisDetector()

    # --- Crisis detection ---
    def test_detects_suicidal_ideation(self):
        result = self.detector.detect("i want to kill myself")
        assert result.is_crisis is True
        assert result.severity in ("HIGH", "CRITICAL")
        assert len(result.matched_crisis_keywords) > 0

    def test_detects_self_harm(self):
        result = self.detector.detect("i have been cutting myself")
        assert result.is_crisis is True
        assert "cutting myself" in result.matched_crisis_keywords

    def test_detects_lethal_means(self):
        result = self.detector.detect("thinking about overdose")
        assert result.is_crisis is True

    def test_detects_goodbye_messages(self):
        result = self.detector.detect("goodbye forever everyone")
        assert result.is_crisis is True

    def test_critical_severity(self):
        result = self.detector.detect("i am going to kill myself tonight")
        assert result.severity == "CRITICAL"

    # --- Non-crisis text ---
    def test_normal_text_no_crisis(self):
        result = self.detector.detect("i had a nice day at work today")
        assert result.is_crisis is False
        assert result.severity in ("NONE", "LOW")

    def test_mild_stress_not_crisis(self):
        result = self.detector.detect("feeling a bit stressed about my exam")
        assert result.is_crisis is False

    # --- Depression / anxiety / stress detection ---
    def test_detects_depression_keywords(self):
        result = self.detector.detect("i feel hopeless and worthless every day")
        assert len(result.matched_depression_keywords) > 0

    def test_detects_anxiety_keywords(self):
        result = self.detector.detect("i am having panic attacks and cannot breathe")
        assert len(result.matched_anxiety_keywords) > 0

    def test_detects_stress_keywords(self):
        result = self.detector.detect("i am overwhelmed and burned out from work")
        assert len(result.matched_stress_keywords) > 0

    # --- Severity levels ---
    def test_medium_severity_multiple_signals(self):
        result = self.detector.detect("i feel depressed hopeless and extremely anxious")
        assert result.severity in ("MEDIUM", "HIGH")

    def test_low_severity_single_signal(self):
        result = self.detector.detect("i feel a little sad today")
        assert result.severity in ("LOW", "MEDIUM")

    # --- Intensifiers ---
    def test_detects_intensifiers(self):
        result = self.detector.detect("i am extremely depressed and utterly hopeless")
        assert result.has_intensifiers is True

    # --- Edge cases ---
    def test_empty_input(self):
        result = self.detector.detect("")
        assert result.is_crisis is False
        assert result.severity == "NONE"

    def test_none_input(self):
        result = self.detector.detect(None)
        assert result.is_crisis is False

    # --- Quick check method ---
    def test_is_crisis_quick_check(self):
        assert self.detector.is_crisis("i want to end my life") is True
        assert self.detector.is_crisis("nice weather today") is False

    # --- Result serialization ---
    def test_to_dict(self):
        result = self.detector.detect("i feel depressed")
        d = result.to_dict()
        assert "is_crisis" in d
        assert "severity" in d
        assert "matched_crisis_keywords" in d
        assert "matched_depression_keywords" in d
        assert isinstance(d, dict)


# =============================================================================
# Keyword Bank Tests
# =============================================================================

class TestKeywordBanks:
    """Tests to verify keyword banks are properly defined."""

    def test_crisis_keywords_not_empty(self):
        assert len(CRISIS_KEYWORDS) > 50

    def test_depression_keywords_not_empty(self):
        assert len(DEPRESSION_KEYWORDS) > 50

    def test_anxiety_keywords_not_empty(self):
        assert len(ANXIETY_KEYWORDS) > 50

    def test_stress_keywords_not_empty(self):
        assert len(STRESS_KEYWORDS) > 50

    def test_negation_words_not_empty(self):
        assert len(NEGATION_WORDS) > 10

    def test_intensifiers_not_empty(self):
        assert len(EMOTION_INTENSIFIERS) > 20

    def test_crisis_contains_key_phrases(self):
        assert "kill myself" in CRISIS_KEYWORDS
        assert "suicide" in CRISIS_KEYWORDS
        assert "end my life" in CRISIS_KEYWORDS

    def test_negation_contains_critical_words(self):
        assert "not" in NEGATION_WORDS
        assert "never" in NEGATION_WORDS
        assert "no" in NEGATION_WORDS
        assert "cannot" in NEGATION_WORDS


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests to verify configuration constants."""

    def test_model_names_defined(self):
        assert EMOTION_MODEL_NAME == "mental/mental-roberta-base"
        assert STRESS_MODEL_NAME == "jnyx74/stress-prediction"
        assert DEPRESSION_MODEL_NAME == "poudel/Depression_and_Non-Depression_Classifier"

    def test_all_model_names_dict(self):
        assert "emotion" in ALL_MODEL_NAMES
        assert "stress" in ALL_MODEL_NAMES
        assert "depression" in ALL_MODEL_NAMES
        assert len(ALL_MODEL_NAMES) == 3


# =============================================================================
# PreprocessingPipeline Tests (text-only mode — no model loading)
# =============================================================================

class TestPreprocessingPipeline:
    """Tests for the PreprocessingPipeline in text-only mode."""

    def setup_method(self):
        self.pipeline = PreprocessingPipeline()

    def test_process_text_only_basic(self):
        result = self.pipeline.process_text_only("I feel sad today")
        assert result.original_text == "I feel sad today"
        assert result.cleaned_text == "i feel sad today"
        assert result.crisis_result is not None
        assert result.crisis_result.is_crisis is False

    def test_process_text_only_with_urls(self):
        result = self.pipeline.process_text_only(
            "Can't sleep check https://example.com"
        )
        assert "https" not in result.cleaned_text
        assert "cannot" in result.cleaned_text

    def test_process_text_only_crisis_detection(self):
        result = self.pipeline.process_text_only("I want to kill myself")
        assert result.crisis_result.is_crisis is True
        assert result.crisis_result.severity in ("HIGH", "CRITICAL")

    def test_process_text_only_empty_input(self):
        result = self.pipeline.process_text_only("")
        assert result.original_text == ""
        assert result.cleaned_text == ""

    def test_process_text_only_complex(self):
        raw = (
            "I CAN'T do this anymore @therapist!!! "
            "Everything is POINTLESS https://help.com "
            "I've been sooooo depressed #mentalhealth"
        )
        result = self.pipeline.process_text_only(raw)

        # Check cleaning worked
        assert result.cleaned_text == result.cleaned_text.lower()
        assert "https" not in result.cleaned_text
        assert "@therapist" not in result.cleaned_text
        assert "cannot" in result.cleaned_text
        assert "soo" in result.cleaned_text  # Repeated chars collapsed

        # Check crisis/keyword detection
        assert result.crisis_result is not None
        assert len(result.crisis_result.matched_depression_keywords) > 0

    def test_process_text_only_preserves_emotional_content(self):
        raw = "I never feel happy, I'm not okay"
        result = self.pipeline.process_text_only(raw)
        assert "never" in result.cleaned_text
        assert "not" in result.cleaned_text
        assert "happy" in result.cleaned_text

    def test_result_to_dict(self):
        result = self.pipeline.process_text_only("I feel anxious")
        d = result.to_dict()
        assert "original_text" in d
        assert "cleaned_text" in d
        assert "crisis_result" in d
        assert isinstance(d, dict)

    def test_pipeline_repr(self):
        s = repr(self.pipeline)
        assert "PreprocessingPipeline" in s


# =============================================================================
# Integration test — full flow without models
# =============================================================================

class TestIntegrationTextOnly:
    """End-to-end integration test for the text-only pipeline."""

    def test_full_mental_health_conversation(self):
        pipeline = PreprocessingPipeline()

        messages = [
            ("Hello, I'm feeling a bit down today", False, "NONE"),
            ("I can't stop worrying about everything", False, None),
            ("I feel extremely depressed and hopeless", False, "MEDIUM"),
            ("I want to end it all, nothing matters", True, None),
            ("Had a good day at the park!", False, "NONE"),
        ]

        for raw_text, expected_crisis, expected_severity in messages:
            result = pipeline.process_text_only(raw_text)

            # Basic checks
            assert result.original_text == raw_text
            assert len(result.cleaned_text) > 0
            assert result.crisis_result is not None

            # Crisis check
            if expected_crisis:
                assert result.crisis_result.is_crisis is True, (
                    f"Expected crisis for: {raw_text}"
                )

            # Severity check (when specified)
            if expected_severity:
                assert result.crisis_result.severity == expected_severity or \
                    (expected_severity in ("LOW", "MEDIUM") and
                     result.crisis_result.severity in ("LOW", "MEDIUM")), (
                    f"Unexpected severity {result.crisis_result.severity} "
                    f"for: {raw_text}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
