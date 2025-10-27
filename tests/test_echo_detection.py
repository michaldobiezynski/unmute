"""Tests for echo detection functionality in unmute_handler."""

import sys
from pathlib import Path

# Add parent directory to path to import unmute modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from unmute.unmute_handler import (
    compute_text_similarity,
    is_text_reading_back_response,
    normalize_text_for_comparison,
)


def test_normalize_text_for_comparison():
    """Test text normalization removes punctuation and standardizes whitespace."""
    assert normalize_text_for_comparison("Hello, World!") == "hello world"
    assert normalize_text_for_comparison("What's up?") == "whats up"
    assert normalize_text_for_comparison("Multiple   spaces") == "multiple spaces"
    assert normalize_text_for_comparison("  Leading trailing  ") == "leading trailing"


def test_compute_text_similarity_identical():
    """Test similarity for identical texts."""
    text = "The quick brown fox jumps over the lazy dog"
    assert compute_text_similarity(text, text) == 1.0


def test_compute_text_similarity_different():
    """Test similarity for completely different texts."""
    text1 = "Hello world"
    text2 = "Goodbye universe"
    similarity = compute_text_similarity(text1, text2)
    assert similarity < 0.3


def test_compute_text_similarity_similar():
    """Test similarity for similar texts with minor differences."""
    text1 = "The weather is nice today"
    text2 = "The weather is very nice today"
    similarity = compute_text_similarity(text1, text2)
    assert similarity > 0.7


def test_is_text_reading_back_response_exact():
    """Test detection of exact match."""
    assistant_text = "The capital of France is Paris"
    user_text = "The capital of France is Paris"
    assert is_text_reading_back_response(user_text, assistant_text)


def test_is_text_reading_back_response_with_punctuation():
    """Test detection when punctuation differs."""
    assistant_text = "Hello, how are you today?"
    user_text = "Hello how are you today"
    assert is_text_reading_back_response(user_text, assistant_text)


def test_is_text_reading_back_response_partial():
    """Test detection when user reads beginning of response."""
    assistant_text = "The Python programming language is great for data science and machine learning"
    user_text = "The Python programming language is great"
    assert is_text_reading_back_response(user_text, assistant_text)


def test_is_text_reading_back_response_not_similar():
    """Test that dissimilar texts are not detected as echo."""
    assistant_text = "What is your favorite color?"
    user_text = "I like pizza"
    assert not is_text_reading_back_response(user_text, assistant_text)


def test_is_text_reading_back_response_empty():
    """Test handling of empty strings."""
    assert not is_text_reading_back_response("", "some text")
    assert not is_text_reading_back_response("some text", "")
    assert not is_text_reading_back_response("", "")


def test_is_text_reading_back_response_short():
    """Test that very short phrases don't trigger false positives."""
    assistant_text = "Yes"
    user_text = "No"
    # Short texts might have high similarity just by chance
    # Our implementation requires at least 3 words for substring matching
    assert not is_text_reading_back_response(user_text, assistant_text)


def test_is_text_reading_back_response_case_insensitive():
    """Test that comparison is case-insensitive."""
    assistant_text = "THE QUICK BROWN FOX"
    user_text = "the quick brown fox"
    assert is_text_reading_back_response(user_text, assistant_text)


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_normalize_text_for_comparison,
        test_compute_text_similarity_identical,
        test_compute_text_similarity_different,
        test_compute_text_similarity_similar,
        test_is_text_reading_back_response_exact,
        test_is_text_reading_back_response_with_punctuation,
        test_is_text_reading_back_response_partial,
        test_is_text_reading_back_response_not_similar,
        test_is_text_reading_back_response_empty,
        test_is_text_reading_back_response_short,
        test_is_text_reading_back_response_case_insensitive,
    ]
    
    failed = 0
    passed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: Unexpected error: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)

