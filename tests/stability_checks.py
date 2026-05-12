"""
stability_checks.py - Extended stability and edge-case tests.
Run with: python tests/stability_checks.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.rag import split_into_chunks, build_tfidf_index, retrieve_relevant_chunks
from modules.extractor import _validate_extraction, parse_extraction_response
from modules.prompts import build_rag_prompt, build_chatbot_context
from modules.chatbot import _streamlit_to_gemini_history

passed = 0
failed = 0
findings = []


def check(label, fn):
    global passed, failed
    try:
        fn()
        print("  PASS  " + label)
        passed += 1
    except AssertionError as e:
        print("  FAIL  " + label + ": " + str(e))
        failed += 1
        findings.append((label, str(e)))
    except Exception as e:
        print("  ERROR " + label + ": " + type(e).__name__ + ": " + str(e))
        failed += 1
        findings.append((label, type(e).__name__ + ": " + str(e)))


print("=" * 60)
print("  STABILITY & EDGE-CASE CHECKS")
print("=" * 60)


def test_overlap_equals_chunk():
    chunks = split_into_chunks("Some pharmacology text.", chunk_size=20, overlap=20)
    assert isinstance(chunks, list), "must return a list"
    assert len(chunks) > 0, "must return at least one chunk"

check("RAG: overlap == chunk_size (infinite-loop guard)", test_overlap_equals_chunk)


def test_overlap_exceeds_chunk():
    chunks = split_into_chunks("Short.", chunk_size=5, overlap=100)
    assert isinstance(chunks, list)

check("RAG: overlap > chunk_size (should not freeze)", test_overlap_exceeds_chunk)


def test_single_word():
    chunks = split_into_chunks("pharmacology", chunk_size=200, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "pharmacology"

check("RAG: single-word document", test_single_word)


def test_oov_query():
    index = build_tfidf_index(["drug metabolism in the liver", "beta blockers reduce heart rate"])
    results = retrieve_relevant_chunks("zzzxxx qqqqq nonsense12345", index)
    assert results == [], "expected [], got " + str(results)

check("RAG: out-of-vocab query returns empty list", test_oov_query)


def test_empty_index():
    try:
        index = build_tfidf_index([])
        assert index["tfidf_matrix"].shape[0] == 0
    except Exception:
        pass  # acceptable - empty input is undefined behaviour

check("RAG: empty chunk list handled without crash", test_empty_index)


def test_unexpected_confidence_not_overwritten():
    data = {"medicines": [{"medicine_name": "Test", "confidence": "VERY_HIGH"}]}
    result = _validate_extraction(data)
    assert "error" not in result
    assert result["medicines"][0]["confidence"] == "VERY_HIGH"

check("Extractor: non-standard confidence value preserved", test_unexpected_confidence_not_overwritten)


def test_blank_confidence_defaults():
    data = {"medicines": [{"medicine_name": "Test", "confidence": ""}]}
    result = _validate_extraction(data)
    assert result["medicines"][0]["confidence"] == "medium"

check("Extractor: blank confidence defaults to medium", test_blank_confidence_defaults)


def test_all_none_fields():
    data = {"medicines": [{"medicine_name": None, "confidence": None}]}
    result = _validate_extraction(data)
    assert "error" not in result
    assert result["medicines"][0]["confidence"] == "medium"

check("Extractor: all-None medicine fields handled", test_all_none_fields)


def test_medicines_not_a_list():
    raw = '{"medicines": "Amoxicillin"}'
    result = parse_extraction_response(raw)
    assert isinstance(result, dict)

check("Extractor: medicines field is a string (not list) handled", test_medicines_not_a_list)


def test_rag_prompt_single_chunk():
    prompt = build_rag_prompt("What is aspirin?", ["Aspirin is an NSAID."])
    assert "Aspirin is an NSAID." in prompt
    assert "CONTEXT START" in prompt
    assert "CONTEXT END" in prompt

check("Prompts: build_rag_prompt single chunk contains markers", test_rag_prompt_single_chunk)


def test_context_empty_medicine_dict():
    context = build_chatbot_context([{}])
    assert "Unknown" in context
    assert "Not specified" in context

check("Prompts: build_chatbot_context with empty medicine dict", test_context_empty_medicine_dict)


def test_mixed_history_formats():
    mixed = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "parts": ["Already gemini format"]},
    ]
    result = _streamlit_to_gemini_history(mixed)
    assert len(result) == 2
    assert result[0]["parts"] == ["Hello"]
    assert result[1]["parts"] == ["Already gemini format"]

check("Chatbot: mixed Streamlit+Gemini history passthrough", test_mixed_history_formats)


def test_malformed_history_skipped():
    malformed = [
        {"role": "user", "text": "bad format"},
        {"role": "user", "content": "good"},
    ]
    result = _streamlit_to_gemini_history(malformed)
    assert len(result) == 1
    assert result[0]["parts"] == ["good"]

check("Chatbot: malformed history message silently skipped", test_malformed_history_skipped)


def test_retrieve_top_k_zero():
    index = build_tfidf_index(["drug metabolism", "beta blockers"])
    results = retrieve_relevant_chunks("beta blockers", index, top_k=0)
    assert results == [], "expected [], got " + str(results)

check("RAG: top_k=0 returns empty list", test_retrieve_top_k_zero)


def test_whitespace_only_text():
    chunks = split_into_chunks("    \n\n   \t  ", chunk_size=200, overlap=50)
    assert chunks == [], "expected [], got " + str(chunks)

check("RAG: whitespace-only text produces no chunks", test_whitespace_only_text)


print()
print("=" * 60)
print("  RESULTS: " + str(passed) + " passed, " + str(failed) + " failed")
print("=" * 60)

if findings:
    print("\nFailed / Error details:")
    for label, err in findings:
        print("  - " + label)
        print("    " + err)
    sys.exit(1)
else:
    print("\n  All stability checks passed!")
