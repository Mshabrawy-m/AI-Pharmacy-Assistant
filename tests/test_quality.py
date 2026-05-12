"""
test_quality.py - Quality, stability and regression tests for AI Pharmacy Assistant.

Tests cover:
  - RAG pipeline (chunking, indexing, retrieval)
  - JSON extraction parser (all 3 strategies + repair)
  - Prompt builders
  - History format conversion
  - Edge cases and error handling

Run with:
    python -m pytest project/tests/test_quality.py -v
  or:
    python project/tests/test_quality.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.rag import (
    split_into_chunks,
    build_tfidf_index,
    retrieve_relevant_chunks,
    _tokenize,
)
from modules.extractor import (
    parse_extraction_response,
    _validate_extraction,
    _repair_truncated_json,
)
from modules.prompts import (
    build_chatbot_context,
    build_rag_prompt,
)
from modules.chatbot import (
    format_history_for_gemini,
    _streamlit_to_gemini_history,
)



SAMPLE_TEXT = """
[Page 1]
Pharmacokinetics is the study of drug absorption, distribution, metabolism,
and excretion. It describes how the body affects a drug after administration.

[Page 2]
Beta blockers are drugs that block the effects of adrenaline on beta-adrenergic
receptors. They are used to treat high blood pressure, angina, and heart failure.

[Page 3]
NSAIDs work by inhibiting cyclooxygenase enzymes. Common examples include
ibuprofen, aspirin, and naproxen. They reduce inflammation and pain.

[Page 4]
Drug metabolism primarily occurs in the liver via cytochrome P450 enzymes.
First-pass metabolism reduces bioavailability of orally administered drugs.
"""

VALID_EXTRACTION_JSON = {
    "medicines": [
        {
            "medicine_name": "Amoxicillin",
            "dosage": "500mg",
            "frequency": "3 times daily",
            "duration": "7 days",
            "route": "oral",
            "confidence": "high",
        }
    ],
    "prescription_notes": None,
    "doctor_name": "Dr. Smith",
    "patient_name": "John Doe",
    "date": "2026-05-12",
}

SAMPLE_MEDICINES = [
    {
        "medicine_name": "Amoxicillin",
        "dosage": "500mg",
        "frequency": "3x daily",
        "duration": "7 days",
        "route": "oral",
        "confidence": "high",
    },
    {
        "medicine_name": "Ibuprofen",
        "dosage": "400mg",
        "frequency": "as needed",
        "duration": None,
        "route": None,
        "confidence": "medium",
    },
]



def test_tokenize_basic():
    tokens = _tokenize("Hello World 123")
    assert "hello" in tokens
    assert "world" in tokens
    assert "123" in tokens
    print("  PASS test_tokenize_basic")


def test_tokenize_removes_single_chars():
    tokens = _tokenize("a b c drug")
    assert "a" not in tokens
    assert "b" not in tokens
    assert "drug" in tokens
    print("  PASS test_tokenize_removes_single_chars")


def test_tokenize_empty():
    assert _tokenize("") == []
    assert _tokenize("   ") == []
    print("  PASS test_tokenize_empty")


def test_tokenize_special_chars():
    tokens = _tokenize("beta-blocker (500mg)")
    assert "beta" in tokens
    assert "blocker" in tokens
    assert "500mg" in tokens
    print("  PASS test_tokenize_special_chars")



def test_split_creates_chunks():
    chunks = split_into_chunks(SAMPLE_TEXT, chunk_size=200, overlap=50)
    assert len(chunks) > 0, "Should create at least one chunk"
    print(f"  PASS test_split_creates_chunks ({len(chunks)} chunks)")


def test_split_chunk_size_respected():
    chunks = split_into_chunks(SAMPLE_TEXT, chunk_size=150, overlap=30)
    for chunk in chunks:
        assert len(chunk) <= 300, f"Chunk too long: {len(chunk)} chars"
    print("  PASS test_split_chunk_size_respected")


def test_split_overlap_creates_more_chunks():
    chunks_no_overlap  = split_into_chunks(SAMPLE_TEXT, chunk_size=300, overlap=0)
    chunks_with_overlap = split_into_chunks(SAMPLE_TEXT, chunk_size=300, overlap=100)
    assert len(chunks_with_overlap) >= len(chunks_no_overlap), \
        "Overlap should produce >= chunks"
    print("  PASS test_split_overlap_creates_more_chunks")


def test_split_empty_text():
    chunks = split_into_chunks("", chunk_size=200, overlap=50)
    assert chunks == [], "Empty text should produce no chunks"
    print("  PASS test_split_empty_text")


def test_split_short_text():
    chunks = split_into_chunks("Short text.", chunk_size=200, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."
    print("  PASS test_split_short_text")


def test_split_no_empty_chunks():
    chunks = split_into_chunks(SAMPLE_TEXT, chunk_size=100, overlap=20)
    for chunk in chunks:
        assert chunk.strip() != "", "No chunk should be empty"
    print("  PASS test_split_no_empty_chunks")



def test_build_index_structure():
    chunks = split_into_chunks(SAMPLE_TEXT, chunk_size=200, overlap=50)
    index = build_tfidf_index(chunks)
    assert "tfidf_matrix" in index
    assert "vocab" in index
    assert "idf" in index
    assert "chunks" in index
    print("  PASS test_build_index_structure")


def test_build_index_matrix_shape():
    chunks = split_into_chunks(SAMPLE_TEXT, chunk_size=200, overlap=50)
    index = build_tfidf_index(chunks)
    n_chunks = len(chunks)
    vocab_size = len(index["vocab"])
    assert index["tfidf_matrix"].shape == (n_chunks, vocab_size)
    print(f"  PASS test_build_index_matrix_shape ({n_chunks}x{vocab_size})")


def test_build_index_normalized_rows():
    import numpy as np
    chunks = split_into_chunks(SAMPLE_TEXT, chunk_size=200, overlap=50)
    index = build_tfidf_index(chunks)
    norms = np.linalg.norm(index["tfidf_matrix"], axis=1)
    for norm in norms:
        assert abs(norm - 1.0) < 1e-5 or norm == 0.0, \
            f"Row not normalised: norm={norm}"
    print("  PASS test_build_index_normalized_rows")


def test_build_index_single_chunk():
    index = build_tfidf_index(["Only one chunk here."])
    assert index["tfidf_matrix"].shape[0] == 1
    print("  PASS test_build_index_single_chunk")



def test_retrieve_returns_relevant_chunk():
    chunks = split_into_chunks(SAMPLE_TEXT, chunk_size=200, overlap=50)
    index = build_tfidf_index(chunks)
    results = retrieve_relevant_chunks("beta blockers blood pressure", index)
    assert len(results) > 0, "Should return at least one result"
    found = any("beta" in r.lower() or "blocker" in r.lower() for r in results)
    assert found, "Relevant chunk should be in top results"
    print("  PASS test_retrieve_returns_relevant_chunk")


def test_retrieve_nsaids_query():
    chunks = split_into_chunks(SAMPLE_TEXT, chunk_size=200, overlap=50)
    index = build_tfidf_index(chunks)
    results = retrieve_relevant_chunks("NSAIDs ibuprofen inflammation", index)
    assert len(results) > 0
    found = any("nsaid" in r.lower() or "ibuprofen" in r.lower() for r in results)
    assert found, "NSAID chunk should be retrieved"
    print("  PASS test_retrieve_nsaids_query")


def test_retrieve_empty_query_fallback():
    chunks = split_into_chunks(SAMPLE_TEXT, chunk_size=200, overlap=50)
    index = build_tfidf_index(chunks)
    results = retrieve_relevant_chunks("", index)
    assert isinstance(results, list), "Should return a list"
    print(f"  PASS test_retrieve_empty_query_fallback ({len(results)} fallback chunks)")


def test_retrieve_no_match_returns_empty():
    chunks = split_into_chunks(SAMPLE_TEXT, chunk_size=200, overlap=50)
    index = build_tfidf_index(chunks)
    results = retrieve_relevant_chunks("xyzzy quantum flux capacitor", index)
    assert isinstance(results, list)
    assert len(results) == 0, "No-match query should return empty list"
    print("  PASS test_retrieve_no_match_returns_empty")


def test_retrieve_top_k_respected():
    chunks = split_into_chunks(SAMPLE_TEXT, chunk_size=100, overlap=20)
    index = build_tfidf_index(chunks)
    results = retrieve_relevant_chunks("drug metabolism liver", index, top_k=2)
    assert len(results) <= 2, "Should not return more than top_k results"
    print("  PASS test_retrieve_top_k_respected")



def test_parse_clean_json():
    raw = json.dumps(VALID_EXTRACTION_JSON)
    result = parse_extraction_response(raw)
    assert "error" not in result
    assert len(result["medicines"]) == 1
    assert result["medicines"][0]["medicine_name"] == "Amoxicillin"
    print("  PASS test_parse_clean_json")


def test_parse_json_with_markdown_fences():
    raw = "```json\n" + json.dumps(VALID_EXTRACTION_JSON) + "\n```"
    result = parse_extraction_response(raw)
    assert "error" not in result
    assert result["medicines"][0]["medicine_name"] == "Amoxicillin"
    print("  PASS test_parse_json_with_markdown_fences")


def test_parse_json_with_preamble():
    raw = "Here is the extracted data:\n\n" + json.dumps(VALID_EXTRACTION_JSON)
    result = parse_extraction_response(raw)
    assert "error" not in result
    print("  PASS test_parse_json_with_preamble")


def test_parse_empty_response():
    result = parse_extraction_response("")
    assert "error" in result
    assert "empty" in result["error"].lower()
    print("  PASS test_parse_empty_response")


def test_parse_whitespace_only():
    result = parse_extraction_response("   \n\n  ")
    assert "error" in result
    print("  PASS test_parse_whitespace_only")


def test_parse_missing_medicines_key():
    raw = json.dumps({"doctor_name": "Dr. X", "patient_name": "Jane"})
    result = parse_extraction_response(raw)
    assert "error" in result
    print("  PASS test_parse_missing_medicines_key")


def test_parse_empty_medicines_list():
    raw = json.dumps({"medicines": [], "doctor_name": None})
    result = parse_extraction_response(raw)
    assert "error" in result
    assert "no medicines" in result["error"].lower()
    print("  PASS test_parse_empty_medicines_list")


def test_parse_missing_fields_filled_with_none():
    raw = json.dumps({
        "medicines": [{"medicine_name": "Aspirin"}],
        "doctor_name": None,
    })
    result = parse_extraction_response(raw)
    assert "error" not in result
    med = result["medicines"][0]
    assert med["dosage"] is None
    assert med["frequency"] is None
    assert med["confidence"] == "medium"   # default
    print("  PASS test_parse_missing_fields_filled_with_none")


def test_parse_truncated_json_repair():
    full = json.dumps(VALID_EXTRACTION_JSON)
    truncated = full[:len(full) - 20]   # cut off the end
    result = parse_extraction_response(truncated)
    assert isinstance(result, dict)
    print(f"  PASS test_parse_truncated_json_repair (result: {'repaired' if 'error' not in result else 'error returned cleanly'})")


def test_repair_truncated_json_closes_braces():
    partial = '{"medicines": [{"medicine_name": "Aspirin", "dosage": "100mg"'
    repaired = _repair_truncated_json(partial)
    assert repaired.count("{") == repaired.count("}")
    assert repaired.count("[") == repaired.count("]")
    print("  PASS test_repair_truncated_json_closes_braces")



def test_build_chatbot_context_with_medicines():
    context = build_chatbot_context(SAMPLE_MEDICINES)
    assert "Amoxicillin" in context
    assert "Ibuprofen" in context
    assert "500mg" in context
    assert "3x daily" in context
    print("  PASS test_build_chatbot_context_with_medicines")


def test_build_chatbot_context_empty():
    context = build_chatbot_context([])
    assert "no medicines" in context.lower()
    print("  PASS test_build_chatbot_context_empty")


def test_build_chatbot_context_null_fields():
    meds = [{"medicine_name": "Aspirin", "dosage": None,
             "frequency": None, "duration": None, "route": None}]
    context = build_chatbot_context(meds)
    assert "Aspirin" in context
    assert "Not specified" in context
    print("  PASS test_build_chatbot_context_null_fields")


def test_build_rag_prompt_contains_context_and_question():
    chunks = ["Beta blockers reduce heart rate.", "NSAIDs reduce inflammation."]
    question = "What are beta blockers?"
    prompt = build_rag_prompt(question, chunks)
    assert "Beta blockers" in prompt
    assert "NSAIDs" in prompt
    assert question in prompt
    assert "CONTEXT START" in prompt
    assert "CONTEXT END" in prompt
    print("  PASS test_build_rag_prompt_contains_context_and_question")


def test_build_rag_prompt_empty_chunks():
    prompt = build_rag_prompt("What is pharmacology?", [])
    assert "What is pharmacology?" in prompt
    print("  PASS test_build_rag_prompt_empty_chunks")



def test_format_history_user_message():
    messages = [{"role": "user", "content": "Hello"}]
    result = format_history_for_gemini(messages)
    assert result[0]["role"] == "user"
    assert result[0]["parts"] == ["Hello"]
    print("  PASS test_format_history_user_message")


def test_format_history_assistant_to_model():
    messages = [{"role": "assistant", "content": "Hi there"}]
    result = format_history_for_gemini(messages)
    assert result[0]["role"] == "model"
    assert result[0]["parts"] == ["Hi there"]
    print("  PASS test_format_history_assistant_to_model")


def test_format_history_multi_turn():
    messages = [
        {"role": "user",      "content": "What is aspirin?"},
        {"role": "assistant", "content": "Aspirin is an NSAID."},
        {"role": "user",      "content": "Any side effects?"},
    ]
    result = format_history_for_gemini(messages)
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "model"
    assert result[2]["role"] == "user"
    print("  PASS test_format_history_multi_turn")


def test_format_history_empty():
    result = format_history_for_gemini([])
    assert result == []
    print("  PASS test_format_history_empty")


def test_format_history_already_gemini_format():
    messages = [{"role": "user", "parts": ["Already converted"]}]
    result = _streamlit_to_gemini_history(messages)
    assert result[0]["role"] == "user"
    assert result[0]["parts"] == ["Already converted"]
    print("  PASS test_format_history_already_gemini_format")



def test_validate_extraction_valid():
    result = _validate_extraction(VALID_EXTRACTION_JSON.copy())
    assert "error" not in result
    assert result["medicines"][0]["medicine_name"] == "Amoxicillin"
    print("  PASS test_validate_extraction_valid")


def test_validate_extraction_missing_key_raises():
    try:
        _validate_extraction({"doctor_name": "Dr. X"})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  PASS test_validate_extraction_missing_key_raises")


def test_validate_extraction_non_dict_raises():
    try:
        _validate_extraction([1, 2, 3])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  PASS test_validate_extraction_non_dict_raises")



def run_all():
    groups = [
        ("Tokenizer",            [test_tokenize_basic, test_tokenize_removes_single_chars,
                                   test_tokenize_empty, test_tokenize_special_chars]),
        ("Text Chunking",        [test_split_creates_chunks, test_split_chunk_size_respected,
                                   test_split_overlap_creates_more_chunks, test_split_empty_text,
                                   test_split_short_text, test_split_no_empty_chunks]),
        ("TF-IDF Index",         [test_build_index_structure, test_build_index_matrix_shape,
                                   test_build_index_normalized_rows, test_build_index_single_chunk]),
        ("Retrieval",            [test_retrieve_returns_relevant_chunk, test_retrieve_nsaids_query,
                                   test_retrieve_empty_query_fallback, test_retrieve_no_match_returns_empty,
                                   test_retrieve_top_k_respected]),
        ("JSON Parser",          [test_parse_clean_json, test_parse_json_with_markdown_fences,
                                   test_parse_json_with_preamble, test_parse_empty_response,
                                   test_parse_whitespace_only, test_parse_missing_medicines_key,
                                   test_parse_empty_medicines_list, test_parse_missing_fields_filled_with_none,
                                   test_parse_truncated_json_repair, test_repair_truncated_json_closes_braces]),
        ("Prompt Builders",      [test_build_chatbot_context_with_medicines, test_build_chatbot_context_empty,
                                   test_build_chatbot_context_null_fields, test_build_rag_prompt_contains_context_and_question,
                                   test_build_rag_prompt_empty_chunks]),
        ("History Conversion",   [test_format_history_user_message, test_format_history_assistant_to_model,
                                   test_format_history_multi_turn, test_format_history_empty,
                                   test_format_history_already_gemini_format]),
        ("Validate Extraction",  [test_validate_extraction_valid, test_validate_extraction_missing_key_raises,
                                   test_validate_extraction_non_dict_raises]),
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for group_name, tests in groups:
        print(f"\n{'=' * 55}")
        print(f"  {group_name}")
        print(f"{'=' * 55}")
        for test_fn in tests:
            try:
                test_fn()
                total_passed += 1
            except Exception as e:
                total_failed += 1
                failures.append((test_fn.__name__, str(e)))
                print(f"  FAIL {test_fn.__name__}: {e}")

    print(f"\n{'=' * 55}")
    print(f"  RESULTS: {total_passed} passed, {total_failed} failed")
    print(f"{'=' * 55}")

    if failures:
        print("\nFailed tests:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\n  All tests passed ✅")


if __name__ == "__main__":
    run_all()
