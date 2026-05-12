"""
rag.py - TF-IDF RAG engine for the Clinical Pharmacology reference book.

The PDF is bundled with the app and indexed automatically at startup.
No uploads, no torch, no external downloads — pure Python + numpy.

Pipeline:
  1. load_default_pdf()  → reads PDF from disk
  2. extract_text_from_pdf() → PyMuPDF text extraction with page markers
  3. split_into_chunks() → overlapping character-level chunks
  4. build_tfidf_index() → TF-IDF matrix (numpy)
  5. retrieve_relevant_chunks() → cosine similarity search → top-k chunks
  6. Chunks passed to Gemini in chatbot.py for answer generation
"""

import os
import re
from collections import Counter

import numpy as np
import fitz   # PyMuPDF



CHUNK_SIZE    = 600   # max characters per chunk
CHUNK_OVERLAP = 150   # characters shared between consecutive chunks
TOP_K         = 10    # chunks returned per query

_MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_MODULE_DIR)
PDF_PATH     = os.path.join(_PROJECT_DIR, "Clinical Pharmacology.pdf")



def load_default_pdf() -> dict:
    """
    Load and index the bundled Clinical Pharmacology PDF.
    Result is cached in st.session_state by the caller (app.py).

    Returns:
        dict: {
            "index":        TF-IDF index dict,
            "chunks":       list[str],
            "total_chunks": int,
            "total_pages":  int,
        }
        On failure: {"error": "message"}
    """
    if not os.path.exists(PDF_PATH):
        return {
            "error": (
                f"Clinical Pharmacology PDF not found.\n"
                f"Expected location: {PDF_PATH}\n"
                "Please place 'Clinical Pharmacology.pdf' in the project folder."
            )
        }
    try:
        with open(PDF_PATH, "rb") as f:
            return process_pdf(f.read())
    except Exception as e:
        return {"error": f"Failed to load PDF: {str(e)}"}


def process_pdf(pdf_bytes: bytes) -> dict:
    """
    Full pipeline: raw PDF bytes → TF-IDF index ready for querying.

    Args:
        pdf_bytes (bytes): Raw bytes of the PDF file.

    Returns:
        dict: index, chunks, total_chunks, total_pages — or {"error": "..."}
    """
    try:
        full_text, page_count = extract_text_from_pdf(pdf_bytes)
        chunks = split_into_chunks(full_text)
        if not chunks:
            return {"error": "No text chunks could be created from the PDF."}
        index = build_tfidf_index(chunks)
        return {
            "index":        index,
            "chunks":       chunks,
            "total_chunks": len(chunks),
            "total_pages":  page_count,
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"PDF processing failed: {str(e)}"}



def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, int]:
    """
    Extract plain text from all pages of a PDF.
    Each page is prefixed with a [Page N] marker for citation support.

    Args:
        pdf_bytes (bytes): Raw PDF bytes.

    Returns:
        tuple[str, int]: Full extracted text and total page count.

    Raises:
        ValueError: If no text could be extracted (e.g. scanned/image PDF).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    pages = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"[Page {page_num}]\n{text}")
    doc.close()

    if not pages:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned/image-only PDF. Please use a text-based PDF."
        )
    return "\n\n".join(pages), total_pages



def split_into_chunks(text: str,
                      chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split a long text into overlapping character-level chunks.

    Overlap ensures context at chunk boundaries is not lost.
    Chunks prefer to break at sentence boundaries ('. ' or newline).

    Args:
        text (str): Full document text.
        chunk_size (int): Target max characters per chunk.
        overlap (int): Characters shared between consecutive chunks.

    Returns:
        list[str]: Non-empty text chunks.
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    chunks, start, text_len = [], 0, len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            break_pos = max(
                text.rfind(". ", start, end),
                text.rfind("\n",  start, end),
            )
            if break_pos > start + overlap:
                end = break_pos + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start += max(1, chunk_size - overlap)

    return chunks



def _tokenize(text: str) -> list:
    """Lowercase alphanumeric tokenizer, drops single-character tokens."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


def build_tfidf_index(chunks: list) -> dict:
    """
    Build a TF-IDF index over all text chunks.

    TF-IDF scores each term by how often it appears in a chunk (TF)
    relative to how rare it is across all chunks (IDF). Rows are
    L2-normalised so cosine similarity reduces to a dot product.

    Args:
        chunks (list[str]): Text chunks from the PDF.

    Returns:
        dict: {
            "chunks":       original chunks list,
            "tfidf_matrix": numpy float32 array (n_chunks × vocab_size),
            "vocab":        {word: column_index},
            "idf":          numpy float32 array (vocab_size,),
        }
    """
    n = len(chunks)
    tokenized = [_tokenize(c) for c in chunks]

    vocab = {w: i for i, w in enumerate(
        sorted({w for tokens in tokenized for w in tokens})
    )}
    vocab_size = len(vocab)

    tf = np.zeros((n, vocab_size), dtype=np.float32)
    for i, tokens in enumerate(tokenized):
        if not tokens:
            continue
        counts = Counter(tokens)
        total  = len(tokens)
        for word, cnt in counts.items():
            if word in vocab:
                tf[i, vocab[word]] = cnt / total

    df  = np.sum(tf > 0, axis=0).astype(np.float32)
    idf = np.log((1 + n) / (1 + df)) + 1.0

    tfidf = tf * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0          # avoid division by zero for empty chunks
    tfidf_norm = tfidf / norms

    return {
        "chunks":       chunks,
        "tfidf_matrix": tfidf_norm,
        "vocab":        vocab,
        "idf":          idf,
    }



def retrieve_relevant_chunks(query: str, index: dict,
                              top_k: int = TOP_K) -> list:
    """
    Return the top-k chunks most relevant to the query.

    Uses the same TF-IDF weighting as the index, then computes cosine
    similarity via dot product (both vectors are L2-normalised).

    Args:
        query (str): The user's question.
        index (dict): TF-IDF index from build_tfidf_index().
        top_k (int): Maximum number of chunks to return.

    Returns:
        list[str]: Relevant chunks ordered by descending similarity score.
                   Returns an empty list if no chunk scores above zero.
    """
    vocab        = index["vocab"]
    idf          = index["idf"]
    tfidf_matrix = index["tfidf_matrix"]
    chunks       = index["chunks"]

    tokens = _tokenize(query)
    if not tokens:
        return chunks[:top_k]

    counts     = Counter(tokens)
    total      = len(tokens)
    query_vec  = np.zeros(len(vocab), dtype=np.float32)
    for word, cnt in counts.items():
        if word in vocab:
            query_vec[vocab[word]] = (cnt / total) * idf[vocab[word]]

    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec /= norm

    scores      = tfidf_matrix.dot(query_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [chunks[i] for i in top_indices if scores[i] > 0]
