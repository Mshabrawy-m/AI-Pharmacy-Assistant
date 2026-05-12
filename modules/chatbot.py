"""
chatbot.py - Groq-powered chatbot functions for the AI Pharmacy Assistant.

Groq provides high-throughput LLM inference with a generous free tier
(30 RPM, no daily quota pressure), making it ideal for chat workloads.
Google Gemini is reserved exclusively for vision (prescription images).

Contains:
  - get_chatbot_response      : Multi-turn chatbot for prescription medicines
  - get_rag_response          : RAG chatbot grounded in the Clinical Pharmacology PDF
  - format_history_for_gemini : Utility kept for test-suite compatibility
  - _streamlit_to_gemini_history : Internal utility
"""

from groq import Groq

from modules.prompts import (
    CHATBOT_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    build_chatbot_context,
    build_rag_prompt,
)


_GROQ_MODEL = "llama-3.3-70b-versatile"



def get_chatbot_response(
    user_message: str,
    chat_history: list,
    medicines: list,
    groq_api_key: str,
) -> str:
    """
    Send a user message to the prescription chatbot and return a response.

    Uses Groq (llama-3.3-70b-versatile) for fast, quota-free responses.
    Injects the extracted medicines as context at the start of every session.

    Args:
        user_message (str): The latest message from the user.
        chat_history (list): Previous turns in Streamlit format
                             [{"role": "user"|"assistant", "content": "text"}]
        medicines (list): Extracted medicines from the prescription.
        groq_api_key (str): Groq API key.

    Returns:
        str: Chatbot response text, or a user-friendly error string.
    """
    try:
        client = Groq(api_key=groq_api_key)
    except Exception as e:
        return f"⚠️ Failed to initialize Groq client: {str(e)}"

    medicine_context = build_chatbot_context(medicines)
    messages = [
        {"role": "system", "content": CHATBOT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Prescription context for this session:\n\n{medicine_context}",
        },
        {
            "role": "assistant",
            "content": (
                "Understood. I have noted the prescribed medicines "
                "and I'm ready to answer your questions. How can I help you?"
            ),
        },
    ] + _to_groq_history(chat_history) + [
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        return _handle_groq_error(e, context="Chatbot")



def get_rag_response(
    user_message: str,
    chat_history: list,
    faiss_index: dict,
    chunks: list,       # kept for call-site compatibility, not used directly
    groq_api_key: str,
) -> str:
    """
    Answer a question using RAG: retrieve relevant PDF chunks, then
    send them with the question to Groq for a grounded answer.

    Args:
        user_message (str): The user's question.
        chat_history (list): Previous turns in Streamlit format.
        faiss_index (dict): TF-IDF index dict from rag.build_tfidf_index().
        chunks (list): Unused — the index already contains the chunks.
        groq_api_key (str): Groq API key.

    Returns:
        str: Grounded answer from Groq, or an error string.
    """
    from modules.rag import retrieve_relevant_chunks

    try:
        retrieved = retrieve_relevant_chunks(user_message, faiss_index)
    except Exception as e:
        return f"⚠️ Retrieval error: {str(e)}"

    if not retrieved:
        return (
            "I couldn't find relevant content in the Clinical Pharmacology textbook "
            "for your question. Please try rephrasing, or ask about a topic "
            "covered in the book."
        )

    rag_prompt = build_rag_prompt(user_message, retrieved)

    try:
        client = Groq(api_key=groq_api_key)
    except Exception as e:
        return f"⚠️ Failed to initialize Groq client: {str(e)}"

    messages = (
        [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        + _to_groq_history(chat_history)
        + [{"role": "user", "content": rag_prompt}]
    )

    try:
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=messages,
            temperature=0.2,        # Low = factual, grounded answers
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        return _handle_groq_error(e, context="RAG chatbot")



def format_history_for_gemini(streamlit_messages: list) -> list:
    """
    Convert Streamlit chat messages to Gemini's history format.
    Kept for test-suite compatibility.

    Streamlit: [{"role": "user"|"assistant", "content": "text"}]
    Gemini:    [{"role": "user"|"model",      "parts":   ["text"]}]
    """
    return _streamlit_to_gemini_history(streamlit_messages)



def _to_groq_history(messages: list) -> list:
    """
    Convert Streamlit-format or Gemini-format messages to Groq/OpenAI format.

    Accepts both:
      - Streamlit: {"role": "user"|"assistant", "content": "text"}
      - Gemini:    {"role": "user"|"model",      "parts":  ["text"]}
    """
    result = []
    for msg in messages:
        if "content" in msg:
            role = "assistant" if msg["role"] == "assistant" else "user"
            result.append({"role": role, "content": msg["content"]})
        elif "parts" in msg:
            role = "assistant" if msg["role"] == "model" else "user"
            content = msg["parts"][0] if msg["parts"] else ""
            result.append({"role": role, "content": content})
    return result


def _streamlit_to_gemini_history(messages: list) -> list:
    """
    Convert a list of Streamlit-format messages to Gemini format.
    Handles both Streamlit format (content key) and already-converted
    Gemini format (parts key) gracefully.
    """
    result = []
    for msg in messages:
        if "content" in msg:
            role = "model" if msg["role"] == "assistant" else "user"
            result.append({"role": role, "parts": [msg["content"]]})
        elif "parts" in msg:
            result.append(msg)   # already in Gemini format — pass through
    return result


def _handle_groq_error(error: Exception, context: str = "Groq") -> str:
    """
    Convert a Groq API exception into a user-friendly error message.

    Args:
        error (Exception): The caught exception.
        context (str): Label for the error source.

    Returns:
        str: User-friendly error message with warning prefix.
    """
    msg = str(error)
    if "invalid_api_key" in msg.lower() or "authentication" in msg.lower():
        return "⚠️ System configuration error. Please contact support."
    if "rate_limit" in msg.lower() or "429" in msg:
        return "⚠️ System rate limit reached. Please wait a moment and try again."
    if "context_length" in msg.lower():
        return "⚠️ Conversation is too long. Please clear the chat and start fresh."
    return f"⚠️ {context} error: {msg}"
