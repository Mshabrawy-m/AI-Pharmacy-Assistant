"""
extractor.py - Prescription image analysis using Gemini Vision with Groq fallback.

Primary: Google Gemini 2.0 Flash (best vision quality, multi-key rotation)
Fallback: Groq Llama-4-Scout vision (free, high limits — used when all Google keys fail)

Supports multiple Google API keys with automatic rotation on quota errors.
"""

import base64
import io
import json
import re

import google.generativeai as genai
from PIL import Image

from modules.prompts import EXTRACTION_PROMPT


_GOOGLE_VISION_MODEL = "gemini-2.0-flash"
_GROQ_VISION_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"



def analyze_prescription(image: Image.Image, api_keys, groq_api_key: str = "") -> dict:
    """
    Send a prescription image to Gemini Vision and extract medicine data.

    Strategy:
      1. Try each Google API key in order (rotates on quota / auth errors).
      2. If ALL Google keys fail → fall back to Groq Llama-4-Scout vision.
      3. If Groq also fails → return a clear error message.

    Args:
        image (PIL.Image.Image): The uploaded prescription image.
        api_keys (str | list[str]): One or more Google Gemini API keys.
        groq_api_key (str): Groq API key for the vision fallback.

    Returns:
        dict: Parsed extraction result or {"error": "message"}.
    """
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    api_keys = [k.strip() for k in api_keys if k.strip()]

    last_google_error = "No Google API keys provided."

    for idx, key in enumerate(api_keys):
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name=_GOOGLE_VISION_MODEL)
        except Exception as e:
            last_google_error = f"Key {idx + 1}: failed to configure — {str(e)}"
            continue

        try:
            response = model.generate_content(
                [EXTRACTION_PROMPT, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            return parse_extraction_response(response.text)

        except Exception as e:
            msg = str(e)
            if "quota" in msg.lower():
                last_google_error = f"Key {idx + 1}: quota exceeded."
                continue  # try next key
            if "API_KEY_INVALID" in msg or "invalid" in msg.lower():
                last_google_error = f"Key {idx + 1}: invalid key."
                continue
            if "safety" in msg.lower():
                return {"error": "Image was blocked by safety filters. Please try a different image."}
            return {"error": f"AI Processing Error: {msg}"}

    if groq_api_key and groq_api_key.strip() and groq_api_key != "YOUR_GROQ_API_KEY_HERE":
        return _analyze_with_groq(image, groq_api_key.strip())

    return {
        "error": (
            "The system is currently experiencing high load. "
            "Please try again later."
        )
    }


def get_medicines_list(extraction_result: dict) -> list:
    """
    Safely retrieve the medicines list from an extraction result.

    Args:
        extraction_result (dict): Result from analyze_prescription().

    Returns:
        list: List of medicine dicts, or empty list on error.
    """
    if "error" in extraction_result:
        return []
    return extraction_result.get("medicines", [])



def _analyze_with_groq(image: Image.Image, groq_api_key: str) -> dict:
    """
    Fallback: analyze the prescription image using Groq's vision model.

    Args:
        image (PIL.Image.Image): The prescription image.
        groq_api_key (str): Valid Groq API key.

    Returns:
        dict: Parsed extraction result or {"error": "message"}.
    """
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
    except Exception as e:
        return {"error": f"Failed to initialize Groq client: {str(e)}"}

    try:
        buf = io.BytesIO()
        rgb_image = image.convert("RGB")
        rgb_image.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        return {"error": f"Failed to encode image: {str(e)}"}

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": EXTRACTION_PROMPT,
                },
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model=_GROQ_VISION_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
        )
        raw_text = response.choices[0].message.content
        return parse_extraction_response(raw_text)
    except Exception as e:
        msg = str(e)
        if "rate_limit" in msg.lower() or "429" in msg:
            return {"error": "⚠️ System rate limit reached. Please wait a moment and try again."}
        if "invalid_api_key" in msg.lower() or "authentication" in msg.lower():
            return {"error": "⚠️ System configuration error."}
        return {"error": f"AI Vision Error: {msg}"}



def parse_extraction_response(raw_text: str) -> dict:
    """
    Parse the model's raw text response into a Python dict.

    Tries three strategies in order:
      1. Strip markdown fences → direct JSON parse
      2. Brace-match to extract the outermost JSON object
      3. Repair truncated JSON by closing open brackets/braces

    Args:
        raw_text (str): Raw text from the model.

    Returns:
        dict: Validated extraction data, or {"error": "..."}.
    """
    if not raw_text or not raw_text.strip():
        return {"error": "Model returned an empty response. Please try again."}

    cleaned = raw_text.strip()

    fenced = re.sub(r"```(?:json)?\s*", "", cleaned).strip().rstrip("`").strip()
    try:
        return _validate_extraction(json.loads(fenced))
    except (json.JSONDecodeError, ValueError):
        pass

    start = cleaned.find("{")
    if start != -1:
        depth, end = 0, -1
        for i, ch in enumerate(cleaned[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end != -1:
            try:
                return _validate_extraction(json.loads(cleaned[start:end]))
            except (json.JSONDecodeError, ValueError):
                pass

        repaired = _repair_truncated_json(cleaned[start:])
        if repaired:
            try:
                return _validate_extraction(json.loads(repaired))
            except (json.JSONDecodeError, ValueError):
                pass

    return {
        "error": (
            "Could not parse the prescription data. "
            "Please ensure the image is clear and try again. "
            f"(Response preview: {raw_text[:150]}...)"
        )
    }



def _validate_extraction(data: dict) -> dict:
    """
    Validate structure and sanitize fields of a parsed extraction dict.

    Raises:
        ValueError: If the 'medicines' key is missing or not a list.
    """
    if not isinstance(data, dict) or "medicines" not in data:
        raise ValueError("'medicines' key not found in response")

    if not isinstance(data["medicines"], list):
        raise ValueError("'medicines' must be a list, got " + type(data["medicines"]).__name__)

    if not data["medicines"]:
        return {
            "error": (
                "No medicines were found in the prescription image. "
                "Please ensure the image is clear and contains a valid prescription."
            )
        }

    required_fields = ["medicine_name", "dosage", "frequency",
                       "duration", "route", "confidence"]
    for med in data["medicines"]:
        if not isinstance(med, dict):
            continue  # skip malformed entries gracefully
        for field in required_fields:
            if field not in med:
                med[field] = None
        if not med.get("confidence"):
            med["confidence"] = "medium"

    return data


def _repair_truncated_json(text: str) -> str:
    """
    Attempt to repair a truncated JSON string by closing open structures.

    Args:
        text (str): Potentially truncated JSON starting with '{'.

    Returns:
        str: Repaired JSON string, or empty string if repair not possible.
    """
    text = text.rstrip().rstrip(",")

    quote_pos = text.rfind('"')
    if quote_pos != -1:
        quotes_before = text[:quote_pos].count('"')
        if quotes_before % 2 == 0:
            for i in range(quote_pos - 1, -1, -1):
                if text[i] == ",":
                    text = text[:i]
                    break
                if text[i] in ("{", "["):
                    break

    open_brackets = text.count("[") - text.count("]")
    open_braces   = text.count("{") - text.count("}")
    text += "]" * max(0, open_brackets) + "}" * max(0, open_braces)

    return text
