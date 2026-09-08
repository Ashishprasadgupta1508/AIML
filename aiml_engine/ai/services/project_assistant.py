import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx


DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
DEFAULT_GEMINI_FALLBACK_MODELS = [
    "gemini-3.6-flash",
]
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Temporary provider/rate-limit errors that are safe to retry.
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES_PER_MODEL = 2
RETRY_BACKOFF_SECONDS = 1.5


class AssistantConfigurationError(Exception):
    pass


class AssistantProviderError(Exception):
    pass


def _get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise AssistantConfigurationError(
            "GEMINI_API_KEY is not configured on the AI/ML backend."
        )
    return api_key


def _get_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def _get_models() -> List[str]:
    """Return primary model followed by configurable fallback models."""
    primary = _get_model().strip() or DEFAULT_GEMINI_MODEL

    configured = os.getenv("GEMINI_FALLBACK_MODELS", "")
    if configured.strip():
        fallbacks = [item.strip() for item in configured.split(",") if item.strip()]
    else:
        fallbacks = DEFAULT_GEMINI_FALLBACK_MODELS

    models = []
    for model in [primary, *fallbacks]:
        if model and model not in models:
            models.append(model)
    return models


def _compact(value: Any, max_chars: int = 12000) -> Any:
    """Keep LLM context bounded while preserving useful structured evidence."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result[key] = _compact(item, max_chars=max_chars)
        return result

    if isinstance(value, list):
        # Similar-project lists can be large. Keep the strongest evidence.
        trimmed = value[:15]
        return [_compact(item, max_chars=max_chars) for item in trimmed]

    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "..."

    return value


def build_assistant_context(
    question: str,
    analysis: Optional[Dict[str, Any]] = None,
    projects: Optional[List[Dict[str, Any]]] = None,
) -> str:
    payload = {
        "question": question,
        "project_analysis": _compact(analysis or {}),
        "comparison_projects": _compact(projects or []),
    }

    return json.dumps(payload, ensure_ascii=False, default=str)


def ask_project_assistant(
    question: str,
    analysis: Optional[Dict[str, Any]] = None,
    projects: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Generate a project-intelligence answer grounded only in supplied AI/ML data.

    The service uses Gemini through its REST API so the AIML project does not
    require the Google SDK. The API key stays server-side.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("message is required.")

    api_key = _get_api_key()

    system_instruction = """
You are the Gati Infrastructure Project Intelligence Assistant.

Answer questions using ONLY the project analysis and comparison data supplied
in the request. Do not invent project facts, historical values, causes,
dates, costs, delays, or recommendations that are not supported by the data.

Your job is to explain AI/ML outputs in clear, professional,
infrastructure-management language. You may interpret relationships between
supplied fields, but clearly label an interpretation as an inference when
appropriate.

IMPORTANT RESPONSE-FORMATTING RULES:
- Return ONLY the answer text. Do not return JSON, code fences, or meta-commentary.
- Use clean Markdown so the answer can be rendered directly in a dashboard.
- Start with a short heading: "## Project Intelligence Summary".
- Show the project name/ID and overall risk near the top when available.
- Use concise sections with `###` headings.
- Use **bold** for important values, risks, and actions.
- Use bullet points for lists.
- When enough numeric fields are available, use ONE compact Markdown table titled or introduced as "Key Risk Indicators".
- For multiple risks, give each risk its own `####` subsection and briefly explain why it matters.
- Include these sections when supported by the supplied data:
  1. Executive Summary
  2. Key Risk Indicators
  3. Major Risks
  4. Expected Impact
  5. Recommended Management Actions
  6. Historical Evidence
  7. Overall Assessment
- Keep the answer concise and decision-oriented. Avoid repeating the same number
  in multiple sections unless it adds useful context.
- Do not use emojis or decorative symbols.
- Preserve units exactly and format large Indian currency values clearly when
  the supplied data supports it (for example, ₹6.49 billion).
- Do not turn an estimated value into a fact. Use wording such as "the model
  projects", "the analysis indicates", or "approximately" where appropriate.
- If the supplied data is insufficient, say so clearly and identify what data is needed.
""".strip()

    context = build_assistant_context(
        question=question,
        analysis=analysis,
        projects=projects,
    )

    payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": context}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1600,
        },
    }

    models = _get_models()
    last_error = None

    with httpx.Client(timeout=45.0) as client:
        for model_index, current_model in enumerate(models):
            url = GEMINI_API_URL.format(model=current_model)
            response = None

            for attempt in range(MAX_RETRIES_PER_MODEL + 1):
                try:
                    response = client.post(
                        url,
                        headers={
                            "x-goog-api-key": api_key,
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                except httpx.HTTPError as exc:
                    last_error = exc
                    if attempt < MAX_RETRIES_PER_MODEL:
                        time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                        continue
                    break

                if response.status_code < 400:
                    break

                if response.status_code not in RETRYABLE_STATUS_CODES:
                    try:
                        detail = response.json()
                    except ValueError:
                        detail = response.text[:500]
                    raise AssistantProviderError(
                        f"LLM provider returned HTTP {response.status_code}: {detail}"
                    )

                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text[:500]
                last_error = (response.status_code, detail)

                if attempt < MAX_RETRIES_PER_MODEL:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else 0.0
                    except (TypeError, ValueError):
                        delay = 0.0
                    delay = max(
                        delay,
                        RETRY_BACKOFF_SECONDS * (attempt + 1),
                    )
                    time.sleep(delay)

            else:
                response = None

            if response is not None and response.status_code < 400:
                try:
                    data = response.json()
                    candidate = data["candidates"][0]
                    finish_reason = candidate.get("finishReason", "STOP")
                    parts = candidate["content"]["parts"]
                    answer = "".join(
                        part.get("text", "")
                        for part in parts
                        if part.get("text")
                    ).strip()

                    # A non-STOP finish can produce an incomplete answer.
                    # Retry it instead of returning a cut-off sentence to the client.
                    if finish_reason != "STOP":
                        last_error = AssistantProviderError(
                            f"Gemini returned an incomplete response (finishReason={finish_reason})."
                        )
                        answer = ""
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    last_error = exc
                    answer = ""

                if answer:
                    return {
                        "answer": answer,
                    }

                last_error = AssistantProviderError(
                    "The LLM provider returned an empty or unexpected answer."
                )

            # Try the next configured model after transient failures.
            if model_index < len(models) - 1:
                continue

    if isinstance(last_error, tuple):
        status_code, detail = last_error
        raise AssistantProviderError(
            "All configured Gemini models were temporarily unavailable. "
            f"Last provider response was HTTP {status_code}: {detail}"
        )

    if isinstance(last_error, httpx.HTTPError):
        raise AssistantProviderError(
            "Unable to reach the configured LLM provider after retries and fallbacks."
        ) from last_error

    if isinstance(last_error, AssistantProviderError):
        raise last_error

    if isinstance(last_error, Exception):
        raise AssistantProviderError(
            "The LLM provider returned an unexpected response after retries and fallbacks."
        ) from last_error

    raise AssistantProviderError(
        "All configured Gemini models were unavailable after retries and fallbacks."
    )
