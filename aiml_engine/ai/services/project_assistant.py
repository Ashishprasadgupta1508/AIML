import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

# Keep Gemini as the only LLM provider. Use the lightweight stable model first
# to minimize latency and improve availability, then fall back to stable Flash.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.7-flash",
]
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Do not spend the whole Postman Cloud Agent window waiting on overloaded models.
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES_PER_MODEL = 0
MODEL_TIMEOUT_SECONDS = 6.0
MAX_MODELS_PER_REQUEST = 3


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


def _clean_model_name(model: str) -> str:
    model = (model or "").strip()
    if model.startswith("models/"):
        model = model[len("models/"):]
    return model


def _get_models() -> List[str]:
    """Build a short Gemini-only fallback chain and ignore obsolete models."""
    primary = _clean_model_name(
        os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    ) or DEFAULT_GEMINI_MODEL

    configured = os.getenv("GEMINI_FALLBACK_MODELS", "")
    fallback_values = (
        [item.strip() for item in configured.split(",") if item.strip()]
        if configured.strip()
        else DEFAULT_GEMINI_FALLBACK_MODELS
    )

    # 2.5 is intentionally excluded because the deployed account previously
    # returned a 404 saying it was unavailable to new users.
    obsolete = {"gemini-2.5-flash", "gemini-2.5-flash-lite"}
    models: List[str] = []
    for model in [primary, *fallback_values]:
        model = _clean_model_name(model)
        if not model or model in obsolete or model in models:
            continue
        models.append(model)

    # Never let a long environment fallback list turn one API call into a
    # 30+ second request.
    return models[:MAX_MODELS_PER_REQUEST]


def _compact(value: Any, max_chars: int = 9000) -> Any:
    if isinstance(value, dict):
        return {key: _compact(item, max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact(item, max_chars) for item in value[:10]]
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "..."
    return value


def build_assistant_context(
    question: str,
    analysis: Optional[Dict[str, Any]] = None,
    projects: Optional[List[Dict[str, Any]]] = None,
) -> str:
    return json.dumps(
        {
            "question": question,
            "project_analysis": _compact(analysis or {}),
            "comparison_projects": _compact(projects or []),
        },
        ensure_ascii=False,
        default=str,
    )


def ask_project_assistant(
    question: str,
    analysis: Optional[Dict[str, Any]] = None,
    projects: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    question = (question or "").strip()
    if not question:
        raise ValueError("message is required.")

    api_key = _get_api_key()

    system_instruction = """
You are the Gati Infrastructure Project Intelligence Assistant.

Answer only from the supplied project analysis and comparison data. Do not
invent facts, dates, costs, causes, delays, or recommendations.

Return only a professional Markdown answer, not JSON or code fences.
Start with "## Project Intelligence Summary". Keep it concise and useful for
project management. Include project/risk summary, key indicators, major risks,
expected impact, recommended actions, historical evidence, and overall
assessment when the supplied data supports them. Use bold values and bullets.
Do not repeat information unnecessarily. Clearly label estimates as estimates.
""".strip()

    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": build_assistant_context(
                            question, analysis, projects
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 900,
        },
    }

    models = _get_models()
    last_error: Any = None

    # Each model gets one short attempt. A 503/429 immediately moves to the
    # next Gemini model instead of waiting through repeated retries.
    with httpx.Client(timeout=MODEL_TIMEOUT_SECONDS) as client:
        for current_model in models:
            url = GEMINI_API_URL.format(model=current_model)
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
                continue

            if response.status_code in RETRYABLE_STATUS_CODES:
                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text[:500]
                last_error = (response.status_code, detail)
                continue

            if response.status_code >= 400:
                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text[:500]
                raise AssistantProviderError(
                    f"LLM provider returned HTTP {response.status_code}: {detail}"
                )

            try:
                data = response.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    feedback = data.get("promptFeedback") or {}
                    last_error = AssistantProviderError(
                        f"Gemini returned no candidates. promptFeedback={feedback!r}"
                    )
                    continue

                candidate = candidates[0] or {}
                parts = (candidate.get("content") or {}).get("parts") or []
                answer = "".join(
                    part.get("text", "")
                    for part in parts
                    if isinstance(part, dict) and part.get("text")
                ).strip()

                if answer:
                    return {"answer": answer}

                last_error = AssistantProviderError(
                    "Gemini returned no answer text. "
                    f"finishReason={candidate.get('finishReason', 'UNKNOWN')}"
                )
            except (TypeError, ValueError) as exc:
                last_error = exc

    if isinstance(last_error, tuple):
        status_code, detail = last_error
        raise AssistantProviderError(
            "Gemini is temporarily unavailable after trying the configured "
            f"Gemini Flash models. Last response was HTTP {status_code}: {detail}"
        )
    if isinstance(last_error, httpx.HTTPError):
        raise AssistantProviderError(
            "Gemini could not be reached within the provider timeout."
        ) from last_error
    if isinstance(last_error, AssistantProviderError):
        raise last_error
    raise AssistantProviderError(
        "Gemini did not return a usable answer after trying the configured models."
    )
