import json
import os
import time
import httpx


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

DEFAULT_GEMINI_FALLBACK_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.5-flash",
]

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/{model}:generateContent"
)

MODEL_TIMEOUT_SECONDS = 5.0
MAX_MODELS_PER_REQUEST = 2


class AssistantConfigurationError(Exception):
    pass


class AssistantProviderError(Exception):
    pass


SYSTEM_PROMPT = """
You are the Gati Infrastructure Project Intelligence Assistant.

Answer the user's question using ONLY the project analysis data supplied
by the backend.

Do not invent project facts.

Explain AI/ML outputs in a professional and decision-oriented manner.

Your response must:
- Start with "## Project Intelligence Summary"
- Use clear Markdown headings.
- Use bullet points where appropriate.
- Highlight important values using bold text.
- Use one compact Markdown table when numerical indicators are available.
- Clearly distinguish predictions and estimates from confirmed project facts.
- Explain major risks, expected impact and recommended management actions.
- Mention historical evidence when available.
- Be concise but professionally detailed.
- Do not include JSON.
- Do not use code fences.
- Do not mention internal provider/model details.
- Do not use emojis.
"""


def _get_api_key():
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def _get_models():
    primary = (
        os.getenv("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    )

    fallback_raw = os.getenv(
        "GEMINI_FALLBACK_MODELS",
        ",".join(DEFAULT_GEMINI_FALLBACK_MODELS),
    )

    fallbacks = [
        item.strip()
        for item in fallback_raw.split(",")
        if item.strip()
    ]

    models = []

    for model in [primary] + fallbacks:
        if model and model not in models:
            models.append(model)

    # Never use obsolete Gemini 2.5 fallback.
    models = [
        model
        for model in models
        if model != "gemini-2.5-flash"
    ]

    return models[:MAX_MODELS_PER_REQUEST]


def _build_prompt(question, analysis=None, projects=None):
    context = {
        "analysis": analysis,
        "projects": projects,
    }

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"PROJECT DATA:\n"
        f"{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
        f"USER QUESTION:\n"
        f"{question}"
    )


def _extract_text(data):
    candidates = data.get("candidates") or []

    if not candidates:
        return None

    candidate = candidates[0] or {}

    content = candidate.get("content") or {}
    parts = content.get("parts") or []

    texts = []

    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if text:
                texts.append(text)

    answer = "\n".join(texts).strip()

    if not answer:
        return None

    return answer


def _call_gemini(model, api_key, prompt):
    url = GEMINI_API_URL.format(model=model)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1600,
        },
    }

    try:
        with httpx.Client(
            timeout=MODEL_TIMEOUT_SECONDS
        ) as client:

            response = client.post(
                url,
                params={
                    "key": api_key
                },
                json=payload,
            )

    except Exception as exc:
        raise AssistantProviderError(
            f"Gemini request failed: {exc}"
        )

    if response.status_code != 200:
        raise AssistantProviderError(
            f"Gemini provider returned HTTP "
            f"{response.status_code}: {response.text[:1000]}"
        )

    try:
        data = response.json()
    except Exception as exc:
        raise AssistantProviderError(
            f"Invalid Gemini response: {exc}"
        )

    answer = _extract_text(data)

    if not answer:
        raise AssistantProviderError(
            "Gemini returned an empty response."
        )

    return answer


def ask_project_assistant(
    question,
    analysis=None,
    projects=None,
):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is required.")

    api_key = _get_api_key()

    if not api_key:
        raise AssistantConfigurationError(
            "Gemini API key is not configured."
        )

    prompt = _build_prompt(
        question=question.strip(),
        analysis=analysis,
        projects=projects,
    )

    models = _get_models()

    if not models:
        raise AssistantConfigurationError(
            "No Gemini model is configured."
        )

    last_error = None

    for model in models:

        try:
            answer = _call_gemini(
                model=model,
                api_key=api_key,
                prompt=prompt,
            )

            return {
                "answer": answer,
            }

        except AssistantProviderError as exc:
            last_error = exc
            continue

    raise AssistantProviderError(
        str(last_error)
        if last_error
        else "Gemini provider unavailable."
    )