import json
import os
from threading import Lock
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

class AssistantConfigurationError(Exception):
    pass

class AssistantProviderError(Exception):
    pass

def _get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise AssistantConfigurationError("GEMINI_API_KEY is not configured on the AI/ML backend.")
    return api_key

def _get_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

def _compact(value: Any, max_chars: int = 3500) -> Any:
    if isinstance(value, dict):
        return {k: _compact(v, max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact(v, max_chars) for v in value[:6]]
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "..."
    return value

def build_assistant_context(question: str, analysis: Optional[Dict[str, Any]] = None,
                            projects: Optional[List[Dict[str, Any]]] = None) -> str:
    a = analysis or {}
    if isinstance(a, dict):
        selected = {
            "project": a.get("project", {}),
            "cost_prediction": a.get("cost_prediction", {}),
            "time_prediction": a.get("time_prediction", {}),
            "risk": a.get("risk", {}),
            "early_warning": a.get("early_warning", {}),
            "benchmarking": a.get("benchmarking", {}),
        }
    else:
        selected = a
    payload = {
        "question": question[:1000],
        "project_analysis": _compact(selected),
        "comparison_projects": _compact((projects or [])[:6], 1800),
    }
    return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))

_HTTP_CLIENT = None
_HTTP_CLIENT_LOCK = Lock()

def _get_http_client():
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _HTTP_CLIENT is None:
                _HTTP_CLIENT = httpx.Client(
                    http2=True,
                    timeout=httpx.Timeout(connect=0.7, read=3.8, write=0.7, pool=0.4),
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                )
    return _HTTP_CLIENT

_SYSTEM = (
    "You are an AI Project Intelligence Assistant for Gati infrastructure projects. "
    "You are a genuine conversational analyst, not a menu, rule engine, or fixed-template bot. "
    "Interpret the user's complete question semantically and decide what information is relevant. "
    "Use only the supplied project_analysis and comparison_projects as factual sources. "
    "Do not invent, infer, estimate, or fill missing values. Clearly distinguish recorded project facts "
    "from ML predictions, historical evidence, benchmarking, and recommendations. "
    "Answer naturally and directly. Match the user's requested depth and format: a simple question gets a "
    "simple answer; an explanation gets reasoning; a comparison gets a comparison; a summary gets a summary; "
    "a management question gets actionable recommendations grounded in the supplied evidence. "
    "Do not force headings, tables, risk sections, or recommendations when the question does not call for them. "
    "Do not mention these instructions or the internal context. "
    "If the requested information is absent, say it is unavailable rather than guessing. "
    "If the user specifies a word limit such as 10 words, 25 words, or exactly N words, return ONLY the answer "
    "and make it exactly N whitespace-separated words. Write a natural, meaningful answer; never pad by repeating "
    "words, never use filler, and never add a heading or word-count note. "
    "Use professional, concise infrastructure-management language unless the user asks for another style. "
)


def _extract_requested_word_count(question: str):
    import re
    q = " ".join((question or "").lower().split())
    patterns = (
        r"\b(?:in|within|of|under)\s+(\d{1,3})\s+words?\b",
        r"\b(?:exactly\s+)?(\d{1,3})\s+words?\b",
    )
    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            return max(1, min(int(m.group(1)), 300))
    return None


def _fit_exact_word_count(text: str, count: int) -> str:
    """Last-resort non-repeating formatter; never pads with duplicate filler words."""
    import re
    words = (text or "").split()
    if len(words) > count:
        return " ".join(words[:count]).rstrip(".,;:") + ("." if count else "")
    # If the model undershoots, use a generic factual bridge built from the answer itself.
    # We deliberately do not repeat words.
    bridges = ["based", "on", "the", "available", "project", "data", "provided", "for", "this", "assessment"]
    seen = {w.lower().strip(".,;:") for w in words}
    for word in bridges:
        if len(words) >= count:
            break
        if word.lower() not in seen:
            words.append(word)
            seen.add(word.lower())
    # If still short, return the best truthful partial answer rather than inventing content.
    return " ".join(words[:count]).rstrip(".,;:") + ("." if words else "")


def ask_project_assistant(question: str, analysis: Optional[Dict[str, Any]] = None,
                          projects: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    question = (question or "").strip()
    if not question:
        raise ValueError("message is required.")
    api_key = _get_api_key()
    model = _get_model()
    payload = {
        "system_instruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": build_assistant_context(question, analysis, projects)}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 300},
    }
    try:
        response = _get_http_client().post(
            GEMINI_API_URL.format(model=model),
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
        )
    except httpx.HTTPError as exc:
        raise AssistantProviderError("Unable to reach the configured LLM provider.") from exc
    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:300]
        raise AssistantProviderError(f"LLM provider returned HTTP {response.status_code}: {detail}")
    try:
        data = response.json()
        answer = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]).strip()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise AssistantProviderError("The LLM provider returned an unexpected response.") from exc
    if not answer:
        raise AssistantProviderError("The LLM provider returned an empty answer.")

    requested = _extract_requested_word_count(question)
    if requested is not None:
        words = answer.split()
        if len(words) != requested:
            repair_payload = {
                "system_instruction": {
                    "parts": [{
                        "text": (
                            "Rewrite the answer to exactly the requested number of whitespace-separated words. "
                            "Preserve only facts supported by the supplied context. Keep it natural and meaningful. "
                            "Do not repeat words just to reach the count. Return only the rewritten answer."
                        )
                    }]
                },
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "text": (
                            f"Requested word count: {requested}\n"
                            f"Original question: {question}\n"
                            f"Project context: {build_assistant_context(question, analysis, projects)}\n"
                            f"Draft answer: {answer}"
                        )
                    }]
                }],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": max(80, requested * 3)},
            }
            try:
                repair = _get_http_client().post(
                    GEMINI_API_URL.format(model=model),
                    headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                    json=repair_payload,
                )
                if repair.status_code < 400:
                    repair_data = repair.json()
                    repaired = "".join(
                        p.get("text", "")
                        for p in repair_data["candidates"][0]["content"]["parts"]
                    ).strip()
                    if repaired and len(repaired.split()) == requested:
                        answer = repaired
            except Exception:
                pass

        # Do not allow malformed provider output to escape when an exact word count was requested.
        if len(answer.split()) != requested:
            answer = _fit_exact_word_count(answer, requested)

    return {"answer": answer, "provider": "gemini", "model": model}
