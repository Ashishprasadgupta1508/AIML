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
                    timeout=httpx.Timeout(connect=0.8, read=1.8, write=0.8, pool=0.5),
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                )
    return _HTTP_CLIENT

_SYSTEM = (
    "You are the Gati Infrastructure Project Intelligence Assistant. "
    "Use ONLY supplied project analysis/comparison data. Never invent facts. "
    "Answer concisely and professionally. Explain the model indication, key risk, "
    "expected impact, recommended action, and historical evidence when relevant. "
    "If data is insufficient, say so. Use Markdown with short headings and bullets."
)

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
    return {"answer": answer, "provider": "gemini", "model": model}
