import json
import os
import re
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
        raise AssistantConfigurationError(
            "GEMINI_API_KEY is not configured on the AI/ML backend."
        )
    return api_key


def _get_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def _compact(value: Any, max_chars: int = 3500) -> Any:
    if isinstance(value, dict):
        return {k: _compact(v, max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact(v, max_chars) for v in value[:8]]
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "..."
    return value


def build_assistant_context(
    question: str,
    analysis: Optional[Dict[str, Any]] = None,
    projects: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build a factual context packet. Never invents missing analysis."""
    a = analysis if isinstance(analysis, dict) else {}
    selected = {
        "project": a.get("project", {}),
        "cost_prediction": a.get("cost_prediction", {}),
        "time_prediction": a.get("time_prediction", {}),
        "risk": a.get("risk", {}),
        "early_warning": a.get("early_warning", {}),
        "benchmarking": a.get("benchmarking", {}),
        "similar_project_analysis": a.get("similar_project_analysis", {}),
        "source": a.get("source"),
    }
    payload = {
        "user_question": question[:2000],
        "project_analysis": _compact(selected),
        "comparison_projects": _compact((projects or [])[:8], 2200),
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
                    timeout=httpx.Timeout(connect=1.0, read=8.0, write=1.0, pool=1.0),
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                )
    return _HTTP_CLIENT


_SYSTEM = """
You are the AI Project Intelligence Assistant for Gati infrastructure projects.
You are a genuine conversational AI analyst. Do not use keyword-based intent routing,
fixed answers, menus, or canned responses.

Interpret the user's complete question and answer that question using ONLY the supplied
project_analysis and comparison_projects. Treat the supplied data as the source of truth.
Do not invent missing values and do not infer a risk, cost, delay, recommendation, or
historical conclusion when that information is not present in the supplied data.

Important distinction:
- Recorded project facts are not predictions.
- ML predictions are not recorded facts.
- Historical/comparable evidence must be described as historical evidence.
- Recommendations must come from supplied model recommendations or clearly be described
  as validation actions supported by the available data.

Answer the user's actual question. A question asking for the project should describe the
project. A risk question should discuss risk only when a risk assessment is present. A cost
question should discuss the cost analysis when present. A comparison should compare the
supplied projects. A summary should summarize the relevant supplied information.
Do not force headings, tables, risk sections, or recommendations unless useful for the question.

If the requested information is not present, explicitly say that the supplied analysis does
not contain it. Do not replace the missing information with a generic project description.

If the user requests exactly N words, return exactly N whitespace-separated words. Make the
answer natural and meaningful. Never repeat words, truncate a sentence, append filler, or add
a word-count note.

Use concise, professional infrastructure-management language.
Return ONLY the answer text.
""".strip()


def _extract_requested_word_count(question: str) -> Optional[int]:
    q = " ".join((question or "").lower().split())
    patterns = (
        r"\b(?:in|within|of|under|exactly)\s+(\d{1,3})\s+words?\b",
        r"\b(\d{1,3})\s+words?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return max(1, min(int(match.group(1)), 300))
    return None


def _extract_answer(data: Dict[str, Any]) -> str:
    try:
        candidates = data.get("candidates") or []
        if not candidates:
            raise ValueError("No candidates returned")
        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        answer = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)).strip()
        if not answer:
            raise ValueError("Empty candidate text")
        return answer
    except (AttributeError, TypeError, ValueError, IndexError, KeyError) as exc:
        raise AssistantProviderError("The LLM provider returned an unexpected response.") from exc


def _provider_error(response: httpx.Response) -> str:
    try:
        data = response.json()
        message = data.get("error", {}).get("message") if isinstance(data, dict) else None
        if message:
            return str(message)[:500]
    except Exception:
        pass
    return response.text[:500] or f"HTTP {response.status_code}"


def _generate(prompt: str, system_instruction: str, model: str, api_key: str, max_tokens: int) -> str:
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.15,
            "maxOutputTokens": max_tokens,
        },
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
        detail = _provider_error(response)
        print(f"[PROJECT_ASSISTANT] Gemini HTTP {response.status_code} model={model}: {detail}")
        raise AssistantProviderError(
            f"LLM provider returned HTTP {response.status_code}: {detail}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise AssistantProviderError("The LLM provider returned invalid JSON.") from exc
    return _extract_answer(data)


def _fit_exact_word_count(text: str, count: int) -> str:
    """Emergency-only formatter. Never repeats filler or invents project facts."""
    words = (text or "").split()
    if len(words) <= count:
        return " ".join(words)
    return " ".join(words[:count]).rstrip(".,;:")


def ask_project_assistant(
    question: str,
    analysis: Optional[Dict[str, Any]] = None,
    projects: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    question = (question or "").strip()
    if not question:
        raise ValueError("message is required.")

    api_key = _get_api_key()
    model = _get_model()
    context = build_assistant_context(question, analysis, projects)
    requested = _extract_requested_word_count(question)

    answer = _generate(
        prompt=context,
        system_instruction=_SYSTEM,
        model=model,
        api_key=api_key,
        max_tokens=max(300, (requested or 80) * 4),
    )

    if requested is not None and len(answer.split()) != requested:
        repair_prompt = (
            f"Requested exact word count: {requested}\n"
            f"Original user question: {question}\n"
            f"Project data: {context}\n"
            f"Draft answer: {answer}\n\n"
            "Rewrite the draft so it answers the original question naturally in exactly the requested "
            "number of whitespace-separated words. Do not add unsupported facts. Return only the answer."
        )
        repaired = _generate(
            prompt=repair_prompt,
            system_instruction=(
                "You are an exact-length answer editor. Preserve factual meaning from the supplied data. "
                "Return only a natural answer with exactly the requested number of whitespace-separated words. "
                "Never repeat words as filler and never invent facts."
            ),
            model=model,
            api_key=api_key,
            max_tokens=max(80, requested * 3),
        )
        if len(repaired.split()) == requested:
            answer = repaired
        else:
            answer = _fit_exact_word_count(repaired, requested)

    return {"answer": answer, "provider": "gemini", "model": model}
