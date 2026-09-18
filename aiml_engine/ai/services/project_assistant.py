import json
import os
import re
from threading import Lock
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
FALLBACK_GEMINI_MODELS = ("gemini-3.6-flash",)
GEMINI_RETRYABLE_STATUS_CODES = {408, 500, 502, 503, 504}
GEMINI_MAX_RETRIES_PER_MODEL = 1
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


def _compact(value: Any, max_chars: int = 1800) -> Any:
    """
    Keep the assistant context small enough for a fast LLM request while
    preserving the structured project/ML fields needed for arbitrary questions.
    This is context compaction, not keyword-based intent routing.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            # Historical/raw detail lists can become very large. Keep a few
            # representative records while preserving their schema.
            if isinstance(item, list):
                result[key] = [_compact(v, 1200) for v in item[:5]]
            else:
                result[key] = _compact(item, max_chars)
        return result

    if isinstance(value, list):
        return [_compact(v, max_chars) for v in value[:5]]

    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "..."

    return value



def build_assistant_context(
    question: str,
    analysis: Optional[Dict[str, Any]] = None,
    projects: Optional[List[Dict[str, Any]]] = None,
    project: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a factual context packet. Never invents missing analysis."""
    a = analysis if isinstance(analysis, dict) else {}
    # Preserve every analysis section so arbitrary project-related questions
    # can be answered. Large lists/text are compacted rather than discarded.
    payload = {
        "user_question": question[:2000],
        "project_record": _compact(project or a.get("project") or {}, max_chars=1800),
        "project_analysis": _compact(a, max_chars=1800),
        "comparison_projects": _compact((projects or [])[:5], 1200),
    }
    context = json.dumps(
        payload,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )

    # Hard upper bound protects Render/Gemini latency when historical records
    # contain unusually large stored text.
    if len(context) > 24000:
        context = context[:24000] + "..."
    return context


_HTTP_CLIENT = None
_HTTP_CLIENT_LOCK = Lock()


def _get_http_client():
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        with _HTTP_CLIENT_LOCK:
            if _HTTP_CLIENT is None:
                _HTTP_CLIENT = httpx.Client(
                    http2=False,
                    trust_env=False,
                    timeout=httpx.Timeout(connect=2.0, read=18.0, write=2.0, pool=2.0),
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
First understand what the user is asking semantically.
Then select only the minimum relevant facts needed to answer that question.
Do not dump the entire project_analysis into the answer.

Relevance guidance (these are not keyword rules; use the complete meaning of the question):
- For risk questions, prioritize risk level, risk score, reason, detected issues, risk-related warnings, and supplied recommendations.
- For cost questions, prioritize predicted final cost, cost overrun, expected range, confidence, and cost-escalation analysis.
- For schedule questions, prioritize predicted delay, expected delay range, planned duration, confidence, and schedule warnings.
- For historical or comparison questions, prioritize comparable-project evidence, similarity, and benchmarking data.
- For general project questions, prioritize project identity and recorded project details.
- For management/action questions, prioritize the supplied recommended solution and the indicators that justify it.
- For summaries, synthesize the most decision-relevant available facts instead of repeating the full context.

If a requested category exists in the supplied data, answer it directly. If it does not exist, say that
the relevant information is unavailable rather than substituting unrelated project facts.\n\nReturn ONLY the answer text.
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


def _generate(
    prompt: str,
    system_instruction: str,
    model: str,
    api_key: str,
    max_tokens: int,
) -> str:
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
        },
    }

    # Timeouts are not retried: retrying an 18s read timeout can double the
    # request latency. Transient HTTP 5xx responses get one short retry.
    import time

    for attempt in range(GEMINI_MAX_RETRIES_PER_MODEL + 1):
        try:
            response = _get_http_client().post(
                GEMINI_API_URL.format(model=model),
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.ReadTimeout as exc:
            print(
                f"[PROJECT_ASSISTANT] Gemini read timeout "
                f"model={model}: {exc}"
            )
            raise AssistantProviderError(
                f"Unable to reach the configured LLM provider (ReadTimeout)."
            ) from exc
        except httpx.ConnectTimeout as exc:
            print(
                f"[PROJECT_ASSISTANT] Gemini connect timeout "
                f"model={model}: {exc}"
            )
            raise AssistantProviderError(
                f"Unable to reach the configured LLM provider (ConnectTimeout)."
            ) from exc
        except httpx.HTTPError as exc:
            error_type = type(exc).__name__
            print(
                f"[PROJECT_ASSISTANT] Gemini transport error "
                f"model={model}: {error_type}: {exc}"
            )
            raise AssistantProviderError(
                f"Unable to reach the configured LLM provider ({error_type})."
            ) from exc

        if response.status_code >= 400:
            detail = _provider_error(response)
            print(
                f"[PROJECT_ASSISTANT] Gemini HTTP {response.status_code} "
                f"model={model} attempt={attempt + 1}: {detail}"
            )

            # Quota/auth errors should fail immediately.
            if response.status_code in (401, 403, 429):
                raise AssistantProviderError(
                    f"LLM provider returned HTTP {response.status_code}: {detail}"
                )

            # Retry only transient server-side errors once.
            if (
                response.status_code in GEMINI_RETRYABLE_STATUS_CODES
                and attempt < GEMINI_MAX_RETRIES_PER_MODEL
            ):
                time.sleep(0.6)
                continue

            raise AssistantProviderError(
                f"LLM provider returned HTTP {response.status_code}: {detail}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise AssistantProviderError(
                "The LLM provider returned invalid JSON."
            ) from exc

        return _extract_answer(data)

    raise AssistantProviderError("Gemini request failed after retry.")


def _fit_exact_word_count(text: str, count: int) -> str:
    """Emergency-only formatter. Never repeats filler or invents project facts."""
    words = (text or "").split()
    if len(words) <= count:
        return " ".join(words)
    return " ".join(words[:count]).rstrip(".,;:")


def _model_candidates() -> List[str]:
    configured = _get_model()
    models = [configured]
    for candidate in FALLBACK_GEMINI_MODELS:
        if candidate not in models:
            models.append(candidate)
    return models


def ask_project_assistant(
    question: str,
    analysis: Optional[Dict[str, Any]] = None,
    projects: Optional[List[Dict[str, Any]]] = None,
    project: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    question = (question or "").strip()
    if not question:
        raise ValueError("message is required.")

    api_key = _get_api_key()
    context = build_assistant_context(question, analysis, projects, project)
    requested = _extract_requested_word_count(question)

    prompt = (
        "The user is asking a question about the supplied infrastructure project. "
        "Understand the question semantically and answer it directly. Select only the "
        "evidence needed for that question from the supplied context. Do not paste a "
        "generic project description when the question asks about a specific topic. "
        "If the requested information is absent, explicitly say it is unavailable in "
        "the supplied project analysis.\n\n"
        + context
    )

    last_error = None
    answer = None
    used_model = None

    for model in _model_candidates():
        try:
            answer = _generate(
                prompt=prompt,
                system_instruction=_SYSTEM,
                model=model,
                api_key=api_key,
                max_tokens=max(300, (requested or 160) * 4),
            )
            used_model = model
            break
        except AssistantProviderError as exc:
            last_error = exc
            print(
                f"[PROJECT_ASSISTANT] model failed: {model}: "
                f"{type(exc).__name__}: {exc}"
            )

            message = str(exc).lower()

            # Quota/auth/configuration errors should not be hidden by
            # cycling through more models. Return the provider error quickly.
            if "http 429" in message or "http 401" in message or "http 403" in message:
                raise AssistantProviderError(
                    f"Gemini provider error on {model}: {exc}"
                ) from exc

            # For transient 5xx errors and timeouts, move to the next
            # stable fallback after _generate has already performed one
            # short retry. This avoids the old 15s timeout + long model chain.
            continue

    if answer is None:
        raise AssistantProviderError(
            f"All configured Gemini models failed. Last error: {last_error}"
        )

    if requested is not None and len(answer.split()) != requested:
        repair_prompt = (
            f"Requested exact word count: {requested}\n"
            f"Original user question: {question}\n"
            f"Project data: {context}\n"
            f"Draft answer: {answer}\n\n"
            "Rewrite the draft so it answers the original question naturally in exactly "
            "the requested number of whitespace-separated words. Keep only facts relevant "
            "to the original question. Do not add unsupported facts or unrelated fields. "
            "Return only the answer."
        )
        try:
            repaired = _generate(
                prompt=repair_prompt,
                system_instruction=(
                    "You are an exact-length answer editor. Preserve factual meaning from "
                    "the supplied data. Return only a natural answer with exactly the requested "
                    "number of whitespace-separated words. Never invent facts."
                ),
                model=used_model,
                api_key=api_key,
                max_tokens=max(80, requested * 3),
            )
            if len(repaired.split()) == requested:
                answer = repaired
            else:
                answer = _fit_exact_word_count(repaired, requested)
        except AssistantProviderError:
            # Keep the successful primary answer rather than replacing it with a fallback.
            pass

    return {"answer": answer, "provider": "gemini", "model": used_model}

