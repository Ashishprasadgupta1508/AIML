import json
import os
import re
from threading import Lock
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_OPENROUTER_MODEL = "google/gemini-3.6-flash"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MAX_RETRIES = 1
OPENROUTER_RETRYABLE_STATUS_CODES = {408, 500, 502, 503, 504}


class AssistantConfigurationError(Exception):
    pass


class AssistantProviderError(Exception):
    pass


def _get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise AssistantConfigurationError(
            "OPENROUTER_API_KEY is not configured on the AI/ML backend."
        )
    return api_key


def _get_model() -> str:
    return (
        os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip()
        or DEFAULT_OPENROUTER_MODEL
    )


def _compact(value: Any, max_chars: int = 1800) -> Any:
    """Bound the prompt while preserving structured project/ML evidence."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
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

    if len(context) > 14000:
        context = context[:14000] + "..."
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
                    timeout=httpx.Timeout(
                        connect=3.0,
                        read=18.0,
                        write=3.0,
                        pool=3.0,
                    ),
                    limits=httpx.Limits(
                        max_connections=10,
                        max_keepalive_connections=5,
                    ),
                )
    return _HTTP_CLIENT


_SYSTEM = """
You are the AI Project Intelligence Assistant for Gati infrastructure projects.
You are a genuine conversational AI analyst. Do not use keyword-based intent routing,
fixed answers, menus, or canned responses.

Interpret the user's complete question and answer that question using ONLY the supplied
project_record, project_analysis, and comparison_projects. Treat supplied data as the
source of truth. Do not invent missing values and do not infer a risk, cost, delay,
recommendation, location, date, or historical conclusion when that information is not
present in the supplied data.

Important distinction:
- Recorded project facts are not predictions.
- ML predictions are not recorded facts.
- Historical/comparable evidence must be described as historical evidence.
- Recommendations must come from supplied model recommendations or clearly be described
  as validation/management actions supported by the available data.

Answer the user's actual question. Do not paste the same project summary for unrelated
questions. Do not force headings, tables, risk sections, or recommendations unless they
help answer the question.

If the requested information is not present, explicitly say that the supplied project
information/analysis does not contain it. Do not replace missing information with a
generic project description.

If the user requests exactly N words, return exactly N whitespace-separated words. Make
the answer natural and meaningful. Never repeat filler, truncate a sentence, append a
word-count note, or invent facts to reach the count.

Use concise, professional infrastructure-management language. First understand what the
user is asking semantically, then select only the minimum relevant facts needed to answer.
Do not dump the entire context into the answer.

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
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("No choices returned")

        message = choices[0].get("message") or {}
        content = message.get("content")

        if isinstance(content, str):
            answer = content.strip()
        elif isinstance(content, list):
            answer = "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("text")
            ).strip()
        else:
            answer = ""

        if not answer:
            raise ValueError("Empty assistant content")
        return answer
    except (AttributeError, TypeError, ValueError, IndexError, KeyError) as exc:
        raise AssistantProviderError(
            "The OpenRouter provider returned an unexpected response."
        ) from exc


def _provider_error(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])[:700]
            if error:
                return str(error)[:700]
    except Exception:
        pass
    return response.text[:700] or f"HTTP {response.status_code}"


def _generate(
    prompt: str,
    system_instruction: str,
    model: str,
    api_key: str,
    max_tokens: int,
    reasoning_enabled: bool = True,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "reasoning": {
            "enabled": reasoning_enabled,
        },
        "max_tokens": min(max_tokens, 384),
        "temperature": 0.2,
        "stream": False,
    }

    import time

    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        try:
            response = _get_http_client().post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://aiml-77m5.onrender.com",
                    "X-Title": "Gati Infrastructure Project Intelligence Assistant",
                },
                json=payload,
            )
        except httpx.ReadTimeout as exc:
            print(
                f"[PROJECT_ASSISTANT] OpenRouter read timeout "
                f"model={model}: {exc}"
            )
            raise AssistantProviderError(
                "Unable to reach the configured LLM provider (ReadTimeout)."
            ) from exc
        except httpx.ConnectTimeout as exc:
            print(
                f"[PROJECT_ASSISTANT] OpenRouter connect timeout "
                f"model={model}: {exc}"
            )
            raise AssistantProviderError(
                "Unable to reach the configured LLM provider (ConnectTimeout)."
            ) from exc
        except httpx.HTTPError as exc:
            error_type = type(exc).__name__
            print(
                f"[PROJECT_ASSISTANT] OpenRouter transport error "
                f"model={model}: {error_type}: {exc}"
            )
            raise AssistantProviderError(
                f"Unable to reach the configured LLM provider ({error_type})."
            ) from exc

        if response.status_code >= 400:
            detail = _provider_error(response)
            print(
                f"[PROJECT_ASSISTANT] OpenRouter HTTP {response.status_code} "
                f"model={model} attempt={attempt + 1}: {detail}"
            )

            # Authentication and quota errors should fail immediately rather
            # than wasting time on a retry that cannot succeed.
            if response.status_code in (401, 402, 403, 429):
                raise AssistantProviderError(
                    f"LLM provider returned HTTP {response.status_code}: {detail}"
                )

            if (
                response.status_code in OPENROUTER_RETRYABLE_STATUS_CODES
                and attempt < OPENROUTER_MAX_RETRIES
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

    raise AssistantProviderError("OpenRouter request failed after retry.")


def _fit_exact_word_count(text: str, count: int) -> str:
    words = (text or "").split()
    if len(words) <= count:
        return " ".join(words)
    return " ".join(words[:count]).rstrip(".,;:")


def ask_project_assistant(
    question: str,
    analysis: Optional[Dict[str, Any]] = None,
    projects: Optional[List[Dict[str, Any]]] = None,
    project: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a genuine semantic project answer through OpenRouter/Gemini."""
    question = (question or "").strip()
    if not question:
        raise ValueError("message is required.")

    api_key = _get_api_key()
    model = _get_model()
    context = build_assistant_context(
        question=question,
        analysis=analysis,
        projects=projects,
        project=project,
    )
    requested = _extract_requested_word_count(question)

    prompt = (
        "The user is asking a question about the supplied infrastructure project. "
        "Understand the question semantically and answer it directly. Select only the "
        "evidence needed for that question from the supplied context. Do not paste a "
        "generic project description when the question asks about a specific topic. "
        "If the requested information is absent, explicitly say it is unavailable in "
        "the supplied project information or analysis.\n\n"
        + context
    )

    # Keep the normal/general-answer path unchanged. Exact word-count requests
    # use a separate output budget and validation path. Gemini 3.6 Flash via
    # this OpenRouter route requires reasoning to remain enabled.
    answer = _generate(
        prompt=prompt,
        system_instruction=_SYSTEM,
        model=model,
        api_key=api_key,
        max_tokens=(
            max(128, min(requested * 8, 256))
            if requested is not None
            else max(220, min((requested or 120) * 3, 384))
        ),
        reasoning_enabled=True,
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
                model=model,
                api_key=api_key,
                max_tokens=max(128, min(requested * 8, 256)),
                reasoning_enabled=True,
            )
            if len(repaired.split()) == requested:
                answer = repaired
            else:
                answer = _fit_exact_word_count(repaired, requested)
        except AssistantProviderError:
            pass

    return {
        "answer": answer,
        "provider": "openrouter",
        "model": model,
    }
