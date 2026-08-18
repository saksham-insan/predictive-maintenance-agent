"""
Gemini AI Reasoning Module for Predictive Maintenance Agent.

Takes the technical SHAP explanation and diagnosis details and asks Gemini
to produce a clear, natural-language insight for a maintenance manager.

Includes:
- Non-blocking asynchronous execution via background thread pool.
- Event-based caching and deduplication to prevent duplicate API requests.
- Strict timeout handling and fallback to technical SHAP explanation on failure.
"""

import os
import threading
import concurrent.futures
from typing import Optional, Callable

try:
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
except ImportError:
    genai = None

# Thread-safe global cache and in-flight tracking
_AI_INSIGHTS_CACHE: dict[str, str] = {}
_IN_FLIGHT_KEYS: set[str] = set()
_CACHE_LOCK = threading.Lock()
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="gemini_reasoning")

# Session-level quota tracking
_QUOTA_EXHAUSTED: bool = False
_QUOTA_WARNED: bool = False


def reset_quota_status():
    """Resets the quota exhausted flag (useful for testing or application restarts)."""
    global _QUOTA_EXHAUSTED, _QUOTA_WARNED
    with _CACHE_LOCK:
        _QUOTA_EXHAUSTED = False
        _QUOTA_WARNED = False


def is_quota_exhausted() -> bool:
    """Returns True if the session has encountered a quota exhausted error."""
    return _QUOTA_EXHAUSTED


def get_client():
    """
    Returns a Gemini client if the Google GenAI library is installed
    and GEMINI_API_KEY is available.
    """
    if _QUOTA_EXHAUSTED:
        return None

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key or genai is None:
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"[WARN] Failed to initialize Gemini client: {e}")
        return None


def make_event_key(
    event_id: str,
    diagnosis: dict,
    recommendation: dict = None
) -> str:
    """
    Constructs a deterministic cache key for an event based on its
    event ID, prediction, confidence, explanation, and recommended action.
    """
    pred = diagnosis.get("prediction", "")
    conf = diagnosis.get("confidence", "")
    exp = diagnosis.get("explanation", "")
    rec_action = recommendation.get("action", "") if recommendation else ""
    return f"{event_id}|pred:{pred}|conf:{conf}|exp:{exp}|rec:{rec_action}"


def get_cached_insight(event_key: str) -> Optional[str]:
    """
    Returns the cached AI insight for the given event key if ready, or None.
    """
    with _CACHE_LOCK:
        return _AI_INSIGHTS_CACHE.get(event_key)


_CALL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="gemini_api_call")


def _handle_api_exception(exc: Exception) -> bool:
    """
    Checks if an exception is a quota / 429 error and marks the session
    as quota exhausted with a single concise warning.
    Returns True if it was a quota error.
    """
    global _QUOTA_EXHAUSTED, _QUOTA_WARNED
    err_str = str(exc).lower()
    err_repr = repr(exc).lower()
    is_quota = (
        "429" in err_str or "429" in err_repr
        or "resource_exhausted" in err_str or "resource_exhausted" in err_repr
        or "quota" in err_str or "quota" in err_repr
    )
    if is_quota:
        with _CACHE_LOCK:
            _QUOTA_EXHAUSTED = True
            if not _QUOTA_WARNED:
                _QUOTA_WARNED = True
                print("[WARN] Gemini quota exhausted; using SHAP fallback for remaining events.")
        return True
    return False


def get_llm_reasoning(
    diagnosis: dict,
    recommendation: dict = None,
    timeout: float = 6.0
) -> str:
    """
    Converts a technical SHAP explanation into a natural-language
    AI insight for a maintenance manager.

    Falls back to the original SHAP explanation if Gemini is unavailable,
    quota is exhausted, times out, or encounters an error.
    """
    # Existing technical explanation as fallback
    fallback_text = diagnosis.get("plain_explanation") or diagnosis.get("explanation", "")

    # Fast short-circuit if quota is exhausted
    if _QUOTA_EXHAUSTED:
        return fallback_text

    client = get_client()
    if client is None:
        return fallback_text

    # Recommendation
    rec_action = (
        recommendation.get("action", "Inspect machine")
        if recommendation
        else "Inspect machine"
    )

    # Prediction
    prediction = diagnosis.get("prediction")
    pred_text = (
        "Failure predicted"
        if prediction == 1
        else "No failure predicted"
    )

    # Confidence
    conf_val = diagnosis.get("confidence", 0.0)
    if isinstance(conf_val, (int, float)):
        conf_text = f"{conf_val:.0%}"
    else:
        conf_text = str(conf_val)

    prompt = f"""
You are an AI assistant helping a maintenance engineer understand
a predictive-maintenance result.

Convert the technical information below into ONE clear and useful
AI insight for a maintenance manager.

Be specific about:
- what the model detected,
- the main factors contributing to the risk,
- and what the recommended action means.

Do not invent information.
Do not mention SHAP, machine learning, AI, or technical implementation details.
Do not change the prediction.
Do not give a diagnosis that is not supported by the input.

Prediction: {pred_text}
Confidence: {conf_text}

Technical explanation:
{fallback_text}

Recommended action:
{rec_action}

Respond with ONLY one concise sentence.
"""

    def _call_api() -> str:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        if response and getattr(response, "text", None):
            summary = response.text.strip()
            if summary:
                return summary
        return fallback_text

    try:
        # Enforce timeout so slow API responses never freeze execution
        future = _CALL_EXECUTOR.submit(_call_api)
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        print(f"[WARN] Gemini reasoning call timed out after {timeout}s, falling back to SHAP text")
        return fallback_text
    except Exception as e:
        if not _handle_api_exception(e):
            print(f"[WARN] Gemini reasoning call failed, falling back to SHAP text: {e}")
        return fallback_text


def trigger_async_llm_reasoning(
    event_id: str,
    diagnosis: dict,
    recommendation: dict = None,
    callback: Optional[Callable[[str, str, str], None]] = None,
    timeout: float = 6.0
) -> str:
    """
    Triggers an asynchronous, non-blocking Gemini AI reasoning call in the background.
    - If quota is exhausted, immediately returns the fallback SHAP explanation.
    - If already cached, returns the cached result immediately.
    - If already in flight, returns the fallback SHAP explanation immediately without
      starting a duplicate request.
    - Otherwise, starts a background task and immediately returns the fallback SHAP explanation.
    """
    fallback_text = diagnosis.get("plain_explanation") or diagnosis.get("explanation", "")

    # Fast short-circuit if quota is exhausted
    if _QUOTA_EXHAUSTED:
        return fallback_text

    event_key = make_event_key(event_id, diagnosis, recommendation)

    with _CACHE_LOCK:
        if _QUOTA_EXHAUSTED:
            return fallback_text
        if event_key in _AI_INSIGHTS_CACHE:
            return _AI_INSIGHTS_CACHE[event_key]
        if event_key in _IN_FLIGHT_KEYS:
            return fallback_text
        _IN_FLIGHT_KEYS.add(event_key)

    def _worker():
        try:
            insight = get_llm_reasoning(diagnosis, recommendation, timeout=timeout)
            with _CACHE_LOCK:
                _AI_INSIGHTS_CACHE[event_key] = insight
                _IN_FLIGHT_KEYS.discard(event_key)
            if callback:
                try:
                    callback(event_id, event_key, insight)
                except Exception:
                    pass
        except Exception as exc:
            if not _handle_api_exception(exc):
                print(f"[WARN] Background Gemini worker error for {event_key}: {exc}")
            with _CACHE_LOCK:
                _AI_INSIGHTS_CACHE[event_key] = fallback_text
                _IN_FLIGHT_KEYS.discard(event_key)

    _EXECUTOR.submit(_worker)
    return fallback_text