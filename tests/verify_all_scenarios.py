"""
End-to-end verification script for all 6 required scenarios:
1. Gemini available -> Gemini AI Insight returned, Reason != AI_Insight
2. Gemini quota/error simulated -> Local rule-based AI Insight returned, Reason != AI_Insight
3. High-risk event -> Actionable high-risk phrasing generated, Reason != AI_Insight
4. Low-confidence event -> Actionable low-confidence phrasing generated, Reason != AI_Insight
5. Normal event -> Normal telemetry phrasing generated, Reason != AI_Insight
6. Repeated same event -> Reuses cached insight, deduplicates calls
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_reasoning import (
    get_llm_reasoning,
    trigger_async_llm_reasoning,
    get_cached_insight,
    make_event_key,
    reset_quota_status,
    is_quota_exhausted,
    generate_local_insight,
    _AI_INSIGHTS_CACHE,
    _IN_FLIGHT_KEYS
)
from agents.orchestrator import run_pipeline


def run_all_checks():
    print("=" * 70)
    print("PREDICTIVE MAINTENANCE AGENT: AI INSIGHTS VERIFICATION")
    print("=" * 70)

    # 1. Gemini Available
    print("\n[Scenario 1] Gemini Available:")
    reset_quota_status()
    _AI_INSIGHTS_CACHE.clear()
    _IN_FLIGHT_KEYS.clear()

    diag_high = {
        "prediction": 1,
        "confidence": 0.88,
        "explanation": "Torque [Nm] increased failure risk; Tool wear [min] increased failure risk",
        "plain_explanation": "Torque [Nm] increased failure risk; Tool wear [min] increased failure risk",
    }
    rec_high = {
        "action": "Schedule immediate maintenance",
        "urgency": "High",
        "confidence": 0.88,
        "explanation": diag_high["explanation"],
    }

    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.text = "Gemini Insight: High torque and tool wear indicate critical failure risk; immediate maintenance recommended."
    mock_client.models.generate_content.return_value = mock_res

    with patch("llm_reasoning.get_client", return_value=mock_client):
        gemini_insight = get_llm_reasoning(diag_high, rec_high)
        print(f"  Reason (SHAP)  : {diag_high['explanation']}")
        print(f"  AI Insight (LLM): {gemini_insight}")
        assert gemini_insight == mock_res.text
        assert gemini_insight != diag_high["explanation"]
        print("  -> PASSED: Gemini insight returned and Reason != AI_Insight")

    # 2. Gemini Quota / Error Simulated (429 RESOURCE_EXHAUSTED)
    print("\n[Scenario 2] Gemini Quota / Error Simulated:")
    reset_quota_status()
    _AI_INSIGHTS_CACHE.clear()
    _IN_FLIGHT_KEYS.clear()

    mock_client_quota = MagicMock()
    mock_client_quota.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: quota exceeded for model")

    with patch("llm_reasoning.get_client", return_value=mock_client_quota):
        fallback_insight = get_llm_reasoning(diag_high, rec_high)
        print(f"  Reason (SHAP)  : {diag_high['explanation']}")
        print(f"  AI Insight (Fallback): {fallback_insight}")
        assert is_quota_exhausted(), "Quota exhausted flag should be True"
        assert fallback_insight != diag_high["explanation"], "AI Insight must NEVER equal raw SHAP Reason!"
        assert "Torque [Nm] and Tool wear [min]" in fallback_insight
        assert "schedule immediate maintenance" in fallback_insight.lower()
        print("  -> PASSED: Circuit breaker tripped, local rule-based insight generated, Reason != AI_Insight")

    # 3. High-Risk Event
    print("\n[Scenario 3] High-Risk Event:")
    high_risk_insight = generate_local_insight(diag_high, rec_high)
    print(f"  Reason (SHAP)  : {diag_high['explanation']}")
    print(f"  AI Insight     : {high_risk_insight}")
    assert high_risk_insight != diag_high["explanation"]
    assert "88%" in high_risk_insight
    assert "significant equipment stress" in high_risk_insight
    assert "Prioritize inspection" in high_risk_insight
    print("  -> PASSED: Actionable high-risk phrasing correctly generated, Reason != AI_Insight")

    # 4. Low-Confidence Event
    print("\n[Scenario 4] Low-Confidence Event:")
    diag_low = {
        "prediction": 0,
        "confidence": 0.32,
        "explanation": "Rotational speed [rpm] increased failure risk",
        "plain_explanation": "Rotational speed [rpm] increased failure risk",
    }
    rec_low = {
        "action": "No action needed",
        "urgency": "Low",
        "confidence": 0.32,
        "explanation": diag_low["explanation"],
    }
    low_conf_insight = generate_local_insight(diag_low, rec_low)
    print(f"  Reason (SHAP)  : {diag_low['explanation']}")
    print(f"  AI Insight     : {low_conf_insight}")
    assert low_conf_insight != diag_low["explanation"]
    assert "32%" in low_conf_insight
    assert "evidence for imminent failure remains low" in low_conf_insight
    assert "monitoring" in low_conf_insight.lower()
    print("  -> PASSED: Actionable low-confidence monitoring phrasing generated, Reason != AI_Insight")

    # 5. Normal Event
    print("\n[Scenario 5] Normal Event:")
    diag_norm = {
        "prediction": 0,
        "confidence": 0.04,
        "explanation": "",
        "plain_explanation": "",
    }
    rec_norm = {
        "action": "No action needed",
        "urgency": "Low",
        "confidence": 0.04,
        "explanation": "",
    }
    norm_insight = generate_local_insight(diag_norm, rec_norm)
    print(f"  Reason (SHAP)  : {diag_norm['explanation']}")
    print(f"  AI Insight     : {norm_insight}")
    assert "normal tolerances" in norm_insight
    assert "4%" in norm_insight
    print("  -> PASSED: Normal telemetry phrasing generated")

    # 6. Repeated Same Event
    print("\n[Scenario 6] Repeated Same Event (Caching & Deduplication):")
    reset_quota_status()
    _AI_INSIGHTS_CACHE.clear()
    _IN_FLIGHT_KEYS.clear()

    mock_client_repeat = MagicMock()
    mock_res_rep = MagicMock()
    mock_res_rep.text = "Cached async LLM insight for repeat testing."
    mock_client_repeat.models.generate_content.return_value = mock_res_rep

    with patch("llm_reasoning.get_client", return_value=mock_client_repeat):
        event_id = "test_event_100"
        key = make_event_key(event_id, diag_high, rec_high)

        # Trigger 1
        res1 = trigger_async_llm_reasoning(event_id, diag_high, rec_high)
        # Should return local insight immediately
        assert res1 != diag_high["explanation"]

        # Wait for background worker
        for _ in range(50):
            cached = get_cached_insight(key)
            if cached:
                break
            time.sleep(0.02)

        assert get_cached_insight(key) == mock_res_rep.text

        # Trigger 2 (same event)
        res2 = trigger_async_llm_reasoning(event_id, diag_high, rec_high)
        assert res2 == mock_res_rep.text
        # Assert API was only called once
        assert mock_client_repeat.models.generate_content.call_count == 1
        print("  -> PASSED: Deduplicated and reused cached insight without repeating API calls")

    print("\n" + "=" * 70)
    print("ALL 6 SCENARIOS VERIFIED SUCCESSFULLY! Reason and AI Insight are NEVER identical.")
    print("=" * 70)


if __name__ == "__main__":
    run_all_checks()
