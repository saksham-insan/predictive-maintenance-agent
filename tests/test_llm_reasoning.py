"""
Unit and integration tests for the LLM Reasoning module, local rule-based fallback,
caching, deduplication, and agent pipeline.
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

# Add src to python path
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
from agents.recommendation_agent import recommendation_agent


class TestLLMReasoning(unittest.TestCase):
    def setUp(self):
        self.sample_diagnosis = {
            "prediction": 1,
            "confidence": 0.85,
            "explanation": "Torque [Nm] increased failure risk; Tool wear [min] increased failure risk",
            "plain_explanation": "Torque [Nm] increased failure risk; Tool wear [min] increased failure risk",
        }
        self.sample_recommendation = {
            "action": "Schedule immediate maintenance",
            "urgency": "High",
            "confidence": 0.85,
            "explanation": self.sample_diagnosis["explanation"],
        }
        _AI_INSIGHTS_CACHE.clear()
        _IN_FLIGHT_KEYS.clear()
        reset_quota_status()

    def test_local_insight_high_risk(self):
        """Test that generate_local_insight creates an actionable high-risk insight with risk factors and confidence."""
        insight = generate_local_insight(self.sample_diagnosis, self.sample_recommendation)
        self.assertIn("Torque [Nm] and Tool wear [min]", insight)
        self.assertIn("85%", insight)
        self.assertIn("schedule immediate maintenance", insight.lower())
        self.assertIn("Prioritize inspection", insight)
        # Ensure it is NOT identical to the raw technical SHAP string
        self.assertNotEqual(insight, self.sample_diagnosis["explanation"])

    def test_local_insight_low_confidence(self):
        """Test that generate_local_insight creates distinct wording for low-confidence anomalies recommending monitoring."""
        low_conf_diagnosis = {
            "prediction": 0,
            "confidence": 0.35,
            "explanation": "Tool wear [min] increased failure risk",
            "plain_explanation": "Tool wear [min] increased failure risk",
        }
        low_conf_recommendation = {
            "action": "No action needed",
            "urgency": "Low",
            "confidence": 0.35,
            "explanation": low_conf_diagnosis["explanation"],
        }
        insight = generate_local_insight(low_conf_diagnosis, low_conf_recommendation)
        self.assertIn("Tool wear [min]", insight)
        self.assertIn("35%", insight)
        self.assertIn("monitoring", insight.lower())
        self.assertIn("evidence for imminent failure remains low", insight)
        self.assertNotEqual(insight, low_conf_diagnosis["explanation"])

    def test_local_insight_moderate_risk(self):
        """Test that generate_local_insight handles moderate confidence / medium urgency cases."""
        mod_diagnosis = {
            "prediction": 1,
            "confidence": 0.55,
            "explanation": "Torque [Nm] increased failure risk",
            "plain_explanation": "Torque [Nm] increased failure risk",
        }
        mod_recommendation = {
            "action": "Schedule maintenance within 48 hours",
            "urgency": "Medium",
            "confidence": 0.55,
            "explanation": mod_diagnosis["explanation"],
        }
        insight = generate_local_insight(mod_diagnosis, mod_recommendation)
        self.assertIn("Torque [Nm]", insight)
        self.assertIn("55%", insight)
        self.assertIn("48 hours", insight)
        self.assertNotEqual(insight, mod_diagnosis["explanation"])

    def test_local_insight_normal(self):
        """Test that generate_local_insight handles normal readings."""
        normal_diagnosis = {
            "prediction": 0,
            "confidence": 0.05,
            "explanation": "",
            "plain_explanation": "",
        }
        normal_recommendation = {
            "action": "No action needed",
            "urgency": "Low",
            "confidence": 0.05,
            "explanation": "",
        }
        insight = generate_local_insight(normal_diagnosis, normal_recommendation)
        self.assertIn("normal tolerances", insight)
        self.assertIn("5%", insight)

    def test_fallback_when_no_api_key(self):
        """Test that get_llm_reasoning falls back to dynamic local insight when GEMINI_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_llm_reasoning(self.sample_diagnosis, self.sample_recommendation)
            expected_local = generate_local_insight(self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(result, expected_local)
            self.assertNotEqual(result, self.sample_diagnosis["explanation"])

    def test_fallback_on_api_exception(self):
        """Test that get_llm_reasoning safely catches any API error and returns local insight."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("General network error")

        with patch("llm_reasoning.get_client", return_value=mock_client):
            result = get_llm_reasoning(self.sample_diagnosis, self.sample_recommendation)
            expected_local = generate_local_insight(self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(result, expected_local)
            self.assertNotEqual(result, self.sample_diagnosis["explanation"])

    def test_quota_exhaustion_circuit_breaker(self):
        """Test that HTTP 429 / RESOURCE_EXHAUSTED marks session as quota exhausted, returns local insight, and prevents repeated calls."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: quota exceeded for model")

        with patch("llm_reasoning.get_client", return_value=mock_client):
            expected_local = generate_local_insight(self.sample_diagnosis, self.sample_recommendation)

            # First call triggers 429, returns local insight, and marks quota exhausted
            result1 = get_llm_reasoning(self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(result1, expected_local)
            self.assertNotEqual(result1, self.sample_diagnosis["explanation"])
            self.assertTrue(is_quota_exhausted())
            self.assertEqual(mock_client.models.generate_content.call_count, 1)

            # Subsequent call immediately short-circuits without calling API
            result2 = get_llm_reasoning(self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(result2, expected_local)
            # Call count should STILL be 1!
            self.assertEqual(mock_client.models.generate_content.call_count, 1)

            # Async trigger also immediately short-circuits with local insight
            result3 = trigger_async_llm_reasoning("t=99", self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(result3, expected_local)
            self.assertEqual(mock_client.models.generate_content.call_count, 1)

    def test_fallback_on_timeout(self):
        """Test that get_llm_reasoning safely times out and returns the local insight."""
        mock_client = MagicMock()

        def slow_generate(*args, **kwargs):
            time.sleep(0.5)
            mock_res = MagicMock()
            mock_res.text = "Slow response"
            return mock_res

        mock_client.models.generate_content.side_effect = slow_generate

        with patch("llm_reasoning.get_client", return_value=mock_client):
            result = get_llm_reasoning(self.sample_diagnosis, self.sample_recommendation, timeout=0.05)
            expected_local = generate_local_insight(self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(result, expected_local)
            self.assertNotEqual(result, self.sample_diagnosis["explanation"])

    def test_successful_llm_call(self):
        """Test that get_llm_reasoning returns the Gemini-generated summary when API succeeds."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "High torque and tool wear indicate critical impending failure; inspect machine immediately."
        mock_client.models.generate_content.return_value = mock_response

        with patch("llm_reasoning.get_client", return_value=mock_client):
            result = get_llm_reasoning(self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(
                result,
                "High torque and tool wear indicate critical impending failure; inspect machine immediately."
            )
            self.assertNotEqual(result, self.sample_diagnosis["explanation"])

    def test_recommendation_agent_is_non_blocking_and_preserves_plain_explanation(self):
        """Test that recommendation_agent attaches plain_explanation and does not block."""
        diagnosis_copy = dict(self.sample_diagnosis)
        rec = recommendation_agent(diagnosis_copy)

        self.assertIn("plain_explanation", rec)
        self.assertIn("plain_explanation", diagnosis_copy)
        self.assertEqual(rec["plain_explanation"], self.sample_diagnosis["explanation"])
        self.assertEqual(diagnosis_copy["plain_explanation"], self.sample_diagnosis["explanation"])
        self.assertEqual(rec["action"], "Schedule immediate maintenance")
        self.assertEqual(rec["urgency"], "High")

    def test_async_llm_reasoning_and_deduplication(self):
        """Test that async reasoning returns local insight immediately, updates cache, and prevents duplicate API calls."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Gemini async insight text."
        mock_client.models.generate_content.return_value = mock_response

        with patch("llm_reasoning.get_client", return_value=mock_client):
            event_id = "t=42"
            key = make_event_key(event_id, self.sample_diagnosis, self.sample_recommendation)
            expected_local = generate_local_insight(self.sample_diagnosis, self.sample_recommendation)

            # First trigger: returns local insight immediately without waiting
            first_return = trigger_async_llm_reasoning(event_id, self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(first_return, expected_local)
            self.assertNotEqual(first_return, self.sample_diagnosis["explanation"])

            # Wait briefly for worker thread to complete
            for _ in range(50):
                cached = get_cached_insight(key)
                if cached:
                    break
                time.sleep(0.02)

            self.assertEqual(get_cached_insight(key), "Gemini async insight text.")

            # Second trigger with same event: returns cached result immediately without duplicate API call
            second_return = trigger_async_llm_reasoning(event_id, self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(second_return, "Gemini async insight text.")

            # Ensure API was only called once
            self.assertEqual(mock_client.models.generate_content.call_count, 1)

    def test_reason_and_ai_insight_never_identical_across_all_cases(self):
        """Comprehensive verification that Reason (SHAP) and AI_Insight are never identical in any scenario."""
        cases = [
            # High risk
            ({
                "prediction": 1,
                "confidence": 0.92,
                "explanation": "Torque [Nm] increased failure risk; Tool wear [min] increased failure risk",
                "plain_explanation": "Torque [Nm] increased failure risk; Tool wear [min] increased failure risk",
            }, {
                "action": "Schedule immediate maintenance",
                "urgency": "High",
            }),
            # Medium risk
            ({
                "prediction": 1,
                "confidence": 0.58,
                "explanation": "Rotational speed [rpm] increased failure risk",
                "plain_explanation": "Rotational speed [rpm] increased failure risk",
            }, {
                "action": "Schedule maintenance within 48 hours",
                "urgency": "Medium",
            }),
            # Low confidence
            ({
                "prediction": 0,
                "confidence": 0.28,
                "explanation": "Process temperature [K] increased failure risk",
                "plain_explanation": "Process temperature [K] increased failure risk",
            }, {
                "action": "No action needed",
                "urgency": "Low",
            }),
        ]

        for diag, rec in cases:
            reason = diag["explanation"]
            local_insight = generate_local_insight(diag, rec)
            self.assertNotEqual(reason, local_insight, f"Reason and local insight were identical for diag: {diag}")

    def test_csv_export_structure(self):
        """Test that high-risk and low-confidence export rows contain all required columns including AI_Insight."""
        high_rows = [{
            "Time": 5,
            "Confidence": "85%",
            "Action": "Schedule immediate maintenance",
            "Reason": "Torque [Nm] increased failure risk; Tool wear [min] increased failure risk",
            "AI_Insight": "Elevated Torque [Nm] and Tool wear [min] indicates significant equipment stress.",
            "_key": "k1"
        }]
        high_export = [
            {
                "Time": r["Time"],
                "Confidence": r["Confidence"],
                "Action": r.get("Action", ""),
                "Reason": r["Reason"],
                "AI_Insight": r.get("AI_Insight", "")
            }
            for r in high_rows
        ]
        high_df = pd.DataFrame(high_export)
        self.assertEqual(list(high_df.columns), ["Time", "Confidence", "Action", "Reason", "AI_Insight"])
        self.assertEqual(high_df["Reason"].iloc[0], "Torque [Nm] increased failure risk; Tool wear [min] increased failure risk")
        self.assertEqual(high_df["AI_Insight"].iloc[0], "Elevated Torque [Nm] and Tool wear [min] indicates significant equipment stress.")
        self.assertNotEqual(high_df["Reason"].iloc[0], high_df["AI_Insight"].iloc[0])

        low_rows = [{
            "Time": 8,
            "Confidence": "35%",
            "Prediction": "Failure",
            "Reason": "Torque [Nm] increased failure risk",
            "AI_Insight": "Current readings show minor variances in Torque [Nm], but evidence for imminent failure remains low (35% confidence).",
            "_key": "k2"
        }]
        low_export = [
            {
                "Time": r["Time"],
                "Confidence": r["Confidence"],
                "Prediction": r.get("Prediction", ""),
                "Reason": r["Reason"],
                "AI_Insight": r.get("AI_Insight", "")
            }
            for r in low_rows
        ]
        low_df = pd.DataFrame(low_export)
        self.assertEqual(list(low_df.columns), ["Time", "Confidence", "Prediction", "Reason", "AI_Insight"])
        self.assertEqual(low_df["Reason"].iloc[0], "Torque [Nm] increased failure risk")
        self.assertEqual(low_df["AI_Insight"].iloc[0], "Current readings show minor variances in Torque [Nm], but evidence for imminent failure remains low (35% confidence).")
        self.assertNotEqual(low_df["Reason"].iloc[0], low_df["AI_Insight"].iloc[0])


if __name__ == "__main__":
    unittest.main()


