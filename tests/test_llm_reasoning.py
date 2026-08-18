"""
Unit and integration tests for the LLM Reasoning module, caching, deduplication,
and agent pipeline.
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
    _AI_INSIGHTS_CACHE,
    _IN_FLIGHT_KEYS
)
from agents.recommendation_agent import recommendation_agent


class TestLLMReasoning(unittest.TestCase):
    def setUp(self):
        self.sample_diagnosis = {
            "prediction": 1,
            "confidence": 0.85,
            "explanation": "Torque increased failure risk; Tool wear increased failure risk",
            "plain_explanation": "Torque increased failure risk; Tool wear increased failure risk",
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

    def test_fallback_when_no_api_key(self):
        """Test that get_llm_reasoning falls back to raw SHAP explanation when GEMINI_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_llm_reasoning(self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(result, self.sample_diagnosis["explanation"])

    def test_fallback_on_api_exception(self):
        """Test that get_llm_reasoning safely catches any API error and returns the fallback."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("General network error")

        with patch("llm_reasoning.get_client", return_value=mock_client):
            result = get_llm_reasoning(self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(result, self.sample_diagnosis["explanation"])

    def test_quota_exhaustion_circuit_breaker(self):
        """Test that HTTP 429 / RESOURCE_EXHAUSTED marks session as quota exhausted and prevents repeated calls."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: quota exceeded for model")

        with patch("llm_reasoning.get_client", return_value=mock_client):
            # First call triggers 429 and marks quota exhausted
            result1 = get_llm_reasoning(self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(result1, self.sample_diagnosis["explanation"])
            self.assertTrue(is_quota_exhausted())
            self.assertEqual(mock_client.models.generate_content.call_count, 1)

            # Subsequent call immediately short-circuits without calling API
            result2 = get_llm_reasoning(self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(result2, self.sample_diagnosis["explanation"])
            # Call count should STILL be 1!
            self.assertEqual(mock_client.models.generate_content.call_count, 1)

            # Async trigger also immediately short-circuits
            result3 = trigger_async_llm_reasoning("t=99", self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(result3, self.sample_diagnosis["explanation"])
            self.assertEqual(mock_client.models.generate_content.call_count, 1)

    def test_fallback_on_timeout(self):
        """Test that get_llm_reasoning safely times out and returns the fallback."""
        mock_client = MagicMock()

        def slow_generate(*args, **kwargs):
            time.sleep(0.5)
            mock_res = MagicMock()
            mock_res.text = "Slow response"
            return mock_res

        mock_client.models.generate_content.side_effect = slow_generate

        with patch("llm_reasoning.get_client", return_value=mock_client):
            result = get_llm_reasoning(self.sample_diagnosis, self.sample_recommendation, timeout=0.05)
            self.assertEqual(result, self.sample_diagnosis["explanation"])

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
        """Test that async reasoning returns immediately with fallback, updates cache, and prevents duplicate API calls."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Gemini async insight text."
        mock_client.models.generate_content.return_value = mock_response

        with patch("llm_reasoning.get_client", return_value=mock_client):
            event_id = "t=42"
            key = make_event_key(event_id, self.sample_diagnosis, self.sample_recommendation)

            # First trigger: returns fallback immediately without waiting
            first_return = trigger_async_llm_reasoning(event_id, self.sample_diagnosis, self.sample_recommendation)
            self.assertEqual(first_return, self.sample_diagnosis["explanation"])

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

    def test_csv_export_structure(self):
        """Test that high-risk and low-confidence export rows contain all required columns including AI_Insight."""
        high_rows = [{
            "Time": 5,
            "Confidence": "85%",
            "Action": "Schedule immediate maintenance",
            "Reason": "SHAP reason text",
            "AI_Insight": "Gemini natural language text",
            "_key": "k1"
        }]
        high_export = [
            {
                "Time": r["Time"],
                "Confidence": r["Confidence"],
                "Action": r.get("Action", ""),
                "Reason": r["Reason"],
                "AI_Insight": r.get("AI_Insight", r["Reason"])
            }
            for r in high_rows
        ]
        high_df = pd.DataFrame(high_export)
        self.assertEqual(list(high_df.columns), ["Time", "Confidence", "Action", "Reason", "AI_Insight"])
        self.assertEqual(high_df["AI_Insight"].iloc[0], "Gemini natural language text")

        low_rows = [{
            "Time": 8,
            "Confidence": "35%",
            "Prediction": "Failure",
            "Reason": "SHAP reason text",
            "AI_Insight": "Low confidence explanation",
            "_key": "k2"
        }]
        low_export = [
            {
                "Time": r["Time"],
                "Confidence": r["Confidence"],
                "Prediction": r.get("Prediction", ""),
                "Reason": r["Reason"],
                "AI_Insight": r.get("AI_Insight", r["Reason"])
            }
            for r in low_rows
        ]
        low_df = pd.DataFrame(low_export)
        self.assertEqual(list(low_df.columns), ["Time", "Confidence", "Prediction", "Reason", "AI_Insight"])
        self.assertEqual(low_df["AI_Insight"].iloc[0], "Low confidence explanation")


if __name__ == "__main__":
    unittest.main()

