"""
Unit tests for ReAct LLM orchestrator and agentic pipeline.
"""

import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from memory import MachineMemory
from llm_client import MockLLMClient, BaseLLMClient
from agents.llm_orchestrator import run_agentic_orchestrator
from agents.agentic_pipeline import run_pipeline_agentic


class BrokenJSONClient(BaseLLMClient):
    """Returns invalid non-JSON strings to test failsafe resilience."""
    def chat(self, messages, system=None):
        return "I am an AI assistant and I think the machine is broken, but this is not valid JSON!"


class TestLLMOrchestrator(unittest.TestCase):

    def setUp(self):
        self.mock_client = MockLLMClient()
        self.memory = MachineMemory()

    def test_react_loop_osf_flow(self):
        row = {
            "Type": "M",
            "Air temperature [K]": 300.5,
            "Process temperature [K]": 311.2,
            "Rotational speed [rpm]": 1350,
            "Torque [Nm]": 65.0,
            "Tool wear [min]": 220
        }
        diag = {
            "prediction": 1,
            "confidence": 0.95,
            "explanation": "Torque increased failure risk; Tool wear increased failure risk"
        }
        self.memory.observe(row)

        res = run_agentic_orchestrator(row, diag, self.memory, llm_client=self.mock_client)

        self.assertEqual(res["status"], "anomaly_diagnosed")
        self.assertIn("osf_overstrain_failure.md", res["grounding_sources"])
        self.assertGreater(len(res["reasoning_trace"]), 0)
        self.assertGreater(len(res["tool_calls_made"]), 0)
        self.assertEqual(res["tool_calls_made"][0]["tool_called"], "check_spare_parts_inventory")
        self.assertIn("Overstrain", res["final_answer"]["summary"])

    def test_react_loop_borderline_trend_flow(self):
        # Create borderline readings
        for w in [180, 184, 188, 192, 196]:
            self.memory.observe({
                "Type": "M",
                "Air temperature [K]": 298.0,
                "Process temperature [K]": 308.0,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 45.0,
                "Tool wear [min]": w
            })

        row = {
            "Type": "M",
            "Air temperature [K]": 298.0,
            "Process temperature [K]": 308.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 45.0,
            "Tool wear [min]": 196
        }
        diag = {
            "prediction": 1,
            "confidence": 0.48,
            "explanation": "Tool wear approaching threshold; borderline trend"
        }

        res = run_agentic_orchestrator(row, diag, self.memory, llm_client=self.mock_client)
        self.assertEqual(res["status"], "anomaly_diagnosed")
        self.assertGreater(len(res["tool_calls_made"]), 0)
        self.assertEqual(res["tool_calls_made"][0]["tool_called"], "request_more_sensor_data")

    def test_react_loop_failsafe_recovery(self):
        broken_client = BrokenJSONClient()
        row = {
            "Type": "L",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1400,
            "Torque [Nm]": 50.0,
            "Tool wear [min]": 100
        }
        diag = {"prediction": 1, "confidence": 0.6, "explanation": "General anomaly"}
        mem = MachineMemory()

        # Orchestrator must not crash; it should catch parse errors and trigger failsafe escalation
        res = run_agentic_orchestrator(row, diag, mem, llm_client=broken_client)
        self.assertEqual(res["status"], "anomaly_diagnosed")
        self.assertEqual(res["final_answer"]["action_taken"], "Escalate to Engineer")

    def test_run_pipeline_agentic_normal_fast_path(self):
        normal_row = {
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0
        }
        mem = MachineMemory()
        res = run_pipeline_agentic(normal_row, mem, llm_client=self.mock_client)
        self.assertEqual(res["status"], "normal")
        self.assertEqual(res["action"], "No action needed")
        self.assertEqual(len(res["reasoning_trace"]), 0)


if __name__ == "__main__":
    unittest.main()
