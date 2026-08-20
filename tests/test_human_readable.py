import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from human_readable import translate_to_human_readable


class TestHumanReadable(unittest.TestCase):

    def test_normal_telemetry(self):
        row = {"Tool wear [min]": 10, "Torque [Nm]": 40, "Rotational speed [rpm]": 1500, "Air temperature [K]": 298, "Process temperature [K]": 308, "Type": "L"}
        res = translate_to_human_readable(row, status="normal")
        self.assertEqual(res["health_status"], "Optimal / Healthy")
        self.assertEqual(res["reliability_score"], "99%")
        self.assertIn("smoothly", res["headline"])

    def test_twf_telemetry(self):
        row = {"Tool wear [min]": 220, "Torque [Nm]": 45, "Rotational speed [rpm]": 1400, "Air temperature [K]": 298, "Process temperature [K]": 308, "Type": "L"}
        diag = {"confidence": 0.88, "prediction": 1, "explanation": "Tool wear increased risk"}
        res = translate_to_human_readable(row, status="anomaly_detected", diagnosis=diag)
        self.assertIn("CRITICAL", res["health_status"])
        self.assertIn("Cutting Tool Wear", res["mode_title"])
        self.assertGreater(len(res["operator_checklist"]), 0)


if __name__ == "__main__":
    unittest.main()
