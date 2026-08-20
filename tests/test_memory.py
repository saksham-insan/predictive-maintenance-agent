"""
Unit tests for MachineMemory temporal tracking and tool-change lifecycle reset.
"""

import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from memory import MachineMemory


class TestMachineMemory(unittest.TestCase):

    def test_observe_and_slopes(self):
        mem = MachineMemory(maxlen=10)
        # Rising wear sequence: 100, 105, 110, 115, 120 (slope ~ +5 min/step)
        for w in [100, 105, 110, 115, 120]:
            snap = mem.observe({
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": w
            })

        self.assertEqual(snap.window_size, 5)
        self.assertEqual(snap.tool_wear_current, 120)
        self.assertAlmostEqual(snap.tool_wear_slope, 5.0, places=2)
        self.assertFalse(snap.is_tool_change_detected)
        # 120 min to 200 min at 5 min/step is 16 steps
        self.assertIsNotNone(snap.est_readings_to_twf_band)
        self.assertAlmostEqual(snap.est_readings_to_twf_band, 16.0, places=1)

    def test_tool_change_reset(self):
        mem = MachineMemory(maxlen=10)
        # Feed high wear
        mem.observe({"Tool wear [min]": 210, "Torque [Nm]": 45, "Type": "L", "Air temperature [K]": 298, "Process temperature [K]": 308, "Rotational speed [rpm]": 1500})
        mem.observe({"Tool wear [min]": 215, "Torque [Nm]": 45, "Type": "L", "Air temperature [K]": 298, "Process temperature [K]": 308, "Rotational speed [rpm]": 1500})

        self.assertEqual(len(mem.history), 2)

        # Tool swap: drops from 215 to 0 (>50 min drop)
        snap = mem.observe({"Tool wear [min]": 0, "Torque [Nm]": 35, "Type": "L", "Air temperature [K]": 298, "Process temperature [K]": 308, "Rotational speed [rpm]": 1500})

        self.assertTrue(snap.is_tool_change_detected)
        self.assertEqual(snap.window_size, 1)
        self.assertEqual(snap.tool_wear_current, 0.0)
        self.assertIn("Tool replacement detected", snap.trend_description)

    def test_context_string_formatting(self):
        mem = MachineMemory()
        mem.observe({"Tool wear [min]": 205, "Torque [Nm]": 50, "Type": "H", "Air temperature [K]": 298, "Process temperature [K]": 308, "Rotational speed [rpm]": 1500})
        ctx = mem.context_string()
        self.assertIn("Current Tool Wear: 205.0 min", ctx)
        self.assertIn("CRITICAL: Tool wear at 205.0 min is inside the high-risk TWF band", ctx)


if __name__ == "__main__":
    unittest.main()
