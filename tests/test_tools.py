"""
Unit tests for agent tools and persistent JSONL audit trail.
"""

import os
import sys
import json
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from agents.tools import (
    create_maintenance_ticket,
    check_spare_parts_inventory,
    escalate_to_engineer,
    request_more_sensor_data,
    execute_tool,
    TOOL_LOG_PATH
)


class TestAgentTools(unittest.TestCase):

    def test_create_ticket(self):
        res = create_maintenance_ticket("LINE_02", "High", "Critical tool wear detected")
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["ticket_id"].startswith("TICK-"))
        self.assertEqual(res["machine_id"], "LINE_02")
        self.assertEqual(res["urgency"], "High")

    def test_check_inventory(self):
        res = check_spare_parts_inventory("tool_insert")
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["found"])
        self.assertGreater(res["details"]["in_stock"], 0)

        # Unmatched part
        res_unknown = check_spare_parts_inventory("non_existent_part_xyz")
        self.assertEqual(res_unknown["status"], "success")
        self.assertFalse(res_unknown["found"])

    def test_escalate_engineer(self):
        res = escalate_to_engineer("LINE_03", "Ambiguous vibration harmonic")
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["escalation_id"].startswith("ESC-"))
        self.assertTrue(res["notification_sent"])

    def test_request_more_sensor_data(self):
        res = request_more_sensor_data("LINE_01", memory_context="Window of 15 readings, wear rising at +2 min/step")
        self.assertEqual(res["status"], "success")
        self.assertIn("Window of 15 readings", res["memory_telemetry"])

    def test_execute_tool_dispatcher(self):
        res = execute_tool("create_maintenance_ticket", {
            "machine_id": "CNC_01",
            "urgency": "Medium",
            "notes": "Test ticket"
        })
        self.assertEqual(res["status"], "success")

        # Unknown tool
        res_err = execute_tool("fake_tool", {})
        self.assertEqual(res_err["status"], "error")

    def test_audit_log_written(self):
        self.assertTrue(os.path.exists(TOOL_LOG_PATH))
        with open(TOOL_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 0)
        last_record = json.loads(lines[-1])
        self.assertIn("timestamp", last_record)
        self.assertIn("tool", last_record)
        self.assertIn("args", last_record)
        self.assertIn("result", last_record)


if __name__ == "__main__":
    unittest.main()
