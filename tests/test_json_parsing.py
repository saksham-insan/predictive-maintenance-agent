import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from agents.llm_orchestrator import _clean_and_parse_json


class TestJSONParsing(unittest.TestCase):

    def test_clean_json(self):
        s = '{"action": "final_answer", "summary": "Tool replaced", "reasoning": "Physics verified"}'
        res = _clean_and_parse_json(s)
        self.assertEqual(res["action"], "final_answer")
        self.assertEqual(res["summary"], "Tool replaced")

    def test_markdown_wrapped_json(self):
        s = '```json\n{"action": "call_tool", "tool": "check_spare_parts_inventory", "args": {"part_type": "tool_insert"}, "reasoning": "Need parts"}\n```'
        res = _clean_and_parse_json(s)
        self.assertEqual(res["action"], "call_tool")
        self.assertEqual(res["tool"], "check_spare_parts_inventory")

    def test_truncated_or_unclosed_json(self):
        # Exact malformed case from user screenshot
        s = '{ "action": "final_answer", "summary": "Tool Wear Failure (TWF) detected. Replace worn-out tool insert, inspect spindle shaft, collet chuck, and machine guide rails, and verify work-holding fixture rigidity.", "reasoning": "Based on the physics formulas, trajectory risk evaluation, and tool se)'
        res = _clean_and_parse_json(s)
        self.assertEqual(res["action"], "final_answer")
        self.assertIn("Tool Wear Failure", res["summary"])
        self.assertIn("physics formulas", res["reasoning"])


if __name__ == "__main__":
    unittest.main()
