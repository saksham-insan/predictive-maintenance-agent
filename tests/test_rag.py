"""
Unit tests for RAG failure mode knowledge base retrieval.
"""

import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from rag import get_knowledge_base, FailureModeKnowledgeBase


class TestRAGKnowledgeBase(unittest.TestCase):

    def setUp(self):
        self.kb = get_knowledge_base()

    def test_indexing(self):
        self.assertEqual(len(self.kb.documents), 5)
        doc_ids = {d["doc_id"] for d in self.kb.documents}
        expected_ids = {
            "twf_tool_wear_failure",
            "hdf_heat_dissipation_failure",
            "pwf_power_failure",
            "osf_overstrain_failure",
            "rnf_random_failure"
        }
        self.assertEqual(doc_ids, expected_ids)

    def test_ranking_sanity_check(self):
        # Query: tool wear 215 min torque 55 Nm rising, Type M
        # Requirement: should rank OSF and TWF as top two hits
        query = "tool wear 215 min torque 55 Nm rising, Type M"
        results = self.kb.retrieve(query, top_k=2)

        self.assertEqual(len(results), 2)
        top_files = [r.filename for r in results]
        self.assertIn("osf_overstrain_failure.md", top_files)
        self.assertIn("twf_tool_wear_failure.md", top_files)

    def test_hdf_query_ranking(self):
        query = "temperature differential low 7.8 K spindle speed 1250 rpm cooling heat dissipation"
        results = self.kb.retrieve(query, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].filename, "hdf_heat_dissipation_failure.md")

    def test_pwf_query_ranking(self):
        query = "spindle power overload 9500 W drive inverter current spike"
        results = self.kb.retrieve(query, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].filename, "pwf_power_failure.md")

    def test_context_string_formatting(self):
        query = "tool wear 220 min"
        context = self.kb.retrieve_context_string(query, top_k=2)
        self.assertIn("--- SOURCE:", context)
        self.assertIn("(relevance=", context)


if __name__ == "__main__":
    unittest.main()
