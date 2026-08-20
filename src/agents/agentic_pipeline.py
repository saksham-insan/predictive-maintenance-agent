"""
Agentic Pipeline Entrypoint for Predictive Maintenance.

Wires together the ML detection and diagnosis baseline with the ReAct Agentic Layer:
1. Monitoring Agent (Isolation Forest) — checks for statistical anomalies.
2. If normal: returns immediately with zero LLM overhead.
3. If anomalous:
   - Diagnosis Agent (Random Forest + SHAP)
   - MachineMemory (temporal trajectory & slope tracking)
   - FailureModeKnowledgeBase (TF-IDF RAG domain physics grounding)
   - LLM Orchestrator (multi-turn ReAct reasoning and tool execution loop)
"""

import os
import sys
from typing import Dict, Any, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from agents.monitoring_agent import monitoring_agent
from agents.diagnosis_agent import diagnosis_agent
from agents.llm_orchestrator import run_agentic_orchestrator
from memory import MachineMemory
from llm_client import BaseLLMClient


def run_pipeline_agentic(
    row: Dict[str, Any],
    memory: MachineMemory,
    llm_client: Optional[BaseLLMClient] = None
) -> Dict[str, Any]:
    """
    Runs one sensor row through the full agentic AI pipeline.

    Returns:
      - For normal readings:
        {"status": "normal", "action": "No action needed", "reasoning_trace": []}
      - For anomalous readings:
        {
          "status": "anomaly_detected",
          "diagnosis": {...},
          "trend_snapshot": {...},
          "grounding_sources": [...],
          "reasoning_trace": [...],
          "tool_calls_made": [...],
          "final_answer": {
            "summary": "...",
            "reasoning": "...",
            "action_taken": "...",
            "urgency": "..."
          }
        }
    """
    # 1. Update temporal memory for continuous trajectory tracking
    trend = memory.observe(row)

    # 2. Fast-path anomaly check via Isolation Forest
    is_anomaly = monitoring_agent(row)

    if not is_anomaly:
        return {
            "status": "normal",
            "action": "No action needed",
            "trend_snapshot": trend,
            "reasoning_trace": [],
            "tool_calls_made": [],
            "final_answer": {
                "summary": "Reading nominal across all vibration, thermal, and torque sensors.",
                "reasoning": "Isolation Forest model detected no multivariate anomaly.",
                "action_taken": "None",
                "urgency": "Low"
            }
        }

    # 3. Anomaly detected — invoke Diagnosis Agent (Random Forest + SHAP)
    diagnosis = diagnosis_agent(row)

    # 4. Invoke LLM-driven ReAct Orchestrator
    orchestrator_result = run_agentic_orchestrator(
        row=row,
        diagnosis=diagnosis,
        memory=memory,
        llm_client=llm_client
    )

    return {
        "status": "anomaly_detected",
        "diagnosis": diagnosis,
        "trend_snapshot": trend,
        "grounding_sources": orchestrator_result["grounding_sources"],
        "reasoning_trace": orchestrator_result["reasoning_trace"],
        "tool_calls_made": orchestrator_result["tool_calls_made"],
        "final_answer": orchestrator_result["final_answer"]
    }


if __name__ == "__main__":
    from llm_client import MockLLMClient

    print("Testing run_pipeline_agentic on normal vs anomalous rows...")
    mem = MachineMemory()
    mock_llm = MockLLMClient()

    normal_row = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 10
    }
    anom_row = {
        "Type": "L",
        "Air temperature [K]": 300.5,
        "Process temperature [K]": 311.2,
        "Rotational speed [rpm]": 1350,
        "Torque [Nm]": 65.0,
        "Tool wear [min]": 220
    }

    res_norm = run_pipeline_agentic(normal_row, mem, llm_client=mock_llm)
    print("Normal row status:", res_norm["status"])

    res_anom = run_pipeline_agentic(anom_row, mem, llm_client=mock_llm)
    print("Anomalous row status:", res_anom["status"])
    print("Grounding used:", res_anom["grounding_sources"])
    print("Final answer summary:", res_anom["final_answer"]["summary"])
