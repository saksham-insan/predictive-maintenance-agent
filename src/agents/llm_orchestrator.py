"""
ReAct Agentic Orchestrator for Predictive Maintenance.

This module implements a genuine ReAct (Reasoning + Acting) autonomous agent loop
that analyzes sensor telemetry, grounds decisions in engineering physics formulas
via TF-IDF RAG, leverages temporal machine memory degradation trajectories, and
takes real autonomous actions using domain tools.
"""

import os
import sys
import json
import re
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rag import get_knowledge_base
from memory import MachineMemory, TrendSnapshot
from llm_client import BaseLLMClient, get_llm_client
from agents.tools import execute_tool, TOOL_SCHEMAS


def _clean_and_parse_json(response_text: str) -> Dict[str, Any]:
    """
    Robust JSON parser that strips markdown code fences, fixes unescaped newlines/quotes,
    repairs truncated JSON objects, and falls back to regex field extraction.
    """
    if not response_text or not isinstance(response_text, str):
        raise ValueError("Empty or invalid response from LLM")

    text = response_text.strip()

    # 1. Strip markdown ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # 2. Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Try extracting outermost JSON object {...}
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Repair unclosed JSON string (e.g. truncated responses or unclosed quote/brace)
    first_brace = text.find("{")
    if first_brace != -1:
        substring = text[first_brace:]
        # Try closing quotes and braces
        for suffix in ['"}', '"}\n}', '"}', '}']:
            try:
                candidate = substring.rstrip(' \n\r\t,.)') + suffix
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # 5. Heuristic regex-based field extractor fallback
    # Extracts action, summary, reasoning, tool, args from freeform or partially malformed JSON
    action_match = re.search(r'"action"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if not action_match:
        action_match = re.search(r'action\s*[:=]\s*"?(\w+)"?', text, re.IGNORECASE)

    if action_match:
        action_val = action_match.group(1).strip().lower()

        # Extract summary
        summary_val = ""
        summary_m = re.search(r'"summary"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text, re.DOTALL)
        if not summary_m:
            summary_m = re.search(r'"summary"\s*:\s*"([^"\n\r]*)', text)
        if summary_m:
            summary_val = summary_m.group(1).replace("\\n", "\n").replace('\\"', '"').strip()

        # Extract reasoning
        reasoning_val = ""
        reasoning_m = re.search(r'"reasoning"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text, re.DOTALL)
        if not reasoning_m:
            reasoning_m = re.search(r'"reasoning"\s*:\s*"([^"\n\r]*)', text)
        if reasoning_m:
            reasoning_val = reasoning_m.group(1).replace("\\n", "\n").replace('\\"', '"').strip()

        # Extract tool
        tool_val = ""
        tool_m = re.search(r'"tool"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        if tool_m:
            tool_val = tool_m.group(1).strip()

        # Extract args
        args_dict = {}
        args_m = re.search(r'"args"\s*:\s*(\{[^}]*\})', text, re.DOTALL)
        if args_m:
            try:
                args_dict = json.loads(args_m.group(1))
            except Exception:
                pass

        if "final" in action_val or action_val == "final_answer":
            if not summary_val and not reasoning_val:
                summary_val = re.sub(r'[\{\}"]', '', text).strip()
            return {
                "action": "final_answer",
                "summary": summary_val or "Maintenance directive synthesized from sensor telemetry.",
                "reasoning": reasoning_val or "Evaluated degradation trend and physics formulas."
            }
        elif "tool" in action_val or action_val == "call_tool":
            return {
                "action": "call_tool",
                "tool": tool_val or "check_spare_parts_inventory",
                "args": args_dict or {"part_type": "tool_insert"},
                "reasoning": reasoning_val or "Verifying spare parts inventory and telemetry data."
            }

    raise ValueError(f"Could not parse valid JSON object from LLM response:\n{response_text[:300]}")


def build_system_prompt(
    row: Dict[str, Any],
    diagnosis: Dict[str, Any],
    trend: TrendSnapshot,
    rag_context: str
) -> str:
    """
    Constructs the grounded industrial expert system prompt.
    """
    tools_doc = json.dumps(TOOL_SCHEMAS, indent=2)

    # Compute instantaneous power and temp diff for quick reference
    air_temp = float(row.get("Air temperature [K]", 0))
    proc_temp = float(row.get("Process temperature [K]", 0))
    rpm = float(row.get("Rotational speed [rpm]", 0))
    torque = float(row.get("Torque [Nm]", 0))
    wear = float(row.get("Tool wear [min]", 0))
    m_type = str(row.get("Type", "M"))

    temp_diff = round(proc_temp - air_temp, 2)
    power_watts = round(torque * rpm * (2 * 3.1415926535 / 60), 1)
    wear_torque_product = round(wear * torque, 1)

    return f"""You are the Industrial Reliability AI Agent for an automated CNC manufacturing facility.
Your role is to diagnose anomalous machine sensor readings, ground your reasoning in exact physics formulas and operating thresholds, evaluate degradation trajectories, and take necessary maintenance actions using your tools.

=== VERIFIED DOMAIN ENGINEERING KNOWLEDGE (RAG GROUNDING) ===
{rag_context}

=== CURRENT SENSOR TELEMETRY ===
- Machine Type: {m_type}
- Air Temperature: {air_temp} K
- Process Temperature: {proc_temp} K  (Temp Difference ΔT = {temp_diff} K)
- Spindle Rotational Speed: {rpm} RPM
- Spindle Torque: {torque} Nm
- Cumulative Tool Wear: {wear} min
- Instantaneous Power: {power_watts} W
- ToolWear × Torque Product: {wear_torque_product} min·Nm

=== ML DIAGNOSIS BASELINE ===
- ML Failure Prediction: {'FAILURE RISK (1)' if diagnosis.get('prediction') == 1 else 'No Failure (0)'}
- ML Model Confidence: {diagnosis.get('confidence', 0):.0%}
- SHAP Feature Explanation: {diagnosis.get('explanation', 'N/A')}

=== TEMPORAL MEMORY & TRAJECTORY ===
- Lifecycle Window: {trend.window_size} readings observed
- Tool Wear Slope: {trend.tool_wear_slope:+.3f} min/reading
- Torque Slope: {trend.torque_slope:+.3f} Nm/reading
- Est. Readings to 200 min TWF Danger Band: {f'{trend.est_readings_to_twf_band:.0f}' if trend.est_readings_to_twf_band is not None else 'N/A'}
- Trajectory Analysis: {trend.trend_description}

=== AVAILABLE TOOLS ===
{tools_doc}

=== OUTPUT CONTRACT ===
You must respond with ONLY a valid JSON object on every turn. Do NOT include markdown backticks or extra commentary outside the JSON.

If you need to use a tool to check parts, gather history, create a work order, or escalate:
{{
  "action": "call_tool",
  "tool": "<tool_name>",
  "args": {{ "<arg_name>": "<value>" }},
  "reasoning": "<Your engineering rationale connecting telemetry, formulas, and tool choice>"
}}

When your investigation is complete and you have your final decision:
{{
  "action": "final_answer",
  "summary": "<Comprehensive industrial diagnosis and operational directive>",
  "reasoning": "<Detailed engineering synthesis referencing specific formula thresholds and trajectory>"
}}
"""


def run_agentic_orchestrator(
    row: Dict[str, Any],
    diagnosis: Dict[str, Any],
    memory: MachineMemory,
    llm_client: Optional[BaseLLMClient] = None,
    max_iterations: int = 5
) -> Dict[str, Any]:
    """
    Executes the multi-turn ReAct reasoning loop on an anomalous sensor reading.
    """
    if llm_client is None:
        llm_client = get_llm_client()

    kb = get_knowledge_base()

    # 1. Update temporal memory and extract trend snapshot
    trend = memory.last_snapshot or memory.observe(row)

    # 2. Retrieve grounded RAG context combining ML explanation + temporal trajectory
    query_text = f"{diagnosis.get('explanation', '')} {trend.trend_description} Type {row.get('Type', 'M')} torque {row.get('Torque [Nm]')} tool wear {row.get('Tool wear [min]')}"
    retrieved_docs = kb.retrieve(query_text, top_k=2)
    rag_context = kb.retrieve_context_string(query_text, top_k=2)
    grounding_sources = [d.filename for d in retrieved_docs]

    # 3. Assemble system prompt and initial user prompt
    system_prompt = build_system_prompt(row, diagnosis, trend, rag_context)

    user_prompt = (
        "Anomalous telemetry has been detected. Perform root-cause diagnosis based on the physics formulas, "
        "evaluate trajectory risk, call any appropriate tools, and issue a final maintenance decision."
    )

    messages = [{"role": "user", "content": user_prompt}]
    reasoning_trace = []
    tool_calls_made = []

    final_result = None
    parse_failures = 0

    for iteration in range(1, max_iterations + 1):
        try:
            llm_response = llm_client.chat(messages, system=system_prompt)
        except Exception as e:
            # Handle backend connection errors safely
            err_reason = f"LLM backend communication failure: {str(e)}"
            execute_tool("escalate_to_engineer", {
                "machine_id": "LINE_01",
                "reason": f"Agent fallback: LLM client offline. {err_reason}"
            })
            reasoning_trace.append({
                "iteration": iteration,
                "reasoning": f"LLM client error encountered: {err_reason}. Triggering safe engineering fallback.",
                "tool_called": "escalate_to_engineer",
                "tool_args": {"machine_id": "LINE_01", "reason": err_reason},
                "tool_result": {"status": "escalated_due_to_backend_error"}
            })
            final_result = {
                "summary": f"Automated diagnosis interrupted: {err_reason}. Telemetry escalated directly to on-duty reliability engineer.",
                "reasoning": "Fallback safety handler activated to ensure no unmonitored anomalies occur during LLM backend unavailability.",
                "action_taken": "Escalate to Engineer",
                "urgency": "High"
            }
            break

        # Parse JSON output from model
        try:
            parsed = _clean_and_parse_json(llm_response)
        except Exception as parse_err:
            parse_failures += 1
            if parse_failures >= 2:
                # Fail safe by escalating to engineer
                fallback_tool_res = execute_tool("escalate_to_engineer", {
                    "machine_id": "LINE_01",
                    "reason": f"Agent output unparseable after retries: {str(parse_err)}"
                })
                reasoning_trace.append({
                    "iteration": iteration,
                    "reasoning": f"Model emitted malformed JSON ({parse_err}). Failsafe escalation triggered.",
                    "tool_called": "escalate_to_engineer",
                    "tool_args": {"machine_id": "LINE_01", "reason": "Unparseable LLM output"},
                    "tool_result": fallback_tool_res
                })
                final_result = {
                    "summary": "Anomaly flagged by ML baseline, but automated agent reasoning required fallback. Escalated to on-duty engineer.",
                    "reasoning": "Model output format validation failed; safe fallback executed to maintain industrial protocol.",
                    "action_taken": "Escalate to Engineer",
                    "urgency": "Medium"
                }
                break

            # Reprompt for clean JSON
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({
                "role": "user",
                "content": "Error: Your response was not valid JSON. You must reply ONLY with a single JSON object with either action='call_tool' or action='final_answer'."
            })
            continue

        action = parsed.get("action")
        reasoning = parsed.get("reasoning", "")

        if action == "call_tool":
            tool_name = parsed.get("tool", "")
            tool_args = parsed.get("args", {})

            # Execute tool
            tool_output = execute_tool(
                tool_name,
                tool_args,
                memory_context=memory.context_string()
            )

            step_record = {
                "iteration": iteration,
                "reasoning": reasoning,
                "tool_called": tool_name,
                "tool_args": tool_args,
                "tool_result": tool_output
            }
            reasoning_trace.append(step_record)
            tool_calls_made.append(step_record)

            # Feed observation back to conversation
            messages.append({"role": "assistant", "content": json.dumps(parsed)})
            messages.append({
                "role": "user",
                "content": f"Observation from {tool_name}:\n{json.dumps(tool_output, indent=2)}\n\nContinue with your next tool call or provide your final_answer JSON."
            })

        elif action == "final_answer":
            summary = parsed.get("summary", "")
            step_record = {
                "iteration": iteration,
                "reasoning": reasoning,
                "tool_called": None,
                "tool_args": None,
                "tool_result": None
            }
            reasoning_trace.append(step_record)

            # Determine high level action category for UI summary
            action_taken = "Maintenance Directive Issued"
            urgency = "Medium"
            if "immediate" in summary.lower() or "critical" in summary.lower() or "overstrain" in summary.lower():
                urgency = "High"
                action_taken = "Immediate Maintenance"
            elif "escalat" in summary.lower():
                action_taken = "Escalated to Engineering"
                urgency = "Medium"
            elif "request_more" in str(tool_calls_made).lower() or "trend" in summary.lower():
                action_taken = "Temporal Monitoring / Shift Planning"
                urgency = "Low"

            final_result = {
                "summary": summary,
                "reasoning": reasoning,
                "action_taken": action_taken,
                "urgency": urgency
            }
            break
        else:
            # Unknown action, prompt retry
            messages.append({"role": "assistant", "content": json.dumps(parsed)})
            messages.append({
                "role": "user",
                "content": "Invalid action specified. Must be 'call_tool' or 'final_answer'."
            })

    # If loop capped out without final_answer
    if final_result is None:
        final_result = {
            "summary": "Agent reached maximum iteration limit. Scheduled routine inspection to verify anomalous readings.",
            "reasoning": "Investigation capped at 5 turns; defensive maintenance alert generated.",
            "action_taken": "Inspection Scheduled",
            "urgency": "Medium"
        }

    return {
        "status": "anomaly_diagnosed",
        "diagnosis": diagnosis,
        "trend_snapshot": trend,
        "grounding_sources": grounding_sources,
        "reasoning_trace": reasoning_trace,
        "tool_calls_made": tool_calls_made,
        "final_answer": final_result
    }


if __name__ == "__main__":
    print("Testing LLM Orchestrator with MockLLMClient on an OSF-like sample...")
    from llm_client import MockLLMClient

    sample_row = {
        "Type": "M",
        "Air temperature [K]": 300.5,
        "Process temperature [K]": 311.2,
        "Rotational speed [rpm]": 1350,
        "Torque [Nm]": 65.0,
        "Tool wear [min]": 220
    }
    sample_diagnosis = {
        "prediction": 1,
        "confidence": 0.97,
        "explanation": "Power increased failure risk; Torque increased failure risk; Tool wear increased failure risk"
    }
    mem = MachineMemory()
    mem.observe(sample_row)

    mock_client = MockLLMClient()
    result = run_agentic_orchestrator(sample_row, sample_diagnosis, mem, llm_client=mock_client)

    print("\n--- AGENTIC ORCHESTRATOR RESULT ---")
    print(f"Grounding sources: {result['grounding_sources']}")
    print(f"Reasoning trace steps: {len(result['reasoning_trace'])}")
    for step in result["reasoning_trace"]:
        print(f"\nStep {step['iteration']}:")
        print(f"  Reasoning: {step['reasoning']}")
        if step['tool_called']:
            print(f"  Tool Called: {step['tool_called']} (Args: {step['tool_args']})")
            print(f"  Tool Result: {step['tool_result']}")
    print(f"\nFinal Summary:\n{result['final_answer']['summary']}")
