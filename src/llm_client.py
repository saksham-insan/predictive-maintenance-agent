"""
Pluggable LLM Client Module.

Provides a unified interface for chat completions across multiple backends:
1. Ollama (Default): Local inference (e.g., llama3.2) via http://localhost:11434
2. Groq (Optional): Fast hosted inference via GROQ_API_KEY and LLM_BACKEND=groq
3. Mock: Scripted deterministic multi-turn ReAct responses for offline testing & CI
"""

import os
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class BaseLLMClient(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        """
        Sends a list of messages to the model and returns the raw assistant response string.
        Each message is a dict with {"role": "user"|"assistant"|"system", "content": "..."}.
        """
        pass


class OllamaClient(BaseLLMClient):
    """
    Local backend interfacing with Ollama's HTTP REST API (/api/chat).
    """

    def __init__(self, host: Optional[str] = None, model: Optional[str] = None, timeout: int = 45):
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        url = f"{self.host}/api/chat"

        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        for m in messages:
            formatted_messages.append({"role": m["role"], "content": m["content"]})

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 1024
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("message", {}).get("content", "")
        except urllib.error.URLError as e:
            err_msg = (
                f"\n[Ollama Connection Error] Could not connect to Ollama at '{self.host}'.\n"
                f"Troubleshooting steps:\n"
                f"1. Ensure Ollama is installed and running (`ollama serve`).\n"
                f"2. Pull the target model: `ollama pull {self.model}`.\n"
                f"3. Check that the port (default 11434) is accessible.\n"
                f"Underlying error: {e}"
            )
            raise ConnectionError(err_msg) from e
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}") from e


class GroqClient(BaseLLMClient):
    """
    Hosted cloud backend using Groq's OpenAI-compatible chat completion endpoint.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required to use GroqClient.")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        for m in messages:
            formatted_messages.append({"role": m["role"], "content": m["content"]})

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": 0.1,
            "max_tokens": 1024
        }

        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"Groq API HTTP {e.code} Error: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {e}") from e


class MockLLMClient(BaseLLMClient):
    """
    Deterministic scripted ReAct mock agent for local verification and automated testing.
    Analyzes prompt contents to emit realistic multi-step tool calls followed by final answers.
    """

    def __init__(self):
        self.call_count = 0

    def chat(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        self.call_count += 1
        eval_text = " ".join([m.get("content", "") for m in messages])
        if system and "=== CURRENT SENSOR TELEMETRY ===" in system:
            telemetry_section = system.split("=== CURRENT SENSOR TELEMETRY ===")[-1].split("=== AVAILABLE TOOLS ===")[0]
            eval_text += " " + telemetry_section
        elif system:
            eval_text += " " + system

        # Determine turn in conversation
        has_tool_result = any("observation" in m.get("content", "").lower() or "tool result" in m.get("content", "").lower() for m in messages)

        # 1. Check for Borderline / Rising Trend scenario (tool wear rising towards TWF band without crossing OSF/TWF yet)
        if "borderline" in eval_text.lower() or "rising fast" in eval_text.lower() or "approaching threshold" in eval_text.lower():
            if not has_tool_result:
                return json.dumps({
                    "action": "call_tool",
                    "tool": "request_more_sensor_data",
                    "args": {"machine_id": "LINE_01"},
                    "reasoning": "Sensor reading is borderline anomalous with a rising tool wear trajectory approaching the 200 min TWF threshold. Requesting additional temporal sensor history from memory buffer to confirm wear slope before issuing a premature maintenance order."
                })
            else:
                return json.dumps({
                    "action": "final_answer",
                    "summary": "Borderline anomaly evaluated against temporal trajectory. Tool wear is escalating and will reach the 200 min critical TWF band within upcoming cycles. Recommend scheduling tool replacement during the upcoming scheduled shift break rather than immediate emergency shutdown.",
                    "reasoning": "Temporal trend analysis proves the machine can safely complete current cycle but requires planned intervention before the next shift."
                })

        # 2. Check for Overstrain Failure (OSF) scenario: high torque + high tool wear
        elif "overstrain" in eval_text.lower() or ("torque" in eval_text.lower() and "tool wear" in eval_text.lower() and ("danger band" in eval_text.lower() or "220" in eval_text.lower())):
            if not has_tool_result:
                return json.dumps({
                    "action": "call_tool",
                    "tool": "check_spare_parts_inventory",
                    "args": {"part_type": "tool_insert"},
                    "reasoning": "Telemetry and SHAP explainability indicate high torque combined with tool wear in the danger zone, matching the Overstrain Failure (OSF) formula (ToolWear * Torque > limit). Checking spare parts inventory for available tool inserts before scheduling replacement."
                })
            else:
                return json.dumps({
                    "action": "final_answer",
                    "summary": "Confirmed Overstrain Failure (OSF) risk. Mechanical stress product exceeds safe limit for machine type. Inventory confirmed replacement tool inserts in stock. Scheduled immediate maintenance ticket and alerted shift supervisor to throttle feed rate.",
                    "reasoning": "With inventory confirmed in stock, immediate tool insert replacement is recommended to prevent catastrophic cutter fracture under high torque."
                })

        # 3. Check for Heat Dissipation Failure (HDF) scenario: low temp diff (< 8.6 K) and low RPM (< 1380)
        elif "hdf" in full_text.lower() or "heat dissipation" in full_text.lower() or "temp_diff" in full_text.lower():
            if not has_tool_result:
                return json.dumps({
                    "action": "call_tool",
                    "tool": "check_spare_parts_inventory",
                    "args": {"part_type": "cooling_system"},
                    "reasoning": "Low temperature gradient (ProcTemp - AirTemp < 8.6 K) combined with low rotational speed satisfies the exact Heat Dissipation Failure (HDF) trigger formula. Checking cooling system components."
                })
            else:
                return json.dumps({
                    "action": "final_answer",
                    "summary": "Heat Dissipation Failure (HDF) warning. Thermal convection is compromised due to low speed and insufficient cooling delta. Clean heat exchangers and inspect coolant pump.",
                    "reasoning": "Cooling system verified in inventory; immediate inspection of coolant lines and fan airflow recommended."
                })

        # 4. Check for Power Failure (PWF) scenario: power < 3500 W or > 9000 W
        elif "pwf" in full_text.lower() or "power failure" in full_text.lower() or "power" in full_text.lower():
            if not has_tool_result:
                return json.dumps({
                    "action": "call_tool",
                    "tool": "create_maintenance_ticket",
                    "args": {"machine_id": "LINE_01", "urgency": "High", "notes": "Instantaneous power exceeded 9000W envelope. Investigate drive inverter and mechanical spindle load."},
                    "reasoning": "Calculated power is outside the safe 3500W-9000W envelope, indicating imminent Power Failure (PWF). Creating high-priority maintenance ticket."
                })
            else:
                return json.dumps({
                    "action": "final_answer",
                    "summary": "Power Failure (PWF) condition detected. Spindle drive overload exceeds 9000W safety cutoff. High urgency maintenance ticket dispatched to inspect drive inverter and reduce feed rate.",
                    "reasoning": "Ticket logged with maintenance team to prevent inverter trip and motor burnout."
                })

        # 5. Low-signal / Random Failure (RNF) or default
        else:
            if not has_tool_result:
                return json.dumps({
                    "action": "call_tool",
                    "tool": "escalate_to_engineer",
                    "args": {"machine_id": "LINE_01", "reason": "Anomaly flagged without matching any physical threshold formulas (TWF/HDF/PWF/OSF). Potential stochastic disturbance (RNF) or sensor calibration drift."},
                    "reasoning": "Telemetry does not violate physical failure formulas for tool wear, power, overstrain, or heat dissipation. Grounding knowledge indicates this fits a low-signal stochastic event (RNF). Escalating to engineer for non-destructive check rather than replacing healthy components."
                })
            else:
                return json.dumps({
                    "action": "final_answer",
                    "summary": "No deterministic physical failure mode (TWF/HDF/PWF/OSF) triggered. Anomaly is consistent with Random Failure (RNF, 0.1% background rate) or transient electrical sensor noise. Escalated to on-duty engineer for verification without taking machine offline.",
                    "reasoning": "Grounding knowledge confirms zero correlation with wear or torque; preserving equipment uptime while alerting engineering."
                })


def get_llm_client(backend: Optional[str] = None) -> BaseLLMClient:
    """
    Factory function returning the configured LLM client.
    Selected via backend argument, or LLM_BACKEND env var ('ollama', 'groq', 'mock').
    Defaults to 'ollama'.
    """
    selected = (backend or os.getenv("LLM_BACKEND", "ollama")).lower().strip()

    if selected == "groq":
        return GroqClient()
    elif selected == "mock":
        return MockLLMClient()
    elif selected == "ollama":
        return OllamaClient()
    else:
        # Fallback to Ollama
        return OllamaClient()


if __name__ == "__main__":
    client = MockLLMClient()
    response = client.chat([{"role": "user", "content": "Telemetry shows high torque and tool wear of 215 min for machine type M. OSF risk."}])
    print("Mock Client Response:\n", response)
