"""
Temporal Memory and Trend Analysis Module for Predictive Maintenance.

Framing & Domain Assumptions:
-----------------------------
AI4I 2020 dataset rows do not carry a distinct machine identifier, so sequential row
order is treated as a continuous telemetry timeline for a single production line.
When cutting tools reach end-of-life, operators perform a physical tool insert swap,
causing 'Tool wear [min]' to drop precipitously back to ~0 min. A drop of > 50 min
between consecutive readings is detected as a tool-change lifecycle event, triggering
a memory window reset.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class TrendSnapshot:
    tool_wear_current: float
    tool_wear_slope: float
    torque_current: float
    torque_slope: float
    est_readings_to_twf_band: Optional[float]
    trend_description: str
    window_size: int
    is_tool_change_detected: bool = False


class MachineMemory:
    """
    Maintains a rolling temporal window (maxlen=20) of sensor readings to track
    degradation trajectories, slope rates, and tool lifecycle phases.
    """

    def __init__(self, maxlen: int = 20):
        self.maxlen = maxlen
        self.history: deque = deque(maxlen=maxlen)
        self.last_snapshot: Optional[TrendSnapshot] = None

    def observe(self, row: dict) -> TrendSnapshot:
        """
        Ingests a new raw sensor row, updates the rolling window, checks for
        tool-change events, and computes the current degradation trend snapshot.
        """
        tool_wear = float(row.get("Tool wear [min]", 0.0))
        torque = float(row.get("Torque [Nm]", 0.0))
        air_temp = float(row.get("Air temperature [K]", 0.0))
        proc_temp = float(row.get("Process temperature [K]", 0.0))
        rpm = float(row.get("Rotational speed [rpm]", 0.0))
        m_type = str(row.get("Type", "M"))

        tool_change = False
        if len(self.history) > 0:
            last_wear = self.history[-1]["Tool wear [min]"]
            if (last_wear - tool_wear) > 50.0:
                # Tool change detected: reset window for new lifecycle
                tool_change = True
                self.history.clear()

        cleaned_row = {
            "Type": m_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": proc_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }
        self.history.append(cleaned_row)

        # Compute slopes
        window_len = len(self.history)
        if window_len < 2:
            wear_slope = 0.0
            torque_slope = 0.0
        else:
            x = np.arange(window_len)
            w_vals = [r["Tool wear [min]"] for r in self.history]
            t_vals = [r["Torque [Nm]"] for r in self.history]

            wear_slope = float(np.polyfit(x, w_vals, 1)[0]) if window_len >= 2 else 0.0
            torque_slope = float(np.polyfit(x, t_vals, 1)[0]) if window_len >= 2 else 0.0

        # Heuristic linear extrapolation to TWF danger band (200 min threshold)
        if tool_wear >= 200.0:
            est_twf_steps = 0.0
        elif wear_slope > 0.05:
            est_twf_steps = round((200.0 - tool_wear) / wear_slope, 1)
        else:
            est_twf_steps = None

        # Build descriptive summary
        descriptions = []
        if tool_change:
            descriptions.append("Tool replacement detected (>50 min wear reset). Initializing fresh lifecycle.")

        if tool_wear >= 200.0:
            descriptions.append(f"CRITICAL: Tool wear at {tool_wear:.1f} min is inside the high-risk TWF band (200–240 min).")
        elif est_twf_steps is not None and est_twf_steps <= 15.0:
            descriptions.append(f"ALERT: Tool wear rising fast (+{wear_slope:.2f} min/reading), estimated ~{est_twf_steps:.0f} readings until 200 min TWF threshold.")
        elif wear_slope > 0.1:
            descriptions.append(f"Tool wear increasing steadily (+{wear_slope:.2f} min/reading, current: {tool_wear:.1f} min).")
        else:
            descriptions.append(f"Tool wear stable at {tool_wear:.1f} min.")

        if torque_slope > 0.5:
            descriptions.append(f"Torque trending upward (+{torque_slope:.2f} Nm/reading, current: {torque:.1f} Nm) - potential mechanical resistance.")
        elif torque_slope < -0.5:
            descriptions.append(f"Torque declining ({torque_slope:.2f} Nm/reading, current: {torque:.1f} Nm).")
        else:
            descriptions.append(f"Torque nominal ({torque:.1f} Nm).")

        trend_desc = " ".join(descriptions)

        snapshot = TrendSnapshot(
            tool_wear_current=round(tool_wear, 2),
            tool_wear_slope=round(wear_slope, 3),
            torque_current=round(torque, 2),
            torque_slope=round(torque_slope, 3),
            est_readings_to_twf_band=est_twf_steps,
            trend_description=trend_desc,
            window_size=window_len,
            is_tool_change_detected=tool_change
        )
        self.last_snapshot = snapshot
        return snapshot

    def context_string(self) -> str:
        """
        Formats current temporal state into a concise string for LLM prompting.
        """
        if not self.last_snapshot or not self.history:
            return "No historical sensor telemetry recorded yet."

        s = self.last_snapshot
        twf_str = f"{s.est_readings_to_twf_band:.0f} readings" if s.est_readings_to_twf_band is not None else "N/A (stable/low rate)"

        lines = [
            f"Temporal Window: {s.window_size} readings tracked (recent lifecycle)",
            f"Current Tool Wear: {s.tool_wear_current} min (Slope: {s.tool_wear_slope:+.3f} min/step)",
            f"Current Torque: {s.torque_current} Nm (Slope: {s.torque_slope:+.3f} Nm/step)",
            f"Est. Readings to 200 min TWF Band: {twf_str}",
            f"Trajectory Summary: {s.trend_description}"
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    memory = MachineMemory()
    print("Simulating sequence of tool wear readings...")
    for w in [180, 183, 186, 189, 192, 196, 199, 203]:
        snap = memory.observe({
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 42.0 + (w - 180) * 0.5,
            "Tool wear [min]": w
        })
    print(memory.context_string())

    print("\nSimulating tool replacement drop:")
    snap = memory.observe({
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 0
    })
    print(memory.context_string())
