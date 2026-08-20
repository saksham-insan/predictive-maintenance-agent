"""
Agent Tool Execution and Audit Logging Module.

Provides executable tools for the ReAct Agentic Orchestrator and maintains an
append-only audit log in `data/agent_logs/tool_calls.jsonl`.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "agent_logs")
TOOL_LOG_PATH = os.path.join(LOGS_DIR, "tool_calls.jsonl")

# ---------------------------------------------------------------------------
# Tool Schemas for LLM System Prompt
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "create_maintenance_ticket",
        "description": "Creates an official maintenance work order ticket in the plant CMMS system for scheduled or immediate repair.",
        "parameters": {
            "machine_id": {"type": "string", "description": "Identifier of the machine (e.g., 'LINE_01' or 'CNC_MILL_4')"},
            "urgency": {"type": "string", "enum": ["Critical", "High", "Medium", "Low"], "description": "Urgency priority level of the maintenance ticket"},
            "notes": {"type": "string", "description": "Detailed explanation of diagnosed failure mode, physical telemetry causes, and required maintenance tasks"}
        },
        "required": ["machine_id", "urgency", "notes"]
    },
    {
        "name": "check_spare_parts_inventory",
        "description": "Queries the factory warehouse inventory database for available replacement components, part numbers, stock quantities, and restock lead times.",
        "parameters": {
            "part_type": {
                "type": "string",
                "enum": ["tool_insert", "cooling_system", "drive_motor", "spindle_bearing", "air_filter"],
                "description": "Category of spare part to check in inventory"
            }
        },
        "required": ["part_type"]
    },
    {
        "name": "escalate_to_engineer",
        "description": "Dispatches an urgent alert notification to on-duty mechanical/reliability engineers when telemetry shows ambiguous signals, anomalous patterns without clear physical root causes, or potential safety hazards.",
        "parameters": {
            "machine_id": {"type": "string", "description": "Identifier of the machine requiring engineering review"},
            "reason": {"type": "string", "description": "Detailed technical rationale explaining why engineering investigation is warranted"}
        },
        "required": ["machine_id", "reason"]
    },
    {
        "name": "request_more_sensor_data",
        "description": "Queries the MachineMemory temporal buffer to inspect recent sequential telemetry readings, degradation slope rates, and historical stability when single-reading signal is borderline or inconclusive.",
        "parameters": {
            "machine_id": {"type": "string", "description": "Identifier of the machine whose temporal memory is being queried"}
        },
        "required": ["machine_id"]
    }
]


# ---------------------------------------------------------------------------
# Mock Inventory Database
# ---------------------------------------------------------------------------
SPARE_PARTS_CATALOG = {
    "tool_insert": {
        "part_name": "Carbide CNC Cutting Tool Insert (Grade TiAlN)",
        "part_number": "INS-CARB-9921",
        "in_stock": 14,
        "location": "Warehouse Bin B-12",
        "status": "Available",
        "lead_time_days": 1
    },
    "cooling_system": {
        "part_name": "Spindle High-Pressure Coolant Pump & Heat Exchanger Assembly",
        "part_number": "COOL-PUMP-4410",
        "in_stock": 3,
        "location": "Warehouse Rack D-04",
        "status": "Available",
        "lead_time_days": 2
    },
    "drive_motor": {
        "part_name": "AC Synchronous Spindle Servo Drive Motor (15kW)",
        "part_number": "MOT-SERVO-8800",
        "in_stock": 1,
        "location": "Warehouse Heavy Storage H-01",
        "status": "Low Stock",
        "lead_time_days": 5
    },
    "spindle_bearing": {
        "part_name": "Precision Angular Contact Spindle Bearing Set",
        "part_number": "BRG-PREC-7210",
        "in_stock": 6,
        "location": "Warehouse Bin A-08",
        "status": "Available",
        "lead_time_days": 2
    },
    "air_filter": {
        "part_name": "Industrial Cabinet HEPA Air & Mist Filter Cartridge",
        "part_number": "FLT-HEPA-2020",
        "in_stock": 22,
        "location": "Warehouse Bin C-03",
        "status": "Available",
        "lead_time_days": 1
    }
}


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------
def _log_tool_call(tool_name: str, args: Dict[str, Any], result: Dict[str, Any]):
    """Appends a single tool invocation record to tool_calls.jsonl."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "args": args,
        "result": result
    }
    with open(TOOL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------
def create_maintenance_ticket(machine_id: str, urgency: str, notes: str) -> Dict[str, Any]:
    """Creates a maintenance work order ticket."""
    ticket_id = f"TICK-{int(time.time()*1000) % 1000000:06d}"
    result = {
        "status": "success",
        "ticket_id": ticket_id,
        "machine_id": machine_id,
        "urgency": urgency,
        "dispatch_status": "Scheduled with Maintenance Team",
        "notes": notes,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }
    _log_tool_call("create_maintenance_ticket", {"machine_id": machine_id, "urgency": urgency, "notes": notes}, result)
    return result


def check_spare_parts_inventory(part_type: str) -> Dict[str, Any]:
    """Queries warehouse spare parts catalog."""
    normalized = part_type.lower().strip().replace(" ", "_").replace("-", "_")
    part_info = SPARE_PARTS_CATALOG.get(normalized)

    if part_info:
        result = {
            "status": "success",
            "query": part_type,
            "found": True,
            "details": part_info
        }
    else:
        # Fallback partial matching
        matched_key = next((k for k in SPARE_PARTS_CATALOG if k in normalized or normalized in k), None)
        if matched_key:
            result = {
                "status": "success",
                "query": part_type,
                "found": True,
                "details": SPARE_PARTS_CATALOG[matched_key]
            }
        else:
            result = {
                "status": "success",
                "query": part_type,
                "found": False,
                "message": f"No spare parts found matching category '{part_type}'. Available: {list(SPARE_PARTS_CATALOG.keys())}"
            }

    _log_tool_call("check_spare_parts_inventory", {"part_type": part_type}, result)
    return result


def escalate_to_engineer(machine_id: str, reason: str) -> Dict[str, Any]:
    """Escalates anomalous situation to on-duty reliability engineer."""
    escalation_id = f"ESC-{int(time.time()*1000) % 1000000:06d}"
    result = {
        "status": "success",
        "escalation_id": escalation_id,
        "machine_id": machine_id,
        "assigned_to": "On-Duty Reliability Engineer (Tier 2)",
        "notification_sent": True,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }
    _log_tool_call("escalate_to_engineer", {"machine_id": machine_id, "reason": reason}, result)
    return result


def request_more_sensor_data(machine_id: str, memory_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Queries the temporal memory buffer to retrieve recent sequential readings and degradation slopes.
    """
    result = {
        "status": "success",
        "machine_id": machine_id,
        "memory_telemetry": memory_context or "Temporal Memory Buffer active. Trajectory shows recent consecutive readings within normal variance.",
        "recommendation": "Use trajectory slope and TWF distance to assess whether intervention can wait for scheduled shift or requires immediate stop."
    }
    _log_tool_call("request_more_sensor_data", {"machine_id": machine_id}, result)
    return result


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------
def execute_tool(tool_name: str, args: Dict[str, Any], memory_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Safely executes a tool by name with arguments and returns the result dictionary.
    """
    if not isinstance(args, dict):
        return {
            "status": "error",
            "error": f"Invalid arguments format for tool '{tool_name}'. Expected dict, got {type(args).__name__}."
        }

    try:
        if tool_name == "create_maintenance_ticket":
            return create_maintenance_ticket(
                machine_id=str(args.get("machine_id", "LINE_01")),
                urgency=str(args.get("urgency", "Medium")),
                notes=str(args.get("notes", "Automated ticket created by agent"))
            )
        elif tool_name == "check_spare_parts_inventory":
            return check_spare_parts_inventory(
                part_type=str(args.get("part_type", "tool_insert"))
            )
        elif tool_name == "escalate_to_engineer":
            return escalate_to_engineer(
                machine_id=str(args.get("machine_id", "LINE_01")),
                reason=str(args.get("reason", "Agent escalation due to anomalous telemetry"))
            )
        elif tool_name == "request_more_sensor_data":
            return request_more_sensor_data(
                machine_id=str(args.get("machine_id", "LINE_01")),
                memory_context=memory_context
            )
        else:
            err = f"Unknown tool '{tool_name}'. Available tools: {[s['name'] for s in TOOL_SCHEMAS]}"
            _log_tool_call(tool_name, args, {"status": "error", "error": err})
            return {"status": "error", "error": err}
    except Exception as e:
        err = f"Tool '{tool_name}' failed with exception: {str(e)}"
        _log_tool_call(tool_name, args, {"status": "error", "error": err})
        return {"status": "error", "error": err}


if __name__ == "__main__":
    print("Testing tool executions and JSONL logging...")
    res1 = check_spare_parts_inventory("tool_insert")
    print("Inventory check:", res1)
    res2 = create_maintenance_ticket("LINE_01", "High", "Tool wear exceeded 200 min")
    print("Ticket creation:", res2)
    print(f"Verified tool call log at: {TOOL_LOG_PATH}")
