from llm_reasoning import trigger_async_llm_reasoning


def recommendation_agent(diagnosis: dict, event_id: str = "event") -> dict:
    """
    Converts a diagnosis into a maintenance recommendation.
    """
    if diagnosis["prediction"] == 1 and diagnosis["confidence"] > 0.7:
        action = "Schedule immediate maintenance"
        urgency = "High"
    elif diagnosis["prediction"] == 1:
        action = "Schedule maintenance within 48 hours"
        urgency = "Medium"
    else:
        action = "No action needed"
        urgency = "Low"

    recommendation = {
        "action": action,
        "urgency": urgency,
        "confidence": diagnosis["confidence"],
        "explanation": diagnosis.get("explanation", ""),
    }

    # Actually call the LLM reasoning module now
    plain_explanation = trigger_async_llm_reasoning(
        event_id=event_id,
        diagnosis=diagnosis,
        recommendation=recommendation
    )
    recommendation["plain_explanation"] = plain_explanation
    diagnosis["plain_explanation"] = plain_explanation

    return recommendation