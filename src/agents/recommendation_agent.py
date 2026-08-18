def recommendation_agent(diagnosis: dict) -> dict:
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

    plain_explanation = diagnosis.get("plain_explanation", diagnosis.get("explanation", ""))

    recommendation = {
        "action": action,
        "urgency": urgency,
        "confidence": diagnosis["confidence"],
        "explanation": diagnosis.get("explanation", ""),
        "plain_explanation": plain_explanation
    }
    diagnosis["plain_explanation"] = plain_explanation

    return recommendation