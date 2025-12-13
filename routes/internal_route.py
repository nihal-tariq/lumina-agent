from state import State


def route_internal(state: State):
    """Routing after AI Evaluator."""
    grade = state.get("grade")
    count = state.get("iteration_count", 0)

    # Safety: Force human review if too many auto-retries
    if count >= 4:
        print("--- Max AI retries reached. Escalate to Human. ---")
        return "human_review"

    if grade == "good":
        print("--- Evaluator Approved. Sending to Human. ---")
        return "human_review"
    else:
        print("--- Evaluator Rejected. Regenerating. ---")
        return "regenerate"