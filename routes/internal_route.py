from state import State


def route_internal(state: State):
    """Routing after AI Evaluator."""
    grade = state["grade"]
    count = state["iteration_count"]

    # If max loops reached, force go to human to decide
    if count >= 5:
        print("--- Max AI retries reached. Sending to Human. ---")
        return "human_review"

    if grade == "good":
        return "human_review"
    else:
        return "regenerate"


def route_human(state: State):
    """Routing after Human Review."""
    if state["approved"]:
        return "end"
    else:
        return "regenerate"
