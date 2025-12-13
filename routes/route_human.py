from state import State


def route_human(state: State):
    """Routing after Human Review."""
    is_approved = state.get("approved", False)

    if is_approved:
        print("--- Human Approved. Finishing. ---")
        return "save"
    else:
        print("--- Human Rejected. Regenerating. ---")
        return "regenerate"
