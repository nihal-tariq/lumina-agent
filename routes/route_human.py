from state import State


def route_human(state: State):
    """Routing after Human Review."""
    if state["approved"]:
        return "end"
    else:
        return "regenerate"