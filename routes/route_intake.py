from state import State


def route_intake(state: State):
    """
    If UniversityName and URL are set, proceed to DB check.
    Otherwise, ask human for help.
    """
    if state.get("UniversityName") and state.get("URL"):
        return "check_db_node"
    return "human_assistance"
