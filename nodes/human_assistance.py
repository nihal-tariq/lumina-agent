from state import State


def human_assistance_node(state: State):
    """
    A pass-through node.
    We interrupt BEFORE this node runs.
    When we resume, we update state with user input, so this node effectively just loops back.
    """
    print("🙋 Waiting for Human Input...")
    return {}