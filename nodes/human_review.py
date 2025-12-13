from state import State


def human_review_node(state: State):
    """
    Pass-through node.
    The graph interrupts BEFORE this node.
    When resumed, the state will already have 'approved' and 'human_feedback' populated.
    """
    print("--- Human Review Step Completed ---")
    return