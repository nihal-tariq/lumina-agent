from state import State


def human_review_node(state: State):
    """INTERRUPT: Shows post to user and asks for input."""

    print("\n" + "=" * 40)
    print("👀 HUMAN REVIEW REQUIRED")
    print("=" * 40)
    print(f"TOPIC: {state['topic']}")
    print(f"HEADING: {state['post_heading']}")
    print(f"CONTENT: {state['post_content']}")
    print("-" * 40)

    # Python input() pauses the script execution here
    user_input = input("Is this post acceptable? (yes/no): ").strip().lower()

    if user_input in ['yes', 'y', 'okay', 'ok']:
        return {"approved": True, "human_feedback": None}
    else:
        feedback = input("Please provide your feedback for changes: ")
        return {"approved": False, "human_feedback": feedback}

