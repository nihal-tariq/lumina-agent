import uuid
import sys
from langchain_core.messages import HumanMessage
from Graph import agent


def run_interactive_session():
    print("\n🚀 AI SOCIAL MEDIA AGENT INITIALIZED")

    user_query = input("Enter Topic, University Name, or URL: ").strip()
    try:
        num_posts_input = input("How many posts would you like to generate? (Default 1): ")
        num_posts = int(num_posts_input) if num_posts_input else 1
    except ValueError:
        num_posts = 1

    base_session_id = str(uuid.uuid4())

    for i in range(num_posts):
        current_post_num = i + 1
        thread_id = f"{base_session_id}_post_{current_post_num}"
        config = {"configurable": {"thread_id": thread_id}}

        print(f"\n📝 STARTING POST #{current_post_num} (ID: {thread_id})")

        # Initial payload for the new post
        # We assume the user wants the same topic for all posts, or you can ask inside the loop
        current_input = {"messages": [HumanMessage(content=user_query)]}

        while True:
            try:
                # 1. Run the Graph until it stops (End or Interrupt)
                # Passing 'None' as input resumes from the last state/checkpoint
                events = agent.stream(current_input, config, stream_mode="values")

                for event in events:
                    # Optional: specific logging could go here
                    pass

                # 2. Check the status of the graph
                snapshot = agent.get_state(config)

                # CASE A: Graph Finished (No next steps)
                if not snapshot.next:
                    print(f"\n✅ Post #{current_post_num} Completed!")
                    print(f"📌 Final Heading: {snapshot.values.get('post_heading', 'N/A')}")
                    break

                # CASE B: Graph Interrupted (Human in the Loop)
                next_node = snapshot.next[0]

                # --- INTERRUPT: HUMAN ASSISTANCE (Missing Info) ---
                if next_node == "human_assistance":
                    # The Chat Node asked a question. Look at the last AI message.
                    last_msg = snapshot.values['messages'][-1]
                    print(f"\n🤖 AI Question: {last_msg.content}")

                    user_response = input("User (Provide URL/Info): ")

                    # Update state with the human answer to the chat history
                    # We 'pretend' this update effectively runs the human_assistance node
                    agent.update_state(
                        config,
                        {"messages": [HumanMessage(content=user_response)]},
                        as_node="human_assistance"
                    )

                    print("🔄 Info received. Resuming workflow...")
                    current_input = None  # Resume with existing state
                    continue

                    # --- INTERRUPT: HUMAN REVIEW (Approve/Reject Post) ---
                elif next_node == "human_review":
                    current_values = snapshot.values
                    print(f"\n" + "=" * 50)
                    print(f"👀 REVIEW REQUIRED FOR POST #{current_post_num}")
                    print(f"=" * 50)
                    print(f"TOPIC:   {current_values.get('topic')}")
                    print(f"HEADING: {current_values.get('post_heading')}")
                    print(f"-" * 20)
                    print(f"CONTENT:\n{current_values.get('post_content')}")
                    print(f"=" * 50)

                    user_decision = input(">> Approve this post? (yes/no/comment): ").strip()

                    if user_decision.lower() in ["yes", "y", "ok"]:
                        # Approved: Set flag to True, Clear feedback
                        update_data = {
                            "approved": True,
                            "human_feedback": None
                        }
                        print("✅ Post Approved.")
                    else:
                        # Rejected: Set flag to False, Add feedback text
                        # If user typed 'no', ask for specific feedback
                        if user_decision.lower() in ["no", "n"]:
                            feedback_text = input(">> Please provide specific feedback: ")
                        else:
                            # Assume the input itself was the feedback (e.g., "Make it shorter")
                            feedback_text = user_decision

                        update_data = {
                            "approved": False,
                            "human_feedback": feedback_text
                        }
                        print("🔄 Feedback Recorded. sending back to Generator...")

                    # Update state and pretend 'human_review' node just finished
                    agent.update_state(
                        config,
                        update_data,
                        as_node="human_review"
                    )

                    current_input = None  # Resume with existing state
                    continue

            except KeyboardInterrupt:
                print("\n⚠️ Operation cancelled by user.")
                sys.exit()
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                # Important: Break the inner while loop to move to the next post or exit
                break

    print("\n🎉 All requested posts processing finished.")


if __name__ == "__main__":
    run_interactive_session()