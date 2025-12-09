import uuid
from langchain_core.messages import HumanMessage


from chroma_setup import get_or_create_knowledge_base
from Graph import agent


collection = get_or_create_knowledge_base("lumina_dataset.xlsx")


def run_interactive_session():
    print("\n" + "=" * 50)
    print(" AI SOCIAL MEDIA AGENT ")
    print("=" * 50)

    user_query = input("Enter Topic, University Name, or URL: ").strip()

    try:
        num_posts = int(input("How many posts would you like to generate? "))
    except ValueError:
        num_posts = 1

    # Base session ID to link these posts conceptually (optional)
    base_session_id = str(uuid.uuid4())

    print(f"\n🚀 Initializing workflow for {num_posts} posts...")

    # 2. Generation Loop
    for i in range(num_posts):
        current_post_num = i + 1
        print(f"\n" + "-" * 40)
        print(f"📝 STARTING GENERATION LOOP: Post #{current_post_num}")
        print("-" * 40)

        # A. Unique Thread Configuration
        # Essential to reset the 'iteration_count' and 'feedback' for every new post
        config = {
            "configurable": {
                "thread_id": f"{base_session_id}_post_{current_post_num}"
            }
        }

        # B. Initial State
        # We pass 'user_query' into 'messages'.
        # We LEAVE 'topic' and 'UniversityName' as None.
        # This forces the 'chat_node' to run its logic, call the RAG tool, and fill them in.
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "topic": None,  # chat_node will extract this
            "UniversityName": None,  # chat_node will extract this
            "iteration_count": 0,
            "approved": False,
            "human_feedback": None,
            "evaluator_feedback": None,
            "URL": [],
            "URL_info": []
        }

        try:
            # C. Invoke Graph
            # The code will interactively pause at:
            # 1. chat_node (if it needs to ask you clarifying questions)
            # 2. human_review_node (to approve the final post)
            final_state = agent.invoke(initial_state, config)

            # D. Output
            print(f"\n✅ Post #{current_post_num} Completed Successfully!")
            print(f"📌 Heading: {final_state.get('post_heading', 'N/A')}")
            # print(f"📄 Content: {final_state.get('post_content', 'N/A')}")

        except KeyboardInterrupt:
            print("\n\n⚠️ Process interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"❌ Error generating Post #{current_post_num}: {e}")

    print("\n" + "=" * 50)
    print("🎉 SESSION COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    run_interactive_session()
