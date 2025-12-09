import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage,AIMessage

from utils.BaseModels import IntakeComplete
from tools.RAG_tool import lookup_university_smart
from state import State


load_dotenv(override=True)

api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0.7
)


def chat_node(state: State) -> State:
    """
    Conversational Intake Node using the Smart RAG Tool.
    """
    print("\n" + "=" * 40)
    print("💬 AI INTAKE ASSISTANT")
    print("=" * 40)

    messages = state.get("messages", [])

    # 1. Bind the SMART Tool
    tools = [lookup_university_smart]  # <--- Updated Tool Name
    llm_with_tools = llm.bind_tools(tools)

    # 2. System Prompt
    system_msg = SystemMessage(content="""
    You are a helpful Social Media Assistant.You have a tool "lookup_university_smart" that you have to call when user 
    gives you name of a university
    Your goal is to gather: **University Name**, **Topic**, and **URL**.

    PROTOCOL:
    1. Ask the user for the University/Topic.
    2. Call 'lookup_university_smart' to find the URL. You must call this tool and then analyze its output.
    3. **Analyze Tool Output:**
       - If "found": true and "official_url" is present -> You are ready.
       - If "official_url" is null -> Ask the user: "I found info on [Name], but not the website link. Please provide 
       the URL."
       - If "found": false -> Ask the user for the URL.
    4. Once you have the URL and University Name , generate the final confirmation and end session. 
    5. Dont say any unnecessary stuff.
    """)

    # History handling
    if not messages or not isinstance(messages[0], SystemMessage):
        history = [system_msg] + messages
    else:
        history = messages

    while True:
        # A. Invoke Chat LLM
        response = llm_with_tools.invoke(history)
        history.append(response)

        # B. Handle Tool Calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"⚙️  AI Analyzing Docs for: {tool_call['args']}")

                # Execute the Smart Tool
                tool_result_json = lookup_university_smart.invoke(tool_call['args'])

                # The result is already rich JSON from the internal LLM
                tool_msg = ToolMessage(
                    tool_call_id=tool_call['id'],
                    content=str(tool_result_json)
                )
                history.append(tool_msg)

                # Debug Print for you
                print(f"✅ RAG Result: {tool_result_json}")

            continue

        # C. Check Completion (Validator)
        validator_llm = llm.with_structured_output(IntakeComplete)
        validation_prompt = history + [
            HumanMessage(content="Do we have University Name, Topic, and URL? Output JSON if yes.")]

        try:
            result = validator_llm.invoke(validation_prompt)
        except:
            result = None

        if result and result.url and result.university_name and result.topic:
            print(f"🤖 AI: {result.final_message}")
            history.append(AIMessage(content=result.final_message))

            return {
                "messages": history,
                "UniversityName": result.university_name,
                "topic": result.topic,
                "URL": [result.url],
                "URL_info": [f"Source: {result.url}"],
            }

        # D. Conversation Continue
        print(f"🤖 AI: {response.content}")
        user_input = input("User: ").strip()
        history.append(HumanMessage(content=user_input))
