import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from tools.RAG_tool import lookup_university_smart
from state import State

load_dotenv(override=True)

api_key = os.getenv('GROQ_API_KEY')

# RECOMMENDED: Use Llama 3.3 70B on Groq. It is much better at JSON than gpt-oss-120b.
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)


def extract_json_from_text(text: str):
    """
    Robustly extracts a JSON object from a string, ignoring conversational filler.
    """
    try:
        # Regex to find the largest substring starting with { and ending with }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    except:
        return None


def chat_node(state: State) -> State:
    print("\n" + "=" * 40)
    print("💬 AI INTAKE NODE RUNNING")
    print("=" * 40)

    messages = state.get("messages", [])

    # =========================================================================
    # STEP 1: VALIDATOR (The Fix)
    # We use a standard invoke + regex parsing. This will not crash on 400 errors.
    # =========================================================================
    print("🕵️ Checking if user already provided info...")

    validator_prompt = [
                           SystemMessage(content="""
        You are a Data Extraction Expert.
        Analyze the conversation history. Look for these three things:
        1. University Name
        2. Topic (e.g., Admissions, Sports, Research)
        3. URL (Official Website Link)

        RULES:
        - If the User provided a URL manually (e.g. "https://..."), TRUST IT. Extract it immediately.
        - Do not validate the URL yourself. Just extract the string.

        OUTPUT FORMAT:
        Reply ONLY with a JSON object. Do not add markdown or conversational text.
        {
            "university_name": "string or null",
            "topic": "string or null",
            "url": "string or null"
        }
        """)
                       ] + messages

    # Standard invocation (No tool binding forces)
    response = llm.invoke(validator_prompt)
    extracted_data = extract_json_from_text(response.content)

    if extracted_data:
        uni_name = extracted_data.get("university_name")
        topic = extracted_data.get("topic")
        url = extracted_data.get("url")

        # LOGIC: We need at least the URL to proceed to the database check
        if url and url != "null":
            print(f"✅ Data Extracted! \nURL: {url}")

            # If name is missing but URL is there, we can guess a generic name to unblock the flow
            final_name = uni_name if uni_name and uni_name != "null" else "University (From URL)"
            final_topic = topic if topic and topic != "null" else "General Info"

            return {
                "UniversityName": final_name,
                "topic": final_topic,
                "URL": [url],
                "URL_info": [f"Source: {url}"]
            }

    # =========================================================================
    # STEP 2: RAG TOOL (Only runs if Step 1 failed to find a URL)
    # =========================================================================
    print("🤔 URL missing in chat history. Attempting to use RAG Tool...")

    tools = [lookup_university_smart]
    llm_with_tools = llm.bind_tools(tools)

    system_msg = SystemMessage(content="""
    Protocol:
    1. If you have the university name, call 'lookup_university_smart'.
    2. If the tool fails (429 or error), ASK THE USER FOR THE URL.
    """)

    if not messages or not isinstance(messages[0], SystemMessage):
        history = [system_msg] + messages
    else:
        history = messages

    response = llm_with_tools.invoke(history)

    if response.tool_calls:
        print(f"⚙️  AI Analyzing Docs for: {response.tool_calls[0]['args']}")

        for tool_call in response.tool_calls:
            tool_result_json = lookup_university_smart.invoke(tool_call['args'])
            print(f"✅ RAG Result: {tool_result_json}")

            tool_msg = ToolMessage(tool_call_id=tool_call['id'], content=str(tool_result_json))

            # Failsafe: If tool failed, force ask user
            if "429" in str(tool_result_json) or "found\": false" in str(tool_result_json):
                print("⚠️ Tool failed (Quota/Error). Asking user for help.")
                error_msg = AIMessage(
                    content="I couldn't verify the URL automatically. Please provide the official URL.")
                return {"messages": [response, tool_msg, error_msg]}

            # If tool succeeded, return so Step 1 can catch it on the next loop
            return {"messages": [response, tool_msg]}

    # If no tool call, just return the response (usually a question)
    print(f"🤖 AI Requesting Info: {response.content}")
    return {"messages": [response]}