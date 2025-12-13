import os
import json
import re
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage


# Import your State definition
from state import State

load_dotenv(override=True)

# --- 1. SETUP RESOURCES ONCE (Not inside the node) ---
# This prevents reconnecting to DB and recompiling Graph on every request
api_key = os.getenv('GROQ_API_KEY')
database_url = os.getenv("DATABASE_URL")

if not api_key or not database_url:
    raise ValueError("Missing API Key or Database URL")

db = SQLDatabase.from_uri(database_url)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key
)

# Setup Tools
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = sql_toolkit.get_tools()

# --- 2. COMPILE THE SUB-AGENT ONCE ---
system_prompt = """You are a specialized SQL Agent. 
Your GOAL: Check if a specific URL exists in the 'university' table.

Schema:
    university (
        id SERIAL PRIMARY KEY,
        uni_name TEXT,
        url TEXT,
        summary TEXT,
        time_stamp TIMESTAMPTZ
    );

INSTRUCTIONS:
1. Search the 'university' table for the specific URL provided.
2. If found, return the uni_name, url, summary, and time_stamp.
3. If not found, explicitly state it is not found.

CRITICAL OUTPUT FORMAT:
You must end your response with a JSON block. Do not include markdown formatting like ```json.
The JSON must look exactly like this:
{
    "status": "success",
    "data": {
        "uni_name": "...",
        "url": "...",
        "summary": "...",
        "time_stamp": "YYYY-MM-DD..."
    }
}
OR if not found:
{
    "status": "not_found", 
    "data": null
}
"""

# We compile the graph here, globally.
# This 'sql_agent' is now a reusable Runnable.
sql_agent = create_agent(llm, tools, system_prompt=system_prompt)


# --- 3. THE NODE (Lightweight & Fast) ---
def check_db_node(state: State) -> State:
    """
    LangGraph node that invokes the pre-compiled SQL Agent.
    """
    # 1. Get Input
    urls = state.get("URL", [])
    if not urls:
        return {"TimeStamp": "NULL", "info": "No URL provided.", "summary": "NULL"}

    target_url = urls[-1] if isinstance(urls, list) else urls

    # 2. Invoke the Pre-Compiled Agent
    query_message = f"Check database for this URL: {target_url}"

    # We invoke the agent with its own internal state
    # We use a distinct thread_id if you want isolation, but for a stateless lookup, it's fine.
    agent_result = sql_agent.invoke({"messages": [HumanMessage(content=query_message)]})

    # 3. Parse Output
    last_message = agent_result["messages"][-1].content

    # Robust JSON Extraction
    json_match = re.search(r'\{.*\}', last_message, re.DOTALL)

    extracted_data = {}
    status = "error"

    if json_match:
        try:
            # Clean up potential markdown artifacts just in case
            clean_json = json_match.group(0).replace("```json", "").replace("```", "")
            data_dict = json.loads(clean_json)
            status = data_dict.get("status")
            extracted_data = data_dict.get("data")
        except json.JSONDecodeError:
            print(f"DEBUG: JSON Parse Error. Raw content: {last_message}")
            status = "json_error"
    else:
        print(f"DEBUG: No JSON found in agent response: {last_message}")
        status = "parse_error"

    # 4. Return Updates to Main Graph State
    if status == "success" and extracted_data:
        found_url = extracted_data.get("url")
        found_timestamp = extracted_data.get("time_stamp")
        found_summary = extracted_data.get("summary")

        info_str = f"URL: {found_url}, TIME_STAMP: {found_timestamp}"

        return {
            "TimeStamp": str(found_timestamp),
            "info": info_str,
            "summary": found_summary,
            "URL_info": state.get("URL_info", []) + [info_str]
        }
    else:
        # Default fallback
        null_str = "URL: NULL, TIME_STAMP: NULL"
        return {
            "TimeStamp": "NULL",
            "summary": None,
            "info": null_str,
            "URL_info": state.get("URL_info", []) + [null_str]
        }