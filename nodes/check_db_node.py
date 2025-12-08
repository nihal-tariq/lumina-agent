import os
import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from state import State


def check_db_node(state: State) -> State:
    """
    LangGraph node that queries the 'university' table for a specific URL.
    Returns URL: <url>, TIME_STAMP: <timestamp> or URL: NULL, TIME_STAMP: NULL.
    """

    # 2. Extract the URL to check
    urls = state.get("URL", [])
    if not urls:
        return {"TimeStamp": "NULL", "info": "No URL provided to check.", "summary": "NULL" }

    target_url = urls[-1] if isinstance(urls, list) else urls

    # 3. Database Connection
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        # Fallback for testing if env var is missing
        database_url = "postgresql://neondb_owner:npg_ldWU2aIuHv9R@ep-green-unit-adoucn1n-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

    db = SQLDatabase.from_uri(database_url)

    # 4. LLM & Toolkit Setup
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0
    )

    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = sql_toolkit.get_tools()

    # 5. System Prompt
    # Added explicit instruction to cast Timestamp to string to avoid JSON errors
    system_prompt = f"""
You are an intelligent SQL agent whose ONLY job is to search a PostgreSQL table.

Schema:
    university (
        id SERIAL PRIMARY KEY,
        uni_name TEXT NOT NULL,
        url TEXT NOT NULL,
        summary TEXT NOT NULL,
        time_stamp TIMESTAMPTZ DEFAULT NOW()
    );

Your task:
1. Generate a PostgreSQL SELECT query to find the row where 'url' matches '{target_url}'.
2. Retrieve 'uni_name', 'url', 'summary' and 'time_stamp'.
3. IMPORTANT: The 'time_stamp' returned by the database is an object. You must convert it to a simple ISO STRING in your JSON output.

Structured Output Format:
Return a single valid JSON object in this format:
{{
    "status": "success" or "not_found",
    "data": {{
        "uni_name": "string value",
        "url": "string value",
        "summary": "string value",
        "time_stamp": "YYYY-MM-DDTHH:MM:SS+00:00" 
    }}
}}

If no row is found, set "status" to "not_found" and "data" to null.
Provide ONLY the JSON. No markdown, no explanation.
"""

    # 6. Create the Agent
    # Using create_react_agent from langgraph.prebuilt is standard for nodes
    agent_executor = create_agent(llm, tools, system_prompt=system_prompt)

    # 7. Invoke the Agent
    query_message = f"Check if this URL exists in the database: {target_url}"
    result = agent_executor.invoke({"messages": [HumanMessage(content=query_message)]})

    # 8. Process Response with Robust Parsing
    last_message = result["messages"][-1].content

    # Use Regex to find the first JSON object { ... }
    # This ignores "Here is your JSON:" text that often breaks parsers
    json_match = re.search(r'\{.*\}', last_message, re.DOTALL)

    extracted_data = {}
    status = "error"

    if json_match:
        json_str = json_match.group(0)
        try:
            data_dict = json.loads(json_str)
            status = data_dict.get("status")
            extracted_data = data_dict.get("data")
        except json.JSONDecodeError:
            status = "json_error"
    else:
        # If regex failed, try raw string just in case
        try:
            data_dict = json.loads(last_message)
            status = data_dict.get("status")
            extracted_data = data_dict.get("data")
        except:
            status = "parse_error"

    # 9. Logic Output
    if status == "success" and extracted_data:
        found_url = extracted_data.get("url")
        found_timestamp = extracted_data.get("time_stamp")
        found_summary = extracted_data.get("summary")

        # Format: URL found
        formatted_string = f"URL: {found_url}, TIME_STAMP: {found_timestamp}"

        return {
            "TimeStamp": str(found_timestamp),
            "info": formatted_string,
            "summary": found_summary,
            "URL_info": state.get("URL_info", []) + [formatted_string]
        }
    else:
        # Format: URL not found (covers 'not_found', 'json_error', 'parse_error')
        # This ensures the workflow proceeds to Scrape_with_jina
        null_string = "URL: NULL, TIME_STAMP: NULL"

        return {
            "TimeStamp": "NULL",
            "summary": None,
            "info": null_string,
            "URL_info": state.get("URL_info", []) + [null_string]
        }