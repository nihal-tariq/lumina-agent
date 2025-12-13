import os

from dotenv import load_dotenv

from datetime import datetime, timezone
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from state import State
from db.operations import insert_university

load_dotenv(override=True)

DATABASE_URL = os.getenv('DATABASE_URL')


def summarize(state: State) -> State:
    """
    Generates a structured, detailed summary from the raw content using
    the Gemini model and a specific system prompt for extraction.
    """
    # 1. Retrieve the raw content from the state
    raw_content = state.get("Content")

    if not raw_content:
        print("Warning: 'Content' field is empty. Skipping summarization.")
        state["summary"] = "No content provided for summarization."
        return state

    # 2. Initialize the Gemini Chat Model
    # We use 'gemini-2.5-flash' for fast, production-ready summarization,
    # mapping the user's requested 'gemini-2.5-flash-lite' intention.
    # Temperature is set to 0 for maximum factual accuracy and less creativity.
    try:
        model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    except Exception as e:
        print(f"Error initializing ChatGoogleGenerativeAI: {e}")
        state["summary"] = f"ERROR: Model initialization failed: {str(e)}"
        return state

    # 3. Define the detailed system prompt (provided by the user)
    system_prompt = f'''
You are a highly precise and careful information extraction assistant. Your task is to process raw, 
unstructured input (which may include HTML tags, noisy text, or other formatting artifacts) and 
generate a clean, detailed, and accurate summary. Follow these instructions exactly:

Clean the Input:

Remove all HTML tags, scripts, advertisements, and other non-informative content.
Remove repeated whitespace, broken formatting, and unnecessary symbols.
Preserve only meaningful, readable textual content.

Extract Detailed Information:
Identify and extract all relevant information, including but not limited to:

University Names – any mentioned institutions.
Important Dates – admission deadlines, exam dates, merit list announcements, registration dates, results, etc.
Events – seminars, workshops, orientation sessions, or other relevant events.
Policies or Guidelines – any official notices, instructions, or updates.
Announcements – notifications about exams, admissions, scholarships, merit lists, results, 
or other official communications.
Any Other Important Content – any information that affects students, faculty, or stakeholders.

Rules for extraction:

Do not hallucinate or invent information. Only include what is explicitly present in the content.
Preserve the exact meaning and context of the extracted information.
If a date or detail is ambiguous, note it as such rather than guessing.

Organize the Summary:

Structure the summary in a readable and organized format, for example:

University Name: …
Important Dates: …
Announcements: …
Events: …
Policies/Guidelines: …

Highlight multiple items in each category as a list if needed.

Use complete sentences where possible to make the summary understandable at a glance.

Accuracy and Completeness:

Include all key details present in the input.
Do not add information from outside sources.
Do not make assumptions or predictions.

Output:

Provide a clean, detailed, human-readable summary ready to be read or distributed.

Make sure the output is concise but comprehensive, 
including every important piece of information extracted from the raw input.

Tone & Style:

Professional, formal, and factual.
Avoid repetition.
Prioritize clarity and readability over brevity.'''

    # 4. Construct the messages for the model
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Please summarize the following raw content:\n\n{raw_content}")
    ]

    # 5. Invoke the model to generate the summary
    try:
        response = model.invoke(messages)
        generated_summary = response.content
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        generated_summary = f"API Error: Summarization failed. Details: {str(e)}"

    state["summary"] = generated_summary

    current_time = datetime.now(timezone.utc)
    uni_name = state["UniversityName"]
    url = state["URL"][0] if isinstance(state["URL"], list) else state["URL"]

    insert_university(
        database_url=DATABASE_URL,
        uni_name=uni_name,
        url=url,
        summary=generated_summary,
        time_stamp=current_time
    )

    return state
