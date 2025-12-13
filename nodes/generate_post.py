import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from state import State
from utils.BaseModels import PostDraft

load_dotenv(override=True)

api_key = os.getenv('EVALUATOR_API_KEY')
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key, temperature=0.5)


def generate_post_node(state: State):
    """Generates a structured post with strict adherence to the summary."""

    structured_llm = llm.with_structured_output(PostDraft)

    # Extract context from state
    summary = state.get("summary", "No summary provided.")
    topic = state.get("topic", "General Update")

    # We provide these from state to help the LLM avoid guessing
    ref_urls = ", ".join(state.get("URL", []))
    ref_time = state.get("TimeStamp", "Check website")

    human_feedback = state.get("human_feedback")
    evaluator_feedback = state.get("evaluator_feedback")
    count = state.get("iteration_count", 0)

    system_instruction = (
        "### ROLE\n"
        "You are a University Admissions Official. Your job is to communicate accurate, "
        "helpful updates to students.\n\n"

        "### STRICT CONTENT RULES\n"
        "1. NO HALLUCINATION: You must ONLY use the provided Summary. Do not invent fees, dates, or programs.\n"
        "2. TONE: Professional, encouraging, and clear. Avoid excessive emojis (max 2-3).\n"
        "3. LENGTH: Concise (100-150 words). Not too long, not too short.\n"
        "4. DISCLAIMER: The content body MUST end with: 'Please refer to the official website for more details.'\n"
        "5. DATA FIELDS: Ensure the University Name, URL, and Date are extracted accurately."
    )

    if human_feedback:
        print(f"\n🤖 System: Refining based on HUMAN feedback...")
        prompt = (
            f"{system_instruction}\n\n"
            f"### TASK: FIX THE PREVIOUS DRAFT\n"
            f"The human reviewer rejected the previous post. Fix it strictly based on this feedback:\n"
            f"🚫 FEEDBACK: {human_feedback}\n\n"
            f"### CONTEXT DATA\n"
            f"Topic: {topic}\n"
            f"Source URL: {ref_urls}\n"
            f"Source Date: {ref_time}\n"
            f"Summary: {summary}\n"
        )
        count = 0

    elif evaluator_feedback:
        print(f"\n🤖 System: Refining based on EVALUATOR feedback...")
        last_feedback = evaluator_feedback[-1] if evaluator_feedback else "General refinement needed."
        prompt = (
            f"{system_instruction}\n\n"
            f"### TASK: REFINE THE DRAFT\n"
            f"Your internal editor found issues. Fix them:\n"
            f"🚫 EDITOR ISSUES: {last_feedback}\n\n"
            f"### CONTEXT DATA\n"
            f"Topic: {topic}\n"
            f"Source URL: {ref_urls}\n"
            f"Source Date: {ref_time}\n"
            f"Summary: {summary}\n"
        )
    else:
        print(f"\n🤖 System: Generating first draft...")
        prompt = (
            f"{system_instruction}\n\n"
            f"### TASK: DRAFT NEW POST\n"
            f"Create a social media update for the following topic.\n\n"
            f"### CONTEXT DATA\n"
            f"Topic: {topic}\n"
            f"Source URL: {ref_urls}\n"
            f"Source Date: {ref_time}\n"
            f"Summary: {summary}\n"
        )

    response = structured_llm.invoke(prompt)

    return {
        "university_name": response.university_name,
        "post_heading": response.post_heading,
        "post_content": response.post_content,
        "relevant_url": response.relevant_url,
        "timestamp": response.timestamp,
        "iteration_count": count + 1,
        "human_feedback": None,
        "evaluator_feedback": None
    }