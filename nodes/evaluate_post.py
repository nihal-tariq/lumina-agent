import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from state import State
from utils.BaseModels import Feedback

load_dotenv(override=True)

api_key = os.getenv('EVALUATOR_API_KEY')

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,  # Low temp for strict grading
    api_key=api_key
)


def evaluate_post_node(state: State):
    """Internal AI Evaluator checking for accuracy and required fields."""
    evaluator_llm = llm.with_structured_output(Feedback)

    # Inputs
    summary = state.get("summary")

    # Generated Outputs to Grade
    uni_name = state.get("university_name")
    heading = state.get("post_heading")
    content = state.get("post_content")
    url = state.get("relevant_url")
    timestamp = state.get("timestamp")

    prompt = (
        f"### ROLE: Senior Editor\n"
        f"You are grading a social media post against a source summary.\n\n"

        f"### SOURCE SUMMARY\n"
        f"{summary}\n\n"

        f"### GENERATED POST\n"
        f"University: {uni_name}\n"
        f"Heading: {heading}\n"
        f"Content: {content}\n"
        f"URL: {url}\n"
        f"Time: {timestamp}\n\n"

        f"### CHECKLIST (CRITICAL)\n"
        f"1. ACCURACY: Does the content match the summary? (No hallucinations)\n"
        f"2. DISCLAIMER: Does the content body end with 'refer to the official website'?\n"
        f"3. COMPLETENESS: Are University Name, URL, and Date present?\n"
        f"4. TONE: Is it professional (not too many emojis)?\n\n"

        f"If ANY of these fail, grade 'bad' and explain why. If all pass, grade 'good'."
    )

    response = evaluator_llm.invoke(prompt)

    return {
        "grade": response.grade,
        "evaluator_feedback": [response.feedback] if response.feedback else []
    }