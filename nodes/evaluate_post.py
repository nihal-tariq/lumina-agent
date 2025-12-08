import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


from state import State
from utils.BaseModels import Feedback

load_dotenv(override=True)

api_key = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    api_key=api_key
)


def evaluate_post_node(state: State):
    """Internal AI Evaluator."""
    evaluator_llm = llm.with_structured_output(Feedback)

    summary = state["summary"]
    heading = state["post_heading"]
    content = state["post_content"]

    prompt = (
        f"Compare the generated post to the summary. The post must be accurate to the given summary\n"
        f"Summary: {summary}\n"
        f"Heading: {heading}\n"
        f"Content: {content}\n"
        f"Is this accurate and engaging? Grade it."
    )

    response = evaluator_llm.invoke(prompt)
    return {
        "grade": response.grade,
        "evaluator_feedback": [response.feedback] if response.feedback else []
    }
