import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


from state import State
from utils.BaseModels import PostDraft


load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0.6
)


def generate_post_node(state: State):
    """Generates post taking into account EITHER automated feedback OR human feedback."""

    structured_llm = llm.with_structured_output(PostDraft)

    summary = state["summary"]
    topic = state["topic"]

    human_feedback = state.get("human_feedback")

    evaluator_feedback = state.get("evaluator_feedback")

    count = state.get("iteration_count", 0)

    if human_feedback:
        print(f"\n🤖 System: Refining based on HUMAN feedback...")
        prompt = (
            f"You are a social media expert. The user (human) rejected the previous draft.\n"
            f"Topic: {topic}\n"
            f"Original Summary: {summary}\n"
            f"Human Feedback (CRITICAL - FIX THIS): {human_feedback}\n"
            f"Generate a revised post heading and content."
        )
        # Reset count on human intervention to give the LLM fresh attempts
        count = 0
    elif evaluator_feedback:
        print(f"\n🤖 System: Refining based on EVALUATOR feedback...")
        prompt = (
            f"You are a social media expert. The internal editor rejected the previous draft.\n"
            f"Topic: {topic}\n"
            f"Original Summary: {summary}\n"
            f"Editor Feedback: {evaluator_feedback}\n"
            f"Generate a revised post heading and content."
        )
    else:
        print(f"\n🤖 System: Generating first draft...")
        prompt = (
            f"### ROLE \n"
            f"You are a University Admissions Consultant and Social Media Expert. Your goal is to help students navigate complex admissions information clearly and accurately.\n\n"

            f"### TASK \n"
            f"Draft a high-value social media post about: '{topic}'.\n"
            f"Base your response strictly on the following Context Summary.\n\n"

            f"### CONTEXT SUMMARY \n"
            f"{summary}\n\n"

            f"### CRITICAL GUIDELINES (STRICT COMPLIANCE REQUIRED) \n"
            f"1. NO HALLUCINATIONS: Do not invent dates, fees, eligibility criteria, or URL links. If a piece of information is not explicitly in the summary, do not include it. \n"
            f"2. DATES ARE VITAL: You must identify and list ALL dates mentioned (application deadlines, entry test dates, merit list displays). Do not skip any.\n"
            f"3. CLARITY: Use professional but encouraging language suitable for aspiring university students.\n"

            f"### OUTPUT FORMAT \n"
            f"Return the response in the following structured format:\n\n"

            f"**HEADLINE:** [Create a  relevant headline]\n\n"

            f"**POST CONTENT:**\n"
            f"[Write the engaging body of the post here. Clearly mention the name of the institute in the content body. Explain the opportunity, why it matters, and general requirements based on the summary.]\n\n"

            f"**📅 IMPORTANT DATES:**\n"
            f"[Bulleted list of all specific dates found in the text. If no dates are provided in the summary, write 'Please check the official website for updated schedules.']\n\n"

            f"**📝 ACTION ITEMS:**\n"
            f"[Clear next steps for the student based on the summary (e.g., 'Apply online,' 'Prepare documents').]"
            f" Return Heading and Content"
        )

    response = structured_llm.invoke(prompt)

    return {
        "post_heading": response.post_heading,
        "post_content": response.post_content,
        "iteration_count": count + 1,
        # Clear feedbacks after use so they don't persist wrongly
        "human_feedback": None,
        "evaluator_feedback": None
    }
