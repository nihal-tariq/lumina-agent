from pydantic import BaseModel, Field
from typing import Literal,  Optional


class IntakeComplete(BaseModel):
    """Call this ONLY when you have the University Name, Topic, AND a valid URL."""
    university_name: str = Field(description="The final university name.")
    topic: str = Field(description="The specific topic (e.g., Admissions).")
    url: str = Field(description="The website URL. If RAG failed, this must come from user input.")
    final_message: str = Field(description="A polite closing message to the user.")


class RetrievedKnowledge(BaseModel):
    """The intelligent extraction from the vector database documents."""
    found: bool = Field(description="True if relevant university info was found in the context.")
    official_name: str = Field(description="The canonical name of the university found in text.")
    official_url: Optional[str] = Field(description="The main website URL extracted from the context. Return None if not explicitly present.")
    key_info: str = Field(description="A brief 2-sentence summary of what this entity is based on the text.")


class PostDraft(BaseModel):
    post_heading: str = Field(description="A catchy, engaging headline for the post.")
    post_content: str = Field(description="The main body content of the social media post.")


# The structure required for the Evaluator's output
class Feedback(BaseModel):
    grade: Literal["good", "needs_improvement"] = Field(
        description="Decide if the post is high quality and accurate to the summary."
    )
    feedback: str = Field(
        description="Specific instructions on how to fix wording, tone, or missing details if needs_improvement."
    )