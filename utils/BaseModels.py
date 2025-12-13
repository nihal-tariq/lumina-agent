from pydantic import BaseModel, Field
from typing import Literal,  Optional, List


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
    official_url: Optional[str] = Field(description="The main website URL extracted from the context. "
                                                    "Return None if not explicitly present.")
    key_info: str = Field(description="A brief 2-sentence summary of what this entity is based on the text.")


class PostDraft(BaseModel):
    university_name: str = Field(description="The explicit name of the university or institution.")
    post_heading: str = Field(description="A catchy but professional headline. No clickbait.")
    post_content: str = Field(description="The main body text. Engaging, concise, limited emojis.")
    relevant_url: str = Field(description="The specific URL mentioned in the source for students to visit.")
    timestamp: str = Field(description="Key dates, deadlines, or the timestamp of the news.")


class Feedback(BaseModel):
    grade: str = Field(description="The grade of the post: 'good' or 'bad'.")
    feedback: Optional[str] = Field(description="Specific reasons for rejection if grade is bad. If good, leave empty.")