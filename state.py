from typing import TypedDict, List, Optional, Annotated
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


def add_lists_safe(existing: Optional[List[str]], new: Optional[List[str]]) -> List[str]:
    existing = existing or []
    new = new or []
    return existing + new


class State(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    UniversityName: str
    URL: List[str]
    URL_info: List[str]
    Content: str
    fetched_content: str
    TimeStamp: str
    info: str
    topic: str
    summary: str

    post_heading: str
    post_content: str
    university_name: str
    relevant_url: str
    timestamp: str

    evaluator_feedback: Annotated[List[str], add_lists_safe]

    grade: str
    iteration_count: int

    human_feedback: Optional[str]
    approved: bool
    chroma_context: str
