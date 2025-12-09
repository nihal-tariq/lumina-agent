import os

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from nodes.chat import chat_node
from nodes.check_db_node import check_db_node
from nodes.scrape_with_jina import scrape_with_jina_node
from nodes.evaluate_post import evaluate_post_node
from nodes.generate_post import generate_post_node
from nodes.summarization import summarize
from nodes.human_review import human_review_node
from routes.internal_route import route_internal
from routes.route_human import route_human
from routes.db_route import route_based_on_timestamp
from state import State
from db.db import create_tables

load_dotenv(override=True)

DATABASE_URL = os.getenv('DATABASE_URL')


with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
    checkpointer.setup()

    workflow = StateGraph(State, checkpoint_saver=checkpointer, allow_writes=True)

    workflow.add_node("Chat_node", chat_node)
    workflow.add_node("check_db_node", check_db_node)
    workflow.add_node("scrape_with_jina", scrape_with_jina_node)
    workflow.add_node("Summarize", summarize)
    workflow.add_node("generate_post", generate_post_node)
    workflow.add_node("evaluate_post", evaluate_post_node)
    workflow.add_node("human_review", human_review_node)

    workflow.add_edge(START, "Chat_node")
    workflow.add_edge("Chat_node", "check_db_node")

    # 3. Add Conditional Edge
    # This determines whether to scrape fresh data or fetch cached data
    workflow.add_conditional_edges(
        "check_db_node",
        route_based_on_timestamp,
        {
            "Scrape_with_jina": "scrape_with_jina",
            "generate_post": "generate_post"
        }
    )

    # 4. Converge branches to Summarize
    workflow.add_edge("scrape_with_jina", "Summarize")
    workflow.add_edge("check_db_node", "generate_post")

    workflow.add_edge("Summarize", "generate_post")
    workflow.add_edge("generate_post", "evaluate_post")

    # Add Conditional Edges
    workflow.add_conditional_edges(
        "evaluate_post",
        route_internal,
        {
            "human_review": "human_review",
            "regenerate": "generate_post"
        }
    )
    workflow.add_conditional_edges(
        "human_review",
        route_human,
        {
            "end": END,
            "regenerate": "generate_post"
        }
    )
    create_tables(DATABASE_URL)
    agent = workflow.compile()
