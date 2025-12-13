from state import State
from db.firebase_db import save_post_to_firestore


def save_post_node(state: State):
    """
    Extracts ONLY the required PostDraft fields and saves to Firestore.
    """
    print("\n💾 System: Saving approved post to Firebase...")

    # Strict mapping to your PostDraft requirements
    payload = {
        "university_name": state.get("university_name"),
        "post_heading": state.get("post_heading"),
        "post_content": state.get("post_content"),
        "relevant_url": state.get("relevant_url"),
        "timestamp": state.get("timestamp")
    }

    # Optional: Basic validation to ensure we don't save empty junk
    if not payload["post_content"]:
        print("⚠️ Warning: Post content is empty. Skipping save.")
        return {"info": "Skipped - Empty Content"}

    try:
        doc_id = save_post_to_firestore(payload)
        return {"info": f"Post saved with ID: {doc_id}"}
    except Exception:
        return {"info": "Failed to save post."}