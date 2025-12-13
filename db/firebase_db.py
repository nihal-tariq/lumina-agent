import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os


def initialize_firebase():
    # Only initialize if it hasn't been done already
    if not firebase_admin._apps:
        cred_path = "serviceAccountKey.json"

        if not os.path.exists(cred_path):
            raise FileNotFoundError(f"❌ Missing {cred_path}. Place it in project root.")

        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)


def save_post_to_firestore(post_data: dict):
    """
    Saves the specific post data to the 'university_updates' collection.
    """
    try:
        initialize_firebase()
        db = firestore.client()

        # 1. Change Collection Name here
        # This will automatically create "university_updates" if it doesn't exist
        update_time, post_ref = db.collection("university_updates").add(post_data)

        print(f"✅ Database: Saved to 'university_updates' (ID: {post_ref.id})")
        return post_ref.id
    except Exception as e:
        print(f"❌ Database Error: {e}")
        raise e