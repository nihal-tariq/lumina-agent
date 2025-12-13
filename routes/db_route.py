from typing import Literal


from datetime import datetime, timezone, timedelta
import dateutil.parser


from state import State


def route_based_on_timestamp(state: State) -> Literal["Scrape_with_jina", "generate_post"]:
    """
    Conditional logic to determine the next step based on URL freshness.

    Returns:
        - "Scrape_with_jina": If URL is missing, NULL, or older than 2 days.
        - "generate_post": If URL exists and is fresh (<= 2 days old).
    """

    timestamp_str = state.get("TimeStamp")

    # Condition 1: URL not found or TimeStamp is explicitly "NULL"
    # We also check for None or empty string just to be safe
    if not timestamp_str or timestamp_str == "NULL":
        print("Routing: URL not found or TimeStamp is NULL -> Scrape_with_jina")
        return "Scrape_with_jina"

    try:
        # Parse the timestamp string.
        # We use dateutil.parser because it's robust against various DB string formats.
        stored_time = dateutil.parser.parse(timestamp_str)

        # Ensure stored_time is timezone-aware. If it's naive, assume UTC.
        if stored_time.tzinfo is None:
            stored_time = stored_time.replace(tzinfo=timezone.utc)

        # Get current time in UTC
        current_time = datetime.now(timezone.utc)

        # Calculate the age of the record
        age = current_time - stored_time

        # Define the threshold (2 days)
        freshness_threshold = timedelta(days=2)

        # Condition 2: URL found but older than 2 days
        if age > freshness_threshold:
            print(f"Routing: Data is old ({age.days} days) -> Scrape_with_jina")
            return "Scrape_with_jina"

        # Condition 3: URL found and fresh
        else:
            print(f"Routing: Data is fresh ({age} old) -> generate_post")
            return "generate_post"

    except (ValueError, TypeError) as e:
        # Fallback: If we can't parse the date, assume we need to re-scrape
        print(f"Routing Error: Could not parse timestamp '{timestamp_str}'. Error: {e} -> Scrape_with_jina")
        return "Scrape_with_jina"
