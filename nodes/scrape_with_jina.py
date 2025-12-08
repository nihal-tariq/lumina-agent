from datetime import datetime, timezone

from state import State
from utils.scrape import scrape_urls_with_jina


def scrape_with_jina_node(state: State) -> State:
    """
    Consumes state['URL'] (list) and returns merged 'Content' containing all scraped pages.
    Returns partial state containing 'Content' and 'TimeStamp'.
    """
    urls = state.get("URL", [])
    if not isinstance(urls, list):
        # normalize to list for safety
        urls = [urls] if urls else []

    current_time = datetime.now(timezone.utc)

    if not urls:
        return {"Content": "", "TimeStamp": current_time.isoformat()}

    scraped_map = scrape_urls_with_jina(urls)
    # Combine contents into single Content field (separated by markers)
    parts = []
    for u, c in scraped_map.items():
        parts.append(f"--- START {u} ---\n{c}\n--- END {u} ---\n")
    combined = "\n".join(parts)

    return {
        "Content": combined,
        "TimeStamp": current_time.isoformat()
    }

