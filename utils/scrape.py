import os

import requests
from typing import List, Dict

from utils.normalize_urls import normalize_url

JINA_READER_PREFIX = "https://r.jina.ai/"
JINA_API_KEY = os.getenv("JINA_API_KEY")


def scrape_urls_with_jina(urls: List[str]) -> Dict[str, str]:
    """
    Returns a mapping normalized_url -> content (string) or error string
    """
    out: Dict[str, str] = {}
    if not JINA_API_KEY:
        raise EnvironmentError("JINA_API_KEY not set in environment")

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Engine": "direct",
        "X-Return-Format": "text",
        "X-With-Images-Summary": "all",
        "X-With-Links-Summary": "all"
    }

    for url in urls:
        try:
            n = normalize_url(url)
        except ValueError as e:
            out[url] = f"[INVALID URL] {e}"
            continue

        target = f"{JINA_READER_PREFIX}{n}"
        try:
            resp = requests.get(target, headers=headers, timeout=30)
            resp.raise_for_status()
            out[n] = resp.text
        except requests.RequestException as e:
            out[n] = f"[JINA ERROR] {e}"

    return out
