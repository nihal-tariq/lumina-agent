from urllib.parse import urlparse


def normalize_url(url: str) -> str:
    url = url.strip()
    if url.startswith("www."):
        url = "https://" + url
    if not urlparse(url).scheme:
        url = "https://" + url
    parsed = urlparse(url)
    if not parsed.netloc or "." not in parsed.netloc:
        raise ValueError(f"Invalid URL format: {url}")
    return url
