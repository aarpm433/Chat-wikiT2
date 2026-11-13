import os
import time
import requests
import re
from bs4 import BeautifulSoup
from markdownify import markdownify as md

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "wiki_data")

API_URL = "https://terraria.wiki.gg/api.php"
HEADERS = {"User-Agent": "ChatWikiT-TerrariaIngest/1.0 (local-dev)"}

session = requests.Session()
session.headers.update(HEADERS)


def api_get(params, retries=6):
    """GET with timeout, retries, backoff, and visible HTTP status."""
    for attempt in range(1, retries + 1):
        try:
            res = session.get(API_URL, params=params, timeout=15)
            if res.status_code == 429 or res.status_code >= 500:
                wait = int(res.headers.get("Retry-After", 5 * attempt))
                print(f"HTTP {res.status_code}; waiting {wait}s (attempt {attempt}/{retries})...")
                time.sleep(wait)
                continue
            res.raise_for_status()
            return res.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP {e.response.status_code} for {params.get('page') or params.get('titles') or params.get('list')}; giving up on this one.")
            return None
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"{e.__class__.__name__}; retry {attempt}/{retries} in 5s...")
            time.sleep(5)
    return None


def get_all_article_titles(max_pages=None):
    """Main-namespace, non-redirect page titles."""
    titles = []
    apcontinue = None
    print("Fetching page index from terraria.wiki.gg...")
    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": 0,
            "apfilterredir": "nonredirects",  # redirects have no useful text
            "aplimit": 500,
            "format": "json",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue

        data = api_get(params)
        if data is None:
            print("Could not fetch the page index; stopping with what we have.")
            break

        for p in data.get("query", {}).get("allpages", []):
            titles.append(p["title"])
            if max_pages and len(titles) >= max_pages:
                return titles[:max_pages]

        if "continue" in data:
            apcontinue = data["continue"]["apcontinue"]
            time.sleep(0.5)
        else:
            break
        for p in data.get("query", {}).get("allpages", []):
            if is_translation(p["title"]):
                continue
    titles.append(p["title"])

    print(f"Discovered {len(titles)} total articles.")
    return titles


def safe_filename(title):
    name = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in title)
    return name.replace(" ", "_")[:150] + ".md"  # keep well under filesystem limits



# Language subpages like "Copper Ore/id" are noise for an English chatbot
LANG_SUFFIXES = {
    "ar", "bg", "cs", "da", "de", "el", "es", "fi", "fr", "hu", "id", "it", "ja",
    "ko", "lt", "nl", "no", "pl", "pt", "pt-br", "ro", "ru", "sv", "th", "tr",
    "uk", "vi", "zh", "zh-tw", "zh-cn",
}

def is_translation(title):
    return "/" in title and title.rsplit("/", 1)[1].lower() in LANG_SUFFIXES


def html_to_markdown(html):
    soup = BeautifulSoup(html, "html.parser")
    for sel in (".navbox", ".toc", "#toc", ".mw-editsection", ".noprint",
                ".mw-empty-elt", "script", "style"):
        for el in soup.select(sel):
            el.decompose()
    text = md(str(soup), heading_style="ATX", strip=["img", "a"])
    text = re.sub(r"^\|(?:\s*:?-+:?\s*\|)+\s*$\n?", "", text, flags=re.M)  # drop |---| rows
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def download_page_extract(title):
    """Returns (success, was_skipped)."""
    filepath = os.path.join(OUTPUT_DIR, safe_filename(title))
    if os.path.exists(filepath):
        return True, True

    data = api_get({
        "action": "parse", "page": title, "prop": "text",
        "format": "json", "redirects": 1, "disableeditsection": 1,
    })
    if not data or "parse" not in data:
        return False, False

    body = html_to_markdown(data["parse"]["text"]["*"])
    if not body:
        return False, False

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {data['parse']['title']}\n\n{body}")
    return True, False


def main(max_pages=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    titles = get_all_article_titles(max_pages=max_pages)
    print(f"Checking {len(titles)} pages against '{OUTPUT_DIR}'...")
    new_downloads = skipped = failed = 0

    for i, title in enumerate(titles, start=1):
        success, was_skipped = download_page_extract(title)
        if was_skipped:
            skipped += 1
        elif success:
            new_downloads += 1
            time.sleep(0.5)  # rate limit only when we hit the network
        else:
            failed += 1

        if i % 250 == 0 or i == len(titles):
            print(f"Processed: {i}/{len(titles)} | Skipped: {skipped} | New: {new_downloads} | Empty/failed: {failed}")

    print(f"\nDone. {skipped + new_downloads} files on disk ({new_downloads} newly downloaded).")


if __name__ == "__main__":
    main(max_pages=None)