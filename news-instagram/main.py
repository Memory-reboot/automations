"""
Daily AI News Poster — Stage 2b (Hardened & Timeout-Safe)

Fetches daily headlines from free RSS feeds, removes duplicate titles, and selects
the single most important headline for an Indian audience using a 3-tier fallback:
  1. Primary: Google Gemini (google-genai SDK, gemini-3.8-flash, 10s timeout)
  2. Fallback 1: Groq (groq SDK, llama-3.3-70b-versatile, 10s timeout)
  3. Fallback 2: all_articles[0] (deterministic safe default)
"""

from datetime import datetime, timezone
import io
import json
import os
import re
import sys
import threading
from typing import Any, Callable, TypeVar

import feedparser
from google import genai
from google.genai import types
from groq import Groq
from PIL import Image, ImageDraw, ImageFont
import requests

RSS_FEEDS = [
    "https://news.google.com/rss/search?q=India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/india/rss.xml",
    "https://www.thehindu.com/news/national/feeder/default.rss",
    "https://feeds.feedburner.com/ndtvnews-top-stories",
]

GEMINI_MODEL = "gemini-3.8-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"
NETWORK_TIMEOUT = 10
MAX_ARTICLES_PER_FEED = 10
USER_AGENT = "DailyAINewsPoster/1.0 (+https://github.com/)"

T = TypeVar("T")


def log(message: str) -> None:
    """Print immediately with flush=True so GitHub Actions logs stream in real time."""
    print(message, flush=True)


def run_with_timeout(func: Callable[[], T], timeout_seconds: int, label: str) -> T:
    """Execute a blocking call in a daemon thread so hangs never stall the process."""
    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def target() -> None:
        try:
            result_box["value"] = func()
        except BaseException as exc:
            error_box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds)

    if worker.is_alive():
        raise TimeoutError(f"{label} exceeded strict {timeout_seconds}s wall-clock timeout.")
    if "error" in error_box:
        raise error_box["error"]
    return result_box["value"]


def fetch_rss_articles(feed_urls: list[str]) -> list[dict[str, str]]:
    """Fetch entries from each RSS feed with a strict 10-second timeout per request."""
    log("Fetching feeds...")
    articles: list[dict[str, str]] = []
    headers = {"User-Agent": USER_AGENT}

    for url in feed_urls:
        try:
            log(f"[RSS] Requesting {url} (timeout={NETWORK_TIMEOUT}s)...")
            response = run_with_timeout(
                lambda u=url: requests.get(u, headers=headers, timeout=NETWORK_TIMEOUT),
                NETWORK_TIMEOUT,
                f"RSS {url}",
            )
            log(f"[RSS] Response received from {url} (status={response.status_code}).")
            response.raise_for_status()

            parsed_feed = feedparser.parse(response.content)
            entries = getattr(parsed_feed, "entries", [])
            if getattr(parsed_feed, "bozo", False) and not entries:
                log(f"[RSS] Warning: Parse error for {url}: {getattr(parsed_feed, 'bozo_exception', 'unknown')}")
                continue

            feed_meta = getattr(parsed_feed, "feed", {}) or {}
            feed_title = feed_meta.get("title", url) if isinstance(feed_meta, dict) else url
            count = 0
            for entry in entries[:MAX_ARTICLES_PER_FEED]:
                title = (entry.get("title") or "").strip()
                link = (entry.get("link") or "").strip()
                if not title:
                    continue
                articles.append(
                    {
                        "title": title,
                        "link": link,
                        "source": str(feed_title),
                    }
                )
                count += 1
            log(f"[RSS] Collected {count} headlines from {feed_title}.")
        except Exception as exc:
            log(f"[RSS] Failed fetching {url}: {exc}")

    return articles


def deduplicate_articles(articles: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove articles with duplicate exact title strings while preserving order."""
    log("Deduplicating headlines...")
    seen_titles: set[str] = set()
    unique_articles: list[dict[str, str]] = []

    for article in articles:
        title = article["title"]
        if title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(article)

    log(f"Kept {len(unique_articles)} unique headlines (from {len(articles)} total).")
    return unique_articles


def build_ranking_prompt(articles: list[dict[str, str]]) -> str:
    """Create a strict prompt asking the LLM to score headlines and return only JSON."""
    headlines_list = "\n".join(
        f"{idx}. {article['title']}" for idx, article in enumerate(articles, start=1)
    )
    return (
        "You are a senior news editor for an Indian audience.\n"
        "Evaluate the following news headlines based on national significance, public "
        "impact in India, policy/economic relevance, or major global relevance to Indians.\n\n"
        "Assign each headline an integer importance score from 1 to 10 (10 = most important) "
        "and return the ranked list sorted from highest score to lowest score.\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Return ONLY a valid JSON array of objects.\n"
        '- Each object must have exactly two keys: "title" (string, copied verbatim) '
        'and "score" (integer 1-10).\n'
        "- Do NOT include markdown code fences, commentary, or any text outside the JSON array.\n\n"
        "Example format:\n"
        '[{"title": "Example headline", "score": 9}]\n\n'
        f"Headlines:\n{headlines_list}"
    )


def parse_ranked_json(raw_text: str) -> list[dict[str, Any]]:
    """Parse and validate a JSON array of {'title': str, 'score': number} objects."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response received from AI model.")

    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    start_idx = cleaned.find("[")
    end_idx = cleaned.rfind("]")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError("No JSON array found in AI response.")

    data = json.loads(cleaned[start_idx : end_idx + 1])
    if not isinstance(data, list) or not data:
        raise ValueError("AI response JSON is not a non-empty list.")

    validated: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        score = item.get("score")
        if isinstance(title, str) and title.strip() and not isinstance(score, bool):
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                continue
            validated.append({"title": title.strip(), "score": numeric_score})

    if not validated:
        raise ValueError("AI response contained no valid {'title', 'score'} objects.")

    validated.sort(key=lambda x: x["score"], reverse=True)
    return validated


def rank_with_gemini(prompt: str) -> list[dict[str, Any]]:
    """Primary AI tier: rank headlines via Google Gemini with a strict 10-second timeout."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY environment variable is not set.")

    # google-genai HttpOptions.timeout is in milliseconds (10_000 ms = 10 s)
    http_options_kwargs: dict[str, Any] = {"timeout": NETWORK_TIMEOUT * 1000}
    if hasattr(types, "HttpRetryOptions"):
        http_options_kwargs["retry_options"] = types.HttpRetryOptions(attempts=1)

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(**http_options_kwargs),
    )

    def _call_gemini() -> str:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return response.text or ""

    log(f"[AI - Gemini] Calling {GEMINI_MODEL} (timeout={NETWORK_TIMEOUT}s)...")
    try:
        raw_text = run_with_timeout(_call_gemini, NETWORK_TIMEOUT, "Gemini API")
        log("[AI - Gemini] Response received from Gemini API.")
    except Exception as exc:
        log(f"[AI - Gemini] Call failed: {exc}")
        raise

    return parse_ranked_json(raw_text)


def rank_with_groq(prompt: str) -> list[dict[str, Any]]:
    """Fallback 1 tier: rank headlines via Groq with a strict 10-second timeout."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key, timeout=NETWORK_TIMEOUT, max_retries=0)

    def _call_groq() -> str:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON-only news ranking assistant. "
                        "Output strictly a valid JSON array with no extra prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=4096,
            timeout=NETWORK_TIMEOUT,
        )
        return completion.choices[0].message.content or ""

    log(f"[AI - Groq] Calling {GROQ_MODEL} (timeout={NETWORK_TIMEOUT}s)...")
    try:
        raw_text = run_with_timeout(_call_groq, NETWORK_TIMEOUT, "Groq API")
        log("[AI - Groq] Response received from Groq API.")
    except Exception as exc:
        log(f"[AI - Groq] Call failed: {exc}")
        raise

    return parse_ranked_json(raw_text)


def match_article_by_title(
    selected_title: str, all_articles: list[dict[str, str]]
) -> dict[str, str]:
    """Match an AI-selected title back to original article metadata when possible."""
    for article in all_articles:
        if article["title"] == selected_title:
            return article

    normalized = selected_title.casefold().strip()
    for article in all_articles:
        if article["title"].casefold().strip() == normalized:
            return article

    return {"title": selected_title, "link": "", "source": "AI Selected"}


def select_top_headline(all_articles: list[dict[str, str]]) -> tuple[dict[str, str], str]:
    """Select the single most important headline using the 3-tier AI fallback chain."""
    prompt = build_ranking_prompt(all_articles)

    # Primary: Google Gemini
    try:
        ranked = rank_with_gemini(prompt)
        top_title = ranked[0]["title"]
        log(f"Success! Primary AI selected headline (score={ranked[0]['score']}).")
        return match_article_by_title(top_title, all_articles), f"Gemini ({GEMINI_MODEL})"
    except Exception as exc:
        log(f"Primary AI failed ({exc}), trying Fallback 1...")

    # Fallback 1: Groq
    try:
        ranked = rank_with_groq(prompt)
        top_title = ranked[0]["title"]
        log(f"Success! Fallback 1 selected headline (score={ranked[0]['score']}).")
        return match_article_by_title(top_title, all_articles), f"Groq ({GROQ_MODEL})"
    except Exception as exc:
        log(f"Fallback 1 failed ({exc}).")

    # Fallback 2: Deterministic safe default
    log("AI Failed. Falling back to default")
    return all_articles[0], "Default Fallback (all_articles[0])"


def main() -> None:
    """Run the Stage 2b pipeline and always reach the final top-story output."""
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    log(f"Starting Daily News Bot (Stage 2b) at {datetime.now(timezone.utc).isoformat()}")

    raw_articles = fetch_rss_articles(RSS_FEEDS)
    all_articles = deduplicate_articles(raw_articles)

    if not all_articles:
        log("Warning: All RSS feeds failed or returned 0 articles. Using safe fallback.")
        top_story = {
            "title": "No news headlines available today.",
            "link": "",
            "source": "RSS Fallback",
        }
        method_used = "RSS Empty Fallback"
    else:
        top_story, method_used = select_top_headline(all_articles)

    log("\nTODAY'S TOP STORY...")
    log(f"Method : {method_used}")
    log(f"Title  : {top_story['title']}")
    log(f"Source : {top_story.get('source', 'N/A')}")
    log(f"Link   : {top_story.get('link', 'N/A')}")
    create_news_card(top_story)
    log("Success!")


def create_news_card(story):
    print("--- GENERATING IMAGE ---")
    prompt = "Abstract dark blue gradient background with subtle geometric shapes, soft light in the corner, clean minimal, no text, no people"
    url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?width=1080&height=1350&nologo=true"
    
    print("Downloading background...")
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        bg_image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Failed to download background: {e}")
        bg_image = Image.new("RGB", (1080, 1350), color=(15, 25, 45))

    draw = ImageDraw.Draw(bg_image)
    
    try:
        font_headline = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 70)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    except IOError:
        font_headline = ImageFont.load_default()
        font_small = ImageFont.load_default()

    overlay = Image.new("RGBA", bg_image.size, (0, 0, 0, 150))
    bg_image = Image.alpha_composite(bg_image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(bg_image)

    margin = 80
    draw.text((margin, margin), "DAILY NEWS", font=font_small, fill=(255, 100, 100))
    
    headline = story['title']
    words = headline.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + word + " "
        if draw.textlength(test_line, font=font_headline) < (1080 - (margin * 2)):
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    lines.append(current_line.strip())

    y_text = 400
    for line in lines:
        draw.text((margin, y_text), line, font=font_headline, fill=(255, 255, 255))
        y_text += 90

    date_str = datetime.utcnow().strftime("%d %B %Y")
    draw.text((margin, 1200), f"Source: {story['source']}", font=font_small, fill=(200, 200, 200))
    draw.text((margin, 1250), date_str, font=font_small, fill=(200, 200, 200))

    output_path = "post.jpg"
    bg_image.save(output_path, "JPEG", quality=95)
    print(f"✅ Saved image as {output_path}")


if __name__ == "__main__":
    main()
