"""
Daily AI News Poster — Stage 2b: RSS Fetching, Deduplication & Multi-Tier AI Ranking

Fetches daily headlines from free RSS feeds, removes duplicate titles, and selects
the single most important headline for an Indian audience using a 3-tier fallback:
  1. Primary: Google Gemini (google-genai SDK, gemini-3.8-flash)
  2. Fallback 1: Groq (groq SDK, llama-3.3-70b-versatile)
  3. Fallback 2: First deduplicated headline (deterministic safe default)
"""

from datetime import datetime, timezone
import json
import os
import re
from typing import Any

import feedparser
from google import genai
from google.genai import types
from groq import Groq
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
REQUEST_TIMEOUT_SECONDS = 15
MAX_ARTICLES_PER_FEED = 10
USER_AGENT = "DailyAINewsPoster/1.0 (+https://github.com/)"


def fetch_rss_articles(feed_urls: list[str]) -> list[dict[str, str]]:
    """Fetch entries from each RSS feed without failing the whole run if one feed errors."""
    print("Fetching feeds...")
    articles: list[dict[str, str]] = []
    headers = {"User-Agent": USER_AGENT}

    for url in feed_urls:
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            parsed_feed = feedparser.parse(response.content)

            if getattr(parsed_feed, "bozo", False) and not parsed_feed.entries:
                print(f"  Warning: Could not parse feed {url}: {parsed_feed.bozo_exception}")
                continue

            feed_title = parsed_feed.feed.get("title", url)
            count = 0
            for entry in parsed_feed.entries[:MAX_ARTICLES_PER_FEED]:
                title = (entry.get("title") or "").strip()
                link = (entry.get("link") or "").strip()
                if not title:
                    continue
                articles.append(
                    {
                        "title": title,
                        "link": link,
                        "source": feed_title,
                    }
                )
                count += 1
            print(f"  Fetched {count} headlines from {feed_title}")
        except requests.RequestException as exc:
            print(f"  Warning: Network error fetching {url}: {exc}")
        except Exception as exc:
            print(f"  Warning: Unexpected error processing {url}: {exc}")

    return articles


def deduplicate_articles(articles: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove articles with duplicate exact title strings while preserving order."""
    print("Deduplicating headlines...")
    seen_titles: set[str] = set()
    unique_articles: list[dict[str, str]] = []

    for article in articles:
        title = article["title"]
        if title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(article)

    print(f"  Kept {len(unique_articles)} unique headlines (from {len(articles)} total).")
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
        "- Each object must have exactly two keys: \"title\" (string, copied verbatim) "
        "and \"score\" (integer 1-10).\n"
        "- Do NOT include markdown code fences, commentary, or any text outside the JSON array.\n\n"
        "Example format:\n"
        '[{"title": "Example headline", "score": 9}]\n\n'
        f"Headlines:\n{headlines_list}"
    )


def parse_ranked_json(raw_text: str) -> list[dict[str, Any]]:
    """Parse and validate a JSON array of {'title': str, 'score': int/float} objects."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response received from AI model.")

    cleaned = raw_text.strip()
    # Strip optional markdown code fences if a model adds them despite instructions
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Extract the JSON array slice if extra leading/trailing text slipped in
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
    """Primary AI tier: rank headlines using Google Gemini via google-genai SDK."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return parse_ranked_json(response.text or "")


def rank_with_groq(prompt: str) -> list[dict[str, Any]]:
    """Fallback 1 tier: rank headlines using Groq API with llama-3.3-70b-versatile."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)
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
    )
    content = completion.choices[0].message.content or ""
    return parse_ranked_json(content)


def match_article_by_title(
    selected_title: str, articles: list[dict[str, str]]
) -> dict[str, str]:
    """Match an AI-selected title back to the original article metadata when possible."""
    for article in articles:
        if article["title"] == selected_title:
            return article

    normalized = selected_title.casefold().strip()
    for article in articles:
        if article["title"].casefold().strip() == normalized:
            return article

    return {"title": selected_title, "link": "", "source": "AI Selected"}


def select_top_headline(articles: list[dict[str, str]]) -> tuple[dict[str, str], str]:
    """Select the single most important headline using the 3-tier AI fallback chain."""
    prompt = build_ranking_prompt(articles)

    # Primary: Google Gemini
    print(f"Ranking headlines with Primary AI ({GEMINI_MODEL})...")
    try:
        ranked = rank_with_gemini(prompt)
        top_title = ranked[0]["title"]
        print(f"Success! Primary AI selected headline (score={ranked[0]['score']}).")
        return match_article_by_title(top_title, articles), f"Gemini ({GEMINI_MODEL})"
    except Exception as exc:
        print(f"Primary AI failed ({exc}), trying Fallback 1...")

    # Fallback 1: Groq
    print(f"Ranking headlines with Fallback 1 ({GROQ_MODEL})...")
    try:
        ranked = rank_with_groq(prompt)
        top_title = ranked[0]["title"]
        print(f"Success! Fallback 1 selected headline (score={ranked[0]['score']}).")
        return match_article_by_title(top_title, articles), f"Groq ({GROQ_MODEL})"
    except Exception as exc:
        print(f"Fallback 1 failed ({exc}), trying Fallback 2...")

    # Fallback 2: Deterministic safe default
    print("Using Fallback 2: Selecting first deduplicated headline as default.")
    return articles[0], "Fallback 2 (First Deduplicated Headline)"


def main() -> None:
    """Run the Stage 2b pipeline: fetch RSS feeds, deduplicate, and select top headline."""
    print(f"Starting Daily News Bot (Stage 2b) at {datetime.now(timezone.utc).isoformat()}")

    raw_articles = fetch_rss_articles(RSS_FEEDS)
    if not raw_articles:
        print("Error: No articles could be fetched from any RSS feed. Exiting safely.")
        return

    unique_articles = deduplicate_articles(raw_articles)
    if not unique_articles:
        print("Error: No valid headlines remained after deduplication. Exiting safely.")
        return

    top_article, method_used = select_top_headline(unique_articles)

    print("\n--- SELECTED TOP HEADLINE ---")
    print(f"Method : {method_used}")
    print(f"Title  : {top_article['title']}")
    print(f"Source : {top_article.get('source', 'N/A')}")
    print(f"Link   : {top_article.get('link', 'N/A')}")
    print("Success!")


if __name__ == "__main__":
    main()
