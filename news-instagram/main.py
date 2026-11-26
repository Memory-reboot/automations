import feedparser
import os
import json
import time
from google import genai
from google.genai import types
from datetime import datetime

FEEDS = [
    "https://news.google.com/rss/search?q=India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/india/rss.xml",
    "https://www.thehindu.com/news/national/feeder/default.rss",
    "https://feeds.feedburner.com/ndtvnews-top-stories"
]

def fetch_news():
    print("--- STAGE 2b: DEDUPLICATION & RANKING ---")
    all_articles = []
    seen_titles = set()
    
    # 1. Fetch and Deduplicate
    for url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]: 
            title = entry.title
            if title not in seen_titles:
                seen_titles.add(title)
                all_articles.append({
                    "title": title,
                    "link": entry.link,
                    "source": feed.feed.get("title", "Unknown")
                })
                
    print(f"Fetched {len(all_articles)} unique headlines.")
    
    # 2. Set up Gemini
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY is missing.")
        return

    client = genai.Client(api_key=api_key)
    
    headlines_text = "\n".join([f"- {a['title']}" for a in all_articles])
    prompt = f"""
    You are the chief editor of a major Indian news channel.
    Read the following list of today's headlines:
    
    {headlines_text}
    
    Rank them from 1 to 10 based on how important and impactful they are for a general Indian audience.
    Return ONLY a valid JSON array of objects with the keys "title" and "score". 
    Do not include any other text or markdown formatting.
    """
    
    print("Asking Gemini to rank the news...")
    
    # 3. Retry Logic for 503 Errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            
            # If successful, parse and print
            rankings = json.loads(response.text)
            rankings.sort(key=lambda x: x['score'], reverse=True)
            
            print("\n--- 🏆 TODAY'S TOP STORY ---")
            print(f"Headline: {rankings[0]['title']}")
            print(f"Importance Score: {rankings[0]['score']}/10")
            print("\n(Stage 3 will generate the image for this headline next!)")
            
            break # Exit the loop since it worked!
            
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                print(f"⚠️ Google is busy (503). Retrying in 5 seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
            else:
                print(f"❌ Gemini Error: {e}")
                break

if __name__ == "__main__":
    fetch_news()
