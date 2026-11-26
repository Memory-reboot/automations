import feedparser
import os
import json
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
        for entry in feed.entries[:5]: # Get top 5 from each feed
            title = entry.title
            if title not in seen_titles:
                seen_titles.add(title)
                all_articles.append({
                    "title": title,
                    "link": entry.link,
                    "source": feed.feed.get("title", "Unknown")
                })
                
    print(f"Fetched {len(all_articles)} unique headlines.")
    
    # 2. Set up Gemini using the NEW google-genai SDK
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY is missing.")
        return

    # Initialize the new client
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
    
    try:
        # Use the new SDK method and a supported model (gemini-2.0-flash)
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        
        rankings = json.loads(response.text)
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        print("\n--- 🏆 TODAY'S TOP STORY ---")
        print(f"Headline: {rankings[0]['title']}")
        print(f"Importance Score: {rankings[0]['score']}/10")
        print("\n(Stage 3 will generate the image for this headline next!)")
        
    except Exception as e:
        print(f"❌ Gemini Error: {e}")

if __name__ == "__main__":
    fetch_news()
