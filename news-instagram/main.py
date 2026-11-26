import feedparser
import os
import json
import google.generativeai as genai
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
        for entry in feed.entries[:5]: # Get top 5 from each
            title = entry.title
            # Simple deduplication: skip if we've seen a very similar title
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
        print("❌ ERROR: GEMINI_API_KEY is missing. Skipping ranking.")
        return

    genai.configure(api_key=api_key)
    # Use the fast and free 1.5 Flash model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Prepare the prompt
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
        # Ask Gemini and force it to return JSON
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Parse the JSON response
        rankings = json.loads(response.text)
        
        # Sort by highest score
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        print("\n--- 🏆 TODAY'S TOP STORY ---")
        top_story_title = rankings[0]['title']
        print(f"Headline: {top_story_title}")
        print(f"Importance Score: {rankings[0]['score']}/10")
        
        # In the next stage, we will use this top_story_title to generate the image
        print("\n(Stage 3 will generate the image for this headline next!)")
        
    except Exception as e:
        print(f"❌ Gemini Error: {e}")

if __name__ == "__main__":
    fetch_news()
