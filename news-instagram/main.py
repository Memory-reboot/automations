import feedparser
import os
import json
import time
import requests
from datetime import datetime

FEEDS = [
    "https://news.google.com/rss/search?q=India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/india/rss.xml",
    "https://www.thehindu.com/news/national/feeder/default.rss",
    "https://feeds.feedburner.com/ndtvnews-top-stories"
]

def fetch_news():
    print("--- STAGE 2b: DEDUPLICATION & RANKING (GROQ) ---")
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
    
    # 2. Set up Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY is missing.")
        return

    headlines_text = "\n".join([f"- {a['title']}" for a in all_articles])
    
    # Groq Prompt (system prompt ensures JSON)
    messages = [
        {
            "role": "system",
            "content": "You are the chief editor of a major Indian news channel. Rank the provided headlines from 1 to 10 based on importance for a general Indian audience. Return ONLY a valid JSON array of objects with the keys 'title' and 'score'. Do not include any other text or markdown formatting."
        },
        {
            "role": "user",
            "content": f"Here are today's headlines:\n{headlines_text}"
        }
    ]

    print("Asking Groq to rank the news...")
    
    # 3. Retry Logic for Groq
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Call Groq's API (OpenAI compatible)
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile", # A very smart and fast free model
                    "messages": messages,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"}
                },
                timeout=30
            )
            
            response.raise_for_status() # Check for HTTP errors
            data = response.json()
            
            # Extract the text response from the JSON structure
            response_text = data['choices'][0]['message']['content']
            
            # Parse the JSON string returned by Groq
            # (Sometimes models wrap JSON in markdown block, so we clean it)
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json\n", "").replace("```", "").strip()
                
            rankings = json.loads(response_text)
            
            # Handle cases where the model returns a dict instead of a list
            if isinstance(rankings, dict):
                # If it returns {"rankings": [...]}, extract the list
                if "rankings" in rankings:
                    rankings = rankings["rankings"]
                else:
                    # If it returns a single object, put it in a list
                    rankings = [rankings]

            rankings.sort(key=lambda x: x['score'], reverse=True)
            
            print("\n--- 🏆 TODAY'S TOP STORY ---")
            print(f"Headline: {rankings[0]['title']}")
            print(f"Importance Score: {rankings[0]['score']}/10")
            print("\n(Stage 3 will generate the image for this headline next!)")
            
            break # Exit the loop since it worked!
            
        except Exception as e:
            print(f"❌ Groq Error: {e}")
            if attempt < max_retries - 1:
                print(f"⚠️ Retrying in 5 seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
            else:
                print("❌ Max retries reached. Giving up.")

if __name__ == "__main__":
    fetch_news()
