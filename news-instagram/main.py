import feedparser
from datetime import datetime

# List of 100% free RSS feeds (No API keys needed)
FEEDS = [
    "https://news.google.com/rss/search?q=India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/india/rss.xml",
    "https://www.thehindu.com/news/national/feeder/default.rss",
    "https://feeds.feedburner.com/ndtvnews-top-stories"
]

def fetch_news():
    print("--- STAGE 2: FETCHING NEWS ---")
    print(f"Time: {datetime.utcnow()} UTC\n")
    
    all_articles = []
    
    for url in FEEDS:
        print(f"Reading feed: {url}")
        # Parse the RSS feed
        feed = feedparser.parse(url)
        
        # Grab the top 3 articles from each feed to avoid overload
        for entry in feed.entries[:3]:
            article = {
                "title": entry.title,
                "link": entry.link,
                "summary": entry.get("summary", ""),
                "source": feed.feed.get("title", "Unknown Source")
            }
            all_articles.append(article)
    
    print(f"\n✅ Successfully fetched {len(all_articles)} articles.")
    print("\n--- TOP 10 HEADLINES RIGHT NOW ---")
    
    # Print the top 10 headlines to the log
    for i, article in enumerate(all_articles[:10], 1):
        print(f"{i}. [{article['source']}] {article['title']}")
        
    return all_articles

if __name__ == "__main__":
    fetch_news()
