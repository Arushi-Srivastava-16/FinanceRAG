"""
News Fetcher Module
Fetches financial news from multiple sources and prepares them for RAG ingestion
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict
import json

class NewsFetcher:
    def __init__(self, news_api_key: str = None):
        """Initialize the news fetcher with API key"""
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2/everything"
        
    def fetch_financial_news(self, query: str, days_back: int = 7, max_articles: int = 20) -> List[Dict]:
        """
        Fetch financial news articles for a given query
        
        Args:
            query: Search query (e.g., "AAPL", "Tesla", "Bitcoin")
            days_back: Number of days to look back
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of article dictionaries
        """
        if not self.news_api_key:
            print("⚠️ Warning: NEWS_API_KEY not found. Using fallback data.")
            return self._get_fallback_data(query)
        
        # Calculate date range
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Build query - add financial keywords
        enhanced_query = f"{query} AND (stock OR trading OR market OR finance OR earnings)"
        
        params = {
            'q': enhanced_query,
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': max_articles,
            'apiKey': self.news_api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                print(f"✅ Fetched {len(articles)} articles for '{query}'")
                return self._process_articles(articles)
            else:
                print(f"❌ API Error: {data.get('message', 'Unknown error')}")
                return self._get_fallback_data(query)
                
        except Exception as e:
            print(f"❌ Error fetching news: {str(e)}")
            return self._get_fallback_data(query)
    
    def _process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process and clean articles"""
        processed = []
        
        for article in articles:
            # Skip articles without content
            if not article.get('description') or not article.get('title'):
                continue
                
            processed_article = {
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', article.get('description', '')),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'author': article.get('author', 'Unknown'),
                'published_at': article.get('publishedAt', ''),
                'url': article.get('url', ''),
                'full_text': f"{article.get('title', '')}. {article.get('description', '')} {article.get('content', '')}"
            }
            processed.append(processed_article)
        
        return processed
    
    def _get_fallback_data(self, query: str) -> List[Dict]:
        """Provide fallback data when API is unavailable"""
        fallback_articles = [
            {
                'title': f'{query} Shows Strong Market Performance',
                'description': f'Recent analysis shows {query} demonstrating positive momentum in the market with increased investor interest.',
                'content': f'{query} has been showing strong performance indicators with positive sentiment from market analysts. Trading volumes have increased significantly.',
                'source': 'Market Analysis',
                'author': 'Financial Team',
                'published_at': datetime.now().isoformat(),
                'url': '#',
                'full_text': f'{query} Shows Strong Market Performance. Recent analysis shows {query} demonstrating positive momentum in the market with increased investor interest. {query} has been showing strong performance indicators with positive sentiment from market analysts.'
            },
            {
                'title': f'Analysts Discuss {query} Future Outlook',
                'description': f'Market experts weigh in on the future prospects of {query} with mixed but generally optimistic views.',
                'content': f'Financial analysts are closely monitoring {query} with many expressing cautious optimism about future growth potential.',
                'source': 'Financial News',
                'author': 'Market Experts',
                'published_at': (datetime.now() - timedelta(days=1)).isoformat(),
                'url': '#',
                'full_text': f'Analysts Discuss {query} Future Outlook. Market experts weigh in on the future prospects of {query} with mixed but generally optimistic views. Financial analysts are closely monitoring {query}.'
            }
        ]
        print(f"ℹ️ Using {len(fallback_articles)} fallback articles for '{query}'")
        return fallback_articles
    
    def format_for_rag(self, articles: List[Dict]) -> str:
        """Format articles into a single document for RAG ingestion"""
        formatted_text = ""
        
        for i, article in enumerate(articles, 1):
            formatted_text += f"\n\n--- Article {i} ---\n"
            formatted_text += f"Title: {article['title']}\n"
            formatted_text += f"Source: {article['source']}\n"
            formatted_text += f"Published: {article['published_at']}\n"
            formatted_text += f"Content: {article['full_text']}\n"
            formatted_text += f"URL: {article['url']}\n"
        
        return formatted_text


# Test function
if __name__ == "__main__":
    # Test the fetcher
    fetcher = NewsFetcher()
    articles = fetcher.fetch_financial_news("AAPL", days_back=3, max_articles=5)
    
    print(f"\n📰 Fetched {len(articles)} articles")
    for i, article in enumerate(articles[:3], 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Source: {article['source']}")
        print(f"   Published: {article['published_at']}")