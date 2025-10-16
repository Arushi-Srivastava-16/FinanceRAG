"""
Sentiment Analyzer Module
Analyzes sentiment of financial news articles using VADER and TextBlob
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from typing import Dict, List
import statistics

class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzer with VADER"""
        self.vader = SentimentIntensityAnalyzer()
        
        # Financial keywords that amplify sentiment
        self.positive_keywords = [
            'profit', 'gain', 'bull', 'surge', 'rally', 'growth', 'outperform',
            'breakthrough', 'record', 'soar', 'boom', 'optimistic', 'upgrade'
        ]
        self.negative_keywords = [
            'loss', 'decline', 'bear', 'crash', 'plunge', 'fall', 'underperform',
            'concern', 'risk', 'warning', 'downgrade', 'bankruptcy', 'debt'
        ]
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and label
        """
        # VADER analysis (better for social media and short texts)
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob analysis (good for longer texts)
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity
        
        # Combined score (weighted average)
        combined_score = (vader_scores['compound'] * 0.6) + (textblob_score * 0.4)
        
        # Adjust for financial keywords
        keyword_adjustment = self._analyze_keywords(text.lower())
        final_score = combined_score + keyword_adjustment
        
        # Clamp between -1 and 1
        final_score = max(-1, min(1, final_score))
        
        # Determine sentiment label
        sentiment_label = self._get_sentiment_label(final_score)
        
        # Confidence based on score magnitude
        confidence = abs(final_score)
        
        return {
            'score': round(final_score, 3),
            'label': sentiment_label,
            'confidence': round(confidence, 3),
            'vader_compound': round(vader_scores['compound'], 3),
            'vader_positive': round(vader_scores['pos'], 3),
            'vader_negative': round(vader_scores['neg'], 3),
            'vader_neutral': round(vader_scores['neu'], 3),
            'textblob_score': round(textblob_score, 3)
        }
    
    def analyze_articles(self, articles: List[Dict]) -> Dict:
        """
        Analyze sentiment of multiple articles
        
        Args:
            articles: List of article dictionaries with 'full_text' key
            
        Returns:
            Dictionary with aggregate sentiment analysis
        """
        if not articles:
            return self._get_neutral_analysis()
        
        article_sentiments = []
        
        for article in articles:
            text = article.get('full_text', '') or article.get('content', '') or article.get('description', '')
            if text:
                sentiment = self.analyze_text(text)
                sentiment['title'] = article.get('title', 'Unknown')
                sentiment['source'] = article.get('source', 'Unknown')
                sentiment['url'] = article.get('url', '#')
                article_sentiments.append(sentiment)
        
        if not article_sentiments:
            return self._get_neutral_analysis()
        
        # Calculate aggregate metrics
        scores = [s['score'] for s in article_sentiments]
        avg_score = statistics.mean(scores)
        median_score = statistics.median(scores)
        
        # Count sentiments
        positive_count = sum(1 for s in article_sentiments if s['label'] in ['Positive', 'Very Positive'])
        negative_count = sum(1 for s in article_sentiments if s['label'] in ['Negative', 'Very Negative'])
        neutral_count = len(article_sentiments) - positive_count - negative_count
        
        # Overall sentiment
        overall_label = self._get_sentiment_label(avg_score)
        overall_confidence = statistics.mean([s['confidence'] for s in article_sentiments])
        
        return {
            'overall_score': round(avg_score, 3),
            'overall_label': overall_label,
            'overall_confidence': round(overall_confidence, 3),
            'median_score': round(median_score, 3),
            'total_articles': len(article_sentiments),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_distribution': {
                'positive': round(positive_count / len(article_sentiments) * 100, 1),
                'negative': round(negative_count / len(article_sentiments) * 100, 1),
                'neutral': round(neutral_count / len(article_sentiments) * 100, 1)
            },
            'article_sentiments': article_sentiments,
            'most_positive': max(article_sentiments, key=lambda x: x['score']),
            'most_negative': min(article_sentiments, key=lambda x: x['score'])
        }
    
    def _analyze_keywords(self, text: str) -> float:
        """Analyze financial keywords and return adjustment score"""
        adjustment = 0.0
        
        # Check for positive keywords
        for keyword in self.positive_keywords:
            if keyword in text:
                adjustment += 0.05
        
        # Check for negative keywords
        for keyword in self.negative_keywords:
            if keyword in text:
                adjustment -= 0.05
        
        # Clamp adjustment
        return max(-0.3, min(0.3, adjustment))
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert score to label"""
        if score >= 0.5:
            return "Very Positive"
        elif score >= 0.1:
            return "Positive"
        elif score <= -0.5:
            return "Very Negative"
        elif score <= -0.1:
            return "Negative"
        else:
            return "Neutral"
    
    def _get_neutral_analysis(self) -> Dict:
        """Return neutral analysis when no data available"""
        return {
            'overall_score': 0.0,
            'overall_label': 'Neutral',
            'overall_confidence': 0.0,
            'median_score': 0.0,
            'total_articles': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'sentiment_distribution': {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0
            },
            'article_sentiments': []
        }
    
    def get_trading_signal(self, sentiment_analysis: Dict) -> Dict:
        """
        Generate a simple trading signal based on sentiment
        
        Args:
            sentiment_analysis: Result from analyze_articles()
            
        Returns:
            Dictionary with trading signal and reasoning
        """
        score = sentiment_analysis['overall_score']
        confidence = sentiment_analysis['overall_confidence']
        
        # Strong positive sentiment
        if score >= 0.5 and confidence >= 0.6:
            signal = "🟢 STRONG BUY"
            reasoning = "Very positive sentiment with high confidence"
        elif score >= 0.2 and confidence >= 0.5:
            signal = "🟢 BUY"
            reasoning = "Positive sentiment detected"
        # Strong negative sentiment
        elif score <= -0.5 and confidence >= 0.6:
            signal = "🔴 STRONG SELL"
            reasoning = "Very negative sentiment with high confidence"
        elif score <= -0.2 and confidence >= 0.5:
            signal = "🔴 SELL"
            reasoning = "Negative sentiment detected"
        # Neutral or uncertain
        else:
            signal = "🟡 HOLD"
            reasoning = "Mixed or neutral sentiment"
        
        return {
            'signal': signal,
            'reasoning': reasoning,
            'score': score,
            'confidence': confidence
        }


# Test function
if __name__ == "__main__":
    # Test the analyzer
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "Apple stock surges to record high on strong earnings report and positive outlook",
        "Tesla faces major concerns as production delays continue and CEO controversy grows",
        "Market remains stable with mixed signals from various sectors"
    ]
    
    print("🧪 Testing Sentiment Analyzer\n")
    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze_text(text)
        print(f"{i}. Text: {text[:60]}...")
        print(f"   Score: {result['score']} | Label: {result['label']} | Confidence: {result['confidence']}")
        print()