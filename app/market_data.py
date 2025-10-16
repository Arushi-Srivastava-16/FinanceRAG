"""
Market Data Module
Fetches real-time and historical market data using yfinance
Provides technical indicators and price analysis
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

class MarketData:
    def __init__(self):
        """Initialize market data fetcher"""
        self.cache = {}  # Simple cache for repeated queries
        
    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get comprehensive stock information
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
            
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'symbol': ticker.upper(),
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
            }
        except Exception as e:
            print(f"❌ Error fetching info for {ticker}: {str(e)}")
            return self._get_fallback_info(ticker)
    
    def get_price_history(self, ticker: str, period: str = "1mo") -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with historical prices
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            print(f"❌ Error fetching history for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, ticker: str, period: str = "3mo") -> Dict:
        """
        Calculate technical indicators
        
        Args:
            ticker: Stock ticker symbol
            period: Historical period to analyze
            
        Returns:
            Dictionary with technical indicators
        """
        try:
            df = self.get_price_history(ticker, period)
            
            if df.empty:
                return self._get_fallback_indicators(ticker)
            
            # Calculate indicators
            close_prices = df['Close']
            
            # Simple Moving Averages
            sma_20 = close_prices.rolling(window=20).mean().iloc[-1] if len(close_prices) >= 20 else None
            sma_50 = close_prices.rolling(window=50).mean().iloc[-1] if len(close_prices) >= 50 else None
            
            # RSI (Relative Strength Index)
            rsi = self._calculate_rsi(close_prices, period=14)
            
            # MACD
            macd_line, signal_line = self._calculate_macd(close_prices)
            
            # Volatility (standard deviation)
            volatility = close_prices.pct_change().std() * 100
            
            # Price change
            current_price = close_prices.iloc[-1]
            price_7d_ago = close_prices.iloc[-7] if len(close_prices) >= 7 else close_prices.iloc[0]
            price_30d_ago = close_prices.iloc[-30] if len(close_prices) >= 30 else close_prices.iloc[0]
            
            change_7d = ((current_price - price_7d_ago) / price_7d_ago * 100)
            change_30d = ((current_price - price_30d_ago) / price_30d_ago * 100)
            
            return {
                'current_price': round(current_price, 2),
                'sma_20': round(sma_20, 2) if sma_20 else None,
                'sma_50': round(sma_50, 2) if sma_50 else None,
                'rsi': round(rsi, 2) if rsi else None,
                'rsi_signal': self._get_rsi_signal(rsi) if rsi else 'N/A',
                'macd': round(macd_line, 2) if macd_line else None,
                'macd_signal': round(signal_line, 2) if signal_line else None,
                'macd_trend': self._get_macd_signal(macd_line, signal_line) if macd_line and signal_line else 'N/A',
                'volatility': round(volatility, 2),
                'change_7d': round(change_7d, 2),
                'change_30d': round(change_30d, 2),
                'trend_7d': 'Bullish 📈' if change_7d > 0 else 'Bearish 📉',
                'trend_30d': 'Bullish 📈' if change_30d > 0 else 'Bearish 📉',
            }
        except Exception as e:
            print(f"❌ Error calculating indicators for {ticker}: {str(e)}")
            return self._get_fallback_indicators(ticker)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period:
                return None
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1]
        except:
            return None
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD indicator"""
        try:
            if len(prices) < 26:
                return None, None
            
            ema_12 = prices.ewm(span=12, adjust=False).mean()
            ema_26 = prices.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            return macd_line.iloc[-1], signal_line.iloc[-1]
        except:
            return None, None
    
    def _get_rsi_signal(self, rsi: float) -> str:
        """Interpret RSI value"""
        if rsi > 70:
            return "Overbought 🔴"
        elif rsi < 30:
            return "Oversold 🟢"
        else:
            return "Neutral 🟡"
    
    def _get_macd_signal(self, macd: float, signal: float) -> str:
        """Interpret MACD values"""
        if macd > signal:
            return "Bullish 📈"
        else:
            return "Bearish 📉"
    
    def get_market_summary(self, ticker: str) -> str:
        """
        Generate a comprehensive market summary for RAG context
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Formatted string with market data
        """
        info = self.get_stock_info(ticker)
        indicators = self.calculate_technical_indicators(ticker)
        
        summary = f"""
📊 MARKET DATA FOR {ticker.upper()}

Company Information:
- Name: {info['name']}
- Sector: {info['sector']}
- Industry: {info['industry']}

Current Trading Data:
- Current Price: ${info['current_price']}
- Previous Close: ${info['previous_close']}
- Day Range: ${info['day_low']} - ${info['day_high']}
- Volume: {info['volume']:,}
- Market Cap: ${info['market_cap']:,}

Technical Indicators:
- RSI (14): {indicators.get('rsi', 'N/A')} - {indicators.get('rsi_signal', 'N/A')}
- SMA (20): ${indicators.get('sma_20', 'N/A')}
- SMA (50): ${indicators.get('sma_50', 'N/A')}
- MACD Trend: {indicators.get('macd_trend', 'N/A')}
- Volatility: {indicators.get('volatility', 'N/A')}%

Price Performance:
- 7-Day Change: {indicators.get('change_7d', 0)}% {indicators.get('trend_7d', '')}
- 30-Day Change: {indicators.get('change_30d', 0)}% {indicators.get('trend_30d', '')}

Key Metrics:
- P/E Ratio: {info['pe_ratio']}
- Beta: {info['beta']}
- 52-Week Range: ${info['52_week_low']} - ${info['52_week_high']}
"""
        return summary
    
    def _get_fallback_info(self, ticker: str) -> Dict:
        """Fallback data when API fails"""
        return {
            'symbol': ticker.upper(),
            'name': ticker.upper(),
            'sector': 'N/A',
            'industry': 'N/A',
            'current_price': 0,
            'previous_close': 0,
            'open': 0,
            'day_high': 0,
            'day_low': 0,
            'volume': 0,
            'market_cap': 0,
            'pe_ratio': 0,
            '52_week_high': 0,
            '52_week_low': 0,
            'dividend_yield': 0,
            'beta': 0,
        }
    
    def _get_fallback_indicators(self, ticker: str) -> Dict:
        """Fallback indicators when calculation fails"""
        return {
            'current_price': 0,
            'sma_20': None,
            'sma_50': None,
            'rsi': None,
            'rsi_signal': 'N/A',
            'macd': None,
            'macd_signal': None,
            'macd_trend': 'N/A',
            'volatility': 0,
            'change_7d': 0,
            'change_30d': 0,
            'trend_7d': 'N/A',
            'trend_30d': 'N/A',
        }
    
    def compare_stocks(self, tickers: List[str], period: str = "1mo") -> Dict:
        """
        Compare multiple stocks
        
        Args:
            tickers: List of ticker symbols
            period: Time period for comparison
            
        Returns:
            Dictionary with comparison data
        """
        comparison = {}
        
        for ticker in tickers:
            indicators = self.calculate_technical_indicators(ticker, period)
            comparison[ticker] = {
                'current_price': indicators['current_price'],
                'change_30d': indicators.get('change_30d', 0),
                'rsi': indicators.get('rsi', 'N/A'),
                'volatility': indicators.get('volatility', 0)
            }
        
        return comparison


# Test function
if __name__ == "__main__":
    print("🧪 Testing Market Data Module\n")
    
    market = MarketData()
    ticker = "AAPL"
    
    print(f"📊 Fetching data for {ticker}...\n")
    
    # Test stock info
    info = market.get_stock_info(ticker)
    print(f"✅ Company: {info['name']}")
    print(f"   Current Price: ${info['current_price']}")
    print(f"   Market Cap: ${info['market_cap']:,}\n")
    
    # Test technical indicators
    indicators = market.calculate_technical_indicators(ticker)
    print(f"📈 Technical Indicators:")
    print(f"   RSI: {indicators.get('rsi', 'N/A')} - {indicators.get('rsi_signal', 'N/A')}")
    print(f"   7-Day Change: {indicators.get('change_7d', 0)}%")
    print(f"   30-Day Change: {indicators.get('change_30d', 0)}%\n")
    
    # Test market summary
    print("📄 Market Summary:")
    print(market.get_market_summary(ticker))
    