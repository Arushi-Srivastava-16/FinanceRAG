"""
FinanceRAG - AI-Powered Trading Intelligence System
Enhanced RAG system combining news, sentiment analysis, and market data
"""

import streamlit as st
import os
from datetime import datetime
import sys

# Import custom modules
from news_fetcher import NewsFetcher
from sentiment_analyzer import SentimentAnalyzer
from market_data import MarketData
from rag_engine import FinancialRAGEngine
from agents import AgentOrchestrator
# Page configuration
st.set_page_config(
    page_title="FinanceRAG - Trading Intelligence",
    page_icon="❡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        border: 1px solid #3d3d4d;
        color: #fafafa;
    }
    .metric-card strong {
        color: #fafafa;
        font-size: 0.95rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #fafafa;
    }
    .metric-detail {
        font-size: 0.85rem;
        color: #b0b0b0;
        margin-top: 0.25rem;
    }
    .positive {
        color: #00ff00;
        font-weight: bold;
    }
    .negative {
        color: #ff0000;
        font-weight: bold;
    }
    .neutral {
        color: #ffaa00;
        font-weight: bold;
    }
    .source-box {
        background-color: #e8eaf6;
        color: #1a1a1a;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.875rem;
    }
    .welcome-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    .feature-card {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .feature-card h3 {
        color: white;
        margin-bottom: 0.5rem;
    }
    .ticker-card {
        background-color: #1e1e2e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        color: white;
        transition: transform 0.2s;
    }
    .ticker-card:hover {
        transform: translateX(5px);
        border-left-color: #764ba2;
    }
    .stat-box {
        text-align: center;
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: white;
    }
    .stat-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'news_fetcher' not in st.session_state:
    st.session_state.news_fetcher = None
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = None
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'llm_provider_choice' not in st.session_state: # New session state for LLM choice
    st.session_state.llm_provider_choice = "gemini"
if 'agent_orchestrator' not in st.session_state:
    st.session_state.agent_orchestrator = None
if 'agent_analysis' not in st.session_state:
    st.session_state.agent_analysis = None

def initialize_components():
    """Initialize all system components"""
    try:
        if st.session_state.news_fetcher is None:
            st.session_state.news_fetcher = NewsFetcher()
        if st.session_state.sentiment_analyzer is None:
            st.session_state.sentiment_analyzer = SentimentAnalyzer()
        if st.session_state.market_data is None:
            st.session_state.market_data = MarketData()
        
        # --- Modifications for RAG Engine Initialization ---
        # The RAG engine needs to be re-initialized if the provider choice changes
        # or if it hasn't been initialized yet.
        if st.session_state.rag_engine is None or \
           st.session_state.rag_engine.llm_provider != st.session_state.llm_provider_choice:

            google_api_key = os.getenv('GOOGLE_API_KEY')
            openai_api_key = os.getenv('OPENAI_API_KEY')
            
            # Check for API keys based on selected provider
            if st.session_state.llm_provider_choice == "gemini":
                if not google_api_key:
                    st.error("⚠️ GOOGLE_API_KEY not found in environment variables!")
                    st.info("Please set GOOGLE_API_KEY in your .env file")
                    return False
            elif st.session_state.llm_provider_choice == "openai":
                if not openai_api_key:
                    st.error("⚠️ OPENAI_API_KEY not found in environment variables!")
                    st.info("Please set OPENAI_API_KEY in your .env file")
                    return False
            
            st.session_state.rag_engine = FinancialRAGEngine(
                api_key=google_api_key,          # Pass Google API key
                openai_api_key=openai_api_key,    # Pass OpenAI API key
                llm_provider=st.session_state.llm_provider_choice # Pass selected provider
            )
        return True
    except Exception as e:
        st.error(f"❌ Error initializing components: {str(e)}")
        return False

def analyze_stock(ticker: str, days_back: int = 7, max_articles: int = 20):
    """Perform comprehensive stock analysis"""
    
    with st.spinner(f"☞ Analyzing {ticker.upper()}..."):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Fetch market data (20%)
            status_text.text("📊 Fetching market data...")
            progress_bar.progress(20)
            market_info = st.session_state.market_data.get_stock_info(ticker)
            market_indicators = st.session_state.market_data.calculate_technical_indicators(ticker)
            
            # Step 2: Fetch news articles (40%)
            status_text.text("📰 Fetching financial news...")
            progress_bar.progress(40)
            articles = st.session_state.news_fetcher.fetch_financial_news(
                ticker, 
                days_back=days_back, 
                max_articles=max_articles
            )
            
            # Step 3: Analyze sentiment (60%)
            status_text.text("🧠 Analyzing sentiment...")
            progress_bar.progress(60)
            sentiment_analysis = st.session_state.sentiment_analyzer.analyze_articles(articles)
            
            # Step 4: Prepare documents for RAG (80%)
            status_text.text("📝 Preparing knowledge base...")
            progress_bar.progress(80)
            
            # Format documents
            documents = []
            
            # Add market summary
            market_summary = st.session_state.market_data.get_market_summary(ticker)
            documents.append(market_summary)
            
            # Add news articles
            news_formatted = st.session_state.news_fetcher.format_for_rag(articles)
            documents.append(news_formatted)
            
            # Add sentiment summary
            sentiment_summary = f"""
SENTIMENT ANALYSIS FOR {ticker.upper()}

Overall Sentiment: {sentiment_analysis['overall_label']} ({sentiment_analysis['overall_score']})
Confidence Level: {sentiment_analysis['overall_confidence']}
Total Articles Analyzed: {sentiment_analysis['total_articles']}

Distribution:
- Positive Articles: {sentiment_analysis['positive_count']} ({sentiment_analysis['sentiment_distribution']['positive']}%)
- Negative Articles: {sentiment_analysis['negative_count']} ({sentiment_analysis['sentiment_distribution']['negative']}%)
- Neutral Articles: {sentiment_analysis['neutral_count']} ({sentiment_analysis['sentiment_distribution']['neutral']}%)

Trading Signal: {st.session_state.sentiment_analyzer.get_trading_signal(sentiment_analysis)['signal']}
Reasoning: {st.session_state.sentiment_analyzer.get_trading_signal(sentiment_analysis)['reasoning']}
"""
            documents.append(sentiment_summary)
            
            # Step 5: Add to RAG engine (100%)
            status_text.text("🔮 Building AI knowledge base...")
            progress_bar.progress(90)
            
            st.session_state.rag_engine.add_documents(
                documents,
                metadatas=[
                    {"type": "market_data", "ticker": ticker, "timestamp": datetime.now().isoformat()},
                    {"type": "news", "ticker": ticker, "timestamp": datetime.now().isoformat()},
                    {"type": "sentiment", "ticker": ticker, "timestamp": datetime.now().isoformat()}
                ]
            )
            
            # Create conversation chain if not exists
            if not st.session_state.rag_engine.conversation_chain:
                st.session_state.rag_engine.create_conversation_chain()
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            return {
                "market_info": market_info,
                "market_indicators": market_indicators,
                "sentiment_analysis": sentiment_analysis,
                "articles": articles,
                "success": True
            }
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            return {"success": False, "error": str(e)}

def display_market_overview(market_info, market_indicators):
    """Display market data overview"""
    
    st.subheader(f"📊 Market Overview: {market_info['name']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price = market_info['current_price']
        prev_close = market_info['previous_close']
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0
        
        st.metric(
            label="Current Price",
            value=f"${price:.2f}",
            delta=f"{change_pct:+.2f}%"
        )
    
    with col2:
        st.metric(
            label="Volume",
            value=f"{market_info['volume']:,}"
        )
    
    with col3:
        st.metric(
            label="Market Cap",
            value=f"${market_info['market_cap']/1e9:.2f}B"
        )
    
    with col4:
        st.metric(
            label="P/E Ratio",
            value=f"{market_info['pe_ratio']:.2f}" if market_info['pe_ratio'] else "N/A"
        )
    
    # Technical indicators
    st.markdown("### 📈 Technical Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rsi = market_indicators.get('rsi', 0)
        rsi_signal = market_indicators.get('rsi_signal', 'N/A')
        rsi_color = "🟢" if "Oversold" in rsi_signal else "🔴" if "Overbought" in rsi_signal else "🟡"
        rsi_display = f"{rsi:.2f}" if rsi else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <strong>RSI (14)</strong>
            <div class="metric-value">{rsi_color} {rsi_display}</div>
            <div class="metric-detail">{rsi_signal}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        change_7d = market_indicators.get('change_7d', 0)
        trend_7d = market_indicators.get('trend_7d', '')
        st.markdown(f"""
        <div class="metric-card">
            <strong>7-Day Performance</strong>
            <div class="metric-value">{change_7d:+.2f}%</div>
            <div class="metric-detail">{trend_7d}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        change_30d = market_indicators.get('change_30d', 0)
        trend_30d = market_indicators.get('trend_30d', '')
        st.markdown(f"""
        <div class="metric-card">
            <strong>30-Day Performance</strong>
            <div class="metric-value">{change_30d:+.2f}%</div>
            <div class="metric-detail">{trend_30d}</div>
        </div>
        """, unsafe_allow_html=True)

def display_sentiment_analysis(sentiment_analysis):
    """Display sentiment analysis results"""
    
    st.subheader("🧠 Sentiment Analysis")
    
    # Overall sentiment
    overall_score = sentiment_analysis['overall_score']
    overall_label = sentiment_analysis['overall_label']
    
    # Color coding
    if overall_score >= 0.1:
        sentiment_class = "positive"
        badge_color = "#4caf50"
    elif overall_score <= -0.1:
        sentiment_class = "negative"
        badge_color = "#f44336"
    else:
        sentiment_class = "neutral"
        badge_color = "#ff9800"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Overall Sentiment</strong>
            <div class="metric-value">
                <span class="sentiment-badge" style="background-color: {badge_color}; color: white;">
                    {overall_label}
                </span>
            </div>
            <div class="metric-detail">Score: {overall_score:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Articles Analyzed</strong>
            <div class="metric-value">{sentiment_analysis['total_articles']}</div>
            <div class="metric-detail">Confidence: {sentiment_analysis['overall_confidence']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        trading_signal = st.session_state.sentiment_analyzer.get_trading_signal(sentiment_analysis)
        st.markdown(f"""
        <div class="metric-card">
            <strong>Trading Signal</strong>
            <div class="metric-value">{trading_signal['signal']}</div>
            <div class="metric-detail">{trading_signal['reasoning']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sentiment distribution chart
    st.markdown("### 📊 Sentiment Distribution")
    
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Positive', 'Neutral', 'Negative'],
            y=[
                sentiment_analysis['positive_count'],
                sentiment_analysis['neutral_count'],
                sentiment_analysis['negative_count']
            ],
            marker_color=['#4caf50', '#ff9800', '#f44336']
        )
    ])
    
    fig.update_layout(
        title="Article Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Number of Articles",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_top_articles(articles, sentiment_analysis):
    """Display top articles with sentiment scores"""
    
    st.subheader("📰 Recent News Articles")
    
    # Get article sentiments
    article_sentiments = sentiment_analysis.get('article_sentiments', [])
    
    if article_sentiments:
        # Display top 5 articles
        for i, article_sent in enumerate(article_sentiments[:5], 1):
            score = article_sent['score']
            label = article_sent['label']
            
            # Color coding
            if score >= 0.1:
                badge_color = "#4caf50"
            elif score <= -0.1:
                badge_color = "#f44336"
            else:
                badge_color = "#ff9800"
            
            # Find matching article
            article = next((a for a in articles if a['title'] == article_sent['title']), None)
            
            if article:
                with st.expander(f"📄 {i}. {article['title']}", expanded=(i==1)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Source:** {article['source']}")
                        st.markdown(f"**Published:** {article['published_at'][:10]}")
                        st.markdown(f"**Description:** {article['description']}")
                        if article['url'] != '#':
                            st.markdown(f"[Read full article]({article['url']})")
                    
                    with col2:
                        st.markdown(f"""
                        <div class="sentiment-badge" style="background-color: {badge_color}; color: white;">
                            {label}
                        </div>
                        <br>
                        <small>Score: {score:.3f}</small>
                        """, unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">📈 FinanceRAG - AI Trading Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Analysis Settings")
        
        # --- New LLM Provider Selection ---
        llm_provider_selection = st.radio(
            "Select LLM Provider",
            options=["Gemini", "OpenAI"],
            index=0 if st.session_state.llm_provider_choice == "gemini" else 1,
            key="llm_provider_radio"
        )
        # Update session state if choice changes and force re-initialization
        if llm_provider_selection.lower() != st.session_state.llm_provider_choice:
            st.session_state.llm_provider_choice = llm_provider_selection.lower()
            st.session_state.rag_engine = None # Force re-initialization of RAG engine with new provider
            st.session_state.analysis_complete = False # Reset analysis view
            st.session_state.messages = [] # Clear chat history
            st.warning(f"Switched to {llm_provider_selection}. Please re-analyze the stock.")
            st.rerun() # Rerun to reflect the provider change immediately

        st.markdown("---") # Separator
        
        # Stock ticker input
        ticker_input = st.text_input(
            "Stock Ticker Symbol",
            value="AAPL",
            help="Enter stock ticker (e.g., AAPL, TSLA, GOOGL)"
        ).upper()
        
        # Analysis parameters
        days_back = st.slider(
            "News Lookback Period (days)",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to look back for news"
        )
        
        max_articles = st.slider(
            "Maximum Articles",
            min_value=5,
            max_value=50,
            value=20,
            help="Maximum number of articles to analyze"
        )
        
        # Analyze button
# Analyze button
        if st.button("☞ Analyze Stock", type="primary", use_container_width=True):
            if ticker_input:
                # Ensure components are initialized with the latest provider choice
                # This ensures the RAG engine uses the selected provider
                if not initialize_components():
                    st.stop() # Stop if API keys are missing for the selected provider
                
                st.session_state.current_ticker = ticker_input
                st.session_state.analysis_complete = False
                
                # Clear RAG engine memory on new analysis to ensure fresh context
                # and clear previous chat history
                if st.session_state.rag_engine:
                    st.session_state.rag_engine.clear_memory()
                st.session_state.messages = [] # Also clear displayed messages
                
                # Perform analysis
                results = analyze_stock(ticker_input, days_back, max_articles)
                
                if results['success']:
                    st.session_state.analysis_data = results
                    st.session_state.analysis_complete = True
                    st.success(f"✅ Analysis complete for {ticker_input} using {st.session_state.llm_provider_choice.capitalize()}!")
                    st.rerun() # Rerun to display analysis results
            else:
                st.warning("⚠️ Please enter a stock ticker")
        
        st.markdown("---")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.rag_engine:
                st.session_state.rag_engine.clear_memory()
            st.success("Conversation cleared!")
            st.rerun()
        
        # Database stats
        if st.session_state.rag_engine:
            st.markdown("---")
            st.markdown("### 📊 Database Stats")
            stats = st.session_state.rag_engine.get_database_stats()
            st.info(f"""
**Provider:** {stats['provider']}  
**Documents:** {stats['total_documents']}  
**Database:** {stats['persist_directory'].split('/')[-1]}
            """)
        
        # Footer
        st.markdown("---")
        st.markdown(f"""
        <small>
        Built with ❤️ using:<br>
        • <strong>Active: {st.session_state.llm_provider_choice.upper()}</strong><br>
        • LangChain & ChromaDB<br>
        • NewsAPI & yfinance<br>
        • VADER Sentiment Analysis
        </small>
        """, unsafe_allow_html=True)
    
    # Main content area
    if st.session_state.analysis_complete and 'analysis_data' in st.session_state:
        data = st.session_state.analysis_data
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🧠 Sentiment", "📰 News", "💬 Chat", "🤖 Multi-Agent"])
        
        with tab1:
            display_market_overview(data['market_info'], data['market_indicators'])
        
        with tab2:
            display_sentiment_analysis(data['sentiment_analysis'])
        
        with tab3:
            display_top_articles(data['articles'], data['sentiment_analysis'])
        
        with tab4:
            # ... existing chat code ...
            st.subheader(f"💬 Chat with {st.session_state.current_ticker} Analysis")
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Display sources if available
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(message["sources"], 1):
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {i}:</strong><br>
                                    {source['content']}<br>
                                    <small>Type: {source['metadata'].get('type', 'N/A')}</small>
                                </div>
                                """, unsafe_allow_html=True)
            
            # Chat input
            if prompt := st.chat_input(f"Ask me anything about {st.session_state.current_ticker}..."):
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("🔮 Thinking..."):
                        # Prepare context
                        context = {
                            "sentiment": data['sentiment_analysis'],
                            "market_data": data['market_indicators']
                        }
                        
                        # Query RAG engine
                        response = st.session_state.rag_engine.query(prompt, context=context)
                        
                        if response['success']:
                            st.markdown(response['answer'])
                            
                            # Display sources
                            if response['sources']:
                                with st.expander("📚 View Sources"):
                                    for i, source in enumerate(response['sources'], 1):
                                        st.markdown(f"""
                                        <div class="source-box">
                                            <strong>Source {i}:</strong><br>
                                            {source['content']}<br>
                                            <small>Type: {source['metadata'].get('type', 'N/A')}</small>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Save to messages
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response['answer'],
                                "sources": response['sources']
                            })
                        else:
                            st.error(response['answer'])
        
        with tab5:
            st.subheader("🤖 Multi-Agent Analysis System")
            
            st.info("💡 **What is this?** Three specialized AI agents analyze the stock independently and then debate to reach a consensus decision!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <strong>📰 Sentiment Agent</strong>
                    <div class="metric-detail">Analyzes news, social media, and market psychology</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <strong>📊 Technical Agent</strong>
                    <div class="metric-detail">Studies charts, indicators, and price patterns</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <strong>🛡️ Risk Agent</strong>
                    <div class="metric-detail">Assesses risks and portfolio management</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Run Multi-Agent Analysis button
            if st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True):
                with st.spinner("🤖 Agents are analyzing..."):
                    try:
                        # Initialize orchestrator if not exists
                        if st.session_state.agent_orchestrator is None:
                            st.session_state.agent_orchestrator = AgentOrchestrator(
                                llm_provider=st.session_state.llm_provider_choice,
                                google_api_key=os.getenv('GOOGLE_API_KEY'),
                                openai_api_key=os.getenv('OPENAI_API_KEY')
                            )
                        
                        # Prepare data for agents
                        agent_data = {
                            "ticker": st.session_state.current_ticker,
                            "sentiment": data['sentiment_analysis'],
                            "market_info": data['market_info'],
                            "market_indicators": data['market_indicators'],
                            "articles": data['articles']
                        }
                        
                        # Run analysis
                        result = st.session_state.agent_orchestrator.run_analysis(agent_data)
                        st.session_state.agent_analysis = result
                        
                        st.success("✅ Multi-Agent Analysis Complete!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error running agents: {str(e)}")
            
            # Display results if available
            if st.session_state.agent_analysis:
                result = st.session_state.agent_analysis
                
                st.markdown("---")
                st.markdown("### 🎯 Consensus Decision")
                
                # Display consensus in a nice box
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 1rem; color: white; margin: 1rem 0;">
                """, unsafe_allow_html=True)
                
                st.markdown(result['consensus'])
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Individual agent recommendations
                st.markdown("### 📊 Individual Agent Recommendations")
                
                col1, col2, col3 = st.columns(3)
                
                agent_results = result['agent_results']
                
                with col1:
                    sentiment_result = agent_results[0]
                    rec_color = _get_recommendation_color(sentiment_result['recommendation'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>📰 Sentiment Agent</strong>
                        <div class="metric-value" style="color: {rec_color};">
                            {sentiment_result['recommendation']}
                        </div>
                        <div class="metric-detail">Confidence: {sentiment_result['confidence']}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("View Full Analysis"):
                        st.write(sentiment_result['analysis'])
                
                with col2:
                    technical_result = agent_results[1]
                    rec_color = _get_recommendation_color(technical_result['recommendation'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>📊 Technical Agent</strong>
                        <div class="metric-value" style="color: {rec_color};">
                            {technical_result['recommendation']}
                        </div>
                        <div class="metric-detail">Confidence: {technical_result['confidence']}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("View Full Analysis"):
                        st.write(technical_result['analysis'])
                
                with col3:
                    risk_result = agent_results[2]
                    rec_color = _get_recommendation_color(risk_result['recommendation'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>🛡️ Risk Agent</strong>
                        <div class="metric-value" style="color: {rec_color};">
                            {risk_result['recommendation']}
                        </div>
                        <div class="metric-detail">
                            Confidence: {risk_result['confidence']}%<br>
                            Risk: {risk_result.get('risk_level', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("View Full Analysis"):
                        st.write(risk_result['analysis'])
                
                # Average confidence chart
                st.markdown("### 📈 Agent Confidence Comparison")
                
                import plotly.graph_objects as go
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Sentiment Agent', 'Technical Agent', 'Risk Agent'],
                        y=[
                            agent_results[0]['confidence'],
                            agent_results[1]['confidence'],
                            agent_results[2]['confidence']
                        ],
                        marker_color=['#667eea', '#764ba2', '#f093fb'],
                        text=[
                            f"{agent_results[0]['confidence']}%",
                            f"{agent_results[1]['confidence']}%",
                            f"{agent_results[2]['confidence']}%"
                        ],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Agent Confidence Levels",
                    yaxis_title="Confidence (%)",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation agreement
                st.markdown("### 🤝 Recommendation Agreement")
                
                recommendations = result['individual_recommendations']
                unique_recs = set(recommendations.values())
                
                if len(unique_recs) == 1:
                    st.success("✅ **UNANIMOUS AGREEMENT** - All agents recommend the same action!")
                elif len(unique_recs) == 2:
                    st.warning("⚠️ **SPLIT DECISION** - Agents have differing opinions. Review carefully!")
                else:
                    st.error("❌ **NO CONSENSUS** - Major disagreement between agents. High uncertainty!")
                
                # Download report button
                if st.button("📥 Download Full Report", use_container_width=True):
                    report = _generate_report(result)
                    st.download_button(
                        label="Download as TXT",
                        data=report,
                        file_name=f"{st.session_state.current_ticker}_agent_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

    
    else:
        # Welcome screen with professional design
        st.markdown("""
        <div class="welcome-container">
            <h1 style="text-align: center; margin-bottom: 0.5rem;">ℜ FinanceRAG</h1>
            <p style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">
                AI-Powered Trading Intelligence Platform
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="stat-box">
                <div class="stat-number">📰</div>
                <div class="stat-label">Real-Time News</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-box">
                <div class="stat-number">𝓐</div>
                <div class="stat-label">AI Sentiment</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stat-box">
                <div class="stat-number">𝔍</div>
                <div class="stat-label">Technical Analysis</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="stat-box">
                <div class="stat-number">💬</div>
                <div class="stat-label">Smart Chat</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main content in two columns
        col_left, col_right = st.columns([1.5, 1])
        
        with col_left:
            st.markdown("### 🚀 How It Works")
            
            st.markdown("""
            <div class="feature-card">
                <h4>1️⃣ Enter Stock Ticker</h4>
                <p>Simply enter any stock symbol (AAPL, TSLA, GOOGL) in the sidebar to begin analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>2️⃣ AI Analyzes Everything</h4>
                <p>Our system fetches news, analyzes sentiment, calculates technical indicators, and builds a comprehensive knowledge base.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>3️⃣ Get Actionable Insights</h4>
                <p>View detailed analysis, sentiment scores, trading signals, and chat with AI for personalized insights.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 💡 Example Questions")
            st.code("""
• What's the current market sentiment?
• Should I buy or sell based on recent news?
• What are the key technical indicators saying?
• What are the biggest risks right now?
• How does sentiment compare to price action?
            """, language=None)
        
        with col_right:
            st.markdown("### 🔥 Popular Tickers")
            
            popular_tickers = [
                ("AAPL", "Apple Inc.", "Technology"),
                ("TSLA", "Tesla Inc.", "Automotive"),
                ("GOOGL", "Alphabet Inc.", "Technology"),
                ("MSFT", "Microsoft Corp.", "Technology"),
                ("AMZN", "Amazon.com Inc.", "E-commerce"),
                ("NVDA", "NVIDIA Corp.", "Semiconductors"),
                ("META", "Meta Platforms", "Social Media"),
                ("BTC-USD", "Bitcoin", "Cryptocurrency"),
                ("ETH-USD", "Ethereum", "Cryptocurrency"),
                ("JPM", "JPMorgan Chase", "Banking"),
            ]
            
            for ticker, name, sector in popular_tickers:
                st.markdown(f"""
                <div class="ticker-card">
                    <strong style="font-size: 1.1rem; color: #667eea;">{ticker}</strong><br>
                    <span style="font-size: 0.9rem;">{name}</span><br>
                    <span style="font-size: 0.75rem; color: #b0b0b0;">{sector}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("✔︎ **Tip**: Click any ticker above and paste it in the sidebar to analyze!")
        
        st.markdown("---")
        
        # Technology stack
        st.markdown("### ⦿ Powered By")
        
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        
        with tech_col1:
            st.markdown("""
            **AI & LLM**
            - Google Gemini Pro
            - LangChain
            - ChromaDB
            """)
        
        with tech_col2:
            st.markdown("""
            **Data Sources**
            - NewsAPI
            - Yahoo Finance
            - Real-time APIs
            """)
        
        with tech_col3:
            st.markdown("""
            **Analysis**
            - VADER Sentiment
            - TextBlob NLP
            - Technical Indicators
            """)
        
        with tech_col4:
            st.markdown("""
            **Visualization**
            - Streamlit
            - Plotly Charts
            - Custom UI
            """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Disclaimer
        st.warning("⚠️ **Disclaimer**: This tool is for informational and educational purposes only. It does not constitute financial advice. Always conduct your own research and consult with qualified financial advisors before making investment decisions.")
        
        # Call to action
        st.success("☜ **Ready to start?** Enter a stock ticker in the sidebar and click 'Analyze Stock'!")

def _get_recommendation_color(recommendation: str) -> str:
    """Get color for recommendation"""
    if "STRONG BUY" in recommendation or "BUY" in recommendation:
        return "#00ff00"
    elif "STRONG SELL" in recommendation or "SELL" in recommendation:
        return "#ff0000"
    else:
        return "#ffaa00"

def _generate_report(result: dict) -> str:
    """Generate downloadable text report"""
    report = f"""
{'='*80}
MULTI-AGENT TRADING ANALYSIS REPORT
{'='*80}

Stock: {result['ticker']}
Analysis Date: {result['timestamp']}
Average Confidence: {result['average_confidence']:.1f}%

{'='*80}
CONSENSUS DECISION
{'='*80}

{result['consensus']}

{'='*80}
INDIVIDUAL AGENT ANALYSES
{'='*80}

"""
    
    for agent_result in result['agent_results']:
        report += f"""
{'-'*80}
{agent_result['agent'].upper()}
{'-'*80}

Recommendation: {agent_result['recommendation']}
Confidence: {agent_result['confidence']}%

Analysis:
{agent_result['analysis']}

"""
    
    report += f"""
{'='*80}
END OF REPORT
{'='*80}
"""
    
    return report

if __name__ == "__main__":
    main()