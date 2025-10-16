# FinanceRAG

**AI-Powered Trading Intelligence System with Multi-Agent Analysis**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

FinanceRAG is an advanced financial analysis platform that combines Retrieval-Augmented Generation (RAG) with multi-agent AI systems to deliver comprehensive stock market insights. The system integrates real-time news, sentiment analysis, technical indicators, and collaborative AI reasoning to support informed trading decisions.

### Key Features

- **Multi-Agent Analysis**: Three specialized AI agents (Sentiment, Technical, Risk) provide independent analysis and reach consensus decisions
- **Real-Time Data Integration**: Automated fetching of financial news and market data
- **Dual-Engine Sentiment Analysis**: VADER and TextBlob combined for robust sentiment scoring
- **Technical Indicators**: RSI, MACD, moving averages, and volatility metrics
- **Conversational Interface**: Natural language queries with RAG-powered responses
- **Flexible LLM Support**: Compatible with Google Gemini and OpenAI models
- **Vector Search**: ChromaDB for efficient semantic retrieval

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│              (Streamlit Dashboard)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │  News   │   │ Market  │   │Sentiment│
   │ Fetcher │   │  Data   │   │Analyzer │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
        └─────────────┼─────────────┘
                      ↓
            ┌──────────────────┐
            │   RAG ENGINE     │
            │  (ChromaDB +     │
            │   LangChain)     │
            └────────┬─────────┘
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │Sentiment │ │Technical │ │   Risk   │
   │  Agent   │ │  Agent   │ │  Agent   │
   └─────┬────┘ └─────┬────┘ └─────┬────┘
         │            │            │
         └────────────┼────────────┘
                      ↓
            ┌──────────────────┐
            │   ORCHESTRATOR   │
            │   (Consensus)    │
            └──────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- API keys for:
  - Google Gemini or OpenAI
  - NewsAPI (free tier available at [newsapi.org](https://newsapi.org))

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/FinanceRAG.git
   cd FinanceRAG
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m textblob.download_corpora
   ```

4. **Configure environment**
   
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your-google-gemini-api-key
   OPENAI_API_KEY=your-openai-api-key
   NEWS_API_KEY=your-newsapi-key
   ```

5. **Launch application**
   ```bash
   cd app
   streamlit run app.py
   ```

   Access the dashboard at `http://localhost:8501`

---

## Usage

### Stock Analysis

1. Select your preferred LLM provider (Gemini or OpenAI)
2. Enter a stock ticker symbol (e.g., AAPL, TSLA, GOOGL)
3. Configure analysis parameters (lookback period, article count)
4. Run analysis to view comprehensive insights

### Multi-Agent System

The multi-agent analysis mode deploys three specialized AI agents:

- **Sentiment Agent**: Analyzes news and social media sentiment
- **Technical Agent**: Evaluates chart patterns and technical indicators
- **Risk Agent**: Assesses volatility and risk metrics

These agents work independently and then collaborate through an orchestrator to reach a consensus recommendation.

### Chat Interface

Ask natural language questions about market conditions:
- "What's the current sentiment for this stock?"
- "What do the technical indicators suggest?"
- "What are the primary risk factors?"

Responses include source citations from the analyzed documents.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Google Gemini Pro / OpenAI GPT-4 |
| **Embeddings** | Gemini Embeddings / OpenAI Embeddings |
| **Vector Database** | ChromaDB |
| **RAG Framework** | LangChain |
| **Interface** | Streamlit |
| **Sentiment Analysis** | VADER, TextBlob |
| **Market Data** | yfinance |
| **News API** | NewsAPI |
| **Visualization** | Plotly |

---

## Project Structure

```
FinanceRAG/
├── app/
│   ├── app.py                    # Main application
│   ├── agents.py                 # Multi-agent system
│   ├── news_fetcher.py           # News integration
│   ├── sentiment_analyzer.py     # Sentiment engine
│   ├── market_data.py            # Market data & indicators
│   └── rag_engine.py             # RAG implementation
├── Vector_DB_Financial_Gemini/   # Gemini vector store
├── Vector_DB_Financial_OpenAI/   # OpenAI vector store
├── requirements.txt
└── README.md
```

---

## Use Cases

- **Portfolio Management**: Analyze multiple stocks with risk-adjusted recommendations
- **Trading Strategy**: Identify entry/exit points using sentiment and technical signals
- **Market Research**: Track sentiment trends and news impact analysis
- **Educational**: Learn about technical indicators and multi-agent AI systems

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

## Disclaimer

**This tool is for educational and informational purposes only.** It does not constitute financial advice. Trading and investing involve substantial risk of loss. Always conduct your own research and consult qualified financial advisors before making investment decisions.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Google Gemini and OpenAI for LLM capabilities
- LangChain for RAG framework
- NewsAPI for financial news data
- yfinance for market data
- Streamlit community

---

**Built for the AI & Finance community**