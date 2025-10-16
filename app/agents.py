"""
Multi-Agent Trading System
Specialized AI agents that collaborate to make trading decisions
"""

import os
from typing import Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime
import json

class BaseAgent:
    """Base class for all trading agents"""
    
    def __init__(self, llm, name: str, role: str):
        self.llm = llm
        self.name = name
        self.role = role
        self.analysis_history = []
    
    def analyze(self, data: Dict) -> Dict:
        """Override this method in child classes"""
        raise NotImplementedError


class SentimentAgent(BaseAgent):
    """Agent specialized in sentiment analysis and news interpretation"""
    
    def __init__(self, llm):
        super().__init__(llm, "Sentiment Analyst", "News & Social Media Analysis")
        
        self.prompt = PromptTemplate(
            input_variables=["ticker", "sentiment_data", "news_summary"],
            template="""You are a Sentiment Analysis Expert specializing in financial markets.

**Your Role:** Analyze news sentiment and public perception to gauge market psychology.

**Stock:** {ticker}

**Sentiment Data:**
{sentiment_data}

**Recent News Summary:**
{news_summary}

**Your Analysis Must Include:**
1. Overall market sentiment interpretation
2. Key sentiment drivers (positive/negative)
3. Sentiment trend analysis
4. Public perception and social signals
5. Sentiment-based trading recommendation (BUY/SELL/HOLD)
6. Confidence level (0-100%)
7. Key risks from sentiment perspective

Provide your analysis in a structured format:
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def analyze(self, data: Dict) -> Dict:
        """Analyze sentiment and news"""
        try:
            sentiment_summary = f"""
Overall Sentiment: {data['sentiment']['overall_label']} (Score: {data['sentiment']['overall_score']})
Positive Articles: {data['sentiment']['positive_count']}
Negative Articles: {data['sentiment']['negative_count']}
Neutral Articles: {data['sentiment']['neutral_count']}
Confidence: {data['sentiment']['overall_confidence']}
"""
            
            news_summary = "\n".join([
                f"- {article['title']} ({article['source']})"
                for article in data.get('articles', [])[:5]
            ])
            
            response = self.chain.run(
                ticker=data['ticker'],
                sentiment_data=sentiment_summary,
                news_summary=news_summary
            )
            
            result = {
                "agent": self.name,
                "analysis": response,
                "timestamp": datetime.now().isoformat(),
                "recommendation": self._extract_recommendation(response),
                "confidence": self._extract_confidence(response)
            }
            
            self.analysis_history.append(result)
            return result
            
        except Exception as e:
            return {
                "agent": self.name,
                "analysis": f"Error: {str(e)}",
                "recommendation": "HOLD",
                "confidence": 0
            }
    
    def _extract_recommendation(self, text: str) -> str:
        """Extract recommendation from analysis"""
        text_upper = text.upper()
        if "STRONG BUY" in text_upper or "STRONGLY RECOMMEND BUY" in text_upper:
            return "STRONG BUY"
        elif "BUY" in text_upper:
            return "BUY"
        elif "STRONG SELL" in text_upper or "STRONGLY RECOMMEND SELL" in text_upper:
            return "STRONG SELL"
        elif "SELL" in text_upper:
            return "SELL"
        else:
            return "HOLD"
    
    def _extract_confidence(self, text: str) -> int:
        """Extract confidence level from analysis"""
        import re
        confidence_match = re.search(r'confidence.*?(\d+)%?', text.lower())
        if confidence_match:
            return int(confidence_match.group(1))
        return 50  # Default confidence


class TechnicalAgent(BaseAgent):
    """Agent specialized in technical analysis"""
    
    def __init__(self, llm):
        super().__init__(llm, "Technical Analyst", "Chart Patterns & Indicators")
        
        self.prompt = PromptTemplate(
            input_variables=["ticker", "technical_data", "price_info"],
            template="""You are a Technical Analysis Expert with deep knowledge of chart patterns and indicators.

**Your Role:** Analyze price action, technical indicators, and chart patterns.

**Stock:** {ticker}

**Technical Indicators:**
{technical_data}

**Price Information:**
{price_info}

**Your Analysis Must Include:**
1. Technical indicator interpretation (RSI, MACD, Moving Averages)
2. Price trend analysis (support/resistance levels)
3. Chart pattern identification
4. Volume analysis
5. Technical-based trading recommendation (BUY/SELL/HOLD)
6. Entry/exit points if applicable
7. Confidence level (0-100%)
8. Technical risks and stop-loss suggestions

Provide your analysis in a structured format:
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def analyze(self, data: Dict) -> Dict:
        """Analyze technical indicators"""
        try:
            technical_data = f"""
Current Price: ${data['market_indicators']['current_price']}
RSI (14): {data['market_indicators'].get('rsi', 'N/A')} - {data['market_indicators'].get('rsi_signal', 'N/A')}
SMA (20): ${data['market_indicators'].get('sma_20', 'N/A')}
SMA (50): ${data['market_indicators'].get('sma_50', 'N/A')}
MACD: {data['market_indicators'].get('macd', 'N/A')}
MACD Signal: {data['market_indicators'].get('macd_signal', 'N/A')}
MACD Trend: {data['market_indicators'].get('macd_trend', 'N/A')}
Volatility: {data['market_indicators'].get('volatility', 'N/A')}%
7-Day Change: {data['market_indicators'].get('change_7d', 0)}%
30-Day Change: {data['market_indicators'].get('change_30d', 0)}%
"""
            
            price_info = f"""
Current Price: ${data['market_info']['current_price']}
Previous Close: ${data['market_info']['previous_close']}
Day High: ${data['market_info']['day_high']}
Day Low: ${data['market_info']['day_low']}
52-Week High: ${data['market_info']['52_week_high']}
52-Week Low: ${data['market_info']['52_week_low']}
Volume: {data['market_info']['volume']:,}
"""
            
            response = self.chain.run(
                ticker=data['ticker'],
                technical_data=technical_data,
                price_info=price_info
            )
            
            result = {
                "agent": self.name,
                "analysis": response,
                "timestamp": datetime.now().isoformat(),
                "recommendation": self._extract_recommendation(response),
                "confidence": self._extract_confidence(response)
            }
            
            self.analysis_history.append(result)
            return result
            
        except Exception as e:
            return {
                "agent": self.name,
                "analysis": f"Error: {str(e)}",
                "recommendation": "HOLD",
                "confidence": 0
            }
    
    def _extract_recommendation(self, text: str) -> str:
        """Extract recommendation from analysis"""
        text_upper = text.upper()
        if "STRONG BUY" in text_upper:
            return "STRONG BUY"
        elif "BUY" in text_upper:
            return "BUY"
        elif "STRONG SELL" in text_upper:
            return "STRONG SELL"
        elif "SELL" in text_upper:
            return "SELL"
        else:
            return "HOLD"
    
    def _extract_confidence(self, text: str) -> int:
        """Extract confidence level from analysis"""
        import re
        confidence_match = re.search(r'confidence.*?(\d+)%?', text.lower())
        if confidence_match:
            return int(confidence_match.group(1))
        return 50


class RiskAgent(BaseAgent):
    """Agent specialized in risk assessment"""
    
    def __init__(self, llm):
        super().__init__(llm, "Risk Manager", "Risk Assessment & Portfolio Management")
        
        self.prompt = PromptTemplate(
            input_variables=["ticker", "market_data", "sentiment_summary", "volatility"],
            template="""You are a Risk Management Expert focused on protecting capital and managing portfolio risk.

**Your Role:** Assess risks, evaluate position sizing, and provide risk management strategies.

**Stock:** {ticker}

**Market Data:**
{market_data}

**Sentiment Summary:**
{sentiment_summary}

**Volatility:** {volatility}%

**Your Analysis Must Include:**
1. Overall risk assessment (LOW/MEDIUM/HIGH/VERY HIGH)
2. Key risk factors identified
3. Market risk vs. company-specific risk
4. Recommended position size (% of portfolio)
5. Stop-loss level recommendation
6. Risk-adjusted recommendation (BUY/SELL/HOLD)
7. Confidence level (0-100%)
8. Risk mitigation strategies

Provide your analysis in a structured format:
"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def analyze(self, data: Dict) -> Dict:
        """Analyze risk factors"""
        try:
            market_data = f"""
Beta: {data['market_info'].get('beta', 'N/A')}
Market Cap: ${data['market_info']['market_cap']/1e9:.2f}B
P/E Ratio: {data['market_info']['pe_ratio']}
Volume: {data['market_info']['volume']:,}
52-Week Range: ${data['market_info']['52_week_low']} - ${data['market_info']['52_week_high']}
"""
            
            sentiment_summary = f"""
Overall Sentiment: {data['sentiment']['overall_label']} (Score: {data['sentiment']['overall_score']})
Confidence: {data['sentiment']['overall_confidence']}
"""
            
            response = self.chain.run(
                ticker=data['ticker'],
                market_data=market_data,
                sentiment_summary=sentiment_summary,
                volatility=data['market_indicators'].get('volatility', 0)
            )
            
            result = {
                "agent": self.name,
                "analysis": response,
                "timestamp": datetime.now().isoformat(),
                "recommendation": self._extract_recommendation(response),
                "confidence": self._extract_confidence(response),
                "risk_level": self._extract_risk_level(response)
            }
            
            self.analysis_history.append(result)
            return result
            
        except Exception as e:
            return {
                "agent": self.name,
                "analysis": f"Error: {str(e)}",
                "recommendation": "HOLD",
                "confidence": 0,
                "risk_level": "UNKNOWN"
            }
    
    def _extract_recommendation(self, text: str) -> str:
        """Extract recommendation from analysis"""
        text_upper = text.upper()
        if "STRONG BUY" in text_upper:
            return "STRONG BUY"
        elif "BUY" in text_upper:
            return "BUY"
        elif "STRONG SELL" in text_upper:
            return "STRONG SELL"
        elif "SELL" in text_upper:
            return "SELL"
        else:
            return "HOLD"
    
    def _extract_confidence(self, text: str) -> int:
        """Extract confidence level from analysis"""
        import re
        confidence_match = re.search(r'confidence.*?(\d+)%?', text.lower())
        if confidence_match:
            return int(confidence_match.group(1))
        return 50
    
    def _extract_risk_level(self, text: str) -> str:
        """Extract risk level from analysis"""
        text_upper = text.upper()
        if "VERY HIGH" in text_upper or "EXTREME" in text_upper:
            return "VERY HIGH"
        elif "HIGH" in text_upper:
            return "HIGH"
        elif "MEDIUM" in text_upper or "MODERATE" in text_upper:
            return "MEDIUM"
        elif "LOW" in text_upper:
            return "LOW"
        else:
            return "MEDIUM"


class AgentOrchestrator:
    """Orchestrates multiple agents to reach a consensus decision"""
    
    def __init__(self, llm_provider: str = "gemini", google_api_key: str = None, openai_api_key: str = None):
        """Initialize the orchestrator with agents"""
        
        # Initialize LLM based on provider
        if llm_provider == "openai":
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=openai_api_key or os.getenv('OPENAI_API_KEY'),
                temperature=0.3
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=google_api_key or os.getenv('GOOGLE_API_KEY'),
                temperature=0.3,
                convert_system_message_to_human=True
            )
        
        # Initialize agents
        self.sentiment_agent = SentimentAgent(self.llm)
        self.technical_agent = TechnicalAgent(self.llm)
        self.risk_agent = RiskAgent(self.llm)
        
        self.agents = [self.sentiment_agent, self.technical_agent, self.risk_agent]
        
        # Consensus prompt
        self.consensus_prompt = PromptTemplate(
            input_variables=["ticker", "agent_analyses"],
            template="""You are the Chief Investment Officer synthesizing analyses from multiple expert agents.

**Stock:** {ticker}

**Agent Analyses:**
{agent_analyses}

**Your Task:**
Synthesize all agent recommendations into a final consensus decision.

**Provide:**
1. Final Recommendation: STRONG BUY / BUY / HOLD / SELL / STRONG SELL
2. Confidence Level: 0-100%
3. Consensus Reasoning: Why this decision (2-3 sentences)
4. Key Agreement Points: Where agents agree
5. Key Disagreement Points: Where agents disagree
6. Action Plan: Specific next steps for the trader
7. Risk Warning: Critical risks to monitor

Format your response clearly and concisely.
"""
        )
        self.consensus_chain = LLMChain(llm=self.llm, prompt=self.consensus_prompt)
    
    def run_analysis(self, data: Dict) -> Dict:
        """Run all agents and synthesize their analyses"""
        
        print(f"\n{'='*60}")
        print(f"🤖 MULTI-AGENT ANALYSIS: {data['ticker']}")
        print(f"{'='*60}\n")
        
        agent_results = []
        
        # Run each agent
        for agent in self.agents:
            print(f"🔄 Running {agent.name}...")
            result = agent.analyze(data)
            agent_results.append(result)
            print(f"   ✅ {agent.name}: {result['recommendation']} (Confidence: {result['confidence']}%)")
        
        print(f"\n{'='*60}")
        print(f"🧠 SYNTHESIZING CONSENSUS...")
        print(f"{'='*60}\n")
        
        # Format agent analyses for consensus
        agent_analyses_text = "\n\n".join([
            f"**{result['agent']}:**\n"
            f"Recommendation: {result['recommendation']}\n"
            f"Confidence: {result['confidence']}%\n"
            f"Analysis:\n{result['analysis']}"
            for result in agent_results
        ])
        
        # Get consensus
        consensus = self.consensus_chain.run(
            ticker=data['ticker'],
            agent_analyses=agent_analyses_text
        )
        
        return {
            "ticker": data['ticker'],
            "timestamp": datetime.now().isoformat(),
            "agent_results": agent_results,
            "consensus": consensus,
            "individual_recommendations": {
                agent.name: result['recommendation']
                for agent, result in zip(self.agents, agent_results)
            },
            "average_confidence": sum(r['confidence'] for r in agent_results) / len(agent_results)
        }
    
    def get_agent_summary(self) -> Dict:
        """Get summary of all agents"""
        return {
            "total_agents": len(self.agents),
            "agents": [
                {
                    "name": agent.name,
                    "role": agent.role,
                    "analyses_count": len(agent.analysis_history)
                }
                for agent in self.agents
            ]
        }


# Test function
if __name__ == "__main__":
    print("🧪 Testing Multi-Agent System\n")
    
    # Mock data for testing
    test_data = {
        "ticker": "AAPL",
        "sentiment": {
            "overall_label": "Positive",
            "overall_score": 0.65,
            "overall_confidence": 0.78,
            "positive_count": 15,
            "negative_count": 3,
            "neutral_count": 2,
            "total_articles": 20
        },
        "market_info": {
            "current_price": 178.50,
            "previous_close": 175.30,
            "day_high": 179.20,
            "day_low": 177.10,
            "volume": 50000000,
            "market_cap": 2800000000000,
            "pe_ratio": 28.5,
            "52_week_high": 198.23,
            "52_week_low": 164.08,
            "beta": 1.2
        },
        "market_indicators": {
            "current_price": 178.50,
            "rsi": 58,
            "rsi_signal": "Neutral",
            "sma_20": 175.00,
            "sma_50": 172.50,
            "macd": 1.5,
            "macd_signal": 1.2,
            "macd_trend": "Bullish",
            "volatility": 2.3,
            "change_7d": 2.8,
            "change_30d": 5.2
        },
        "articles": [
            {"title": "Apple reports strong Q4 earnings", "source": "CNBC"},
            {"title": "iPhone sales exceed expectations", "source": "Reuters"}
        ]
    }
    
    try:
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(llm_provider="gemini")
        
        # Run analysis
        result = orchestrator.run_analysis(test_data)
        
        print("\n" + "="*60)
        print("📊 FINAL CONSENSUS:")
        print("="*60)
        print(result['consensus'])
        
        print("\n" + "="*60)
        print("📈 INDIVIDUAL RECOMMENDATIONS:")
        print("="*60)
        for agent_name, rec in result['individual_recommendations'].items():
            print(f"  • {agent_name}: {rec}")
        
        print(f"\n  Average Confidence: {result['average_confidence']:.1f}%")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")