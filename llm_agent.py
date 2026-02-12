"""Advanced LLM Agent for Nifty 50 Scalping Research"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from datetime import datetime
import config
from scalping_strategies import TradeSetup, StrategyEngine, SignalType


class ScalpingResearchAgent:
    """
    LLM-powered agent for generating scalping trade ideas and analysis
    
    This agent uses market data, technical indicators, and trading strategies
    to generate comprehensive research reports and trade recommendations.
    """
    
    def __init__(self, provider: str = None):
        self.provider = provider or config.LLM_PROVIDER
        self.client = None
        self._init_client()
        
        # System prompt for the trading agent
        self.system_prompt = """You are an expert Nifty 50 scalping analyst specializing in 5-minute timeframe trading. 
Your role is to analyze market data, technical indicators, and trading setups to provide actionable research and trade ideas.

Key responsibilities:
1. Analyze current market conditions and bias
2. Evaluate trade setups from multiple strategies
3. Provide clear entry, stop-loss, and target levels
4. Assess risk-reward ratios
5. Give confidence ratings for each trade idea
6. Explain the reasoning behind each recommendation

Trading philosophy:
- Focus on high-probability setups with minimum 1.5:1 risk-reward
- Prioritize capital preservation
- Use multiple timeframe confirmation when possible
- Consider volume and momentum for validation
- Be aware of key support/resistance levels

Always provide specific, actionable recommendations with clear levels."""

    def _init_client(self):
        """Initialize the LLM client based on provider"""
        try:
            if self.provider == "anthropic":
                import anthropic
                api_key = config.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = anthropic.Anthropic(api_key=api_key)
            elif self.provider == "openai":
                import openai
                api_key = config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            print(f"Warning: {self.provider} library not installed")
        except Exception as e:
            print(f"Warning: Could not initialize LLM client: {e}")
    
    def generate_research_report(
        self,
        market_data: Dict,
        analysis: Dict,
        setups: List[TradeSetup],
        market_bias: Dict
    ) -> Dict:
        """
        Generate a comprehensive research report using LLM
        
        Args:
            market_data: Current price and volume data
            analysis: Technical indicator analysis
            setups: List of valid trade setups
            market_bias: Overall market bias assessment
            
        Returns:
            Research report with trade recommendations
        """
        # Build context for LLM
        context = self._build_context(market_data, analysis, setups, market_bias)
        
        # Try to get LLM-generated insights
        llm_insights = self._get_llm_analysis(context)
        
        # Build the report
        report = {
            "timestamp": datetime.now().isoformat(),
            "market_snapshot": market_data,
            "market_bias": market_bias,
            "technical_analysis": self._summarize_technicals(analysis),
            "trade_setups": [self._setup_to_dict(s) for s in setups],
            "top_recommendation": self._get_top_recommendation(setups),
            "llm_insights": llm_insights,
            "risk_assessment": self._assess_risk(analysis, market_bias),
            "action_items": self._generate_action_items(setups, market_bias)
        }
        
        return report
    
    def _build_context(
        self,
        market_data: Dict,
        analysis: Dict,
        setups: List[TradeSetup],
        market_bias: Dict
    ) -> str:
        """Build context string for LLM analysis"""
        
        setups_text = ""
        for i, setup in enumerate(setups[:5], 1):  # Top 5 setups
            setups_text += f"""
Setup {i}: {setup.strategy.value}
- Direction: {setup.direction}
- Entry: {setup.entry_price}
- Stop Loss: {setup.stop_loss}
- Target 1: {setup.target_1}
- Target 2: {setup.target_2}
- Confidence: {setup.confidence}%
- Risk-Reward: {setup.risk_reward}
- Reasoning: {setup.reasoning}
"""
        
        context = f"""
=== NIFTY 50 5-MINUTE SCALPING ANALYSIS ===

CURRENT MARKET DATA:
- Symbol: {market_data.get('symbol', 'NIFTY 50')}
- Current Price: {market_data.get('close', 'N/A')}
- Change: {market_data.get('change_pct', 0):.2f}%
- Volume: {market_data.get('volume', 'N/A')}

MARKET BIAS:
- Overall Bias: {market_bias.get('bias', 'NEUTRAL')}
- Strength: {market_bias.get('strength', 50)}%
- EMA Trend: {market_bias.get('details', {}).get('ema_trend', 'N/A')}
- VWAP Position: {market_bias.get('details', {}).get('vwap_position', 'N/A')}
- MACD Trend: {market_bias.get('details', {}).get('macd_trend', 'N/A')}

TECHNICAL INDICATORS:
- RSI: {analysis.get('rsi', {}).get('value', 'N/A')} ({analysis.get('rsi', {}).get('condition', 'N/A')})
- MACD: {analysis.get('macd', {}).get('trend', 'N/A')}
- EMA Trend: {analysis.get('ema', {}).get('trend', 'N/A')}
- VWAP: {analysis.get('vwap', {}).get('value', 'N/A')} (Price {analysis.get('vwap', {}).get('position', 'N/A')})
- ATR: {analysis.get('atr', {}).get('value', 'N/A')} ({analysis.get('atr', {}).get('pct', 0):.2f}%)
- Volume Spike: {analysis.get('volume', {}).get('spike', False)}

BOLLINGER BANDS:
- Upper: {analysis.get('bollinger', {}).get('upper', 'N/A')}
- Middle: {analysis.get('bollinger', {}).get('middle', 'N/A')}
- Lower: {analysis.get('bollinger', {}).get('lower', 'N/A')}
- Squeeze: {analysis.get('bollinger', {}).get('squeeze', False)}

TRADE SETUPS IDENTIFIED:
{setups_text if setups_text else "No valid setups found at this time."}

Based on this analysis, provide:
1. Your assessment of current market conditions
2. The best trade opportunity (if any)
3. Key levels to watch
4. Risk factors to consider
5. Recommended action (WAIT, BUY, SELL, or specific setup)
"""
        return context
    
    def _get_llm_analysis(self, context: str) -> Dict:
        """Get analysis from LLM"""
        if not self.client:
            return self._generate_rule_based_insights(context)
        
        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1500,
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": context}
                    ]
                )
                return {
                    "source": "llm",
                    "analysis": response.content[0].text,
                    "model": "claude-3-sonnet"
                }
            
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    max_tokens=1500,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": context}
                    ]
                )
                return {
                    "source": "llm",
                    "analysis": response.choices[0].message.content,
                    "model": "gpt-4-turbo"
                }
                
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return self._generate_rule_based_insights(context)
        
        return self._generate_rule_based_insights(context)
    
    def _generate_rule_based_insights(self, context: str) -> Dict:
        """Generate insights using rule-based logic when LLM is unavailable"""
        # Parse context for key data points
        insights = []
        
        # Extract bias from context
        if "STRONGLY BULLISH" in context:
            insights.append("Market showing strong bullish momentum. Look for pullback entries on dips to EMAs or VWAP.")
        elif "STRONGLY BEARISH" in context:
            insights.append("Market showing strong bearish momentum. Look for short entries on rallies to resistance.")
        elif "BULLISH" in context:
            insights.append("Market has bullish bias. Favor long setups but be cautious near resistance.")
        elif "BEARISH" in context:
            insights.append("Market has bearish bias. Favor short setups but watch for support bounces.")
        else:
            insights.append("Market in consolidation. Wait for clear breakout or trade range extremes.")
        
        # Check for specific conditions
        if "Squeeze: True" in context:
            insights.append("⚠️ Bollinger Band squeeze detected - volatility expansion imminent. Watch for breakout direction.")
        
        if "Volume Spike: True" in context:
            insights.append("📊 Volume spike detected - current move has participation. Increases conviction on signals.")
        
        if "RSI:" in context:
            if "OVERSOLD" in context:
                insights.append("RSI oversold - watch for bullish reversal patterns.")
            elif "OVERBOUGHT" in context:
                insights.append("RSI overbought - watch for bearish reversal patterns.")
        
        return {
            "source": "rule_based",
            "analysis": "\n\n".join(insights),
            "insights": insights
        }
    
    def _summarize_technicals(self, analysis: Dict) -> Dict:
        """Create a summary of technical conditions"""
        return {
            "trend": analysis.get("ema", {}).get("trend", "NEUTRAL"),
            "momentum": analysis.get("macd", {}).get("trend", "NEUTRAL"),
            "rsi_condition": analysis.get("rsi", {}).get("condition", "NEUTRAL"),
            "vwap_position": analysis.get("vwap", {}).get("position", "AT"),
            "volatility": "HIGH" if analysis.get("atr", {}).get("pct", 0) > 0.5 else "NORMAL",
            "volume_condition": "SPIKE" if analysis.get("volume", {}).get("spike", False) else "NORMAL"
        }
    
    def _setup_to_dict(self, setup: TradeSetup) -> Dict:
        """Convert TradeSetup to dictionary"""
        return {
            "strategy": setup.strategy.value,
            "signal": setup.signal.value,
            "direction": setup.direction,
            "entry_price": setup.entry_price,
            "stop_loss": setup.stop_loss,
            "target_1": setup.target_1,
            "target_2": setup.target_2,
            "confidence": setup.confidence,
            "risk_reward": setup.risk_reward,
            "reasoning": setup.reasoning,
            "timestamp": setup.timestamp,
            "indicators": setup.indicators
        }
    
    def _get_top_recommendation(self, setups: List[TradeSetup]) -> Optional[Dict]:
        """Get the top trade recommendation"""
        if not setups:
            return {
                "action": "WAIT",
                "reason": "No high-confidence setups available. Wait for better opportunity.",
                "confidence": 0
            }
        
        top = setups[0]
        return {
            "action": f"{top.direction} ({top.strategy.value})",
            "entry": top.entry_price,
            "stop_loss": top.stop_loss,
            "target": top.target_1,
            "confidence": top.confidence,
            "risk_reward": top.risk_reward,
            "reason": top.reasoning
        }
    
    def _assess_risk(self, analysis: Dict, market_bias: Dict) -> Dict:
        """Assess current risk conditions"""
        risk_factors = []
        risk_level = "LOW"
        
        # Check for high volatility
        atr_pct = analysis.get("atr", {}).get("pct", 0)
        if atr_pct > 0.7:
            risk_factors.append("High volatility - use smaller position sizes")
            risk_level = "HIGH"
        elif atr_pct > 0.5:
            risk_factors.append("Elevated volatility")
            risk_level = "MEDIUM"
        
        # Check for extreme RSI
        rsi = analysis.get("rsi", {}).get("value", 50)
        if rsi > 75 or rsi < 25:
            risk_factors.append("RSI at extreme levels - reversal risk")
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
        
        # Check for weak trend
        if "NEUTRAL" in market_bias.get("bias", ""):
            risk_factors.append("No clear trend - choppy conditions likely")
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
        
        # Check for squeeze
        if analysis.get("bollinger", {}).get("squeeze", False):
            risk_factors.append("BB squeeze - explosive move possible in either direction")
        
        return {
            "level": risk_level,
            "factors": risk_factors,
            "recommendation": self._get_risk_recommendation(risk_level)
        }
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get position sizing recommendation based on risk"""
        if risk_level == "HIGH":
            return "Reduce position size to 50% of normal. Widen stops or avoid trading."
        elif risk_level == "MEDIUM":
            return "Use standard position size but ensure strict stop-loss adherence."
        else:
            return "Normal position sizing appropriate. Follow standard risk management."
    
    def _generate_action_items(self, setups: List[TradeSetup], market_bias: Dict) -> List[str]:
        """Generate actionable items for the trader"""
        actions = []
        
        if not setups:
            actions.append("⏳ No active setups - wait for valid signal")
            actions.append("👀 Monitor for: EMA crossovers, VWAP tests, or momentum breakouts")
        else:
            top = setups[0]
            if top.confidence >= 75:
                actions.append(f"🎯 HIGH CONFIDENCE: Consider {top.direction} at {top.entry_price}")
                actions.append(f"🛡️ Set stop-loss at {top.stop_loss}")
                actions.append(f"💰 Target: {top.target_1} (T1), {top.target_2} (T2)")
            else:
                actions.append(f"⚡ MODERATE CONFIDENCE: {top.direction} setup at {top.entry_price}")
                actions.append("📊 Wait for additional confirmation before entry")
        
        # Add bias-based actions
        bias = market_bias.get("bias", "NEUTRAL")
        if "BULLISH" in bias:
            actions.append("📈 Bullish bias - favor long positions")
        elif "BEARISH" in bias:
            actions.append("📉 Bearish bias - favor short positions")
        else:
            actions.append("⚖️ Neutral bias - trade range extremes or wait for breakout")
        
        return actions


class QuickScalpIdeas:
    """Generate quick scalp ideas without full LLM analysis"""
    
    def __init__(self, data_with_indicators):
        self.data = data_with_indicators
    
    def get_quick_ideas(self) -> List[Dict]:
        """Generate quick scalp ideas based on current conditions"""
        if self.data.empty:
            return []
        
        ideas = []
        latest = self.data.iloc[-1]
        
        # Quick long ideas
        if latest["rsi"] < 35 and latest["close"] < latest["vwap"]:
            ideas.append({
                "type": "QUICK LONG",
                "condition": "Oversold bounce",
                "entry": f"Near {latest['close']:.2f}",
                "target": f"VWAP at {latest['vwap']:.2f}",
                "stop": f"Below {latest['bb_lower']:.2f}",
                "timeframe": "2-4 candles"
            })
        
        if latest["ema_bullish"]:
            ideas.append({
                "type": "EMA LONG",
                "condition": "Fresh EMA crossover",
                "entry": f"At {latest['close']:.2f}",
                "target": f"Upper BB at {latest['bb_upper']:.2f}",
                "stop": f"Below EMA21 at {latest['ema_slow']:.2f}",
                "timeframe": "4-8 candles"
            })
        
        # Quick short ideas
        if latest["rsi"] > 65 and latest["close"] > latest["vwap"]:
            ideas.append({
                "type": "QUICK SHORT",
                "condition": "Overbought rejection",
                "entry": f"Near {latest['close']:.2f}",
                "target": f"VWAP at {latest['vwap']:.2f}",
                "stop": f"Above {latest['bb_upper']:.2f}",
                "timeframe": "2-4 candles"
            })
        
        if latest["ema_bearish"]:
            ideas.append({
                "type": "EMA SHORT",
                "condition": "Fresh EMA crossover down",
                "entry": f"At {latest['close']:.2f}",
                "target": f"Lower BB at {latest['bb_lower']:.2f}",
                "stop": f"Above EMA21 at {latest['ema_slow']:.2f}",
                "timeframe": "4-8 candles"
            })
        
        # Range ideas
        if latest["bb_squeeze"]:
            ideas.append({
                "type": "BREAKOUT WATCH",
                "condition": "BB Squeeze - breakout imminent",
                "entry": f"On breakout above {latest['bb_upper']:.2f} or below {latest['bb_lower']:.2f}",
                "target": "1.5x BB width from breakout",
                "stop": "Middle BB",
                "timeframe": "Wait for breakout"
            })
        
        return ideas


def create_research_agent() -> ScalpingResearchAgent:
    """Factory function to create a research agent"""
    return ScalpingResearchAgent()
