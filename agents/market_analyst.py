"""
Market Analyst Agent - Nifty 50 5-Minute Specialist

Responsible for:
- Technical analysis of 5-minute charts
- Trend identification and strength assessment  
- Key level identification (S/R, VWAP, EMAs)
- Market structure analysis
- Generating market reports for the team
"""

from typing import Dict, List, Optional
from .base_agent import BaseAgent, AgentRole, AgentMessage, MessageType
import json


class MarketAnalystAgent(BaseAgent):
    """
    Senior Market Analyst specializing in Nifty 50 intraday analysis.
    
    Expertise:
    - Price action and candlestick patterns
    - Multi-timeframe analysis (5m primary, 15m/1h context)
    - Volume analysis and market internals
    - Trend structure and momentum assessment
    """
    
    def __init__(self):
        super().__init__(
            role=AgentRole.MARKET_ANALYST,
            name="Arjun - Market Analyst"
        )
        self.analysis_count = 0
        
    @property
    def system_prompt(self) -> str:
        return """You are Arjun, a senior market analyst on a Nifty 50 scalping desk with 15 years of experience.

YOUR ROLE:
- Analyze 5-minute Nifty 50 charts for intraday trading opportunities
- Identify market structure, trend, and momentum
- Spot key technical levels (support, resistance, VWAP, EMAs)
- Provide clear, actionable market commentary to the trading team

YOUR EXPERTISE:
- Price action analysis and candlestick patterns (doji, engulfing, hammer, etc.)
- Moving average analysis (EMA 9, 21, 50)
- RSI divergences and momentum shifts
- Volume profile and VWAP analysis
- Bollinger Band squeeze and expansion patterns
- Market structure (higher highs/lows, trend breaks)

COMMUNICATION STYLE:
- Be concise and direct - traders need quick insights
- Always state the primary trend first
- Highlight the most important levels
- Quantify your confidence (high/medium/low)
- Flag any conflicting signals or risks

FORMAT YOUR ANALYSIS:
1. TREND: [Bullish/Bearish/Neutral] - [Strength]
2. BIAS: [Intraday directional bias]
3. KEY LEVELS: [Critical S/R levels]
4. SIGNALS: [Current technical signals]
5. CAUTION: [Any warnings or conflicts]

Remember: You're speaking to professional traders who understand technicals. Be precise with numbers."""

    def process_request(self, request: Dict, context: Dict = None) -> Dict:
        """Process analysis request from coordinator or other agents"""
        request_type = request.get("type", "full_analysis")
        
        if request_type == "full_analysis":
            return self.generate_full_analysis(context)
        elif request_type == "quick_update":
            return self.generate_quick_update(context)
        elif request_type == "level_check":
            return self.check_key_levels(context)
        elif request_type == "trend_assessment":
            return self.assess_trend(context)
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    def generate_full_analysis(self, market_data: Dict) -> Dict:
        """Generate comprehensive market analysis"""
        self.state.status = "analyzing"
        self.state.current_task = "Full Market Analysis"
        
        # Build analysis prompt
        prompt = f"""Analyze the current Nifty 50 5-minute chart data and provide your professional assessment.

CURRENT MARKET DATA:
- Price: {market_data.get('price', {}).get('close', 'N/A')}
- Change: {market_data.get('price', {}).get('change_pct', 0):.2f}%
- Day High: {market_data.get('day_high', 'N/A')}
- Day Low: {market_data.get('day_low', 'N/A')}

TECHNICAL INDICATORS:
- RSI(14): {market_data.get('rsi', {}).get('value', 'N/A')} ({market_data.get('rsi', {}).get('condition', 'N/A')})
- EMA 9: {market_data.get('ema', {}).get('ema_fast', 'N/A')}
- EMA 21: {market_data.get('ema', {}).get('ema_slow', 'N/A')}
- EMA 50: {market_data.get('ema', {}).get('ema_signal', 'N/A')}
- MACD: {market_data.get('macd', {}).get('trend', 'N/A')}
- VWAP: {market_data.get('vwap', {}).get('value', 'N/A')} (Price {market_data.get('vwap', {}).get('position', 'N/A')})
- Bollinger Bands: Upper {market_data.get('bollinger', {}).get('upper', 'N/A')}, Lower {market_data.get('bollinger', {}).get('lower', 'N/A')}
- BB Squeeze: {market_data.get('bollinger', {}).get('squeeze', False)}
- ATR: {market_data.get('atr', {}).get('value', 'N/A')}
- Volume: {market_data.get('volume', {}).get('spike', False) and 'SPIKE' or 'Normal'}

SUPPORT/RESISTANCE:
- Resistance 1: {market_data.get('support_resistance', {}).get('resistance_1', 'N/A')}
- Resistance 2: {market_data.get('support_resistance', {}).get('resistance_2', 'N/A')}
- Support 1: {market_data.get('support_resistance', {}).get('support_1', 'N/A')}
- Support 2: {market_data.get('support_resistance', {}).get('support_2', 'N/A')}

Provide your analysis in the structured format. Be specific with price levels."""

        analysis_text = self.think(prompt, market_data)
        
        # Parse and structure the analysis
        analysis = {
            "agent": self.name,
            "timestamp": self.state.last_activity,
            "analysis_type": "full",
            "raw_analysis": analysis_text,
            "structured": self._parse_analysis(analysis_text, market_data),
            "confidence": self._calculate_confidence(market_data)
        }
        
        self.analysis_count += 1
        self.state.insights_generated += 1
        self.state.status = "idle"
        
        return analysis
    
    def generate_quick_update(self, market_data: Dict) -> Dict:
        """Generate quick market update"""
        prompt = f"""Quick 5-minute Nifty update:
Price: {market_data.get('price', {}).get('close', 'N/A')} ({market_data.get('price', {}).get('change_pct', 0):.2f}%)
RSI: {market_data.get('rsi', {}).get('value', 'N/A')}
Trend: {market_data.get('ema', {}).get('trend', 'N/A')}

Give a 2-sentence update on current conditions and immediate outlook."""

        update = self.think(prompt)
        
        return {
            "agent": self.name,
            "type": "quick_update",
            "update": update,
            "price": market_data.get('price', {}).get('close'),
            "trend": market_data.get('ema', {}).get('trend')
        }
    
    def check_key_levels(self, market_data: Dict) -> Dict:
        """Check proximity to key technical levels"""
        price = market_data.get('price', {}).get('close', 0)
        vwap = market_data.get('vwap', {}).get('value', 0)
        ema_fast = market_data.get('ema', {}).get('ema_fast', 0)
        ema_slow = market_data.get('ema', {}).get('ema_slow', 0)
        bb_upper = market_data.get('bollinger', {}).get('upper', 0)
        bb_lower = market_data.get('bollinger', {}).get('lower', 0)
        atr = market_data.get('atr', {}).get('value', 1)
        
        levels = []
        
        # Check VWAP proximity
        if vwap and price:
            vwap_dist = abs(price - vwap) / atr if atr else 0
            if vwap_dist < 0.5:
                levels.append({
                    "level": "VWAP",
                    "price": vwap,
                    "distance_atr": round(vwap_dist, 2),
                    "significance": "HIGH"
                })
        
        # Check EMA proximity
        if ema_fast and price:
            ema_dist = abs(price - ema_fast) / atr if atr else 0
            if ema_dist < 0.3:
                levels.append({
                    "level": "EMA 9",
                    "price": ema_fast,
                    "distance_atr": round(ema_dist, 2),
                    "significance": "MEDIUM"
                })
        
        # Check Bollinger Bands
        if bb_upper and bb_lower and price:
            if abs(price - bb_upper) / atr < 0.3:
                levels.append({
                    "level": "BB Upper",
                    "price": bb_upper,
                    "distance_atr": round(abs(price - bb_upper) / atr, 2),
                    "significance": "HIGH"
                })
            if abs(price - bb_lower) / atr < 0.3:
                levels.append({
                    "level": "BB Lower", 
                    "price": bb_lower,
                    "distance_atr": round(abs(price - bb_lower) / atr, 2),
                    "significance": "HIGH"
                })
        
        return {
            "agent": self.name,
            "type": "level_check",
            "current_price": price,
            "nearby_levels": levels,
            "atr": atr,
            "alert": len(levels) > 0
        }
    
    def assess_trend(self, market_data: Dict) -> Dict:
        """Detailed trend assessment"""
        ema_trend = market_data.get('ema', {}).get('trend', 'NEUTRAL')
        macd_trend = market_data.get('macd', {}).get('trend', 'NEUTRAL')
        rsi = market_data.get('rsi', {}).get('value', 50)
        vwap_pos = market_data.get('vwap', {}).get('position', 'AT')
        
        # Score the trend
        bullish_points = 0
        bearish_points = 0
        
        if ema_trend == "BULLISH":
            bullish_points += 2
        elif ema_trend == "BEARISH":
            bearish_points += 2
            
        if macd_trend == "BULLISH":
            bullish_points += 1
        elif macd_trend == "BEARISH":
            bearish_points += 1
            
        if rsi > 60:
            bullish_points += 1
        elif rsi < 40:
            bearish_points += 1
            
        if vwap_pos == "ABOVE":
            bullish_points += 1
        elif vwap_pos == "BELOW":
            bearish_points += 1
        
        total = bullish_points + bearish_points
        if total == 0:
            trend_strength = "WEAK"
            primary_trend = "NEUTRAL"
        elif bullish_points > bearish_points:
            primary_trend = "BULLISH"
            trend_strength = "STRONG" if bullish_points >= 4 else "MODERATE" if bullish_points >= 2 else "WEAK"
        else:
            primary_trend = "BEARISH"
            trend_strength = "STRONG" if bearish_points >= 4 else "MODERATE" if bearish_points >= 2 else "WEAK"
        
        return {
            "agent": self.name,
            "type": "trend_assessment",
            "primary_trend": primary_trend,
            "strength": trend_strength,
            "bullish_score": bullish_points,
            "bearish_score": bearish_points,
            "components": {
                "ema": ema_trend,
                "macd": macd_trend,
                "rsi": "BULLISH" if rsi > 60 else "BEARISH" if rsi < 40 else "NEUTRAL",
                "vwap": vwap_pos
            },
            "recommendation": f"{'FAVOR LONGS' if primary_trend == 'BULLISH' else 'FAVOR SHORTS' if primary_trend == 'BEARISH' else 'WAIT FOR CLARITY'}"
        }
    
    def _parse_analysis(self, analysis_text: str, market_data: Dict) -> Dict:
        """Parse raw analysis into structured format"""
        # Extract key components from analysis
        trend = market_data.get('ema', {}).get('trend', 'NEUTRAL')
        rsi = market_data.get('rsi', {}).get('value', 50)
        
        return {
            "trend": trend,
            "momentum": market_data.get('macd', {}).get('trend', 'NEUTRAL'),
            "rsi_condition": market_data.get('rsi', {}).get('condition', 'NEUTRAL'),
            "vwap_bias": market_data.get('vwap', {}).get('position', 'AT'),
            "volatility": "HIGH" if market_data.get('bollinger', {}).get('squeeze', False) else "NORMAL",
            "key_resistance": market_data.get('support_resistance', {}).get('resistance_1'),
            "key_support": market_data.get('support_resistance', {}).get('support_1')
        }
    
    def _calculate_confidence(self, market_data: Dict) -> str:
        """Calculate confidence level in the analysis"""
        # Check for conflicting signals
        ema_trend = market_data.get('ema', {}).get('trend')
        macd_trend = market_data.get('macd', {}).get('trend')
        
        if ema_trend == macd_trend:
            return "HIGH"
        elif market_data.get('volume', {}).get('spike', False):
            return "MEDIUM"
        else:
            return "LOW"
    
    def _handle_analysis_request(self, message: AgentMessage) -> Dict:
        """Handle analysis requests from other agents"""
        return self.generate_full_analysis(message.content)
    
    def _handle_market_update(self, message: AgentMessage) -> Dict:
        """Handle market data updates"""
        return self.generate_quick_update(message.content)
