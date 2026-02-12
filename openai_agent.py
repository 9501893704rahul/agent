"""
Advanced OpenAI-Powered Scalping Agent for Nifty 50 5-Minute Trading

This module implements a sophisticated AI agent using OpenAI's GPT models
for real-time scalping research and trade signal generation.
"""

import json
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import config


class AgentRole(Enum):
    ANALYST = "analyst"
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    STRATEGIST = "strategist"


@dataclass
class AgentMessage:
    role: str
    content: str
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TradeIdea:
    direction: str  # LONG, SHORT, NEUTRAL
    strategy: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: Optional[float]
    confidence: int  # 0-100
    risk_reward: float
    reasoning: str
    key_levels: Dict[str, float]
    invalidation: str
    timeframe: str


class OpenAIScalpingAgent:
    """
    Advanced OpenRouter/OpenAI-powered agent for Nifty 50 scalping analysis.
    
    Features:
    - Multi-turn conversation for deep analysis
    - Function calling for structured outputs
    - Streaming responses for real-time updates
    - Context-aware recommendations
    - OpenRouter support for multiple model providers
    """
    
    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        # OpenRouter configuration
        self.api_key = api_key or config.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
        self.model = model or config.OPENROUTER_MODEL
        self.base_url = base_url or config.OPENROUTER_BASE_URL
        self.client = None
        self.conversation_history: List[AgentMessage] = []
        self.tools = self._define_tools()
        
        self._initialize_client()
        
        # System prompts for different roles
        self.system_prompts = {
            AgentRole.ANALYST: self._get_analyst_prompt(),
            AgentRole.TRADER: self._get_trader_prompt(),
            AgentRole.RISK_MANAGER: self._get_risk_manager_prompt(),
            AgentRole.STRATEGIST: self._get_strategist_prompt()
        }
    
    def _initialize_client(self):
        """Initialize OpenRouter client (OpenAI-compatible)"""
        if not self.api_key:
            print("⚠️ OpenRouter API key not configured. Running in offline mode.")
            return
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers={
                    "HTTP-Referer": "https://nifty-scalping-agent.app",
                    "X-Title": "Nifty 50 Scalping Agent"
                }
            )
            print(f"✅ OpenRouter client initialized with model: {self.model}")
        except ImportError:
            print("❌ OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            print(f"❌ Error initializing OpenRouter client: {e}")
    
    def _get_analyst_prompt(self) -> str:
        return """You are an expert technical analyst specializing in Nifty 50 index scalping on the 5-minute timeframe.

Your expertise includes:
- Price action analysis and candlestick patterns
- Technical indicators (RSI, MACD, EMA, Bollinger Bands, VWAP)
- Volume analysis and market microstructure
- Support/resistance identification
- Trend analysis and momentum assessment

When analyzing market data:
1. First assess the overall market structure and trend
2. Identify key technical levels (support, resistance, VWAP, EMAs)
3. Evaluate momentum and volume conditions
4. Look for confluence of multiple indicators
5. Consider the risk-reward of potential setups

Always provide specific price levels and be quantitative in your analysis.
Format your analysis clearly with sections for trend, momentum, key levels, and signals."""

    def _get_trader_prompt(self) -> str:
        return """You are a professional Nifty 50 scalper executing trades on the 5-minute timeframe.

Your trading rules:
- Minimum risk-reward ratio: 1.5:1
- Maximum trade duration: 60 minutes (12 candles)
- Position sizing based on ATR-calculated stop loss
- Never risk more than 1% per trade
- Use limit orders for entries when possible

For each trade idea, you MUST provide:
1. Exact entry price or zone
2. Stop-loss level (mandatory)
3. Target 1 (partial profit)
4. Target 2 (full exit)
5. Trade management rules
6. Invalidation criteria

Be decisive. If there's a valid setup, recommend action. If not, clearly say WAIT."""

    def _get_risk_manager_prompt(self) -> str:
        return """You are a risk manager for a Nifty 50 scalping desk.

Your responsibilities:
- Evaluate trade risk before execution
- Assess position sizing recommendations
- Identify potential risks in current market conditions
- Monitor correlation with broader market moves
- Flag high-risk scenarios

Risk factors to consider:
1. Volatility regime (ATR-based)
2. Time of day (opening, closing, lunch)
3. News/event risk
4. Trend strength vs. choppy conditions
5. Liquidity conditions

Provide clear risk ratings: LOW, MEDIUM, HIGH, EXTREME
For each rating, explain why and suggest position adjustments."""

    def _get_strategist_prompt(self) -> str:
        return """You are a trading strategist developing scalping strategies for Nifty 50.

Available strategies to evaluate:
1. VWAP Bounce - Trade pullbacks to VWAP in trending markets
2. EMA Crossover - Trade momentum shifts using EMA 9/21
3. RSI Divergence - Catch reversals at extremes
4. Bollinger Squeeze - Trade volatility expansions
5. Momentum Breakout - Trade strong directional moves
6. Pullback Entry - Enter trends on EMA pullbacks
7. Scalp Reversal - Quick trades at overbought/oversold extremes

For each market condition, recommend:
- Best strategy to deploy
- Why this strategy fits current conditions
- Expected win rate and risk-reward
- When to avoid this strategy"""

    def _define_tools(self) -> List[Dict]:
        """Define function calling tools for structured outputs"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_trade_idea",
                    "description": "Generate a structured trade idea with entry, stop-loss, and targets",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["LONG", "SHORT", "NEUTRAL"],
                                "description": "Trade direction"
                            },
                            "strategy": {
                                "type": "string",
                                "description": "Strategy name (e.g., VWAP Bounce, EMA Crossover)"
                            },
                            "entry_price": {
                                "type": "number",
                                "description": "Recommended entry price"
                            },
                            "stop_loss": {
                                "type": "number",
                                "description": "Stop-loss price level"
                            },
                            "target_1": {
                                "type": "number",
                                "description": "First profit target"
                            },
                            "target_2": {
                                "type": "number",
                                "description": "Second profit target"
                            },
                            "confidence": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                                "description": "Confidence level 0-100"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Detailed reasoning for the trade"
                            },
                            "invalidation": {
                                "type": "string",
                                "description": "When to invalidate this trade idea"
                            }
                        },
                        "required": ["direction", "strategy", "entry_price", "stop_loss", "target_1", "confidence", "reasoning"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_market_condition",
                    "description": "Analyze current market conditions and bias",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "trend": {
                                "type": "string",
                                "enum": ["STRONG_UPTREND", "UPTREND", "NEUTRAL", "DOWNTREND", "STRONG_DOWNTREND"],
                                "description": "Current trend assessment"
                            },
                            "momentum": {
                                "type": "string",
                                "enum": ["STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"],
                                "description": "Momentum assessment"
                            },
                            "volatility": {
                                "type": "string",
                                "enum": ["LOW", "NORMAL", "HIGH", "EXTREME"],
                                "description": "Volatility regime"
                            },
                            "bias": {
                                "type": "string",
                                "enum": ["BULLISH", "NEUTRAL", "BEARISH"],
                                "description": "Overall market bias"
                            },
                            "key_levels": {
                                "type": "object",
                                "properties": {
                                    "resistance_1": {"type": "number"},
                                    "resistance_2": {"type": "number"},
                                    "support_1": {"type": "number"},
                                    "support_2": {"type": "number"},
                                    "vwap": {"type": "number"}
                                },
                                "description": "Key price levels to watch"
                            },
                            "analysis": {
                                "type": "string",
                                "description": "Detailed market analysis"
                            }
                        },
                        "required": ["trend", "momentum", "volatility", "bias", "analysis"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "risk_assessment",
                    "description": "Assess risk for current market conditions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "risk_level": {
                                "type": "string",
                                "enum": ["LOW", "MEDIUM", "HIGH", "EXTREME"],
                                "description": "Overall risk level"
                            },
                            "position_size_adjustment": {
                                "type": "number",
                                "description": "Recommended position size multiplier (0.25-1.0)"
                            },
                            "risk_factors": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of identified risk factors"
                            },
                            "recommendations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Risk management recommendations"
                            }
                        },
                        "required": ["risk_level", "position_size_adjustment", "risk_factors"]
                    }
                }
            }
        ]
    
    def analyze(
        self,
        market_data: Dict,
        technical_analysis: Dict,
        trade_setups: List[Dict],
        role: AgentRole = AgentRole.TRADER
    ) -> Dict:
        """
        Run comprehensive analysis using OpenAI
        
        Args:
            market_data: Current OHLCV data
            technical_analysis: Indicator values and signals
            trade_setups: Pre-computed trade setups from strategies
            role: Agent role to use for analysis
            
        Returns:
            Analysis results with trade recommendations
        """
        if not self.client:
            return self._offline_analysis(market_data, technical_analysis, trade_setups)
        
        # Build the analysis prompt
        prompt = self._build_analysis_prompt(market_data, technical_analysis, trade_setups)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompts[role]},
                    {"role": "user", "content": prompt}
                ],
                tools=self.tools,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=2000
            )
            
            return self._process_response(response)
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._offline_analysis(market_data, technical_analysis, trade_setups)
    
    def generate_scalp_ideas(
        self,
        market_data: Dict,
        technical_analysis: Dict,
        num_ideas: int = 3
    ) -> List[TradeIdea]:
        """Generate multiple scalp trade ideas"""
        if not self.client:
            return self._offline_scalp_ideas(market_data, technical_analysis)
        
        prompt = f"""Based on the following Nifty 50 5-minute data, generate {num_ideas} scalping trade ideas.

MARKET DATA:
{json.dumps(market_data, indent=2)}

TECHNICAL ANALYSIS:
{json.dumps(technical_analysis, indent=2)}

For each idea, call the generate_trade_idea function with complete details.
Prioritize ideas by confidence level. Include at least one contrarian idea if applicable."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompts[AgentRole.TRADER]},
                    {"role": "user", "content": prompt}
                ],
                tools=self.tools,
                tool_choice={"type": "function", "function": {"name": "generate_trade_idea"}},
                temperature=0.4,
                max_tokens=3000,
                n=1
            )
            
            ideas = []
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.function.name == "generate_trade_idea":
                        args = json.loads(tool_call.function.arguments)
                        idea = TradeIdea(
                            direction=args.get("direction", "NEUTRAL"),
                            strategy=args.get("strategy", "Unknown"),
                            entry_price=args.get("entry_price", 0),
                            stop_loss=args.get("stop_loss", 0),
                            target_1=args.get("target_1", 0),
                            target_2=args.get("target_2"),
                            confidence=args.get("confidence", 50),
                            risk_reward=self._calc_rr(args),
                            reasoning=args.get("reasoning", ""),
                            key_levels={},
                            invalidation=args.get("invalidation", ""),
                            timeframe="5-minute"
                        )
                        ideas.append(idea)
            
            # Sort by confidence
            ideas.sort(key=lambda x: x.confidence, reverse=True)
            return ideas
            
        except Exception as e:
            print(f"Error generating ideas: {e}")
            return self._offline_scalp_ideas(market_data, technical_analysis)
    
    def chat(self, message: str, context: Dict = None) -> str:
        """Interactive chat with the agent"""
        if not self.client:
            return "OpenAI client not initialized. Please provide API key."
        
        # Add context if provided
        full_message = message
        if context:
            full_message = f"""Context:
{json.dumps(context, indent=2)}

User Query: {message}"""
        
        # Add to conversation history
        self.conversation_history.append(AgentMessage(role="user", content=message))
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompts[AgentRole.TRADER]}
            ]
            
            # Add recent conversation history
            for msg in self.conversation_history[-10:]:
                messages.append({"role": msg.role, "content": msg.content})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=500
            )
            
            assistant_message = response.choices[0].message.content
            self.conversation_history.append(AgentMessage(role="assistant", content=assistant_message))
            
            return assistant_message
            
        except Exception as e:
            return f"Error: {e}"
    
    def get_market_commentary(self, market_data: Dict, technical_analysis: Dict) -> str:
        """Get natural language market commentary"""
        if not self.client:
            return self._offline_commentary(market_data, technical_analysis)
        
        prompt = f"""Provide a brief, professional market commentary for Nifty 50 based on this data:

Current Price: {market_data.get('close', 'N/A')}
Change: {market_data.get('change_pct', 0):.2f}%

Technical Snapshot:
- RSI: {technical_analysis.get('rsi', {}).get('value', 'N/A')}
- MACD: {technical_analysis.get('macd', {}).get('trend', 'N/A')}
- VWAP Position: {technical_analysis.get('vwap', {}).get('position', 'N/A')}
- EMA Trend: {technical_analysis.get('ema', {}).get('trend', 'N/A')}

Write 2-3 sentences summarizing current conditions and outlook for scalpers."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a concise market commentator for trading desks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Commentary error: {e}")
            return self._offline_commentary(market_data, technical_analysis)
    
    def _build_analysis_prompt(
        self,
        market_data: Dict,
        technical_analysis: Dict,
        trade_setups: List[Dict]
    ) -> str:
        """Build comprehensive analysis prompt"""
        setups_text = ""
        for i, setup in enumerate(trade_setups[:5], 1):
            setups_text += f"""
Setup {i}: {setup.get('strategy', 'Unknown')}
- Direction: {setup.get('direction', 'N/A')}
- Entry: {setup.get('entry_price', 'N/A')}
- Stop Loss: {setup.get('stop_loss', 'N/A')}
- Target: {setup.get('target_1', 'N/A')}
- Confidence: {setup.get('confidence', 0)}%
- R:R: {setup.get('risk_reward', 0)}
"""
        
        return f"""Analyze the following Nifty 50 5-minute scalping data and provide trade recommendations.

=== CURRENT MARKET DATA ===
Symbol: {market_data.get('symbol', 'NIFTY 50')}
Price: {market_data.get('close', 'N/A')}
Open: {market_data.get('open', 'N/A')}
High: {market_data.get('high', 'N/A')}
Low: {market_data.get('low', 'N/A')}
Change: {market_data.get('change_pct', 0):.2f}%
Volume: {market_data.get('volume', 'N/A')}

=== TECHNICAL INDICATORS ===
RSI: {technical_analysis.get('rsi', {}).get('value', 'N/A')} ({technical_analysis.get('rsi', {}).get('condition', 'N/A')})
MACD: {technical_analysis.get('macd', {}).get('macd', 'N/A')} (Signal: {technical_analysis.get('macd', {}).get('signal', 'N/A')})
MACD Trend: {technical_analysis.get('macd', {}).get('trend', 'N/A')}
EMA Fast (9): {technical_analysis.get('ema', {}).get('ema_fast', 'N/A')}
EMA Slow (21): {technical_analysis.get('ema', {}).get('ema_slow', 'N/A')}
EMA Trend: {technical_analysis.get('ema', {}).get('trend', 'N/A')}
VWAP: {technical_analysis.get('vwap', {}).get('value') or 'N/A'}
Price vs VWAP: {technical_analysis.get('vwap', {}).get('position', 'N/A')} ({(technical_analysis.get('vwap', {}).get('deviation') or 0):.2f}%)
Bollinger Upper: {technical_analysis.get('bollinger', {}).get('upper', 'N/A')}
Bollinger Lower: {technical_analysis.get('bollinger', {}).get('lower', 'N/A')}
BB Squeeze: {technical_analysis.get('bollinger', {}).get('squeeze', False)}
ATR: {technical_analysis.get('atr', {}).get('value', 'N/A')} ({(technical_analysis.get('atr', {}).get('pct') or 0):.2f}%)
Volume Spike: {technical_analysis.get('volume', {}).get('spike', False)}

=== PRE-COMPUTED TRADE SETUPS ===
{setups_text if setups_text else "No setups identified by rule-based strategies."}

=== YOUR TASK ===
1. Call analyze_market_condition function with your market assessment
2. Call risk_assessment function with current risk evaluation  
3. Call generate_trade_idea function with your best trade recommendation (if any valid setup exists)

If no good setup exists, recommend NEUTRAL direction with reasoning."""
    
    def _process_response(self, response) -> Dict:
        """Process OpenAI response with function calls"""
        result = {
            "raw_response": None,
            "market_analysis": None,
            "risk_assessment": None,
            "trade_idea": None,
            "commentary": ""
        }
        
        message = response.choices[0].message
        
        # Get text response if any
        if message.content:
            result["commentary"] = message.content
            result["raw_response"] = message.content
        
        # Process tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                if func_name == "analyze_market_condition":
                    result["market_analysis"] = args
                elif func_name == "risk_assessment":
                    result["risk_assessment"] = args
                elif func_name == "generate_trade_idea":
                    args["risk_reward"] = self._calc_rr(args)
                    result["trade_idea"] = args
        
        return result
    
    def _calc_rr(self, args: Dict) -> float:
        """Calculate risk-reward ratio"""
        entry = args.get("entry_price", 0)
        sl = args.get("stop_loss", 0)
        t1 = args.get("target_1", 0)
        
        if entry and sl and t1:
            risk = abs(entry - sl)
            reward = abs(t1 - entry)
            return round(reward / risk, 2) if risk > 0 else 0
        return 0
    
    def _offline_analysis(
        self,
        market_data: Dict,
        technical_analysis: Dict,
        trade_setups: List[Dict]
    ) -> Dict:
        """Provide analysis when OpenAI is not available"""
        # Determine bias
        ema_trend = technical_analysis.get("ema", {}).get("trend", "NEUTRAL")
        macd_trend = technical_analysis.get("macd", {}).get("trend", "NEUTRAL")
        vwap_pos = technical_analysis.get("vwap", {}).get("position", "AT")
        rsi = technical_analysis.get("rsi", {}).get("value", 50)
        
        bullish_count = sum([
            ema_trend == "BULLISH",
            macd_trend == "BULLISH",
            vwap_pos == "ABOVE",
            rsi > 50
        ])
        
        if bullish_count >= 3:
            bias = "BULLISH"
            trend = "UPTREND"
        elif bullish_count <= 1:
            bias = "BEARISH"
            trend = "DOWNTREND"
        else:
            bias = "NEUTRAL"
            trend = "NEUTRAL"
        
        return {
            "market_analysis": {
                "trend": trend,
                "momentum": macd_trend,
                "volatility": "HIGH" if technical_analysis.get("atr", {}).get("pct", 0) > 0.5 else "NORMAL",
                "bias": bias,
                "analysis": f"Market showing {bias.lower()} bias with {ema_trend.lower()} EMA trend."
            },
            "risk_assessment": {
                "risk_level": "MEDIUM",
                "position_size_adjustment": 0.75,
                "risk_factors": ["Operating in offline mode - limited analysis"]
            },
            "trade_idea": trade_setups[0] if trade_setups else None,
            "commentary": f"Nifty 50 is currently {bias.lower()} with price {vwap_pos.lower()} VWAP."
        }
    
    def _offline_scalp_ideas(
        self,
        market_data: Dict,
        technical_analysis: Dict
    ) -> List[TradeIdea]:
        """Generate ideas without OpenAI"""
        ideas = []
        
        price = market_data.get("close", 0)
        rsi = technical_analysis.get("rsi", {}).get("value", 50)
        vwap = technical_analysis.get("vwap", {}).get("value", price)
        atr = technical_analysis.get("atr", {}).get("value", price * 0.005)
        
        # RSI-based idea
        if rsi < 35:
            ideas.append(TradeIdea(
                direction="LONG",
                strategy="RSI Oversold Bounce",
                entry_price=price,
                stop_loss=price - (atr * 1.5),
                target_1=vwap,
                target_2=price + (atr * 2.5),
                confidence=65,
                risk_reward=1.5,
                reasoning=f"RSI at {rsi:.1f} indicates oversold conditions. Look for bounce to VWAP.",
                key_levels={"vwap": vwap, "stop": price - (atr * 1.5)},
                invalidation="Close below stop loss level",
                timeframe="5-minute"
            ))
        elif rsi > 65:
            ideas.append(TradeIdea(
                direction="SHORT",
                strategy="RSI Overbought Fade",
                entry_price=price,
                stop_loss=price + (atr * 1.5),
                target_1=vwap,
                target_2=price - (atr * 2.5),
                confidence=65,
                risk_reward=1.5,
                reasoning=f"RSI at {rsi:.1f} indicates overbought conditions. Look for fade to VWAP.",
                key_levels={"vwap": vwap, "stop": price + (atr * 1.5)},
                invalidation="Close above stop loss level",
                timeframe="5-minute"
            ))
        
        return ideas
    
    def _offline_commentary(self, market_data: Dict, technical_analysis: Dict) -> str:
        """Generate commentary without OpenAI"""
        price = market_data.get("close", 0)
        change = market_data.get("change_pct", 0)
        rsi = technical_analysis.get("rsi", {}).get("value", 50)
        trend = technical_analysis.get("ema", {}).get("trend", "NEUTRAL")
        
        direction = "higher" if change > 0 else "lower" if change < 0 else "flat"
        
        return f"Nifty 50 trading {direction} at {price:.2f} ({change:+.2f}%). RSI at {rsi:.1f} with {trend.lower()} EMA structure. {'Momentum intact.' if trend != 'NEUTRAL' else 'Awaiting directional clarity.'}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


def create_openai_agent(api_key: str = None) -> OpenAIScalpingAgent:
    """Factory function to create OpenAI agent"""
    return OpenAIScalpingAgent(api_key=api_key)


if __name__ == "__main__":
    # Test the agent
    agent = create_openai_agent()
    
    # Sample data for testing
    sample_market_data = {
        "symbol": "NIFTY 50",
        "close": 22150.50,
        "open": 22100.00,
        "high": 22180.00,
        "low": 22050.00,
        "change_pct": 0.45,
        "volume": 125000
    }
    
    sample_analysis = {
        "rsi": {"value": 58, "condition": "NEUTRAL"},
        "macd": {"macd": 15.5, "signal": 12.3, "trend": "BULLISH"},
        "ema": {"ema_fast": 22140, "ema_slow": 22100, "trend": "BULLISH"},
        "vwap": {"value": 22120, "position": "ABOVE", "deviation": 0.14},
        "bollinger": {"upper": 22200, "lower": 22050, "squeeze": False},
        "atr": {"value": 45, "pct": 0.20},
        "volume": {"spike": False}
    }
    
    print("=" * 50)
    print("OpenAI Scalping Agent Test")
    print("=" * 50)
    
    # Test commentary
    commentary = agent.get_market_commentary(sample_market_data, sample_analysis)
    print(f"\nMarket Commentary:\n{commentary}")
    
    # Test full analysis
    result = agent.analyze(sample_market_data, sample_analysis, [])
    print(f"\nAnalysis Result:\n{json.dumps(result, indent=2)}")
