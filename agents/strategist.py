"""
Strategy Agent - Nifty 50 5-Minute Scalping Strategist

Responsible for:
- Evaluating multiple scalping strategies
- Matching market conditions to optimal strategies
- Generating trade setups with specific parameters
- Strategy performance tracking
"""

from typing import Dict, List, Optional
from .base_agent import BaseAgent, AgentRole, AgentMessage, MessageType
import json


class StrategyAgent(BaseAgent):
    """
    Chief Strategist specializing in Nifty 50 scalping strategies.
    
    Strategies in arsenal:
    1. VWAP Bounce - Trade pullbacks to VWAP
    2. EMA Crossover - Momentum continuation
    3. RSI Divergence - Reversal plays
    4. Bollinger Squeeze - Volatility breakouts
    5. Momentum Breakout - Strong directional moves
    6. Pullback Entry - Trend continuation
    7. Scalp Reversal - Quick reversal trades
    """
    
    def __init__(self):
        super().__init__(
            role=AgentRole.STRATEGIST,
            name="Priya - Chief Strategist"
        )
        self.strategies = self._init_strategies()
        self.active_setups = []
        
    @property
    def system_prompt(self) -> str:
        return """You are Priya, the Chief Strategist on a Nifty 50 scalping desk with expertise in systematic trading.

YOUR ROLE:
- Evaluate market conditions and select optimal scalping strategies
- Design specific trade setups with entry, stop-loss, and targets
- Match strategy to current market regime (trending, ranging, volatile)
- Provide clear rationale for each strategy recommendation

YOUR STRATEGY ARSENAL:
1. VWAP BOUNCE: Trade pullbacks to VWAP in trending markets. Best when trend is clear.
2. EMA CROSSOVER: Trade EMA 9/21 crossovers with momentum confirmation. Good for trend starts.
3. RSI DIVERGENCE: Catch reversals when price/RSI diverge. Best at extremes (RSI <30 or >70).
4. BOLLINGER SQUEEZE: Trade breakouts after low volatility periods. Watch for volume confirmation.
5. MOMENTUM BREAKOUT: Trade strong moves breaking recent highs/lows. Needs volume spike.
6. PULLBACK ENTRY: Enter trends on pullbacks to EMA 9 or 21. Best in strong trends.
7. SCALP REVERSAL: Quick trades at overbought/oversold extremes. Tight stops required.

STRATEGY SELECTION RULES:
- TRENDING MARKET: VWAP Bounce, Pullback Entry, EMA Crossover
- RANGING MARKET: Scalp Reversal, Bollinger Squeeze
- HIGH VOLATILITY: Momentum Breakout, Bollinger Squeeze
- LOW VOLATILITY: Avoid trading or use Squeeze breakout

FOR EACH SETUP PROVIDE:
1. Strategy name and type (LONG/SHORT)
2. Entry zone (specific price or condition)
3. Stop-loss (in points and as price level)
4. Target 1 (partial exit) and Target 2 (full exit)
5. Risk-Reward ratio (minimum 1.5:1)
6. Confidence level (1-100%)
7. Invalidation criteria
8. Time horizon (number of 5-min candles)

Be specific with numbers. Round to nearest 0.05 for Nifty levels."""

    def _init_strategies(self) -> Dict:
        """Initialize strategy configurations"""
        return {
            "vwap_bounce": {
                "name": "VWAP Bounce",
                "market_condition": ["trending"],
                "min_rr": 1.5,
                "typical_duration": "4-8 candles",
                "best_when": "Clear trend with price pulling back to VWAP"
            },
            "ema_crossover": {
                "name": "EMA Crossover",
                "market_condition": ["trending", "early_trend"],
                "min_rr": 2.0,
                "typical_duration": "6-12 candles",
                "best_when": "Fresh crossover with MACD confirmation"
            },
            "rsi_divergence": {
                "name": "RSI Divergence",
                "market_condition": ["reversal", "extreme"],
                "min_rr": 1.5,
                "typical_duration": "4-8 candles",
                "best_when": "RSI at extremes with divergence pattern"
            },
            "bollinger_squeeze": {
                "name": "Bollinger Squeeze Breakout",
                "market_condition": ["low_volatility", "consolidation"],
                "min_rr": 2.0,
                "typical_duration": "4-10 candles",
                "best_when": "BB width at 20-period low, awaiting expansion"
            },
            "momentum_breakout": {
                "name": "Momentum Breakout",
                "market_condition": ["high_volatility", "breakout"],
                "min_rr": 2.0,
                "typical_duration": "2-6 candles",
                "best_when": "Breaking key level with volume spike"
            },
            "pullback_entry": {
                "name": "Pullback Entry",
                "market_condition": ["strong_trend"],
                "min_rr": 1.5,
                "typical_duration": "4-8 candles",
                "best_when": "Strong trend with price at EMA support/resistance"
            },
            "scalp_reversal": {
                "name": "Scalp Reversal",
                "market_condition": ["extreme", "ranging"],
                "min_rr": 1.5,
                "typical_duration": "2-4 candles",
                "best_when": "Multiple indicators at extreme with reversal candle"
            }
        }

    def process_request(self, request: Dict, context: Dict = None) -> Dict:
        """Process strategy request"""
        request_type = request.get("type", "recommend_strategy")
        
        if request_type == "recommend_strategy":
            return self.recommend_strategy(context)
        elif request_type == "generate_setup":
            return self.generate_trade_setup(request.get("strategy"), context)
        elif request_type == "evaluate_all":
            return self.evaluate_all_strategies(context)
        elif request_type == "validate_setup":
            return self.validate_setup(request.get("setup"), context)
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    def recommend_strategy(self, market_data: Dict) -> Dict:
        """Recommend best strategy for current market conditions"""
        self.state.status = "analyzing"
        self.state.current_task = "Strategy Selection"
        
        # Analyze market conditions
        conditions = self._analyze_conditions(market_data)
        
        # Score each strategy
        strategy_scores = {}
        for strategy_id, strategy in self.strategies.items():
            score = self._score_strategy(strategy, conditions)
            strategy_scores[strategy_id] = score
        
        # Get best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best_strategy]
        
        # Use LLM for detailed recommendation
        prompt = f"""Based on current Nifty 50 5-minute conditions, recommend the best scalping strategy.

MARKET CONDITIONS:
{json.dumps(conditions, indent=2)}

STRATEGY SCORES:
{json.dumps(strategy_scores, indent=2)}

TOP STRATEGY: {self.strategies[best_strategy]['name']} (Score: {best_score})

Explain why this strategy is optimal right now and what specific setup to look for.
If no good setup exists, recommend waiting."""

        recommendation = self.think(prompt, market_data)
        
        self.state.insights_generated += 1
        self.state.status = "idle"
        
        return {
            "agent": self.name,
            "type": "strategy_recommendation",
            "market_conditions": conditions,
            "recommended_strategy": best_strategy,
            "strategy_name": self.strategies[best_strategy]['name'],
            "score": best_score,
            "all_scores": strategy_scores,
            "reasoning": recommendation,
            "action": "TRADE" if best_score >= 60 else "WAIT"
        }
    
    def generate_trade_setup(self, strategy_id: str, market_data: Dict) -> Dict:
        """Generate specific trade setup for a strategy"""
        self.state.status = "generating"
        self.state.current_task = f"Setup for {strategy_id}"
        
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return {"error": f"Unknown strategy: {strategy_id}"}
        
        price = market_data.get('price', {}).get('close', 0)
        atr = market_data.get('atr', {}).get('value', 20)
        trend = market_data.get('ema', {}).get('trend', 'NEUTRAL')
        
        # Determine direction based on conditions
        direction = self._determine_direction(strategy_id, market_data)
        
        # Calculate levels
        if direction == "LONG":
            entry = price
            stop_loss = price - (atr * 1.5)
            target_1 = price + (atr * 2)
            target_2 = price + (atr * 3)
        else:  # SHORT
            entry = price
            stop_loss = price + (atr * 1.5)
            target_1 = price - (atr * 2)
            target_2 = price - (atr * 3)
        
        risk = abs(entry - stop_loss)
        reward = abs(target_1 - entry)
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        
        # Get LLM to refine the setup
        prompt = f"""Refine this {strategy['name']} trade setup for Nifty 50:

Direction: {direction}
Entry: {entry:.2f}
Stop Loss: {stop_loss:.2f}
Target 1: {target_1:.2f}
Target 2: {target_2:.2f}
R:R: {rr_ratio}

Current conditions:
- Price: {price}
- ATR: {atr}
- Trend: {trend}
- RSI: {market_data.get('rsi', {}).get('value', 50)}
- VWAP: {market_data.get('vwap', {}).get('value', price)}

Provide:
1. Confidence level (0-100)
2. Key reasons for this trade
3. What would invalidate the setup
4. Optimal entry timing"""

        refinement = self.think(prompt, market_data)
        
        setup = {
            "agent": self.name,
            "type": "trade_setup",
            "strategy": strategy_id,
            "strategy_name": strategy['name'],
            "direction": direction,
            "entry": round(entry, 2),
            "stop_loss": round(stop_loss, 2),
            "target_1": round(target_1, 2),
            "target_2": round(target_2, 2),
            "risk_points": round(risk, 2),
            "reward_points": round(reward, 2),
            "risk_reward": rr_ratio,
            "confidence": self._estimate_confidence(market_data, strategy_id),
            "duration": strategy['typical_duration'],
            "refinement": refinement,
            "timestamp": self.state.last_activity
        }
        
        self.active_setups.append(setup)
        self.state.insights_generated += 1
        self.state.status = "idle"
        
        return setup
    
    def evaluate_all_strategies(self, market_data: Dict) -> Dict:
        """Evaluate all strategies and rank them"""
        self.state.status = "evaluating"
        
        conditions = self._analyze_conditions(market_data)
        evaluations = []
        
        for strategy_id, strategy in self.strategies.items():
            score = self._score_strategy(strategy, conditions)
            evaluation = {
                "strategy_id": strategy_id,
                "name": strategy['name'],
                "score": score,
                "suitable": score >= 50,
                "conditions_met": self._check_conditions(strategy, conditions),
                "typical_duration": strategy['typical_duration']
            }
            evaluations.append(evaluation)
        
        # Sort by score
        evaluations.sort(key=lambda x: x['score'], reverse=True)
        
        self.state.status = "idle"
        
        return {
            "agent": self.name,
            "type": "strategy_evaluation",
            "market_conditions": conditions,
            "evaluations": evaluations,
            "top_3": evaluations[:3],
            "tradeable": any(e['suitable'] for e in evaluations)
        }
    
    def validate_setup(self, setup: Dict, market_data: Dict) -> Dict:
        """Validate a trade setup against current conditions"""
        issues = []
        
        # Check R:R ratio
        if setup.get('risk_reward', 0) < 1.5:
            issues.append("Risk-reward below minimum (1.5:1)")
        
        # Check direction vs trend
        trend = market_data.get('ema', {}).get('trend', 'NEUTRAL')
        if setup.get('direction') == 'LONG' and trend == 'BEARISH':
            issues.append("Long setup against bearish EMA trend")
        elif setup.get('direction') == 'SHORT' and trend == 'BULLISH':
            issues.append("Short setup against bullish EMA trend")
        
        # Check stop loss distance
        atr = market_data.get('atr', {}).get('value', 20)
        sl_distance = abs(setup.get('entry', 0) - setup.get('stop_loss', 0))
        if sl_distance > atr * 2.5:
            issues.append(f"Stop loss too wide ({sl_distance:.0f} points, ATR is {atr:.0f})")
        
        return {
            "agent": self.name,
            "type": "validation",
            "setup": setup.get('strategy_name'),
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendation": "PROCEED" if len(issues) == 0 else "REVISE" if len(issues) == 1 else "REJECT"
        }
    
    def _analyze_conditions(self, market_data: Dict) -> Dict:
        """Analyze current market conditions"""
        trend = market_data.get('ema', {}).get('trend', 'NEUTRAL')
        rsi = market_data.get('rsi', {}).get('value', 50)
        squeeze = market_data.get('bollinger', {}).get('squeeze', False)
        volume_spike = market_data.get('volume', {}).get('spike', False)
        atr_pct = market_data.get('atr', {}).get('pct', 0.1)
        
        # Determine market regime
        if trend in ['BULLISH', 'BEARISH'] and abs(rsi - 50) > 10:
            regime = "trending"
        elif squeeze:
            regime = "consolidation"
        elif atr_pct > 0.3:
            regime = "high_volatility"
        else:
            regime = "ranging"
        
        return {
            "regime": regime,
            "trend": trend,
            "trend_strength": "strong" if abs(rsi - 50) > 20 else "moderate" if abs(rsi - 50) > 10 else "weak",
            "volatility": "high" if atr_pct > 0.3 else "low" if atr_pct < 0.1 else "normal",
            "squeeze_active": squeeze,
            "volume_elevated": volume_spike,
            "rsi_extreme": rsi < 30 or rsi > 70,
            "tradeable": regime in ["trending", "high_volatility"] or squeeze
        }
    
    def _score_strategy(self, strategy: Dict, conditions: Dict) -> int:
        """Score a strategy based on current conditions"""
        score = 50  # Base score
        
        # Market regime match
        if conditions['regime'] in strategy['market_condition']:
            score += 30
        
        # Trend alignment
        if conditions['trend_strength'] == 'strong':
            if strategy['name'] in ['Pullback Entry', 'VWAP Bounce', 'Momentum Breakout']:
                score += 15
        
        # Volatility match
        if conditions['volatility'] == 'high' and 'breakout' in strategy['name'].lower():
            score += 10
        
        # Squeeze condition
        if conditions['squeeze_active'] and 'squeeze' in strategy['name'].lower():
            score += 20
        
        # Volume confirmation
        if conditions['volume_elevated']:
            score += 5
        
        # RSI extreme for reversal strategies
        if conditions['rsi_extreme'] and 'reversal' in strategy['name'].lower():
            score += 15
        
        return min(100, score)
    
    def _check_conditions(self, strategy: Dict, conditions: Dict) -> List[str]:
        """Check which conditions are met for a strategy"""
        met = []
        if conditions['regime'] in strategy['market_condition']:
            met.append(f"Market regime ({conditions['regime']})")
        if conditions['tradeable']:
            met.append("Market is tradeable")
        if conditions['volume_elevated']:
            met.append("Volume confirmation")
        return met
    
    def _determine_direction(self, strategy_id: str, market_data: Dict) -> str:
        """Determine trade direction based on strategy and conditions"""
        trend = market_data.get('ema', {}).get('trend', 'NEUTRAL')
        rsi = market_data.get('rsi', {}).get('value', 50)
        vwap_pos = market_data.get('vwap', {}).get('position', 'AT')
        
        # Reversal strategies go against trend at extremes
        if strategy_id in ['rsi_divergence', 'scalp_reversal']:
            if rsi > 70:
                return "SHORT"
            elif rsi < 30:
                return "LONG"
        
        # Trend-following strategies
        if trend == 'BULLISH':
            return "LONG"
        elif trend == 'BEARISH':
            return "SHORT"
        
        # Default to VWAP position
        return "LONG" if vwap_pos == "BELOW" else "SHORT"
    
    def _estimate_confidence(self, market_data: Dict, strategy_id: str) -> int:
        """Estimate confidence level for a setup"""
        base = 50
        
        # Add for trend alignment
        trend = market_data.get('ema', {}).get('trend')
        macd = market_data.get('macd', {}).get('trend')
        if trend == macd:
            base += 15
        
        # Add for volume
        if market_data.get('volume', {}).get('spike'):
            base += 10
        
        # Add for VWAP position
        if market_data.get('vwap', {}).get('position') in ['ABOVE', 'BELOW']:
            base += 5
        
        # Strategy-specific adjustments
        if strategy_id == 'momentum_breakout' and market_data.get('volume', {}).get('spike'):
            base += 15
        
        return min(95, base)
    
    def _handle_strategy_proposal(self, message: AgentMessage) -> Dict:
        """Handle strategy proposals from coordinator"""
        return self.recommend_strategy(message.content)
