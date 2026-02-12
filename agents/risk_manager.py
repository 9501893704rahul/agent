"""
Risk Manager Agent - Nifty 50 5-Minute Trading Risk Control

Responsible for:
- Position sizing calculations
- Risk assessment for trade setups
- Portfolio exposure monitoring
- Stop-loss and risk parameter validation
- Real-time risk alerts
"""

from typing import Dict, List, Optional
from .base_agent import BaseAgent, AgentRole, AgentMessage, MessageType
import json


class RiskManagerAgent(BaseAgent):
    """
    Chief Risk Officer for the Nifty 50 scalping desk.
    
    Risk Framework:
    - Max risk per trade: 1% of capital
    - Max daily loss: 3% of capital
    - Max concurrent positions: 2
    - Correlation monitoring
    - Volatility-adjusted sizing
    """
    
    def __init__(self, capital: float = 1000000):  # Default 10L capital
        super().__init__(
            role=AgentRole.RISK_MANAGER,
            name="Vikram - Risk Manager"
        )
        self.capital = capital
        self.max_risk_per_trade = 0.01  # 1%
        self.max_daily_loss = 0.03  # 3%
        self.max_positions = 2
        self.daily_pnl = 0
        self.open_positions = []
        self.risk_events = []
        
    @property
    def system_prompt(self) -> str:
        return """You are Vikram, the Chief Risk Officer on a Nifty 50 scalping desk with 20 years of risk management experience.

YOUR ROLE:
- Protect capital at all costs - preservation is priority #1
- Validate all trade setups for proper risk management
- Calculate position sizes based on volatility and risk tolerance
- Monitor overall portfolio exposure
- Alert the team to dangerous market conditions

RISK PARAMETERS:
- Maximum risk per trade: 1% of capital
- Maximum daily loss limit: 3% of capital
- Maximum concurrent positions: 2
- Minimum risk-reward ratio: 1.5:1
- Stop-loss must be defined for every trade

POSITION SIZING FORMULA:
Position Size = (Capital × Risk%) / (Entry - Stop Loss)
Example: ₹10,00,000 × 1% = ₹10,000 risk
If stop is 25 points away: 10,000 / 25 = 400 units (8 lots of Nifty)

RISK ASSESSMENT CRITERIA:
1. SETUP RISK: Is the stop-loss logical? Is R:R acceptable?
2. MARKET RISK: Current volatility, time of day, news events
3. PORTFOLIO RISK: Existing exposure, correlation, daily P&L
4. EXECUTION RISK: Liquidity, slippage expectations

RISK RATINGS:
- GREEN: Proceed with full position size
- YELLOW: Proceed with 50-75% position size
- ORANGE: Proceed with 25-50% position size or wait
- RED: Do not trade / Exit existing positions

Always be conservative. It's better to miss a trade than take excessive risk.
Your job is to say NO when necessary - protect the desk."""

    def process_request(self, request: Dict, context: Dict = None) -> Dict:
        """Process risk management request"""
        request_type = request.get("type", "assess_trade")
        
        if request_type == "assess_trade":
            return self.assess_trade_risk(request.get("setup"), context)
        elif request_type == "position_size":
            return self.calculate_position_size(request.get("setup"), context)
        elif request_type == "market_risk":
            return self.assess_market_risk(context)
        elif request_type == "portfolio_status":
            return self.get_portfolio_status()
        elif request_type == "validate_stop":
            return self.validate_stop_loss(request.get("setup"), context)
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    def assess_trade_risk(self, setup: Dict, market_data: Dict) -> Dict:
        """Comprehensive risk assessment for a trade setup"""
        self.state.status = "assessing"
        self.state.current_task = "Trade Risk Assessment"
        
        risk_factors = []
        risk_score = 0  # Lower is better, 0-100
        
        # 1. Setup Risk Assessment
        setup_risk = self._assess_setup_risk(setup)
        risk_factors.extend(setup_risk['factors'])
        risk_score += setup_risk['score']
        
        # 2. Market Risk Assessment
        market_risk = self._assess_market_risk_internal(market_data)
        risk_factors.extend(market_risk['factors'])
        risk_score += market_risk['score']
        
        # 3. Portfolio Risk Assessment
        portfolio_risk = self._assess_portfolio_risk(setup)
        risk_factors.extend(portfolio_risk['factors'])
        risk_score += portfolio_risk['score']
        
        # Determine risk rating
        if risk_score <= 20:
            rating = "GREEN"
            position_multiplier = 1.0
            recommendation = "APPROVED - Full position size"
        elif risk_score <= 40:
            rating = "YELLOW"
            position_multiplier = 0.75
            recommendation = "APPROVED WITH CAUTION - Reduce to 75% size"
        elif risk_score <= 60:
            rating = "ORANGE"
            position_multiplier = 0.5
            recommendation = "HIGH RISK - Reduce to 50% or wait"
        else:
            rating = "RED"
            position_multiplier = 0
            recommendation = "REJECTED - Do not trade"
        
        # Calculate position size
        position_info = self._calculate_position(setup, position_multiplier)
        
        # Get LLM assessment
        prompt = f"""Risk assessment for {setup.get('strategy_name', 'Unknown')} trade:

SETUP:
- Direction: {setup.get('direction')}
- Entry: {setup.get('entry')}
- Stop Loss: {setup.get('stop_loss')}
- Target: {setup.get('target_1')}
- R:R: {setup.get('risk_reward')}

RISK FACTORS IDENTIFIED:
{chr(10).join(f'- {f}' for f in risk_factors)}

RISK SCORE: {risk_score}/100
RATING: {rating}

Provide a brief risk summary (2-3 sentences) and any additional concerns or recommendations."""

        llm_assessment = self.think(prompt, market_data)
        
        assessment = {
            "agent": self.name,
            "type": "risk_assessment",
            "setup": setup.get('strategy_name'),
            "risk_score": risk_score,
            "rating": rating,
            "rating_color": rating.lower(),
            "risk_factors": risk_factors,
            "recommendation": recommendation,
            "position_multiplier": position_multiplier,
            "position_info": position_info,
            "llm_assessment": llm_assessment,
            "approved": rating in ["GREEN", "YELLOW"],
            "timestamp": self.state.last_activity
        }
        
        self.state.insights_generated += 1
        self.state.status = "idle"
        
        if rating == "RED":
            self.risk_events.append({
                "type": "trade_rejected",
                "reason": risk_factors,
                "timestamp": self.state.last_activity
            })
        
        return assessment
    
    def calculate_position_size(self, setup: Dict, market_data: Dict = None) -> Dict:
        """Calculate optimal position size"""
        entry = setup.get('entry', 0)
        stop_loss = setup.get('stop_loss', 0)
        
        if not entry or not stop_loss:
            return {"error": "Missing entry or stop loss"}
        
        # Risk per trade
        risk_amount = self.capital * self.max_risk_per_trade
        
        # Points at risk
        points_risk = abs(entry - stop_loss)
        
        if points_risk == 0:
            return {"error": "Stop loss same as entry"}
        
        # Calculate units (Nifty lot size = 50)
        lot_size = 50
        units = risk_amount / points_risk
        lots = int(units / lot_size)
        
        # Adjust for volatility if market data available
        if market_data:
            atr = market_data.get('atr', {}).get('value', 20)
            atr_pct = market_data.get('atr', {}).get('pct', 0.1)
            
            # Reduce size in high volatility
            if atr_pct > 0.3:
                lots = max(1, int(lots * 0.5))
            elif atr_pct > 0.2:
                lots = max(1, int(lots * 0.75))
        
        position_value = lots * lot_size * entry
        max_loss = lots * lot_size * points_risk
        
        return {
            "agent": self.name,
            "type": "position_size",
            "capital": self.capital,
            "risk_per_trade": f"{self.max_risk_per_trade * 100}%",
            "risk_amount": round(risk_amount, 2),
            "entry": entry,
            "stop_loss": stop_loss,
            "points_risk": round(points_risk, 2),
            "recommended_lots": lots,
            "units": lots * lot_size,
            "position_value": round(position_value, 2),
            "max_loss": round(max_loss, 2),
            "lot_size": lot_size
        }
    
    def assess_market_risk(self, market_data: Dict) -> Dict:
        """Assess overall market risk conditions"""
        self.state.status = "assessing"
        
        risk_level = "NORMAL"
        alerts = []
        
        # Volatility check
        atr_pct = market_data.get('atr', {}).get('pct', 0.1)
        if atr_pct > 0.4:
            risk_level = "EXTREME"
            alerts.append("⚠️ EXTREME volatility - reduce all positions")
        elif atr_pct > 0.3:
            risk_level = "HIGH"
            alerts.append("⚡ High volatility - use reduced size")
        
        # Squeeze detection
        if market_data.get('bollinger', {}).get('squeeze', False):
            alerts.append("📊 BB Squeeze active - expect volatility expansion")
        
        # RSI extremes
        rsi = market_data.get('rsi', {}).get('value', 50)
        if rsi > 80:
            alerts.append("🔴 RSI extremely overbought - reversal risk")
            if risk_level == "NORMAL":
                risk_level = "ELEVATED"
        elif rsi < 20:
            alerts.append("🟢 RSI extremely oversold - bounce possible")
            if risk_level == "NORMAL":
                risk_level = "ELEVATED"
        
        # Volume analysis
        if market_data.get('volume', {}).get('spike', False):
            alerts.append("📈 Volume spike detected - institutional activity")
        
        # Time-based risk (market hours)
        # Opening hour and closing hour are typically more volatile
        
        position_recommendation = {
            "NORMAL": 1.0,
            "ELEVATED": 0.75,
            "HIGH": 0.5,
            "EXTREME": 0.25
        }
        
        self.state.status = "idle"
        
        return {
            "agent": self.name,
            "type": "market_risk",
            "risk_level": risk_level,
            "position_size_multiplier": position_recommendation[risk_level],
            "alerts": alerts,
            "metrics": {
                "volatility": atr_pct,
                "rsi": rsi,
                "squeeze": market_data.get('bollinger', {}).get('squeeze', False),
                "volume_spike": market_data.get('volume', {}).get('spike', False)
            },
            "recommendation": f"Use {int(position_recommendation[risk_level] * 100)}% of normal position size"
        }
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio risk status"""
        return {
            "agent": self.name,
            "type": "portfolio_status",
            "capital": self.capital,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": round(self.daily_pnl / self.capital * 100, 2),
            "open_positions": len(self.open_positions),
            "max_positions": self.max_positions,
            "daily_loss_limit": self.capital * self.max_daily_loss,
            "remaining_risk_budget": self.capital * self.max_daily_loss - abs(min(0, self.daily_pnl)),
            "can_trade": len(self.open_positions) < self.max_positions and self.daily_pnl > -self.capital * self.max_daily_loss,
            "risk_events_today": len(self.risk_events)
        }
    
    def validate_stop_loss(self, setup: Dict, market_data: Dict) -> Dict:
        """Validate stop-loss placement"""
        entry = setup.get('entry', 0)
        stop_loss = setup.get('stop_loss', 0)
        direction = setup.get('direction', 'LONG')
        atr = market_data.get('atr', {}).get('value', 20)
        
        issues = []
        
        # Check stop is on correct side
        if direction == 'LONG' and stop_loss >= entry:
            issues.append("Stop loss must be below entry for long trades")
        elif direction == 'SHORT' and stop_loss <= entry:
            issues.append("Stop loss must be above entry for short trades")
        
        # Check stop distance
        stop_distance = abs(entry - stop_loss)
        
        if stop_distance < atr * 0.5:
            issues.append(f"Stop too tight ({stop_distance:.0f} pts) - likely to get stopped out. Min: {atr * 0.5:.0f} pts")
        elif stop_distance > atr * 3:
            issues.append(f"Stop too wide ({stop_distance:.0f} pts) - excessive risk. Max: {atr * 3:.0f} pts")
        
        # Check against key levels
        vwap = market_data.get('vwap', {}).get('value', 0)
        if vwap:
            if direction == 'LONG' and stop_loss > vwap and entry > vwap:
                issues.append("Consider placing stop below VWAP for better protection")
        
        return {
            "agent": self.name,
            "type": "stop_validation",
            "valid": len(issues) == 0,
            "stop_distance": round(stop_distance, 2),
            "atr": round(atr, 2),
            "stop_atr_multiple": round(stop_distance / atr, 2),
            "issues": issues,
            "recommendation": "VALID" if len(issues) == 0 else "ADJUST"
        }
    
    def _assess_setup_risk(self, setup: Dict) -> Dict:
        """Assess risk of the trade setup itself"""
        factors = []
        score = 0
        
        # R:R ratio check
        rr = setup.get('risk_reward', 0)
        if rr < 1.5:
            factors.append(f"Low R:R ratio ({rr}) - below minimum 1.5")
            score += 25
        elif rr < 2.0:
            factors.append(f"Moderate R:R ratio ({rr})")
            score += 10
        
        # Confidence check
        confidence = setup.get('confidence', 50)
        if confidence < 50:
            factors.append(f"Low confidence ({confidence}%)")
            score += 20
        elif confidence < 70:
            factors.append(f"Moderate confidence ({confidence}%)")
            score += 5
        
        # Stop loss distance
        entry = setup.get('entry', 0)
        sl = setup.get('stop_loss', 0)
        if entry and sl:
            sl_pct = abs(entry - sl) / entry * 100
            if sl_pct > 0.5:
                factors.append(f"Wide stop loss ({sl_pct:.2f}%)")
                score += 15
        
        return {"factors": factors, "score": score}
    
    def _assess_market_risk_internal(self, market_data: Dict) -> Dict:
        """Internal market risk assessment"""
        factors = []
        score = 0
        
        # Volatility
        atr_pct = market_data.get('atr', {}).get('pct', 0.1) if market_data else 0.1
        if atr_pct > 0.3:
            factors.append(f"High market volatility (ATR {atr_pct:.2%})")
            score += 20
        
        # RSI extremes
        rsi = market_data.get('rsi', {}).get('value', 50) if market_data else 50
        if rsi > 75 or rsi < 25:
            factors.append(f"RSI at extreme ({rsi:.0f})")
            score += 10
        
        # Squeeze condition
        if market_data and market_data.get('bollinger', {}).get('squeeze', False):
            factors.append("BB squeeze - volatility expansion expected")
            score += 10
        
        return {"factors": factors, "score": score}
    
    def _assess_portfolio_risk(self, setup: Dict) -> Dict:
        """Assess portfolio-level risk"""
        factors = []
        score = 0
        
        # Check position limit
        if len(self.open_positions) >= self.max_positions:
            factors.append(f"Position limit reached ({self.max_positions})")
            score += 30
        elif len(self.open_positions) == self.max_positions - 1:
            factors.append("Approaching position limit")
            score += 10
        
        # Check daily loss
        daily_loss_pct = abs(min(0, self.daily_pnl)) / self.capital * 100
        if daily_loss_pct > 2:
            factors.append(f"Daily loss at {daily_loss_pct:.1f}% - approaching limit")
            score += 25
        elif daily_loss_pct > 1:
            factors.append(f"Daily loss at {daily_loss_pct:.1f}%")
            score += 10
        
        # Check for correlation with existing positions
        if self.open_positions:
            new_direction = setup.get('direction', 'LONG')
            same_direction = sum(1 for p in self.open_positions if p.get('direction') == new_direction)
            if same_direction > 0:
                factors.append(f"Correlated with {same_direction} existing position(s)")
                score += 15
        
        return {"factors": factors, "score": score}
    
    def _calculate_position(self, setup: Dict, multiplier: float) -> Dict:
        """Calculate position with risk adjustment"""
        if multiplier == 0:
            return {"lots": 0, "reason": "Trade rejected - no position"}
        
        entry = setup.get('entry', 0)
        stop_loss = setup.get('stop_loss', 0)
        
        if not entry or not stop_loss:
            return {"error": "Missing entry or stop loss"}
        
        risk_amount = self.capital * self.max_risk_per_trade * multiplier
        points_risk = abs(entry - stop_loss)
        
        lot_size = 50
        units = risk_amount / points_risk if points_risk > 0 else 0
        lots = max(1, int(units / lot_size))
        
        return {
            "lots": lots,
            "units": lots * lot_size,
            "risk_amount": round(risk_amount, 2),
            "multiplier_applied": multiplier
        }
    
    def _handle_risk_assessment(self, message: AgentMessage) -> Dict:
        """Handle risk assessment requests"""
        setup = message.content.get('setup', {})
        market_data = message.content.get('market_data', {})
        return self.assess_trade_risk(setup, market_data)
    
    def _handle_approval_request(self, message: AgentMessage) -> Dict:
        """Handle trade approval requests"""
        setup = message.content.get('setup', {})
        market_data = message.content.get('market_data', {})
        assessment = self.assess_trade_risk(setup, market_data)
        
        return {
            "approved": assessment['approved'],
            "rating": assessment['rating'],
            "assessment": assessment
        }
