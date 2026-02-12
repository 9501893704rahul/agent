"""
Head Trader Agent - Nifty 50 5-Minute Trading Desk Leader

Responsible for:
- Final trade approval/rejection decisions
- Team coordination and conflict resolution
- Strategy prioritization
- Overall desk performance
"""

from typing import Dict, List, Optional
from .base_agent import BaseAgent, AgentRole, AgentMessage, MessageType
from datetime import datetime
import json


class HeadTraderAgent(BaseAgent):
    """
    Head Trader / Supervisor for the Nifty 50 scalping desk.
    
    Makes final decisions on:
    - Trade approvals after team analysis
    - Strategy selection in conflicting scenarios
    - Risk overrides in special situations
    - Team coordination
    """
    
    def __init__(self):
        super().__init__(
            role=AgentRole.HEAD_TRADER,
            name="Anand - Head Trader"
        )
        self.decisions = []
        self.team_performance = {}
        self.session_stats = {
            'trades_approved': 0,
            'trades_rejected': 0,
            'total_analyzed': 0
        }
        
    @property
    def system_prompt(self) -> str:
        return """You are Anand, the Head Trader running a Nifty 50 scalping desk with 25 years of trading experience.

YOUR ROLE:
- Make final GO/NO-GO decisions on trade recommendations
- Synthesize inputs from Market Analyst, Strategist, Risk Manager, and Execution
- Override team decisions when your experience dictates
- Maintain discipline and consistency in execution

YOUR EXPERIENCE:
- Traded through multiple market cycles (2008 crash, 2020 COVID, etc.)
- Deep understanding of Nifty 50 behavior and nuances
- Know when to be aggressive and when to be defensive
- Understand that preservation of capital enables future opportunities

DECISION FRAMEWORK:
1. MARKET CONTEXT: Is the overall environment favorable?
2. SETUP QUALITY: Does the trade have edge?
3. RISK ALIGNMENT: Is risk properly sized and managed?
4. EXECUTION FEASIBILITY: Can we enter/exit cleanly?
5. GUT CHECK: Does this feel right based on experience?

APPROVAL CRITERIA:
- Analyst confirms favorable technical setup
- Strategist provides clear entry/exit plan
- Risk Manager approves position size
- Execution conditions are favorable
- No conflicting signals from team members

REJECTION REASONS:
- Team disagreement on direction
- Risk parameters exceeded
- Poor market conditions
- Insufficient edge (low confidence)
- Gut feeling says no

COMMUNICATION:
When approving: Be clear on exactly what to do
When rejecting: Explain why and what would change your mind
Always: Log your decision rationale for review

Remember: You're responsible for the P&L. Be decisive but not reckless."""

    def process_request(self, request: Dict, context: Dict = None) -> Dict:
        """Process request from coordinator"""
        request_type = request.get("type", "final_decision")
        
        if request_type == "final_decision":
            return self.make_final_decision(request.get("team_inputs"), context)
        elif request_type == "evaluate_setup":
            return self.evaluate_setup(request.get("setup"), context)
        elif request_type == "morning_brief":
            return self.generate_morning_brief(context)
        elif request_type == "session_review":
            return self.generate_session_review()
        elif request_type == "override":
            return self.handle_override(request.get("override_request"), context)
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    def make_final_decision(self, team_inputs: Dict, market_data: Dict) -> Dict:
        """Make final GO/NO-GO decision based on team inputs"""
        self.state.status = "deciding"
        self.state.current_task = "Final Trade Decision"
        self.session_stats['total_analyzed'] += 1
        
        # Extract team inputs
        analyst_input = team_inputs.get('analyst', {})
        strategy_input = team_inputs.get('strategist', {})
        risk_input = team_inputs.get('risk_manager', {})
        execution_input = team_inputs.get('execution', {})
        
        # Build decision factors
        factors = {
            'analyst_confidence': analyst_input.get('confidence', 'LOW'),
            'strategy_score': strategy_input.get('score', 0),
            'risk_approved': risk_input.get('approved', False),
            'risk_rating': risk_input.get('rating', 'RED'),
            'entry_conditions_met': execution_input.get('conditions_met', False),
            'team_aligned': self._check_team_alignment(team_inputs)
        }
        
        # Calculate approval score
        approval_score = self._calculate_approval_score(factors, team_inputs)
        
        # Make decision
        if approval_score >= 75 and factors['risk_approved']:
            decision = "APPROVED"
            action = "EXECUTE"
            self.session_stats['trades_approved'] += 1
        elif approval_score >= 60 and factors['risk_rating'] in ['GREEN', 'YELLOW']:
            decision = "APPROVED_WITH_CAUTION"
            action = "EXECUTE_REDUCED"
            self.session_stats['trades_approved'] += 1
        elif approval_score >= 40:
            decision = "WAIT"
            action = "MONITOR"
        else:
            decision = "REJECTED"
            action = "NO_TRADE"
            self.session_stats['trades_rejected'] += 1
        
        # Generate rationale using LLM
        prompt = f"""As Head Trader, provide your final decision rationale.

TEAM INPUTS SUMMARY:
- Analyst: {analyst_input.get('raw_analysis', 'N/A')[:200]}...
- Strategy: {strategy_input.get('strategy_name', 'N/A')} (Score: {strategy_input.get('score', 0)})
- Risk: {risk_input.get('rating', 'N/A')} - {risk_input.get('recommendation', 'N/A')}
- Execution: Conditions {'MET' if execution_input.get('conditions_met') else 'NOT MET'}

APPROVAL SCORE: {approval_score}/100
DECISION: {decision}

Provide:
1. Your rationale (2-3 sentences)
2. Key factor that drove the decision
3. What the team should do next"""

        rationale = self.think(prompt, market_data)
        
        # Build final decision
        final_decision = {
            "agent": self.name,
            "type": "final_decision",
            "decision": decision,
            "action": action,
            "approval_score": approval_score,
            "factors": factors,
            "rationale": rationale,
            "setup": {
                "strategy": strategy_input.get('strategy_name'),
                "direction": strategy_input.get('direction'),
                "entry": execution_input.get('entry_price'),
                "stop_loss": execution_input.get('stop_loss'),
                "target": execution_input.get('targets', {}).get('target_1')
            },
            "position_size": risk_input.get('position_info', {}),
            "execution_notes": execution_input.get('execution_notes', ''),
            "timestamp": datetime.now().isoformat()
        }
        
        # Log decision
        self.decisions.append(final_decision)
        
        self.state.insights_generated += 1
        self.state.status = "idle"
        
        return final_decision
    
    def evaluate_setup(self, setup: Dict, market_data: Dict) -> Dict:
        """Quick evaluation of a setup without full team input"""
        self.state.status = "evaluating"
        
        # Quick assessment
        confidence = setup.get('confidence', 50)
        rr_ratio = setup.get('risk_reward', 0)
        direction = setup.get('direction', 'NEUTRAL')
        
        # Check against market conditions
        trend = market_data.get('ema', {}).get('trend', 'NEUTRAL')
        
        issues = []
        
        if rr_ratio < 1.5:
            issues.append("R:R below minimum threshold")
        
        if confidence < 60:
            issues.append("Low confidence setup")
        
        if direction == 'LONG' and trend == 'BEARISH':
            issues.append("Long trade against bearish trend")
        elif direction == 'SHORT' and trend == 'BULLISH':
            issues.append("Short trade against bullish trend")
        
        quick_decision = "PROCEED" if len(issues) == 0 else "REVIEW" if len(issues) == 1 else "SKIP"
        
        self.state.status = "idle"
        
        return {
            "agent": self.name,
            "type": "quick_evaluation",
            "setup": setup.get('strategy_name'),
            "quick_decision": quick_decision,
            "issues": issues,
            "recommendation": f"{'Run full team analysis' if quick_decision == 'REVIEW' else 'Proceed to team analysis' if quick_decision == 'PROCEED' else 'Skip this setup'}"
        }
    
    def generate_morning_brief(self, market_data: Dict) -> Dict:
        """Generate morning trading brief"""
        self.state.status = "briefing"
        
        prompt = f"""Generate a morning brief for the Nifty 50 scalping desk.

MARKET DATA:
- Nifty 50: {market_data.get('price', {}).get('close', 'N/A')}
- Change: {market_data.get('price', {}).get('change_pct', 0):.2f}%
- Trend: {market_data.get('ema', {}).get('trend', 'N/A')}
- RSI: {market_data.get('rsi', {}).get('value', 'N/A')}
- Volatility: ATR {market_data.get('atr', {}).get('value', 'N/A')} ({market_data.get('atr', {}).get('pct', 0):.2%})

Provide:
1. Overall market assessment (1-2 sentences)
2. Key levels to watch today
3. Preferred trading bias (bullish/bearish/neutral)
4. Risk guidance for the session
5. What to look for in setups"""

        brief = self.think(prompt, market_data)
        
        self.state.status = "idle"
        
        return {
            "agent": self.name,
            "type": "morning_brief",
            "brief": brief,
            "market_snapshot": {
                "price": market_data.get('price', {}).get('close'),
                "trend": market_data.get('ema', {}).get('trend'),
                "volatility": market_data.get('atr', {}).get('pct')
            },
            "session_bias": self._determine_session_bias(market_data),
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_session_review(self) -> Dict:
        """Generate end-of-session review"""
        approved = self.session_stats['trades_approved']
        rejected = self.session_stats['trades_rejected']
        total = self.session_stats['total_analyzed']
        
        approval_rate = (approved / total * 100) if total > 0 else 0
        
        return {
            "agent": self.name,
            "type": "session_review",
            "stats": {
                "total_analyzed": total,
                "approved": approved,
                "rejected": rejected,
                "approval_rate": f"{approval_rate:.1f}%"
            },
            "decisions": self.decisions[-10:],  # Last 10 decisions
            "summary": f"Analyzed {total} setups, approved {approved} ({approval_rate:.1f}%)"
        }
    
    def handle_override(self, override_request: Dict, market_data: Dict) -> Dict:
        """Handle override requests"""
        override_type = override_request.get('type')
        reason = override_request.get('reason', '')
        
        # Log override
        override_decision = {
            "agent": self.name,
            "type": "override",
            "override_type": override_type,
            "reason": reason,
            "market_conditions": {
                "price": market_data.get('price', {}).get('close'),
                "trend": market_data.get('ema', {}).get('trend')
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Head trader can override risk manager in specific situations
        if override_type == "risk_override":
            if "volatility_spike" in reason.lower() or "news_event" in reason.lower():
                override_decision['approved'] = True
                override_decision['action'] = "Reduce position to 25% max"
            else:
                override_decision['approved'] = False
                override_decision['action'] = "Override rejected - follow risk manager"
        
        return override_decision
    
    def _check_team_alignment(self, team_inputs: Dict) -> bool:
        """Check if team is aligned on direction"""
        directions = set()
        
        # Get direction from strategy
        if team_inputs.get('strategist', {}).get('direction'):
            directions.add(team_inputs['strategist']['direction'])
        
        # Get trend from analyst
        trend = team_inputs.get('analyst', {}).get('structured', {}).get('trend')
        if trend == 'BULLISH':
            directions.add('LONG')
        elif trend == 'BEARISH':
            directions.add('SHORT')
        
        # Aligned if only one direction
        return len(directions) <= 1
    
    def _calculate_approval_score(self, factors: Dict, team_inputs: Dict) -> int:
        """Calculate approval score from 0-100"""
        score = 50  # Base score
        
        # Analyst confidence
        if factors['analyst_confidence'] == 'HIGH':
            score += 15
        elif factors['analyst_confidence'] == 'MEDIUM':
            score += 5
        
        # Strategy score
        strategy_score = factors['strategy_score']
        if strategy_score >= 80:
            score += 20
        elif strategy_score >= 60:
            score += 10
        elif strategy_score < 40:
            score -= 15
        
        # Risk approval
        if factors['risk_approved']:
            score += 15
        else:
            score -= 20
        
        # Risk rating
        if factors['risk_rating'] == 'GREEN':
            score += 10
        elif factors['risk_rating'] == 'YELLOW':
            score += 5
        elif factors['risk_rating'] == 'RED':
            score -= 25
        
        # Entry conditions
        if factors['entry_conditions_met']:
            score += 10
        else:
            score -= 10
        
        # Team alignment
        if factors['team_aligned']:
            score += 10
        else:
            score -= 15
        
        return max(0, min(100, score))
    
    def _determine_session_bias(self, market_data: Dict) -> str:
        """Determine overall session bias"""
        trend = market_data.get('ema', {}).get('trend', 'NEUTRAL')
        rsi = market_data.get('rsi', {}).get('value', 50)
        vwap_pos = market_data.get('vwap', {}).get('position', 'AT')
        
        bullish = 0
        bearish = 0
        
        if trend == 'BULLISH':
            bullish += 2
        elif trend == 'BEARISH':
            bearish += 2
        
        if rsi > 55:
            bullish += 1
        elif rsi < 45:
            bearish += 1
        
        if vwap_pos == 'ABOVE':
            bullish += 1
        elif vwap_pos == 'BELOW':
            bearish += 1
        
        if bullish > bearish + 1:
            return "BULLISH - Favor long setups"
        elif bearish > bullish + 1:
            return "BEARISH - Favor short setups"
        else:
            return "NEUTRAL - Wait for clear direction"
    
    def _handle_approval_request(self, message: AgentMessage) -> Dict:
        """Handle approval requests from coordinator"""
        return self.make_final_decision(
            message.content.get('team_inputs', {}),
            message.content.get('market_data', {})
        )
