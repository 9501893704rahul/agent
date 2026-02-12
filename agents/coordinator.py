"""
Trading Desk Coordinator - Orchestrates Multi-Agent Collaboration

Responsible for:
- Coordinating all agents in the trading desk
- Managing workflow and information flow
- Aggregating team outputs
- Running full analysis pipeline
"""

from typing import Dict, List, Optional
from datetime import datetime
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_agent import AgentRole, AgentMessage, MessageType
from .market_analyst import MarketAnalystAgent
from .strategist import StrategyAgent
from .risk_manager import RiskManagerAgent
from .execution import ExecutionAgent
from .head_trader import HeadTraderAgent


class TradingDeskCoordinator:
    """
    Orchestrates the multi-agent trading desk for Nifty 50 scalping.
    
    Workflow:
    1. Market Analyst analyzes current conditions
    2. Strategist recommends optimal strategy and generates setup
    3. Risk Manager assesses risk and calculates position size
    4. Execution Agent generates entry signal
    5. Head Trader makes final approval decision
    
    All agents collaborate to produce comprehensive trade research.
    """
    
    def __init__(self, capital: float = 1000000):
        # Initialize all agents
        self.market_analyst = MarketAnalystAgent()
        self.strategist = StrategyAgent()
        self.risk_manager = RiskManagerAgent(capital=capital)
        self.execution = ExecutionAgent()
        self.head_trader = HeadTraderAgent()
        
        # Team roster
        self.team = {
            AgentRole.MARKET_ANALYST: self.market_analyst,
            AgentRole.STRATEGIST: self.strategist,
            AgentRole.RISK_MANAGER: self.risk_manager,
            AgentRole.EXECUTION: self.execution,
            AgentRole.HEAD_TRADER: self.head_trader
        }
        
        # Communication log
        self.message_log: List[AgentMessage] = []
        self.workflow_history: List[Dict] = []
        
        print("=" * 60)
        print("🏢 NIFTY 50 SCALPING DESK - TEAM INITIALIZED")
        print("=" * 60)
        for role, agent in self.team.items():
            print(f"  ✅ {agent.name}")
        print("=" * 60)
    
    def run_full_analysis(self, market_data: Dict) -> Dict:
        """
        Run complete multi-agent analysis pipeline.
        
        This orchestrates all agents to analyze market conditions,
        generate trade setups, assess risk, and make final decisions.
        """
        workflow_id = f"WF_{datetime.now().strftime('%H%M%S')}"
        workflow_start = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"📊 STARTING FULL ANALYSIS - {workflow_id}")
        print(f"{'='*60}\n")
        
        results = {
            "workflow_id": workflow_id,
            "timestamp": workflow_start.isoformat(),
            "stages": {},
            "team_discussion": [],
            "final_decision": None
        }
        
        try:
            # Stage 1: Market Analysis
            print("📈 Stage 1: Market Analyst analyzing conditions...")
            analyst_output = self.market_analyst.generate_full_analysis(market_data)
            results['stages']['market_analysis'] = analyst_output
            results['team_discussion'].append({
                "agent": self.market_analyst.name,
                "stage": "Market Analysis",
                "summary": analyst_output.get('raw_analysis', '')[:300]
            })
            print(f"   ✅ Analysis complete - Trend: {analyst_output.get('structured', {}).get('trend', 'N/A')}")
            
            # Stage 2: Strategy Recommendation
            print("\n🎯 Stage 2: Strategist evaluating strategies...")
            strategy_output = self.strategist.recommend_strategy(market_data)
            results['stages']['strategy_recommendation'] = strategy_output
            results['team_discussion'].append({
                "agent": self.strategist.name,
                "stage": "Strategy Selection",
                "summary": f"Recommended: {strategy_output.get('strategy_name', 'N/A')} (Score: {strategy_output.get('score', 0)})"
            })
            print(f"   ✅ Strategy: {strategy_output.get('strategy_name', 'N/A')} (Score: {strategy_output.get('score', 0)})")
            
            # Stage 3: Generate Trade Setup (if strategy is viable)
            if strategy_output.get('action') == 'TRADE' and strategy_output.get('score', 0) >= 50:
                print("\n📝 Stage 3: Generating trade setup...")
                setup = self.strategist.generate_trade_setup(
                    strategy_output.get('recommended_strategy'),
                    market_data
                )
                results['stages']['trade_setup'] = setup
                results['team_discussion'].append({
                    "agent": self.strategist.name,
                    "stage": "Trade Setup",
                    "summary": f"{setup.get('direction')} @ {setup.get('entry')} | SL: {setup.get('stop_loss')} | T1: {setup.get('target_1')}"
                })
                print(f"   ✅ Setup: {setup.get('direction')} @ {setup.get('entry')}")
                
                # Stage 4: Risk Assessment
                print("\n🛡️ Stage 4: Risk Manager assessing risk...")
                risk_output = self.risk_manager.assess_trade_risk(setup, market_data)
                results['stages']['risk_assessment'] = risk_output
                results['team_discussion'].append({
                    "agent": self.risk_manager.name,
                    "stage": "Risk Assessment",
                    "summary": f"Rating: {risk_output.get('rating')} | {risk_output.get('recommendation')}"
                })
                print(f"   ✅ Risk Rating: {risk_output.get('rating')} - {risk_output.get('recommendation')}")
                
                # Stage 5: Execution Signal
                print("\n⚡ Stage 5: Execution preparing entry signal...")
                execution_output = self.execution.generate_entry_signal(setup, market_data)
                results['stages']['execution_signal'] = execution_output
                results['team_discussion'].append({
                    "agent": self.execution.name,
                    "stage": "Execution Signal",
                    "summary": f"{execution_output.get('action')} @ {execution_output.get('entry_price')} | Type: {execution_output.get('order_type')}"
                })
                print(f"   ✅ Signal: {execution_output.get('action')} @ {execution_output.get('entry_price')}")
                
                # Stage 6: Head Trader Decision
                print("\n👔 Stage 6: Head Trader making final decision...")
                team_inputs = {
                    'analyst': analyst_output,
                    'strategist': strategy_output,
                    'risk_manager': risk_output,
                    'execution': execution_output
                }
                final_decision = self.head_trader.make_final_decision(team_inputs, market_data)
                results['final_decision'] = final_decision
                results['team_discussion'].append({
                    "agent": self.head_trader.name,
                    "stage": "Final Decision",
                    "summary": f"Decision: {final_decision.get('decision')} | Action: {final_decision.get('action')}"
                })
                print(f"   ✅ Decision: {final_decision.get('decision')} - {final_decision.get('action')}")
                
            else:
                print("\n⏳ No viable setup - Team recommends waiting")
                results['final_decision'] = {
                    "decision": "NO_TRADE",
                    "action": "WAIT",
                    "rationale": strategy_output.get('reasoning', 'No high-confidence setup available'),
                    "timestamp": datetime.now().isoformat()
                }
                results['team_discussion'].append({
                    "agent": self.head_trader.name,
                    "stage": "Decision",
                    "summary": "No trade - waiting for better setup"
                })
        
        except Exception as e:
            print(f"\n❌ Error in workflow: {e}")
            results['error'] = str(e)
        
        # Calculate workflow duration
        workflow_end = datetime.now()
        results['duration_seconds'] = (workflow_end - workflow_start).total_seconds()
        
        # Log workflow
        self.workflow_history.append(results)
        
        print(f"\n{'='*60}")
        print(f"✅ ANALYSIS COMPLETE - {results['duration_seconds']:.2f}s")
        print(f"{'='*60}\n")
        
        return results
    
    def get_quick_signal(self, market_data: Dict) -> Dict:
        """
        Quick signal generation without full team discussion.
        For time-sensitive situations.
        """
        # Quick trend assessment
        trend = self.market_analyst.assess_trend(market_data)
        
        # Quick strategy evaluation
        strategies = self.strategist.evaluate_all_strategies(market_data)
        
        if not strategies.get('tradeable'):
            return {
                "signal": "NO_TRADE",
                "reason": "Market conditions not favorable",
                "trend": trend,
                "timestamp": datetime.now().isoformat()
            }
        
        # Get top strategy
        top_strategy = strategies['top_3'][0] if strategies.get('top_3') else None
        
        if top_strategy and top_strategy.get('suitable'):
            # Generate quick setup
            setup = self.strategist.generate_trade_setup(
                top_strategy['strategy_id'],
                market_data
            )
            
            # Quick risk check
            market_risk = self.risk_manager.assess_market_risk(market_data)
            
            return {
                "signal": setup.get('direction'),
                "strategy": setup.get('strategy_name'),
                "entry": setup.get('entry'),
                "stop_loss": setup.get('stop_loss'),
                "target": setup.get('target_1'),
                "confidence": setup.get('confidence'),
                "risk_level": market_risk.get('risk_level'),
                "position_multiplier": market_risk.get('position_size_multiplier'),
                "trend": trend,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "signal": "NO_TRADE",
            "reason": "No suitable strategy found",
            "trend": trend,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_team_status(self) -> Dict:
        """Get status of all team members"""
        return {
            "team_status": {
                agent.name: agent.get_status()
                for agent in self.team.values()
            },
            "workflows_completed": len(self.workflow_history),
            "messages_exchanged": len(self.message_log)
        }
    
    def get_market_brief(self, market_data: Dict) -> Dict:
        """Get morning brief from head trader with team input"""
        # Get analyst's view
        analyst_view = self.market_analyst.generate_quick_update(market_data)
        
        # Get strategist's view
        strategy_eval = self.strategist.evaluate_all_strategies(market_data)
        
        # Get risk assessment
        market_risk = self.risk_manager.assess_market_risk(market_data)
        
        # Head trader brief
        brief = self.head_trader.generate_morning_brief(market_data)
        
        return {
            "brief": brief,
            "analyst_view": analyst_view,
            "top_strategies": strategy_eval.get('top_3', []),
            "risk_level": market_risk.get('risk_level'),
            "risk_alerts": market_risk.get('alerts', []),
            "timestamp": datetime.now().isoformat()
        }
    
    def chat_with_team(self, message: str, market_data: Dict = None) -> Dict:
        """
        Have a conversation with the trading team.
        Routes questions to appropriate agent(s).
        """
        message_lower = message.lower()
        responses = []
        
        # Route to appropriate agents based on keywords
        if any(word in message_lower for word in ['trend', 'analysis', 'technical', 'chart', 'indicator']):
            response = self.market_analyst.think(message, market_data)
            responses.append({
                "agent": self.market_analyst.name,
                "response": response
            })
        
        if any(word in message_lower for word in ['strategy', 'setup', 'trade', 'entry', 'approach']):
            response = self.strategist.think(message, market_data)
            responses.append({
                "agent": self.strategist.name,
                "response": response
            })
        
        if any(word in message_lower for word in ['risk', 'position', 'size', 'stop', 'loss', 'protect']):
            response = self.risk_manager.think(message, market_data)
            responses.append({
                "agent": self.risk_manager.name,
                "response": response
            })
        
        if any(word in message_lower for word in ['execute', 'order', 'fill', 'exit', 'timing']):
            response = self.execution.think(message, market_data)
            responses.append({
                "agent": self.execution.name,
                "response": response
            })
        
        # If no specific routing or asking for decision/opinion
        if not responses or any(word in message_lower for word in ['decision', 'opinion', 'think', 'should', 'recommend']):
            response = self.head_trader.think(message, market_data)
            responses.append({
                "agent": self.head_trader.name,
                "response": response
            })
        
        return {
            "query": message,
            "responses": responses,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_workflow_summary(self) -> Dict:
        """Get summary of recent workflows"""
        recent = self.workflow_history[-10:]
        
        approved = sum(1 for w in recent if w.get('final_decision', {}).get('decision') == 'APPROVED')
        rejected = sum(1 for w in recent if w.get('final_decision', {}).get('decision') == 'REJECTED')
        waiting = sum(1 for w in recent if w.get('final_decision', {}).get('action') == 'WAIT')
        
        return {
            "total_workflows": len(self.workflow_history),
            "recent_10": {
                "approved": approved,
                "rejected": rejected,
                "waiting": waiting
            },
            "avg_duration": sum(w.get('duration_seconds', 0) for w in recent) / len(recent) if recent else 0,
            "last_workflow": recent[-1] if recent else None
        }


def create_trading_desk(capital: float = 1000000) -> TradingDeskCoordinator:
    """Factory function to create a trading desk"""
    return TradingDeskCoordinator(capital=capital)
