"""
Multi-Agent Trading Desk System for Nifty 50 5-Minute Scalping

A team of specialized AI agents working together:
- Market Analyst (Arjun): Technical analysis and market conditions
- Strategist (Priya): Strategy selection and trade setup generation
- Risk Manager (Vikram): Risk assessment and position sizing
- Execution (Ravi): Entry/exit signals and trade management
- Head Trader (Anand): Final decisions and team coordination
"""

from .base_agent import BaseAgent, AgentMessage, AgentRole, MessageType
from .market_analyst import MarketAnalystAgent
from .strategist import StrategyAgent
from .risk_manager import RiskManagerAgent
from .execution import ExecutionAgent
from .head_trader import HeadTraderAgent
from .coordinator import TradingDeskCoordinator, create_trading_desk

__all__ = [
    'BaseAgent',
    'AgentMessage',
    'AgentRole',
    'MessageType',
    'MarketAnalystAgent',
    'StrategyAgent',
    'RiskManagerAgent',
    'ExecutionAgent',
    'HeadTraderAgent',
    'TradingDeskCoordinator',
    'create_trading_desk'
]

__version__ = '1.0.0'
