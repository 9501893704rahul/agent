"""
Real-Time Scalping Agent for Nifty 50

Monitors market continuously and:
- Auto-enters trades based on strategy signals
- Monitors open positions in real-time
- Auto-exits on stop-loss/target
- Trails stop-loss to lock profits
- Provides live updates via callbacks
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import traceback

from data_fetcher import NiftyDataFetcher
from indicators import ScalpingIndicators, get_candle_pattern_analysis
from scalping_strategies import StrategyEngine, TradeSetup
from paper_trading import PaperTradingEngine, Position
from candle_patterns import CandlePatternAnalyzer, PatternSignal
import config


class AgentState(Enum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"


@dataclass
class TradeEvent:
    """Event when a trade action occurs"""
    event_type: str  # ENTRY, EXIT, UPDATE, ALERT
    timestamp: str
    position_id: Optional[str]
    side: Optional[str]
    price: float
    details: Dict
    
    def to_dict(self):
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'position_id': self.position_id,
            'side': self.side,
            'price': self.price,
            'details': self.details
        }


class RealTimeScalpingAgent:
    """
    Real-Time Scalping Agent
    
    Continuously monitors the market and executes paper trades
    based on scalping strategies.
    """
    
    def __init__(self, paper_engine: PaperTradingEngine, 
                 update_interval: int = 5,
                 auto_trade: bool = True,
                 min_confidence: int = 65):
        
        self.paper_engine = paper_engine
        self.update_interval = update_interval  # seconds between updates
        self.auto_trade = auto_trade  # Auto-execute trades
        self.min_confidence = min_confidence  # Minimum confidence to trade
        
        # Components
        self.data_fetcher = NiftyDataFetcher()
        
        # State
        self.state = AgentState.STOPPED
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Current market data
        self.current_price: float = 0.0
        self.last_analysis: Optional[Dict] = None
        self.last_setups: List[TradeSetup] = []
        self.pending_signals: List[TradeSetup] = []
        
        # Candle pattern analysis
        self.last_patterns: Optional[Dict] = None
        self.pattern_confirmation_required = True  # Require pattern confirmation for trades
        self.min_pattern_confidence = 75  # Minimum confidence for pattern signals
        
        # Trade events log
        self.events: List[TradeEvent] = []
        self.max_events = 100
        
        # Callbacks for real-time updates
        self.on_price_update: Optional[Callable] = None
        self.on_signal: Optional[Callable] = None
        self.on_trade_entry: Optional[Callable] = None
        self.on_trade_exit: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Trading rules
        self.trading_hours_only = True
        self.market_open = "09:15"
        self.market_close = "15:30"
        self.cooldown_after_trade = 300  # 5 minutes cooldown after a trade
        self.last_trade_time: Optional[datetime] = None
        
        # Trailing stop settings
        self.enable_trailing_stop = True
        self.trailing_trigger_percent = 0.5  # Trigger at 0.5% profit
        self.trailing_stop_percent = 0.3  # Trail by 0.3%
        
        # Wire up paper engine callbacks
        self.paper_engine.on_position_opened = self._on_position_opened
        self.paper_engine.on_position_closed = self._on_position_closed
        
        print("🤖 Real-Time Scalping Agent initialized")
    
    def _log_event(self, event_type: str, price: float, details: Dict,
                   position_id: Optional[str] = None, side: Optional[str] = None):
        """Log a trade event"""
        event = TradeEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            position_id=position_id,
            side=side,
            price=price,
            details=details
        )
        self.events.append(event)
        
        # Keep only recent events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        return event
    
    def _on_position_opened(self, position: Position):
        """Callback when position is opened"""
        self.last_trade_time = datetime.now()
        
        event = self._log_event(
            "ENTRY",
            position.entry_price,
            {
                'strategy': position.strategy,
                'quantity': position.quantity,
                'stop_loss': position.stop_loss,
                'target_1': position.target_1
            },
            position.id,
            position.side
        )
        
        if self.on_trade_entry:
            self.on_trade_entry(position, event)
    
    def _on_position_closed(self, trade):
        """Callback when position is closed"""
        event = self._log_event(
            "EXIT",
            trade.exit_price,
            {
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'reason': trade.exit_reason,
                'result': trade.result.value,
                'duration_minutes': trade.duration_minutes
            },
            trade.id,
            trade.side
        )
        
        if self.on_trade_exit:
            self.on_trade_exit(trade, event)
    
    def _is_trading_hours(self) -> bool:
        """Check if market is open"""
        if not self.trading_hours_only:
            return True
        
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        # Check if weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        return self.market_open <= current_time <= self.market_close
    
    def _is_cooldown_active(self) -> bool:
        """Check if cooldown period is active after a trade"""
        if self.last_trade_time is None:
            return False
        
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        return elapsed < self.cooldown_after_trade
    
    def _update_trailing_stops(self):
        """Update trailing stops for profitable positions"""
        if not self.enable_trailing_stop:
            return
        
        for position in self.paper_engine.get_open_positions():
            if position.side == "LONG":
                profit_percent = ((self.current_price - position.entry_price) / position.entry_price) * 100
                
                if profit_percent >= self.trailing_trigger_percent:
                    new_stop = self.current_price * (1 - self.trailing_stop_percent / 100)
                    
                    if new_stop > position.stop_loss:
                        old_stop = position.stop_loss
                        position.stop_loss = new_stop
                        
                        self._log_event(
                            "TRAILING_STOP",
                            self.current_price,
                            {
                                'old_stop': old_stop,
                                'new_stop': new_stop,
                                'profit_percent': profit_percent
                            },
                            position.id,
                            position.side
                        )
                        print(f"📈 Trailing stop updated: {old_stop:.2f} → {new_stop:.2f}")
            
            else:  # SHORT
                profit_percent = ((position.entry_price - self.current_price) / position.entry_price) * 100
                
                if profit_percent >= self.trailing_trigger_percent:
                    new_stop = self.current_price * (1 + self.trailing_stop_percent / 100)
                    
                    if new_stop < position.stop_loss:
                        old_stop = position.stop_loss
                        position.stop_loss = new_stop
                        
                        self._log_event(
                            "TRAILING_STOP",
                            self.current_price,
                            {
                                'old_stop': old_stop,
                                'new_stop': new_stop,
                                'profit_percent': profit_percent
                            },
                            position.id,
                            position.side
                        )
                        print(f"📈 Trailing stop updated: {old_stop:.2f} → {new_stop:.2f}")
    
    def _analyze_market(self) -> Optional[Dict]:
        """Fetch data and analyze market with candlestick pattern detection"""
        try:
            # Fetch fresh data
            raw_data = self.data_fetcher.fetch_data()
            
            if raw_data.empty:
                return None
            
            # Calculate indicators
            indicators = ScalpingIndicators(raw_data)
            analyzed_data = indicators.calculate_all()
            
            # Get current analysis
            current_analysis = indicators.get_current_analysis()
            market_data = self.data_fetcher.get_current_price()
            
            # Update current price
            self.current_price = market_data.get('close', market_data.get('price', 0))
            
            # Run strategy engine
            strategy_engine = StrategyEngine(analyzed_data)
            setups = strategy_engine.run_all_strategies()
            market_bias = strategy_engine.get_market_bias()
            
            # === CANDLESTICK PATTERN ANALYSIS ===
            pattern_analyzer = CandlePatternAnalyzer(analyzed_data)
            pattern_summary = pattern_analyzer.get_pattern_summary()
            self.last_patterns = pattern_summary
            
            # Get high-confidence actionable patterns
            actionable_patterns = pattern_analyzer.get_actionable_signals(self.min_pattern_confidence)
            
            # Enhance setups with pattern confirmation
            enhanced_setups = self._enhance_setups_with_patterns(setups, actionable_patterns, pattern_summary)
            
            self.last_setups = enhanced_setups
            self.last_analysis = {
                'market_data': market_data,
                'analysis': current_analysis,
                'setups': enhanced_setups,
                'market_bias': market_bias,
                'candle_patterns': pattern_summary,
                'actionable_patterns': [
                    {
                        'type': p.pattern_type.value,
                        'signal': p.signal.value,
                        'confidence': p.confidence,
                        'description': p.description,
                        'entry': p.entry_suggestion,
                        'stop_loss': p.stop_loss_suggestion,
                        'target': p.target_suggestion
                    }
                    for p in actionable_patterns[:5]
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            # Log high-confidence patterns
            if actionable_patterns:
                best_pattern = actionable_patterns[0]
                print(f"📊 Pattern: {best_pattern.pattern_type.value} ({best_pattern.signal.value}) - {best_pattern.confidence}% confidence")
            
            return self.last_analysis
            
        except Exception as e:
            print(f"❌ Error analyzing market: {e}")
            traceback.print_exc()
            if self.on_error:
                self.on_error(str(e))
            return None
    
    def _enhance_setups_with_patterns(self, setups: List[TradeSetup], 
                                       patterns: list, 
                                       pattern_summary: Dict) -> List[TradeSetup]:
        """
        Enhance trade setups with candlestick pattern confirmation.
        Boosts confidence when patterns align with technical signals.
        """
        if not patterns:
            return setups
        
        enhanced_setups = []
        pattern_bias = pattern_summary.get('overall_bias', 'NEUTRAL')
        
        for setup in setups:
            # Check if any pattern confirms the setup direction
            confirming_patterns = []
            
            for pattern in patterns:
                pattern_signal = pattern.signal
                
                # Check alignment
                if setup.direction == "LONG" and pattern_signal in [PatternSignal.STRONG_BULLISH, PatternSignal.BULLISH]:
                    confirming_patterns.append(pattern)
                elif setup.direction == "SHORT" and pattern_signal in [PatternSignal.STRONG_BEARISH, PatternSignal.BEARISH]:
                    confirming_patterns.append(pattern)
            
            # Boost confidence if patterns confirm
            if confirming_patterns:
                best_confirming = max(confirming_patterns, key=lambda p: p.confidence)
                
                # Calculate confidence boost
                pattern_boost = min(15, best_confirming.confidence * 0.15)
                new_confidence = min(98, setup.confidence + pattern_boost)
                
                # Update setup with enhanced confidence
                setup.confidence = new_confidence
                setup.reasoning += f" | Pattern confirmation: {best_confirming.pattern_type.value} ({best_confirming.confidence}%)"
                setup.indicators['pattern_confirmed'] = True
                setup.indicators['confirming_pattern'] = best_confirming.pattern_type.value
                setup.indicators['pattern_confidence'] = best_confirming.confidence
            else:
                setup.indicators['pattern_confirmed'] = False
            
            # Also check if pattern bias aligns
            if (setup.direction == "LONG" and pattern_bias == "BULLISH") or \
               (setup.direction == "SHORT" and pattern_bias == "BEARISH"):
                setup.confidence = min(98, setup.confidence + 5)
                setup.indicators['bias_aligned'] = True
            else:
                setup.indicators['bias_aligned'] = False
            
            enhanced_setups.append(setup)
        
        # Re-sort by enhanced confidence
        enhanced_setups.sort(key=lambda x: x.confidence, reverse=True)
        
        return enhanced_setups
    
    def _evaluate_signals(self, setups: List[TradeSetup]):
        """Evaluate trade signals and execute if valid"""
        
        if not self.auto_trade:
            return
        
        if not self._is_trading_hours():
            return
        
        if self._is_cooldown_active():
            return
        
        # Filter by confidence
        valid_setups = [s for s in setups if s.confidence >= self.min_confidence]
        
        if not valid_setups:
            return
        
        # Get best setup
        best_setup = valid_setups[0]  # Already sorted by confidence
        
        # Check if we should enter
        open_positions = self.paper_engine.get_open_positions()
        
        # Don't enter if already have position in same direction
        for pos in open_positions:
            if pos.side == best_setup.direction:
                return
        
        # Log signal
        self._log_event(
            "SIGNAL",
            best_setup.entry_price,
            {
                'strategy': best_setup.strategy.value,
                'direction': best_setup.direction,
                'confidence': best_setup.confidence,
                'risk_reward': best_setup.risk_reward,
                'reasoning': best_setup.reasoning
            }
        )
        
        if self.on_signal:
            self.on_signal(best_setup)
        
        # Execute trade
        print(f"\n🎯 SIGNAL: {best_setup.strategy.value} - {best_setup.direction}")
        print(f"   Entry: {best_setup.entry_price:.2f}, SL: {best_setup.stop_loss:.2f}, T1: {best_setup.target_1:.2f}")
        print(f"   Confidence: {best_setup.confidence}%, R:R = {best_setup.risk_reward}")
        
        self.paper_engine.open_position(
            side=best_setup.direction,
            entry_price=best_setup.entry_price,
            stop_loss=best_setup.stop_loss,
            target_1=best_setup.target_1,
            target_2=best_setup.target_2,
            strategy=best_setup.strategy.value
        )
    
    def _monitoring_loop(self):
        """Main monitoring loop - runs in separate thread"""
        print(f"🚀 Agent started - Monitoring every {self.update_interval}s")
        
        while not self._stop_event.is_set():
            try:
                # Analyze market
                analysis = self._analyze_market()
                
                if analysis and self.current_price > 0:
                    # Update paper engine with current price
                    # This checks stop-loss and targets
                    self.paper_engine.update_price(self.current_price)
                    
                    # Update trailing stops
                    self._update_trailing_stops()
                    
                    # Evaluate new signals
                    if analysis.get('setups'):
                        self._evaluate_signals(analysis['setups'])
                    
                    # Callback for price update
                    if self.on_price_update:
                        self.on_price_update(
                            self.current_price,
                            self.paper_engine.get_open_positions(),
                            analysis
                        )
                    
                    # Callback for position update
                    if self.on_position_update:
                        positions = self.paper_engine.get_open_positions()
                        if positions:
                            self.on_position_update(positions)
                
                # Wait for next update
                self._stop_event.wait(self.update_interval)
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                traceback.print_exc()
                if self.on_error:
                    self.on_error(str(e))
                time.sleep(5)  # Wait before retrying
    
    def start(self):
        """Start the real-time monitoring agent"""
        if self.state == AgentState.RUNNING:
            print("⚠️ Agent already running")
            return
        
        self._stop_event.clear()
        self.state = AgentState.RUNNING
        
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        
        print("✅ Real-Time Scalping Agent STARTED")
        print(f"   Auto-trade: {self.auto_trade}")
        print(f"   Min confidence: {self.min_confidence}%")
        print(f"   Update interval: {self.update_interval}s")
    
    def stop(self):
        """Stop the real-time monitoring agent"""
        if self.state != AgentState.RUNNING:
            print("⚠️ Agent not running")
            return
        
        self._stop_event.set()
        self.state = AgentState.STOPPED
        
        if self._thread:
            self._thread.join(timeout=5)
        
        print("🛑 Real-Time Scalping Agent STOPPED")
    
    def pause(self):
        """Pause auto-trading (monitoring continues)"""
        self.auto_trade = False
        self.state = AgentState.PAUSED
        print("⏸️ Auto-trading PAUSED")
    
    def resume(self):
        """Resume auto-trading"""
        self.auto_trade = True
        self.state = AgentState.RUNNING
        print("▶️ Auto-trading RESUMED")
    
    def manual_entry(self, direction: str, entry_price: Optional[float] = None,
                     stop_loss: Optional[float] = None, target: Optional[float] = None) -> bool:
        """Manually enter a trade"""
        
        if entry_price is None:
            entry_price = self.current_price
        
        if entry_price <= 0:
            print("❌ Invalid entry price")
            return False
        
        # Calculate default SL and target if not provided
        atr = 25  # Default ATR estimate
        
        if direction.upper() == "LONG":
            if stop_loss is None:
                stop_loss = entry_price - (atr * 1.5)
            if target is None:
                target = entry_price + (atr * 2)
        else:
            if stop_loss is None:
                stop_loss = entry_price + (atr * 1.5)
            if target is None:
                target = entry_price - (atr * 2)
        
        position = self.paper_engine.open_position(
            side=direction.upper(),
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=target,
            target_2=None,
            strategy="MANUAL"
        )
        
        return position is not None
    
    def manual_exit(self, position_id: str, reason: str = "MANUAL") -> bool:
        """Manually exit a position"""
        trade = self.paper_engine.close_position(
            position_id, 
            self.current_price,
            reason
        )
        return trade is not None
    
    def close_all_positions(self, reason: str = "CLOSE_ALL"):
        """Close all open positions"""
        positions = self.paper_engine.get_open_positions()
        
        for position in positions:
            self.paper_engine.close_position(
                position.id,
                self.current_price,
                reason
            )
        
        print(f"🔒 Closed {len(positions)} positions")
    
    def get_status(self) -> Dict:
        """Get current agent status"""
        positions = self.paper_engine.get_open_positions()
        stats = self.paper_engine.get_statistics()
        
        return {
            'state': self.state.value,
            'auto_trade': self.auto_trade,
            'current_price': self.current_price,
            'trading_hours': self._is_trading_hours(),
            'cooldown_active': self._is_cooldown_active(),
            'open_positions': len(positions),
            'positions': [p.to_dict() for p in positions],
            'statistics': stats,
            'last_update': self.last_analysis.get('timestamp') if self.last_analysis else None,
            'recent_events': [e.to_dict() for e in self.events[-10:]]
        }
    
    def get_events(self, limit: int = 50) -> List[Dict]:
        """Get recent events"""
        return [e.to_dict() for e in self.events[-limit:]]


# Global agent instance
_agent_instance: Optional[RealTimeScalpingAgent] = None
_paper_engine: Optional[PaperTradingEngine] = None


def get_paper_engine() -> PaperTradingEngine:
    """Get or create paper trading engine"""
    global _paper_engine
    if _paper_engine is None:
        _paper_engine = PaperTradingEngine(
            initial_capital=50000.0,
            max_positions=3,
            risk_per_trade=0.02
        )
    return _paper_engine


def get_realtime_agent() -> RealTimeScalpingAgent:
    """Get or create real-time agent"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RealTimeScalpingAgent(
            paper_engine=get_paper_engine(),
            update_interval=10,
            auto_trade=True,
            min_confidence=65
        )
    return _agent_instance
