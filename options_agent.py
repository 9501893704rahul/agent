"""
Real-Time Options Trading Agent for Nifty 50

Autonomous agent that:
- Monitors market in real-time
- Auto-trades CE (Calls) and PE (Puts)
- Manages positions with trailing stops
- Uses AI for signal confirmation
"""

import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import traceback

from data_fetcher import NiftyDataFetcher
from indicators import ScalpingIndicators
from scalping_strategies import StrategyEngine
from options_trading import (
    NiftyOptionsChain, OptionsPaperTrading, OptionsScalpingStrategy,
    OptionContract, OptionPosition, OptionType, print_options_chain
)
import config


class AgentMode(Enum):
    AGGRESSIVE = "AGGRESSIVE"   # More trades, lower confidence threshold
    MODERATE = "MODERATE"       # Balanced approach
    CONSERVATIVE = "CONSERVATIVE"  # Fewer trades, higher confidence


@dataclass
class OptionsSignal:
    timestamp: str
    signal_type: str  # MOMENTUM_CE, MOMENTUM_PE, REVERSAL_CE, REVERSAL_PE, etc.
    option: OptionContract
    action: str  # BUY or SELL
    confidence: int
    reasoning: str
    spot_price: float
    target_pct: float
    stoploss_pct: float


class RealTimeOptionsAgent:
    """
    Autonomous Options Trading Agent
    
    Continuously monitors market and executes options trades
    based on technical signals and market conditions.
    """
    
    def __init__(self, 
                 initial_capital: float = 50000.0,
                 mode: AgentMode = AgentMode.MODERATE,
                 update_interval: int = 10,
                 max_positions: int = 3):
        
        # Trading engine
        self.options_chain = NiftyOptionsChain()
        self.paper_trading = OptionsPaperTrading(initial_capital)
        self.strategy = OptionsScalpingStrategy(self.options_chain)
        self.data_fetcher = NiftyDataFetcher()
        
        # Agent settings
        self.mode = mode
        self.update_interval = update_interval
        self.max_positions = max_positions
        self.initial_capital = initial_capital
        
        # Mode-specific settings
        self._configure_mode()
        
        # State
        self.is_running = False
        self.is_paused = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Market data
        self.current_spot = 0.0
        self.current_chain: Optional[Dict] = None
        self.last_analysis: Optional[Dict] = None
        
        # Signals and events
        self.signals_history: List[OptionsSignal] = []
        self.last_trade_time: Optional[datetime] = None
        self.cooldown_seconds = 300  # 5 minutes between trades
        
        # Callbacks
        self.on_signal: Optional[Callable] = None
        self.on_trade_open: Optional[Callable] = None
        self.on_trade_close: Optional[Callable] = None
        self.on_price_update: Optional[Callable] = None
        
        print(f"🤖 Options Agent initialized")
        print(f"   Mode: {mode.value}")
        print(f"   Capital: ₹{initial_capital:,.0f}")
        print(f"   Max Positions: {max_positions}")
    
    def _configure_mode(self):
        """Configure settings based on agent mode"""
        if self.mode == AgentMode.AGGRESSIVE:
            self.min_confidence = 60
            self.target_pct = 25
            self.stoploss_pct = 35
            self.cooldown_seconds = 180  # 3 min
        elif self.mode == AgentMode.MODERATE:
            self.min_confidence = 70
            self.target_pct = 20
            self.stoploss_pct = 30
            self.cooldown_seconds = 300  # 5 min
        else:  # CONSERVATIVE
            self.min_confidence = 80
            self.target_pct = 15
            self.stoploss_pct = 25
            self.cooldown_seconds = 600  # 10 min
    
    def _is_trading_hours(self) -> bool:
        """Check if within market hours"""
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
        current_time = now.strftime("%H:%M")
        return "09:15" <= current_time <= "15:25"
    
    def _is_cooldown_active(self) -> bool:
        """Check if cooldown period is active"""
        if self.last_trade_time is None:
            return False
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        return elapsed < self.cooldown_seconds
    
    def _fetch_market_data(self) -> bool:
        """Fetch latest market data"""
        try:
            raw_data = self.data_fetcher.fetch_data()
            if raw_data.empty:
                return False
            
            # Calculate indicators
            indicators = ScalpingIndicators(raw_data)
            analyzed_data = indicators.calculate_all()
            
            # Get analysis
            analysis = indicators.get_current_analysis()
            market_data = self.data_fetcher.get_current_price()
            
            # Get market bias
            engine = StrategyEngine(analyzed_data)
            bias = engine.get_market_bias()
            
            # Update spot price
            self.current_spot = market_data.get('close', market_data.get('price', 0))
            
            # Get options chain
            self.current_chain = self.options_chain.get_option_chain(self.current_spot)
            
            self.last_analysis = {
                'timestamp': datetime.now().isoformat(),
                'spot': self.current_spot,
                'rsi': analysis['rsi']['value'],
                'macd': analysis['macd'],
                'ema': analysis['ema'],
                'bias': bias,
                'chain': self.current_chain
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            traceback.print_exc()
            return False
    
    def _generate_signals(self) -> List[OptionsSignal]:
        """Generate trading signals based on current analysis"""
        signals = []
        
        if not self.last_analysis or not self.current_chain:
            return signals
        
        bias = self.last_analysis['bias']
        rsi = self.last_analysis['rsi']
        spot = self.current_spot
        
        # Momentum signal
        momentum_signal = self.strategy.momentum_signal(
            bias['bias'], 
            bias['strength'],
            self.current_chain
        )
        
        if momentum_signal:
            confidence = 70 if 'STRONGLY' in bias['bias'] else 60
            signals.append(OptionsSignal(
                timestamp=datetime.now().isoformat(),
                signal_type=momentum_signal['strategy'],
                option=momentum_signal['option'],
                action=momentum_signal['action'],
                confidence=confidence,
                reasoning=momentum_signal['reasoning'],
                spot_price=spot,
                target_pct=self.target_pct,
                stoploss_pct=self.stoploss_pct
            ))
        
        # Reversal signal
        at_support = rsi < 35
        at_resistance = rsi > 65
        
        reversal_signal = self.strategy.scalp_reversal_signal(
            rsi, at_support, at_resistance,
            self.current_chain
        )
        
        if reversal_signal:
            signals.append(OptionsSignal(
                timestamp=datetime.now().isoformat(),
                signal_type=reversal_signal['strategy'],
                option=reversal_signal['option'],
                action=reversal_signal['action'],
                confidence=reversal_signal.get('confidence', 70),
                reasoning=reversal_signal['reasoning'],
                spot_price=spot,
                target_pct=self.target_pct,
                stoploss_pct=self.stoploss_pct
            ))
        
        # Filter by confidence
        signals = [s for s in signals if s.confidence >= self.min_confidence]
        
        return signals
    
    def _should_trade(self, signal: OptionsSignal) -> bool:
        """Determine if we should execute a trade"""
        # Check max positions
        open_positions = self.paper_trading.get_open_positions()
        if len(open_positions) >= self.max_positions:
            return False
        
        # Check cooldown
        if self._is_cooldown_active():
            return False
        
        # Check if already have similar position
        for pos in open_positions:
            if pos.contract.option_type == signal.option.option_type:
                # Already have same type (CE or PE)
                return False
        
        # Check capital
        required = signal.option.ltp * self.options_chain.NIFTY_LOT_SIZE
        if required > self.paper_trading.current_capital * 0.5:
            return False  # Don't use more than 50% capital per trade
        
        return True
    
    def _execute_trade(self, signal: OptionsSignal) -> bool:
        """Execute a trade based on signal"""
        try:
            trade_signal = {
                'strategy': signal.signal_type,
                'action': signal.action,
                'option': signal.option,
                'quantity': self.options_chain.NIFTY_LOT_SIZE,
                'target_pct': signal.target_pct,
                'stoploss_pct': signal.stoploss_pct,
                'reasoning': signal.reasoning
            }
            
            position = self.paper_trading.open_position(trade_signal)
            
            if position:
                self.last_trade_time = datetime.now()
                self.signals_history.append(signal)
                
                if self.on_trade_open:
                    self.on_trade_open(position, signal)
                
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error executing trade: {e}")
            return False
    
    def _update_positions(self):
        """Update all positions and check exits"""
        if self.current_spot <= 0:
            return
        
        # Update position prices
        self.paper_trading.update_positions(self.current_spot)
        
        # Check exit conditions
        positions_before = len(self.paper_trading.get_open_positions())
        self.paper_trading.check_exit_conditions(
            target_pct=self.target_pct,
            stoploss_pct=self.stoploss_pct
        )
        positions_after = len(self.paper_trading.get_open_positions())
        
        # Callback if position was closed
        if positions_after < positions_before and self.on_trade_close:
            if self.paper_trading.trade_history:
                last_trade = self.paper_trading.trade_history[-1]
                self.on_trade_close(last_trade)
    
    def _agent_loop(self):
        """Main agent loop"""
        print(f"🚀 Options Agent started - Monitoring every {self.update_interval}s")
        
        while not self._stop_event.is_set():
            try:
                if self.is_paused:
                    time.sleep(1)
                    continue
                
                # Fetch data
                if not self._fetch_market_data():
                    time.sleep(5)
                    continue
                
                # Update positions
                self._update_positions()
                
                # Price update callback
                if self.on_price_update:
                    self.on_price_update(
                        self.current_spot,
                        self.paper_trading.get_open_positions(),
                        self.last_analysis
                    )
                
                # Generate signals (only during trading hours)
                if self._is_trading_hours():
                    signals = self._generate_signals()
                    
                    for signal in signals:
                        if self.on_signal:
                            self.on_signal(signal)
                        
                        if self._should_trade(signal):
                            self._execute_trade(signal)
                
                # Wait for next update
                self._stop_event.wait(self.update_interval)
                
            except Exception as e:
                print(f"❌ Agent loop error: {e}")
                traceback.print_exc()
                time.sleep(5)
    
    def start(self):
        """Start the agent"""
        if self.is_running:
            print("⚠️ Agent already running")
            return
        
        self._stop_event.clear()
        self.is_running = True
        self.is_paused = False
        
        self._thread = threading.Thread(target=self._agent_loop, daemon=True)
        self._thread.start()
        
        print("✅ Options Agent STARTED")
    
    def stop(self):
        """Stop the agent"""
        if not self.is_running:
            print("⚠️ Agent not running")
            return
        
        self._stop_event.set()
        self.is_running = False
        
        if self._thread:
            self._thread.join(timeout=5)
        
        print("🛑 Options Agent STOPPED")
    
    def pause(self):
        """Pause trading (monitoring continues)"""
        self.is_paused = True
        print("⏸️ Options Agent PAUSED")
    
    def resume(self):
        """Resume trading"""
        self.is_paused = False
        print("▶️ Options Agent RESUMED")
    
    def manual_buy_ce(self, strike: Optional[float] = None) -> bool:
        """Manually buy a Call option"""
        if self.current_spot <= 0:
            self._fetch_market_data()
        
        if strike is None:
            strike = self.options_chain.get_atm_strike(self.current_spot)
        
        option = self.options_chain.get_option_by_strike(
            strike, OptionType.CALL, self.current_chain
        )
        
        if option:
            signal = {
                'strategy': 'MANUAL_CE',
                'action': 'BUY',
                'option': option,
                'quantity': self.options_chain.NIFTY_LOT_SIZE,
                'target_pct': self.target_pct,
                'stoploss_pct': self.stoploss_pct,
                'reasoning': f'Manual CE buy at strike {strike}'
            }
            return self.paper_trading.open_position(signal) is not None
        return False
    
    def manual_buy_pe(self, strike: Optional[float] = None) -> bool:
        """Manually buy a Put option"""
        if self.current_spot <= 0:
            self._fetch_market_data()
        
        if strike is None:
            strike = self.options_chain.get_atm_strike(self.current_spot)
        
        option = self.options_chain.get_option_by_strike(
            strike, OptionType.PUT, self.current_chain
        )
        
        if option:
            signal = {
                'strategy': 'MANUAL_PE',
                'action': 'BUY',
                'option': option,
                'quantity': self.options_chain.NIFTY_LOT_SIZE,
                'target_pct': self.target_pct,
                'stoploss_pct': self.stoploss_pct,
                'reasoning': f'Manual PE buy at strike {strike}'
            }
            return self.paper_trading.open_position(signal) is not None
        return False
    
    def close_position(self, position_id: str) -> bool:
        """Close a specific position"""
        return self.paper_trading.close_position(position_id, 'MANUAL') is not None
    
    def close_all(self):
        """Close all open positions"""
        for pos in self.paper_trading.get_open_positions():
            self.paper_trading.close_position(pos.id, 'CLOSE_ALL')
    
    def get_status(self) -> Dict:
        """Get current agent status"""
        positions = self.paper_trading.get_open_positions()
        stats = self.paper_trading.get_statistics()
        
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'mode': self.mode.value,
            'spot_price': self.current_spot,
            'trading_hours': self._is_trading_hours(),
            'cooldown_active': self._is_cooldown_active(),
            'open_positions': len(positions),
            'positions': [p.to_dict() for p in positions],
            'statistics': stats,
            'last_update': self.last_analysis['timestamp'] if self.last_analysis else None,
            'recent_signals': len(self.signals_history)
        }
    
    def get_options_chain(self) -> Dict:
        """Get current options chain"""
        if self.current_chain:
            return {
                'spot': self.current_chain['spot'],
                'expiry': self.current_chain['expiry'].isoformat(),
                'calls': [c.to_dict() for c in self.current_chain['calls']],
                'puts': [p.to_dict() for p in self.current_chain['puts']]
            }
        return {}


def run_options_agent_simulation(duration_minutes: int = 5, mode: AgentMode = AgentMode.MODERATE):
    """Run options agent simulation"""
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║       🤖 REAL-TIME OPTIONS TRADING AGENT                         ║
║       Autonomous CE/PE Trading with Auto Entry/Exit              ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Create agent
    agent = RealTimeOptionsAgent(
        initial_capital=50000,
        mode=mode,
        update_interval=5,
        max_positions=2
    )
    
    # Set callbacks
    def on_signal(signal):
        print(f"   📡 SIGNAL: {signal.signal_type} | {signal.option.symbol} @ ₹{signal.option.ltp:.2f}")
    
    def on_trade_open(position, signal):
        print(f"   🎯 OPENED: {position.contract.symbol} @ ₹{position.entry_price:.2f}")
    
    def on_trade_close(trade):
        emoji = '✅' if trade['pnl'] >= 0 else '❌'
        print(f"   {emoji} CLOSED: {trade['symbol']} | P&L: ₹{trade['pnl']:+,.2f}")
    
    def on_price_update(spot, positions, analysis):
        bias = analysis['bias']['bias'] if analysis else 'N/A'
        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} | Spot: {spot:.2f} | {bias}")
        for pos in positions:
            pnl_emoji = '🟢' if pos.unrealized_pnl >= 0 else '🔴'
            print(f"   📈 {pos.contract.symbol} | {pnl_emoji} P&L: ₹{pos.unrealized_pnl:+,.2f}")
    
    agent.on_signal = on_signal
    agent.on_trade_open = on_trade_open
    agent.on_trade_close = on_trade_close
    agent.on_price_update = on_price_update
    
    # Start agent
    print(f"🚀 Starting agent for {duration_minutes} minutes...")
    print("="*60)
    agent.start()
    
    # Run for specified duration
    try:
        for i in range(duration_minutes * 6):  # Check every 10 seconds
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    # Stop and close all
    agent.stop()
    agent.close_all()
    
    # Print summary
    stats = agent.paper_trading.get_statistics()
    print(f"""
{'='*60}
📊 OPTIONS AGENT SUMMARY
{'='*60}

💰 CAPITAL:
   Starting: ₹{stats['initial_capital']:,.2f}
   Current:  ₹{stats['current_capital']:,.2f}
   P&L:      ₹{stats['total_pnl']:+,.2f} ({stats['total_pnl_pct']:+.2f}%)

📊 STATISTICS:
   Mode:           {mode.value}
   Total Trades:   {stats['total_trades']}
   Winning:        {stats['winning_trades']}
   Losing:         {stats['losing_trades']}
   Win Rate:       {stats['win_rate']}%
""")
    
    if agent.paper_trading.trade_history:
        print("📜 TRADE LOG:")
        for t in agent.paper_trading.trade_history:
            emoji = '✅' if t['result'] == 'WIN' else '❌'
            print(f"   {emoji} {t['symbol']} | ₹{t['entry_price']:.2f} → ₹{t['exit_price']:.2f} | P&L: ₹{t['pnl']:+,.2f}")
    
    print("="*60)
    
    return agent


if __name__ == "__main__":
    run_options_agent_simulation(duration_minutes=2, mode=AgentMode.AGGRESSIVE)
