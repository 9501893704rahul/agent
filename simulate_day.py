"""
Simulate Today's Trading Day from 9:15 AM

Replays Nifty 50 5-minute candles and lets the agent trade in real-time simulation.
"""

import time
import sys
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

from data_fetcher import NiftyDataFetcher
from indicators import ScalpingIndicators
from scalping_strategies import StrategyEngine, TradeSetup
from paper_trading import PaperTradingEngine


class DaySimulator:
    """Simulates a full trading day with candle-by-candle replay"""
    
    def __init__(self, initial_capital: float = 50000.0, speed: float = 1.0):
        """
        Args:
            initial_capital: Starting capital for paper trading
            speed: Simulation speed (1.0 = 1 candle per second, 0.5 = 2 candles per second)
        """
        self.paper_engine = PaperTradingEngine(
            initial_capital=initial_capital,
            max_positions=3,
            risk_per_trade=0.02,
            data_file="simulation_trades.json"
        )
        self.paper_engine.reset()
        
        self.speed = speed
        self.data_fetcher = NiftyDataFetcher()
        self.full_data = None
        self.min_confidence = 65
        self.cooldown_candles = 3  # Wait 3 candles (15 min) after a trade
        self.last_trade_candle = -999
        
        # Statistics
        self.signals_generated = 0
        self.trades_taken = 0
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch today's data"""
        print("📊 Fetching Nifty 50 data...")
        self.full_data = self.data_fetcher.fetch_data()
        
        if self.full_data.empty:
            print("❌ Failed to fetch data")
            return pd.DataFrame()
        
        print(f"✅ Fetched {len(self.full_data)} candles")
        print(f"   From: {self.full_data.index[0]}")
        print(f"   To: {self.full_data.index[-1]}")
        
        return self.full_data
    
    def get_today_data(self) -> pd.DataFrame:
        """Filter to get only today's data starting from 9:15"""
        if self.full_data is None or self.full_data.empty:
            return pd.DataFrame()
        
        # Get the latest date in the data
        latest_date = self.full_data.index[-1].date()
        
        # Filter for that date
        today_data = self.full_data[self.full_data.index.date == latest_date]
        
        # Filter for market hours (9:15 to 15:30)
        today_data = today_data.between_time('09:15', '15:30')
        
        return today_data
    
    def analyze_candle(self, data_slice: pd.DataFrame) -> Dict:
        """Analyze up to current candle"""
        if len(data_slice) < 20:
            return {'setups': [], 'bias': None}
        
        indicators = ScalpingIndicators(data_slice)
        analyzed = indicators.calculate_all()
        
        engine = StrategyEngine(analyzed)
        setups = engine.run_all_strategies()
        bias = engine.get_market_bias()
        
        return {
            'setups': setups,
            'bias': bias,
            'analysis': indicators.get_current_analysis()
        }
    
    def check_position_exit(self, current_price: float):
        """Check if any position should be closed"""
        self.paper_engine.update_price(current_price)
    
    def should_enter_trade(self, candle_index: int) -> bool:
        """Check if we can enter a new trade (cooldown check)"""
        return (candle_index - self.last_trade_candle) >= self.cooldown_candles
    
    def print_candle_summary(self, candle: pd.Series, candle_num: int, total: int):
        """Print candle information"""
        change = ((candle['close'] - candle['open']) / candle['open']) * 100
        direction = "🟢" if change >= 0 else "🔴"
        
        print(f"\n{'='*70}")
        print(f"📍 Candle {candle_num}/{total} | {candle.name.strftime('%H:%M')}")
        print(f"   O: {candle['open']:.2f} | H: {candle['high']:.2f} | L: {candle['low']:.2f} | C: {candle['close']:.2f}")
        print(f"   {direction} Change: {change:+.2f}%")
    
    def print_position_status(self):
        """Print current position status"""
        positions = self.paper_engine.get_open_positions()
        
        if not positions:
            print("   📭 No open positions")
            return
        
        for pos in positions:
            pnl_emoji = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
            print(f"   📈 {pos.side} @ {pos.entry_price:.2f} | Current: {pos.current_price:.2f}")
            print(f"      SL: {pos.stop_loss:.2f} | Target: {pos.target_1:.2f}")
            print(f"      {pnl_emoji} P&L: ₹{pos.unrealized_pnl:+.2f}")
    
    def print_trade_event(self, event_type: str, details: Dict):
        """Print trade event"""
        if event_type == "ENTRY":
            print(f"\n   🎯 {'='*50}")
            print(f"   🎯 NEW TRADE: {details['direction']} @ {details['price']:.2f}")
            print(f"   🎯 Strategy: {details['strategy']}")
            print(f"   🎯 SL: {details['sl']:.2f} | Target: {details['target']:.2f}")
            print(f"   🎯 Confidence: {details['confidence']}%")
            print(f"   🎯 {'='*50}")
        elif event_type == "EXIT":
            emoji = "✅" if details['pnl'] >= 0 else "❌"
            print(f"\n   {emoji} {'='*50}")
            print(f"   {emoji} TRADE CLOSED: {details['reason']}")
            print(f"   {emoji} Exit Price: {details['price']:.2f}")
            print(f"   {emoji} P&L: ₹{details['pnl']:+.2f} ({details['pnl_pct']:+.2f}%)")
            print(f"   {emoji} {'='*50}")
    
    def run_simulation(self):
        """Run the full day simulation"""
        # Fetch data
        if not self.fetch_data().empty:
            today_data = self.get_today_data()
        else:
            return
        
        if today_data.empty:
            print("❌ No data available for today")
            return
        
        print(f"\n{'='*70}")
        print(f"🚀 STARTING DAY SIMULATION")
        print(f"{'='*70}")
        print(f"📅 Date: {today_data.index[0].strftime('%Y-%m-%d')}")
        print(f"📊 Candles: {len(today_data)} (9:15 - 15:30)")
        print(f"💰 Starting Capital: ₹{self.paper_engine.current_capital:,.2f}")
        print(f"⚡ Speed: {self.speed}x (1 candle = {self.speed:.1f}s)")
        print(f"🎯 Min Confidence: {self.min_confidence}%")
        print(f"{'='*70}")
        
        print("\n🚀 Starting simulation...\n")
        
        # Need minimum candles for indicators
        warmup_candles = 30
        
        # Get historical data for warmup
        historical_start_idx = self.full_data.index.get_loc(today_data.index[0])
        if historical_start_idx >= warmup_candles:
            warmup_data = self.full_data.iloc[historical_start_idx - warmup_candles:historical_start_idx]
        else:
            warmup_data = pd.DataFrame()
        
        # Track closed trades for notifications
        prev_trade_count = 0
        
        # Simulate candle by candle
        for i, (timestamp, candle) in enumerate(today_data.iterrows()):
            candle_num = i + 1
            
            # Build data slice (warmup + candles so far)
            if not warmup_data.empty:
                data_slice = pd.concat([warmup_data, today_data.iloc[:i+1]])
            else:
                data_slice = today_data.iloc[:i+1]
            
            current_price = candle['close']
            
            # Print candle info
            self.print_candle_summary(candle, candle_num, len(today_data))
            
            # Check for position exits first
            self.check_position_exit(current_price)
            
            # Check if any trade was closed
            current_trade_count = self.paper_engine.total_trades
            if current_trade_count > prev_trade_count:
                # A trade was just closed
                history = self.paper_engine.get_trade_history(1)
                if history:
                    last_trade = history[-1]
                    self.print_trade_event("EXIT", {
                        'reason': last_trade['exit_reason'],
                        'price': last_trade['exit_price'],
                        'pnl': last_trade['pnl'],
                        'pnl_pct': last_trade['pnl_percent']
                    })
                prev_trade_count = current_trade_count
            
            # Analyze market
            analysis = self.analyze_candle(data_slice)
            
            # Print market bias
            if analysis['bias']:
                bias = analysis['bias']
                print(f"   📊 Market: {bias['bias']} | Bullish: {bias['bullish_factors']}/5 | Bearish: {bias['bearish_factors']}/5")
            
            # Print position status
            self.print_position_status()
            
            # Check for signals
            setups = analysis.get('setups', [])
            valid_setups = [s for s in setups if s.confidence >= self.min_confidence]
            
            if valid_setups:
                self.signals_generated += 1
                best = valid_setups[0]
                print(f"   📡 SIGNAL: {best.strategy.value} - {best.direction}")
                print(f"      Entry: {best.entry_price:.2f} | Confidence: {best.confidence}%")
                
                # Check if we can trade
                open_positions = self.paper_engine.get_open_positions()
                can_trade = self.should_enter_trade(candle_num)
                no_same_direction = not any(p.side == best.direction for p in open_positions)
                
                if can_trade and no_same_direction and len(open_positions) < 3:
                    # Enter the trade
                    position = self.paper_engine.open_position(
                        side=best.direction,
                        entry_price=best.entry_price,
                        stop_loss=best.stop_loss,
                        target_1=best.target_1,
                        target_2=best.target_2,
                        strategy=best.strategy.value
                    )
                    
                    if position:
                        self.trades_taken += 1
                        self.last_trade_candle = candle_num
                        self.print_trade_event("ENTRY", {
                            'direction': best.direction,
                            'price': best.entry_price,
                            'strategy': best.strategy.value,
                            'sl': best.stop_loss,
                            'target': best.target_1,
                            'confidence': best.confidence
                        })
                elif not can_trade:
                    print(f"      ⏳ Cooldown active ({self.cooldown_candles - (candle_num - self.last_trade_candle)} candles left)")
                elif not no_same_direction:
                    print(f"      ⚠️ Already have {best.direction} position")
            
            # Wait based on speed
            time.sleep(self.speed)
        
        # Close any remaining positions at end of day
        print(f"\n{'='*70}")
        print("🔔 MARKET CLOSE - Closing all positions")
        print(f"{'='*70}")
        
        final_price = today_data.iloc[-1]['close']
        for pos in self.paper_engine.get_open_positions():
            trade = self.paper_engine.close_position(pos.id, final_price, "MARKET_CLOSE")
            if trade:
                self.print_trade_event("EXIT", {
                    'reason': 'MARKET_CLOSE',
                    'price': final_price,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_percent
                })
        
        # Print final summary
        self.print_final_summary(today_data)
    
    def print_final_summary(self, today_data: pd.DataFrame):
        """Print end of day summary"""
        stats = self.paper_engine.get_statistics()
        history = self.paper_engine.get_trade_history(50)
        
        print(f"\n{'='*70}")
        print(f"📊 DAY SIMULATION COMPLETE")
        print(f"{'='*70}")
        
        # Market summary
        open_price = today_data.iloc[0]['open']
        close_price = today_data.iloc[-1]['close']
        day_high = today_data['high'].max()
        day_low = today_data['low'].min()
        day_change = ((close_price - open_price) / open_price) * 100
        
        print(f"\n📈 MARKET SUMMARY:")
        print(f"   Open: {open_price:.2f} | Close: {close_price:.2f}")
        print(f"   High: {day_high:.2f} | Low: {day_low:.2f}")
        print(f"   Day Change: {day_change:+.2f}%")
        
        # Trading summary
        print(f"\n💰 TRADING SUMMARY:")
        print(f"   Starting Capital: ₹{stats['initial_capital']:,.2f}")
        print(f"   Final Capital: ₹{stats['current_capital']:,.2f}")
        print(f"   Total P&L: ₹{stats['total_pnl']:+,.2f} ({stats['total_pnl_percent']:+.2f}%)")
        print(f"   Max Drawdown: {stats['max_drawdown']:.2f}%")
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"   Signals Generated: {self.signals_generated}")
        print(f"   Trades Taken: {stats['total_trades']}")
        print(f"   Winning Trades: {stats['winning_trades']}")
        print(f"   Losing Trades: {stats['losing_trades']}")
        print(f"   Win Rate: {stats['win_rate']}%")
        print(f"   Avg Win: ₹{stats['avg_win']:+.2f}")
        print(f"   Avg Loss: ₹{stats['avg_loss']:+.2f}")
        print(f"   Profit Factor: {stats['profit_factor']:.2f}")
        
        # Trade details
        if history:
            print(f"\n📜 TRADE LOG:")
            print(f"   {'Side':<6} {'Entry':>10} {'Exit':>10} {'P&L':>12} {'Result':>8} {'Strategy':<20}")
            print(f"   {'-'*70}")
            for t in history:
                result_emoji = "✅" if t['result'] == 'WIN' else "❌"
                print(f"   {t['side']:<6} {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} ₹{t['pnl']:>+10.2f} {result_emoji:>8} {t['strategy']:<20}")
        
        print(f"\n{'='*70}")


def main():
    import sys
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║       🎯 NIFTY 50 DAY TRADING SIMULATOR 🎯                       ║
    ║                                                                  ║
    ║   Replays today's market data candle-by-candle                   ║
    ║   Agent automatically trades based on scalping strategies        ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Get speed from command line or use fast default
    speed = 0.01  # Instant by default
    if len(sys.argv) > 1:
        try:
            speed = float(sys.argv[1])
        except:
            pass
    
    print(f"⚡ Running at speed: {speed}s per candle")
    
    # Create and run simulator
    simulator = DaySimulator(initial_capital=50000.0, speed=speed)
    
    try:
        simulator.run_simulation()
    except KeyboardInterrupt:
        print("\n\n⚠️ Simulation interrupted by user")
        simulator.print_final_summary(simulator.get_today_data())


if __name__ == "__main__":
    main()
