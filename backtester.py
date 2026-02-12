"""
Backtesting & Accuracy Module for Nifty 50 Scalping Agent

Measures:
- Win Rate (Accuracy)
- Profit Factor
- Sharpe Ratio
- Max Drawdown
- Average R:R Achieved
- Total P&L
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json

from data_fetcher import NiftyDataFetcher
from indicators import ScalpingIndicators


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: datetime
    direction: str  # LONG or SHORT
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    strategy: str
    pnl_points: float = 0
    pnl_pct: float = 0
    result: str = ""  # WIN, LOSS, BREAKEVEN
    exit_reason: str = ""  # TARGET, STOP_LOSS, TIME_EXIT
    

class Backtester:
    """
    Backtests the scalping strategies on historical data
    """
    
    def __init__(self, initial_capital: float = 1000000, risk_per_trade: float = 0.01):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.data_fetcher = NiftyDataFetcher()
        
    def run_backtest(self, days: int = 30, strategies: List[str] = None) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            days: Number of days to backtest
            strategies: List of strategies to test (None = all)
        """
        print(f"\n{'='*60}")
        print(f"📊 BACKTESTING NIFTY 50 SCALPING STRATEGIES")
        print(f"{'='*60}")
        print(f"Period: Last {days} days")
        print(f"Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"Risk per Trade: {self.risk_per_trade*100}%")
        print(f"{'='*60}\n")
        
        # Fetch historical data
        print("📥 Fetching historical data...")
        df = self.data_fetcher.fetch_data(period=f"{days}d", interval="5m")
        
        if df.empty:
            return {"error": "Failed to fetch data"}
        
        print(f"✅ Loaded {len(df)} candles")
        
        # Calculate indicators for full dataset
        indicators = ScalpingIndicators(df)
        df = indicators.calculate_all()
        
        # Simulate trading
        print("\n🔄 Running simulation...")
        
        position = None
        entry_idx = 0
        
        for i in range(50, len(df) - 10):  # Start after warmup, leave room for exit
            current_bar = df.iloc[i]
            
            # Skip if we have an open position
            if position:
                # Check exit conditions
                exit_result = self._check_exit(position, df, i)
                if exit_result:
                    self._close_trade(position, exit_result)
                    position = None
                continue
            
            # Get current analysis for this bar
            analysis = self._get_bar_analysis(df, i)
            
            # Check for signals using our own logic
            signals = self._generate_signals(analysis, df, i)
            
            if signals.get('signals'):
                # Get the strongest signal
                best_signal = max(signals['signals'], key=lambda x: x.get('confidence', 0))
                
                if best_signal.get('confidence', 0) >= 60:
                    # Filter by strategy if specified
                    if strategies and best_signal.get('strategy') not in strategies:
                        continue
                    
                    # Open position
                    position = self._open_trade(best_signal, df, i)
                    entry_idx = i
        
        # Close any remaining position
        if position:
            self._close_trade(position, {
                'exit_price': df.iloc[-1]['close'],
                'exit_time': df.iloc[-1].name,
                'reason': 'END_OF_DATA'
            })
        
        # Calculate results
        results = self._calculate_results()
        
        self._print_results(results)
        
        return results
    
    def _generate_signals(self, analysis: Dict, df: pd.DataFrame, idx: int) -> Dict:
        """Generate trading signals based on analysis"""
        signals = []
        
        price = analysis['price']['close']
        rsi = analysis['rsi']['value']
        ema_trend = analysis['ema']['trend']
        macd_trend = analysis['macd']['trend']
        vwap = analysis['vwap']['value']
        vwap_pos = analysis['vwap']['position']
        bb_upper = analysis['bollinger']['upper']
        bb_lower = analysis['bollinger']['lower']
        squeeze = analysis['bollinger']['squeeze']
        volume_spike = analysis['volume']['spike']
        
        # Strategy 1: VWAP Bounce
        vwap_dist = abs(price - vwap) / analysis['atr']['value']
        if vwap_dist < 0.3:  # Price near VWAP
            if ema_trend == 'BULLISH' and price > vwap:
                signals.append({
                    'strategy': 'vwap_bounce',
                    'direction': 'LONG',
                    'confidence': 65 + (10 if volume_spike else 0)
                })
            elif ema_trend == 'BEARISH' and price < vwap:
                signals.append({
                    'strategy': 'vwap_bounce',
                    'direction': 'SHORT',
                    'confidence': 65 + (10 if volume_spike else 0)
                })
        
        # Strategy 2: RSI Reversal
        if rsi < 25:
            signals.append({
                'strategy': 'rsi_reversal',
                'direction': 'LONG',
                'confidence': 70 + (15 if macd_trend == 'BULLISH' else 0)
            })
        elif rsi > 75:
            signals.append({
                'strategy': 'rsi_reversal',
                'direction': 'SHORT',
                'confidence': 70 + (15 if macd_trend == 'BEARISH' else 0)
            })
        
        # Strategy 3: EMA Crossover
        ema_fast = analysis['ema']['ema_fast']
        ema_slow = analysis['ema']['ema_slow']
        prev_bar = df.iloc[idx-1]
        prev_ema_fast = prev_bar.get('ema_fast', ema_fast)
        prev_ema_slow = prev_bar.get('ema_slow', ema_slow)
        
        # Bullish crossover
        if prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow:
            signals.append({
                'strategy': 'ema_crossover',
                'direction': 'LONG',
                'confidence': 68 + (12 if macd_trend == 'BULLISH' else 0)
            })
        # Bearish crossover
        elif prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow:
            signals.append({
                'strategy': 'ema_crossover',
                'direction': 'SHORT',
                'confidence': 68 + (12 if macd_trend == 'BEARISH' else 0)
            })
        
        # Strategy 4: Bollinger Squeeze Breakout
        if squeeze:
            if price > bb_upper * 0.998:
                signals.append({
                    'strategy': 'bb_squeeze',
                    'direction': 'LONG',
                    'confidence': 72 + (10 if volume_spike else 0)
                })
            elif price < bb_lower * 1.002:
                signals.append({
                    'strategy': 'bb_squeeze',
                    'direction': 'SHORT',
                    'confidence': 72 + (10 if volume_spike else 0)
                })
        
        # Strategy 5: Momentum Breakout (high confidence only with volume)
        if volume_spike and ema_trend == macd_trend:
            if ema_trend == 'BULLISH' and rsi > 55 and rsi < 75:
                signals.append({
                    'strategy': 'momentum',
                    'direction': 'LONG',
                    'confidence': 75
                })
            elif ema_trend == 'BEARISH' and rsi < 45 and rsi > 25:
                signals.append({
                    'strategy': 'momentum',
                    'direction': 'SHORT',
                    'confidence': 75
                })
        
        return {'signals': signals}
    
    def _get_bar_analysis(self, df: pd.DataFrame, idx: int) -> Dict:
        """Get analysis for a specific bar"""
        bar = df.iloc[idx]
        prev_bar = df.iloc[idx-1]
        
        return {
            'price': {
                'close': bar['close'],
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'change_pct': (bar['close'] - prev_bar['close']) / prev_bar['close'] * 100
            },
            'rsi': {
                'value': bar.get('rsi', 50),
                'condition': 'OVERBOUGHT' if bar.get('rsi', 50) > 70 else 'OVERSOLD' if bar.get('rsi', 50) < 30 else 'NEUTRAL'
            },
            'ema': {
                'ema_fast': bar.get('ema_fast', bar['close']),
                'ema_slow': bar.get('ema_slow', bar['close']),
                'trend': 'BULLISH' if bar.get('ema_fast', 0) > bar.get('ema_slow', 0) else 'BEARISH'
            },
            'macd': {
                'value': bar.get('macd', 0),
                'signal': bar.get('macd_signal', 0),
                'histogram': bar.get('macd_histogram', 0),
                'trend': 'BULLISH' if bar.get('macd_histogram', 0) > 0 else 'BEARISH'
            },
            'bollinger': {
                'upper': bar.get('bb_upper', bar['close'] + 20),
                'lower': bar.get('bb_lower', bar['close'] - 20),
                'middle': bar.get('bb_middle', bar['close']),
                'squeeze': bar.get('bb_squeeze', False)
            },
            'vwap': {
                'value': bar.get('vwap', bar['close']),
                'position': 'ABOVE' if bar['close'] > bar.get('vwap', bar['close']) else 'BELOW'
            },
            'atr': {
                'value': bar.get('atr', 20),
                'pct': bar.get('atr', 20) / bar['close'] * 100
            },
            'volume': {
                'current': bar['volume'],
                'avg': df['volume'].iloc[max(0,idx-20):idx].mean(),
                'spike': bar['volume'] > df['volume'].iloc[max(0,idx-20):idx].mean() * 1.5
            }
        }
    
    def _open_trade(self, signal: Dict, df: pd.DataFrame, idx: int) -> Dict:
        """Open a new trade"""
        bar = df.iloc[idx]
        atr = bar.get('atr', 20)
        
        direction = signal.get('direction', signal.get('action', 'LONG'))
        if direction == 'BUY':
            direction = 'LONG'
        elif direction == 'SELL':
            direction = 'SHORT'
        
        entry_price = bar['close']
        
        # Calculate stop loss and target
        if direction == 'LONG':
            stop_loss = entry_price - (atr * 1.5)
            target = entry_price + (atr * 2.5)
        else:
            stop_loss = entry_price + (atr * 1.5)
            target = entry_price - (atr * 2.5)
        
        return {
            'entry_time': bar.name,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target': target,
            'strategy': signal.get('strategy', 'unknown'),
            'atr': atr
        }
    
    def _check_exit(self, position: Dict, df: pd.DataFrame, idx: int) -> Dict:
        """Check if position should be exited"""
        bar = df.iloc[idx]
        entry_idx = df.index.get_loc(position['entry_time'])
        bars_held = idx - entry_idx
        
        direction = position['direction']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        target = position['target']
        
        # Check stop loss
        if direction == 'LONG':
            if bar['low'] <= stop_loss:
                return {
                    'exit_price': stop_loss,
                    'exit_time': bar.name,
                    'reason': 'STOP_LOSS'
                }
            if bar['high'] >= target:
                return {
                    'exit_price': target,
                    'exit_time': bar.name,
                    'reason': 'TARGET'
                }
        else:  # SHORT
            if bar['high'] >= stop_loss:
                return {
                    'exit_price': stop_loss,
                    'exit_time': bar.name,
                    'reason': 'STOP_LOSS'
                }
            if bar['low'] <= target:
                return {
                    'exit_price': target,
                    'exit_time': bar.name,
                    'reason': 'TARGET'
                }
        
        # Time-based exit (max 10 candles / 50 minutes)
        if bars_held >= 10:
            return {
                'exit_price': bar['close'],
                'exit_time': bar.name,
                'reason': 'TIME_EXIT'
            }
        
        return None
    
    def _close_trade(self, position: Dict, exit_info: Dict):
        """Close trade and record results"""
        direction = position['direction']
        entry_price = position['entry_price']
        exit_price = exit_info['exit_price']
        
        # Calculate P&L
        if direction == 'LONG':
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price
        
        pnl_pct = pnl_points / entry_price * 100
        
        # Determine result
        if pnl_points > 0:
            result = 'WIN'
        elif pnl_points < 0:
            result = 'LOSS'
        else:
            result = 'BREAKEVEN'
        
        # Calculate position size based on risk
        risk_amount = self.capital * self.risk_per_trade
        risk_points = abs(entry_price - position['stop_loss'])
        lot_size = 50  # Nifty lot size
        lots = max(1, int(risk_amount / (risk_points * lot_size)))
        
        # Update capital
        trade_pnl = pnl_points * lots * lot_size
        self.capital += trade_pnl
        self.equity_curve.append(self.capital)
        
        # Record trade
        trade = Trade(
            entry_time=position['entry_time'],
            exit_time=exit_info['exit_time'],
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=position['stop_loss'],
            target=position['target'],
            strategy=position['strategy'],
            pnl_points=pnl_points,
            pnl_pct=pnl_pct,
            result=result,
            exit_reason=exit_info['reason']
        )
        self.trades.append(trade)
    
    def _calculate_results(self) -> Dict:
        """Calculate backtesting results"""
        if not self.trades:
            return {"error": "No trades executed"}
        
        # Basic counts
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.result == 'WIN')
        losses = sum(1 for t in self.trades if t.result == 'LOSS')
        breakeven = sum(1 for t in self.trades if t.result == 'BREAKEVEN')
        
        # Win rate (Accuracy)
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        
        # P&L calculations
        total_pnl_points = sum(t.pnl_points for t in self.trades)
        gross_profit = sum(t.pnl_points for t in self.trades if t.pnl_points > 0)
        gross_loss = abs(sum(t.pnl_points for t in self.trades if t.pnl_points < 0))
        
        # Profit Factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade
        avg_win = gross_profit / wins if wins > 0 else 0
        avg_loss = gross_loss / losses if losses > 0 else 0
        avg_trade = total_pnl_points / total_trades if total_trades > 0 else 0
        
        # Risk-Reward achieved
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
        
        # Drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = np.max(drawdown)
        
        # Sharpe Ratio (simplified)
        returns = np.diff(equity) / equity[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 75) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # By strategy
        strategy_stats = {}
        for strategy in set(t.strategy for t in self.trades):
            strat_trades = [t for t in self.trades if t.strategy == strategy]
            strat_wins = sum(1 for t in strat_trades if t.result == 'WIN')
            strategy_stats[strategy] = {
                'trades': len(strat_trades),
                'wins': strat_wins,
                'win_rate': strat_wins / len(strat_trades) * 100 if strat_trades else 0,
                'total_pnl': sum(t.pnl_points for t in strat_trades)
            }
        
        # By exit reason
        exit_stats = {}
        for reason in set(t.exit_reason for t in self.trades):
            reason_trades = [t for t in self.trades if t.exit_reason == reason]
            exit_stats[reason] = {
                'count': len(reason_trades),
                'pct': len(reason_trades) / total_trades * 100
            }
        
        return {
            'summary': {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'breakeven': breakeven,
                'win_rate': round(win_rate, 2),
                'accuracy': f"{win_rate:.1f}%"
            },
            'pnl': {
                'total_pnl_points': round(total_pnl_points, 2),
                'gross_profit': round(gross_profit, 2),
                'gross_loss': round(gross_loss, 2),
                'net_pnl_inr': round(self.capital - self.initial_capital, 2),
                'return_pct': round((self.capital - self.initial_capital) / self.initial_capital * 100, 2)
            },
            'ratios': {
                'profit_factor': round(profit_factor, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'avg_trade': round(avg_trade, 2),
                'risk_reward_achieved': round(avg_rr, 2),
                'expectancy': round(expectancy, 2),
                'sharpe_ratio': round(sharpe_ratio, 2)
            },
            'risk': {
                'max_drawdown_pct': round(max_drawdown, 2),
                'final_capital': round(self.capital, 2),
                'initial_capital': self.initial_capital
            },
            'by_strategy': strategy_stats,
            'by_exit_reason': exit_stats,
            'equity_curve': self.equity_curve
        }
    
    def _print_results(self, results: Dict):
        """Print formatted results"""
        print(f"\n{'='*60}")
        print("📊 BACKTEST RESULTS")
        print(f"{'='*60}")
        
        summary = results.get('summary', {})
        print(f"\n📈 TRADE SUMMARY:")
        print(f"   Total Trades: {summary.get('total_trades', 0)}")
        print(f"   Wins: {summary.get('wins', 0)} | Losses: {summary.get('losses', 0)} | BE: {summary.get('breakeven', 0)}")
        print(f"   ✅ WIN RATE (ACCURACY): {summary.get('accuracy', '0%')}")
        
        pnl = results.get('pnl', {})
        print(f"\n💰 P&L:")
        print(f"   Total Points: {pnl.get('total_pnl_points', 0)}")
        print(f"   Net P&L: ₹{pnl.get('net_pnl_inr', 0):,.0f}")
        print(f"   Return: {pnl.get('return_pct', 0)}%")
        
        ratios = results.get('ratios', {})
        print(f"\n📊 KEY RATIOS:")
        print(f"   Profit Factor: {ratios.get('profit_factor', 0)}")
        print(f"   Risk-Reward Achieved: {ratios.get('risk_reward_achieved', 0)}")
        print(f"   Expectancy: {ratios.get('expectancy', 0)} pts/trade")
        print(f"   Sharpe Ratio: {ratios.get('sharpe_ratio', 0)}")
        
        risk = results.get('risk', {})
        print(f"\n⚠️ RISK METRICS:")
        print(f"   Max Drawdown: {risk.get('max_drawdown_pct', 0)}%")
        print(f"   Final Capital: ₹{risk.get('final_capital', 0):,.0f}")
        
        print(f"\n📋 BY STRATEGY:")
        for strategy, stats in results.get('by_strategy', {}).items():
            print(f"   {strategy}: {stats['trades']} trades, {stats['win_rate']:.1f}% win rate, {stats['total_pnl']:.0f} pts")
        
        print(f"\n{'='*60}")
    
    def get_trade_log(self) -> List[Dict]:
        """Get detailed trade log"""
        return [
            {
                'entry_time': str(t.entry_time),
                'exit_time': str(t.exit_time),
                'direction': t.direction,
                'strategy': t.strategy,
                'entry': t.entry_price,
                'exit': t.exit_price,
                'pnl_points': t.pnl_points,
                'result': t.result,
                'exit_reason': t.exit_reason
            }
            for t in self.trades
        ]


def run_accuracy_test(days: int = 30) -> Dict:
    """Quick function to test accuracy"""
    backtester = Backtester()
    return backtester.run_backtest(days=days)


if __name__ == "__main__":
    # Run backtest
    results = run_accuracy_test(days=30)
    
    # Save results
    with open('backtest_results.json', 'w') as f:
        # Convert non-serializable items
        save_results = {k: v for k, v in results.items() if k != 'equity_curve'}
        json.dump(save_results, f, indent=2, default=str)
    
    print("\n✅ Results saved to backtest_results.json")
