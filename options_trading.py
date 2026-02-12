"""
Options Trading Module for Nifty 50 Scalping Agent

Supports:
- ATM/ITM/OTM strike selection
- Call (CE) and Put (PE) trading
- Options Greeks calculation
- Options scalping strategies
- Paper trading for options
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import yfinance as yf

import config


class OptionType(Enum):
    CALL = "CE"  # Call Option
    PUT = "PE"   # Put Option


class StrikeType(Enum):
    ATM = "ATM"      # At The Money
    ITM = "ITM"      # In The Money
    OTM = "OTM"      # Out of The Money
    ITM_1 = "ITM_1"  # 1 strike ITM
    ITM_2 = "ITM_2"  # 2 strikes ITM
    OTM_1 = "OTM_1"  # 1 strike OTM
    OTM_2 = "OTM_2"  # 2 strikes OTM


@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float  # Implied Volatility


@dataclass
class OptionContract:
    symbol: str
    strike: float
    option_type: OptionType
    expiry: datetime
    spot_price: float
    ltp: float  # Last Traded Price
    bid: float
    ask: float
    volume: int
    oi: int  # Open Interest
    greeks: Optional[OptionGreeks] = None
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'strike': self.strike,
            'option_type': self.option_type.value,
            'expiry': self.expiry.strftime('%Y-%m-%d'),
            'spot_price': self.spot_price,
            'ltp': self.ltp,
            'bid': self.bid,
            'ask': self.ask,
            'volume': self.volume,
            'oi': self.oi,
            'greeks': {
                'delta': self.greeks.delta,
                'gamma': self.greeks.gamma,
                'theta': self.greeks.theta,
                'vega': self.greeks.vega,
                'iv': self.greeks.iv
            } if self.greeks else None
        }


@dataclass
class OptionPosition:
    id: str
    contract: OptionContract
    quantity: int  # Lot size (Nifty = 25)
    entry_price: float
    current_price: float
    direction: str  # BUY or SELL
    strategy: str
    entry_time: str
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    status: str = "OPEN"
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'contract': self.contract.to_dict(),
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'direction': self.direction,
            'strategy': self.strategy,
            'entry_time': self.entry_time,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'status': self.status
        }


class BlackScholes:
    """Black-Scholes Option Pricing Model for Greeks calculation"""
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter"""
        if T <= 0 or sigma <= 0:
            return 0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter"""
        if T <= 0 or sigma <= 0:
            return 0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)
    
    @staticmethod
    def norm_cdf(x: float) -> float:
        """Cumulative distribution function for standard normal"""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    @staticmethod
    def norm_pdf(x: float) -> float:
        """Probability density function for standard normal"""
        return math.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Call option price"""
        if T <= 0:
            return max(0, S - K)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * BlackScholes.norm_cdf(d1) - K * math.exp(-r * T) * BlackScholes.norm_cdf(d2)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Put option price"""
        if T <= 0:
            return max(0, K - S)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * math.exp(-r * T) * BlackScholes.norm_cdf(-d2) - S * BlackScholes.norm_cdf(-d1)
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                         option_type: OptionType) -> OptionGreeks:
        """Calculate all Greeks for an option"""
        if T <= 0:
            T = 0.0001  # Small time to avoid division by zero
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        # Delta
        if option_type == OptionType.CALL:
            delta = BlackScholes.norm_cdf(d1)
        else:
            delta = BlackScholes.norm_cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = BlackScholes.norm_pdf(d1) / (S * sigma * math.sqrt(T))
        
        # Theta (per day)
        if option_type == OptionType.CALL:
            theta = (-(S * BlackScholes.norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) 
                    - r * K * math.exp(-r * T) * BlackScholes.norm_cdf(d2)) / 365
        else:
            theta = (-(S * BlackScholes.norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) 
                    + r * K * math.exp(-r * T) * BlackScholes.norm_cdf(-d2)) / 365
        
        # Vega (per 1% change in IV)
        vega = S * math.sqrt(T) * BlackScholes.norm_pdf(d1) / 100
        
        return OptionGreeks(
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 2),
            vega=round(vega, 2),
            iv=round(sigma * 100, 2)  # Convert to percentage
        )
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, 
                          r: float, option_type: OptionType, 
                          max_iterations: int = 100, tolerance: float = 0.0001) -> float:
        """Calculate Implied Volatility using Newton-Raphson method"""
        sigma = 0.3  # Initial guess (30%)
        
        for _ in range(max_iterations):
            if option_type == OptionType.CALL:
                price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholes.put_price(S, K, T, r, sigma)
            
            diff = price - market_price
            
            if abs(diff) < tolerance:
                return sigma
            
            # Vega for Newton-Raphson
            d1 = BlackScholes.d1(S, K, T, r, sigma)
            vega = S * math.sqrt(T) * BlackScholes.norm_pdf(d1)
            
            if vega < 0.0001:
                break
            
            sigma = sigma - diff / vega
            
            # Keep sigma in reasonable bounds
            sigma = max(0.01, min(sigma, 5.0))
        
        return sigma


class NiftyOptionsChain:
    """Fetches and manages Nifty 50 Options Chain"""
    
    NIFTY_LOT_SIZE = 25
    STRIKE_INTERVAL = 50  # Nifty strikes are in multiples of 50
    
    def __init__(self):
        self.spot_price = 0.0
        self.options_data: Dict[str, OptionContract] = {}
        self.expiry_dates: List[datetime] = []
        self.risk_free_rate = 0.07  # 7% annual rate
        
    def get_nearest_expiry(self) -> datetime:
        """Get nearest Thursday (weekly expiry)"""
        today = datetime.now()
        days_until_thursday = (3 - today.weekday()) % 7
        if days_until_thursday == 0 and today.hour >= 15:
            days_until_thursday = 7
        nearest_expiry = today + timedelta(days=days_until_thursday)
        return nearest_expiry.replace(hour=15, minute=30, second=0, microsecond=0)
    
    def get_atm_strike(self, spot_price: float) -> float:
        """Get ATM strike price"""
        return round(spot_price / self.STRIKE_INTERVAL) * self.STRIKE_INTERVAL
    
    def get_strike_range(self, spot_price: float, num_strikes: int = 10) -> List[float]:
        """Get range of strikes around ATM"""
        atm = self.get_atm_strike(spot_price)
        strikes = []
        for i in range(-num_strikes, num_strikes + 1):
            strikes.append(atm + (i * self.STRIKE_INTERVAL))
        return strikes
    
    def fetch_spot_price(self) -> float:
        """Fetch current Nifty 50 spot price"""
        try:
            ticker = yf.Ticker(config.NIFTY_SYMBOL)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                self.spot_price = data['Close'].iloc[-1]
            return self.spot_price
        except Exception as e:
            print(f"Error fetching spot price: {e}")
            return self.spot_price
    
    def simulate_option_price(self, strike: float, option_type: OptionType, 
                              spot_price: float, days_to_expiry: float,
                              iv: float = 0.15) -> Tuple[float, OptionGreeks]:
        """
        Simulate option price using Black-Scholes
        (Use this when real options data is not available)
        """
        T = days_to_expiry / 365.0
        r = self.risk_free_rate
        
        if option_type == OptionType.CALL:
            price = BlackScholes.call_price(spot_price, strike, T, r, iv)
        else:
            price = BlackScholes.put_price(spot_price, strike, T, r, iv)
        
        greeks = BlackScholes.calculate_greeks(spot_price, strike, T, r, iv, option_type)
        
        return max(0.05, round(price, 2)), greeks
    
    def get_option_chain(self, spot_price: float = None) -> Dict[str, List[OptionContract]]:
        """
        Get simulated options chain
        Returns dict with 'calls' and 'puts' lists
        """
        if spot_price is None:
            spot_price = self.fetch_spot_price()
        
        if spot_price <= 0:
            spot_price = 25800  # Default fallback
        
        self.spot_price = spot_price
        expiry = self.get_nearest_expiry()
        days_to_expiry = max(0.1, (expiry - datetime.now()).total_seconds() / 86400)
        
        strikes = self.get_strike_range(spot_price, num_strikes=10)
        
        calls = []
        puts = []
        
        # Base IV varies by moneyness
        atm_strike = self.get_atm_strike(spot_price)
        
        for strike in strikes:
            moneyness = abs(strike - spot_price) / spot_price
            # IV smile - higher IV for OTM options
            iv = 0.12 + (moneyness * 0.5)
            
            # Call option
            call_price, call_greeks = self.simulate_option_price(
                strike, OptionType.CALL, spot_price, days_to_expiry, iv
            )
            
            calls.append(OptionContract(
                symbol=f"NIFTY{expiry.strftime('%d%b').upper()}{int(strike)}CE",
                strike=strike,
                option_type=OptionType.CALL,
                expiry=expiry,
                spot_price=spot_price,
                ltp=call_price,
                bid=round(call_price * 0.99, 2),
                ask=round(call_price * 1.01, 2),
                volume=int(np.random.randint(10000, 500000)),
                oi=int(np.random.randint(100000, 5000000)),
                greeks=call_greeks
            ))
            
            # Put option
            put_price, put_greeks = self.simulate_option_price(
                strike, OptionType.PUT, spot_price, days_to_expiry, iv
            )
            
            puts.append(OptionContract(
                symbol=f"NIFTY{expiry.strftime('%d%b').upper()}{int(strike)}PE",
                strike=strike,
                option_type=OptionType.PUT,
                expiry=expiry,
                spot_price=spot_price,
                ltp=put_price,
                bid=round(put_price * 0.99, 2),
                ask=round(put_price * 1.01, 2),
                volume=int(np.random.randint(10000, 500000)),
                oi=int(np.random.randint(100000, 5000000)),
                greeks=put_greeks
            ))
        
        return {'calls': calls, 'puts': puts, 'spot': spot_price, 'expiry': expiry}
    
    def get_option_by_strike(self, strike: float, option_type: OptionType,
                             chain: Dict = None) -> Optional[OptionContract]:
        """Get specific option contract"""
        if chain is None:
            chain = self.get_option_chain()
        
        options = chain['calls'] if option_type == OptionType.CALL else chain['puts']
        
        for opt in options:
            if opt.strike == strike:
                return opt
        return None
    
    def get_atm_options(self, chain: Dict = None) -> Tuple[OptionContract, OptionContract]:
        """Get ATM Call and Put"""
        if chain is None:
            chain = self.get_option_chain()
        
        atm_strike = self.get_atm_strike(chain['spot'])
        
        atm_call = self.get_option_by_strike(atm_strike, OptionType.CALL, chain)
        atm_put = self.get_option_by_strike(atm_strike, OptionType.PUT, chain)
        
        return atm_call, atm_put


class OptionsScalpingStrategy:
    """
    Options Scalping Strategies for Nifty 50
    
    Strategies:
    1. Momentum CE/PE - Buy calls on bullish signals, puts on bearish
    2. Straddle Scalp - Buy ATM straddle on high volatility
    3. Directional Scalp - Buy OTM options in trending market
    4. Reversal Play - Buy options at support/resistance
    """
    
    def __init__(self, options_chain: NiftyOptionsChain):
        self.options_chain = options_chain
        self.lot_size = NiftyOptionsChain.NIFTY_LOT_SIZE
    
    def momentum_signal(self, market_bias: str, confidence: int,
                        chain: Dict) -> Optional[Dict]:
        """
        Momentum-based options trading
        Buy CE on bullish, PE on bearish
        """
        if confidence < 65:
            return None
        
        spot = chain['spot']
        atm_strike = self.options_chain.get_atm_strike(spot)
        
        if "BULLISH" in market_bias:
            # Buy ATM or slightly OTM Call
            strike = atm_strike if confidence >= 75 else atm_strike + 50
            option = self.options_chain.get_option_by_strike(strike, OptionType.CALL, chain)
            
            if option:
                return {
                    'strategy': 'MOMENTUM_CE',
                    'action': 'BUY',
                    'option': option,
                    'quantity': self.lot_size,
                    'target_pct': 20,  # 20% profit target
                    'stoploss_pct': 30,  # 30% stop loss
                    'reasoning': f"Bullish momentum ({market_bias}). Buying {strike}CE"
                }
        
        elif "BEARISH" in market_bias:
            # Buy ATM or slightly OTM Put
            strike = atm_strike if confidence >= 75 else atm_strike - 50
            option = self.options_chain.get_option_by_strike(strike, OptionType.PUT, chain)
            
            if option:
                return {
                    'strategy': 'MOMENTUM_PE',
                    'action': 'BUY',
                    'option': option,
                    'quantity': self.lot_size,
                    'target_pct': 20,
                    'stoploss_pct': 30,
                    'reasoning': f"Bearish momentum ({market_bias}). Buying {strike}PE"
                }
        
        return None
    
    def breakout_signal(self, price_change_pct: float, volume_spike: bool,
                        chain: Dict) -> Optional[Dict]:
        """
        Breakout-based options trading
        Buy options on strong price movement with volume
        """
        if abs(price_change_pct) < 0.3 or not volume_spike:
            return None
        
        spot = chain['spot']
        atm_strike = self.options_chain.get_atm_strike(spot)
        
        if price_change_pct > 0.3:
            # Bullish breakout - buy OTM Call
            strike = atm_strike + 100
            option = self.options_chain.get_option_by_strike(strike, OptionType.CALL, chain)
            
            if option:
                return {
                    'strategy': 'BREAKOUT_CE',
                    'action': 'BUY',
                    'option': option,
                    'quantity': self.lot_size,
                    'target_pct': 30,
                    'stoploss_pct': 40,
                    'reasoning': f"Bullish breakout (+{price_change_pct:.2f}%). Buying OTM {strike}CE"
                }
        
        elif price_change_pct < -0.3:
            # Bearish breakout - buy OTM Put
            strike = atm_strike - 100
            option = self.options_chain.get_option_by_strike(strike, OptionType.PUT, chain)
            
            if option:
                return {
                    'strategy': 'BREAKOUT_PE',
                    'action': 'BUY',
                    'option': option,
                    'quantity': self.lot_size,
                    'target_pct': 30,
                    'stoploss_pct': 40,
                    'reasoning': f"Bearish breakout ({price_change_pct:.2f}%). Buying OTM {strike}PE"
                }
        
        return None
    
    def scalp_reversal_signal(self, rsi: float, at_support: bool, at_resistance: bool,
                              chain: Dict) -> Optional[Dict]:
        """
        Reversal-based options scalping
        Buy options at extreme RSI levels
        """
        spot = chain['spot']
        atm_strike = self.options_chain.get_atm_strike(spot)
        
        if rsi < 30 and at_support:
            # Oversold at support - buy Call
            option = self.options_chain.get_option_by_strike(atm_strike, OptionType.CALL, chain)
            
            if option:
                return {
                    'strategy': 'REVERSAL_CE',
                    'action': 'BUY',
                    'option': option,
                    'quantity': self.lot_size,
                    'target_pct': 15,
                    'stoploss_pct': 25,
                    'reasoning': f"Oversold reversal (RSI={rsi:.1f}). Buying ATM {atm_strike}CE"
                }
        
        elif rsi > 70 and at_resistance:
            # Overbought at resistance - buy Put
            option = self.options_chain.get_option_by_strike(atm_strike, OptionType.PUT, chain)
            
            if option:
                return {
                    'strategy': 'REVERSAL_PE',
                    'action': 'BUY',
                    'option': option,
                    'quantity': self.lot_size,
                    'target_pct': 15,
                    'stoploss_pct': 25,
                    'reasoning': f"Overbought reversal (RSI={rsi:.1f}). Buying ATM {atm_strike}PE"
                }
        
        return None


class OptionsPaperTrading:
    """Paper Trading Engine for Options"""
    
    def __init__(self, initial_capital: float = 50000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, OptionPosition] = {}
        self.trade_history: List[Dict] = []
        self.options_chain = NiftyOptionsChain()
        self.strategy = OptionsScalpingStrategy(self.options_chain)
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
    
    def get_margin_required(self, option: OptionContract, quantity: int, 
                           direction: str) -> float:
        """Calculate margin required for option trade"""
        if direction == "BUY":
            # For buying options, margin = premium paid
            return option.ltp * quantity
        else:
            # For selling options, higher margin (simplified)
            return option.ltp * quantity * 3
    
    def open_position(self, signal: Dict) -> Optional[OptionPosition]:
        """Open a new options position"""
        import uuid
        
        option = signal['option']
        quantity = signal['quantity']
        direction = signal['action']
        
        margin_required = self.get_margin_required(option, quantity, direction)
        
        if margin_required > self.current_capital:
            print(f"⚠️ Insufficient capital. Required: ₹{margin_required:,.2f}")
            return None
        
        position_id = str(uuid.uuid4())[:8]
        
        position = OptionPosition(
            id=position_id,
            contract=option,
            quantity=quantity,
            entry_price=option.ltp,
            current_price=option.ltp,
            direction=direction,
            strategy=signal['strategy'],
            entry_time=datetime.now().isoformat()
        )
        
        self.positions[position_id] = position
        self.current_capital -= margin_required
        
        print(f"✅ OPENED {direction} {option.symbol} @ ₹{option.ltp:.2f} x {quantity}")
        print(f"   Strategy: {signal['strategy']}")
        print(f"   Greeks: Δ={option.greeks.delta:.3f} Θ={option.greeks.theta:.2f}")
        
        return position
    
    def update_positions(self, new_spot_price: float):
        """Update all position prices based on new spot"""
        chain = self.options_chain.get_option_chain(new_spot_price)
        
        for pos_id, position in self.positions.items():
            if position.status != "OPEN":
                continue
            
            # Get updated option price
            updated_option = self.options_chain.get_option_by_strike(
                position.contract.strike,
                position.contract.option_type,
                chain
            )
            
            if updated_option:
                position.current_price = updated_option.ltp
                position.contract = updated_option
                
                # Calculate P&L
                if position.direction == "BUY":
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity
    
    def close_position(self, position_id: str, reason: str) -> Optional[Dict]:
        """Close an options position"""
        if position_id not in self.positions:
            return None
        
        position = self.positions[position_id]
        
        if position.status != "OPEN":
            return None
        
        # Calculate final P&L
        if position.direction == "BUY":
            pnl = (position.current_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - position.current_price) * position.quantity
        
        # Update position
        position.status = "CLOSED"
        position.exit_price = position.current_price
        position.exit_time = datetime.now().isoformat()
        position.exit_reason = reason
        position.realized_pnl = pnl
        
        # Update capital
        if position.direction == "BUY":
            self.current_capital += position.current_price * position.quantity
        else:
            margin_return = position.entry_price * position.quantity * 3
            self.current_capital += margin_return + pnl
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            result = "WIN"
        else:
            self.losing_trades += 1
            result = "LOSS"
        
        trade_record = {
            'id': position.id,
            'symbol': position.contract.symbol,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': position.exit_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'pnl_pct': (pnl / (position.entry_price * position.quantity)) * 100,
            'strategy': position.strategy,
            'exit_reason': reason,
            'result': result
        }
        
        self.trade_history.append(trade_record)
        del self.positions[position_id]
        
        emoji = "🟢" if pnl >= 0 else "🔴"
        print(f"{emoji} CLOSED {position.contract.symbol} @ ₹{position.exit_price:.2f}")
        print(f"   P&L: ₹{pnl:+,.2f} | Reason: {reason}")
        
        return trade_record
    
    def check_exit_conditions(self, target_pct: float = 20, stoploss_pct: float = 30):
        """Check if any position should be closed"""
        positions_to_close = []
        
        for pos_id, position in self.positions.items():
            if position.status != "OPEN":
                continue
            
            pnl_pct = (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100
            
            if pnl_pct >= target_pct:
                positions_to_close.append((pos_id, "TARGET_HIT"))
            elif pnl_pct <= -stoploss_pct:
                positions_to_close.append((pos_id, "STOPLOSS_HIT"))
        
        for pos_id, reason in positions_to_close:
            self.close_position(pos_id, reason)
    
    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': round(self.current_capital, 2),
            'total_pnl': round(self.total_pnl, 2),
            'total_pnl_pct': round((self.total_pnl / self.initial_capital) * 100, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(win_rate, 1),
            'open_positions': len([p for p in self.positions.values() if p.status == "OPEN"])
        }
    
    def get_open_positions(self) -> List[OptionPosition]:
        """Get all open positions"""
        return [p for p in self.positions.values() if p.status == "OPEN"]


def print_options_chain(chain: Dict):
    """Pretty print options chain"""
    print(f"\n{'='*80}")
    print(f"📊 NIFTY OPTIONS CHAIN | Spot: {chain['spot']:.2f} | Expiry: {chain['expiry'].strftime('%d-%b-%Y')}")
    print(f"{'='*80}")
    print(f"{'CALLS':<40} | {'PUTS':>40}")
    print(f"{'Strike':>8} {'LTP':>8} {'Delta':>7} {'IV':>6} | {'Strike':>8} {'LTP':>8} {'Delta':>7} {'IV':>6}")
    print("-" * 80)
    
    for call, put in zip(chain['calls'], chain['puts']):
        atm_marker = " ←ATM" if abs(call.strike - chain['spot']) < 25 else ""
        print(f"{call.strike:>8.0f} {call.ltp:>8.2f} {call.greeks.delta:>7.3f} {call.greeks.iv:>5.1f}% | "
              f"{put.strike:>8.0f} {put.ltp:>8.2f} {put.greeks.delta:>7.3f} {put.greeks.iv:>5.1f}%{atm_marker}")
    
    print("=" * 80)
