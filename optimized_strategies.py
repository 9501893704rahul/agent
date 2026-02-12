"""
Optimized Scalping Strategies for Nifty 50

Based on Accuracy Analysis:
- Increased confidence threshold (75%)
- Focus on profitable strategies only
- Prefer LONG trades (50% win rate vs SHORT 20%)
- Wider stop-loss to avoid premature exits
- Better risk:reward ratio (1:2 minimum)
- Market regime filter
- Time-based filters
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class OptimizedStrategy(Enum):
    PULLBACK_ENTRY = "Pullback Entry"      # Best performer (50% WR)
    TREND_FOLLOWING = "Trend Following"     # New optimized strategy
    VWAP_BOUNCE = "VWAP Bounce"            # Keep but optimize
    MOMENTUM_BREAKOUT = "Momentum Breakout" # Keep but optimize


@dataclass
class OptimizedTradeSetup:
    strategy: OptimizedStrategy
    direction: str  # LONG or SHORT
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    confidence: int
    risk_reward: float
    reasoning: str


class MarketRegime:
    """Detect market regime for filtering trades"""
    
    @staticmethod
    def detect(data: pd.DataFrame) -> dict:
        """
        Detect current market regime
        Returns: trending_up, trending_down, ranging, volatile
        """
        if len(data) < 20:
            return {'regime': 'unknown', 'strength': 0}
        
        # Calculate metrics
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Trend detection using EMA
        ema_9 = close.ewm(span=9).mean()
        ema_21 = close.ewm(span=21).mean()
        ema_50 = close.ewm(span=50).mean() if len(data) >= 50 else ema_21
        
        current_price = close.iloc[-1]
        ema_9_val = ema_9.iloc[-1]
        ema_21_val = ema_21.iloc[-1]
        ema_50_val = ema_50.iloc[-1]
        
        # ADX for trend strength
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Volatility
        returns = close.pct_change()
        volatility = returns.rolling(20).std().iloc[-1] * 100
        
        # Determine regime
        if current_price > ema_9_val > ema_21_val > ema_50_val:
            regime = 'STRONG_UPTREND'
            strength = 90
        elif current_price > ema_21_val and ema_9_val > ema_21_val:
            regime = 'UPTREND'
            strength = 70
        elif current_price < ema_9_val < ema_21_val < ema_50_val:
            regime = 'STRONG_DOWNTREND'
            strength = 90
        elif current_price < ema_21_val and ema_9_val < ema_21_val:
            regime = 'DOWNTREND'
            strength = 70
        else:
            regime = 'RANGING'
            strength = 40
        
        # Check volatility
        is_volatile = volatility > 0.5
        
        return {
            'regime': regime,
            'strength': strength,
            'volatility': volatility,
            'is_volatile': is_volatile,
            'atr': atr,
            'trend_aligned': regime in ['UPTREND', 'STRONG_UPTREND', 'DOWNTREND', 'STRONG_DOWNTREND']
        }


class OptimizedStrategyEngine:
    """
    Optimized Strategy Engine with improved accuracy
    
    Key Optimizations:
    1. Higher confidence threshold (75%)
    2. Only use profitable strategies
    3. Prefer LONG trades
    4. Wider stops (0.4% instead of 0.08%)
    5. Better R:R (1:2 minimum)
    6. Market regime filter
    7. Time filter (avoid first/last 15 min)
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.current_price = data['close'].iloc[-1]
        self.current_time = data.index[-1]
        
        # Calculate indicators
        self._calculate_indicators()
        
        # Get market regime
        self.regime = MarketRegime.detect(data)
        
        # Optimized settings
        self.min_confidence = 75
        self.stop_loss_pct = 0.35  # 0.35% stop loss (wider)
        self.min_risk_reward = 2.0  # Minimum 1:2 R:R
        self.prefer_long = True  # LONG has better win rate
    
    def _calculate_indicators(self):
        """Calculate all required indicators"""
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        volume = self.data['volume'] if 'volume' in self.data else pd.Series([1]*len(close))
        
        # EMAs
        self.ema_9 = close.ewm(span=9).mean()
        self.ema_21 = close.ewm(span=21).mean()
        self.ema_50 = close.ewm(span=50).mean() if len(close) >= 50 else self.ema_21
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.rsi = 100 - (100 / (1 + rs))
        
        # VWAP
        typical_price = (high + low + close) / 3
        self.vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        self.atr = tr.rolling(14).mean()
        
        # Bollinger Bands
        self.bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        self.bb_upper = self.bb_middle + (bb_std * 2)
        self.bb_lower = self.bb_middle - (bb_std * 2)
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        self.macd = ema_12 - ema_26
        self.macd_signal = self.macd.ewm(span=9).mean()
        self.macd_histogram = self.macd - self.macd_signal
        
        # Stochastic
        lowest_14 = low.rolling(14).min()
        highest_14 = high.rolling(14).max()
        self.stoch_k = ((close - lowest_14) / (highest_14 - lowest_14)) * 100
        self.stoch_d = self.stoch_k.rolling(3).mean()
    
    def _is_valid_time(self) -> bool:
        """Check if current time is valid for trading"""
        time_str = self.current_time.strftime('%H:%M')
        
        # Avoid first 15 minutes (high volatility, unpredictable)
        if time_str < '09:30':
            return False
        
        # Avoid last 15 minutes (market close effects)
        if time_str > '15:15':
            return False
        
        # Avoid lunch hour (low liquidity)
        # if '12:30' <= time_str <= '13:30':
        #     return False
        
        return True
    
    def _calculate_stop_loss(self, direction: str) -> float:
        """Calculate optimized stop loss"""
        atr = self.atr.iloc[-1] if not pd.isna(self.atr.iloc[-1]) else self.current_price * 0.003
        
        # Use ATR-based stop or minimum 0.35%
        atr_stop = atr * 1.5
        pct_stop = self.current_price * (self.stop_loss_pct / 100)
        
        stop_distance = max(atr_stop, pct_stop)
        
        if direction == 'LONG':
            return round(self.current_price - stop_distance, 2)
        else:
            return round(self.current_price + stop_distance, 2)
    
    def _calculate_targets(self, entry: float, stop_loss: float, direction: str) -> tuple:
        """Calculate targets with minimum 1:2 R:R"""
        risk = abs(entry - stop_loss)
        
        if direction == 'LONG':
            target_1 = entry + (risk * self.min_risk_reward)
            target_2 = entry + (risk * 3)
        else:
            target_1 = entry - (risk * self.min_risk_reward)
            target_2 = entry - (risk * 3)
        
        return round(target_1, 2), round(target_2, 2)
    
    def pullback_entry_strategy(self) -> Optional[OptimizedTradeSetup]:
        """
        Optimized Pullback Entry - Best performing strategy
        
        LONG: Price pulls back to EMA in uptrend
        SHORT: Price pulls back to EMA in downtrend (reduced frequency)
        """
        if len(self.data) < 30:
            return None
        
        price = self.current_price
        ema_9 = self.ema_9.iloc[-1]
        ema_21 = self.ema_21.iloc[-1]
        rsi = self.rsi.iloc[-1]
        
        # LONG Setup (preferred)
        if self.regime['regime'] in ['UPTREND', 'STRONG_UPTREND']:
            # Price near EMA 21 (within 0.2%)
            distance_to_ema = abs(price - ema_21) / price * 100
            
            if distance_to_ema < 0.2 and price > ema_21:
                if ema_9 > ema_21 and 40 < rsi < 65:
                    stop_loss = self._calculate_stop_loss('LONG')
                    target_1, target_2 = self._calculate_targets(price, stop_loss, 'LONG')
                    
                    confidence = 80 if self.regime['strength'] > 70 else 75
                    
                    return OptimizedTradeSetup(
                        strategy=OptimizedStrategy.PULLBACK_ENTRY,
                        direction='LONG',
                        entry_price=price,
                        stop_loss=stop_loss,
                        target_1=target_1,
                        target_2=target_2,
                        confidence=confidence,
                        risk_reward=self.min_risk_reward,
                        reasoning=f"Pullback to EMA21 in {self.regime['regime']}"
                    )
        
        # SHORT Setup (only in strong downtrend)
        if self.regime['regime'] == 'STRONG_DOWNTREND' and not self.prefer_long:
            distance_to_ema = abs(price - ema_21) / price * 100
            
            if distance_to_ema < 0.2 and price < ema_21:
                if ema_9 < ema_21 and 35 < rsi < 60:
                    stop_loss = self._calculate_stop_loss('SHORT')
                    target_1, target_2 = self._calculate_targets(price, stop_loss, 'SHORT')
                    
                    return OptimizedTradeSetup(
                        strategy=OptimizedStrategy.PULLBACK_ENTRY,
                        direction='SHORT',
                        entry_price=price,
                        stop_loss=stop_loss,
                        target_1=target_1,
                        target_2=target_2,
                        confidence=75,
                        risk_reward=self.min_risk_reward,
                        reasoning=f"Pullback to EMA21 in STRONG_DOWNTREND"
                    )
        
        return None
    
    def trend_following_strategy(self) -> Optional[OptimizedTradeSetup]:
        """
        New Trend Following Strategy
        
        Enter in direction of trend when momentum confirms
        """
        if len(self.data) < 30:
            return None
        
        price = self.current_price
        ema_9 = self.ema_9.iloc[-1]
        ema_21 = self.ema_21.iloc[-1]
        macd = self.macd.iloc[-1]
        macd_signal = self.macd_signal.iloc[-1]
        macd_prev = self.macd.iloc[-2]
        rsi = self.rsi.iloc[-1]
        
        # LONG: Uptrend with MACD crossover
        if self.regime['regime'] in ['UPTREND', 'STRONG_UPTREND']:
            # MACD just crossed above signal
            if macd > macd_signal and macd_prev <= self.macd_signal.iloc[-2]:
                if price > ema_21 and 45 < rsi < 70:
                    stop_loss = self._calculate_stop_loss('LONG')
                    target_1, target_2 = self._calculate_targets(price, stop_loss, 'LONG')
                    
                    confidence = 80 if self.regime['strength'] > 80 else 75
                    
                    return OptimizedTradeSetup(
                        strategy=OptimizedStrategy.TREND_FOLLOWING,
                        direction='LONG',
                        entry_price=price,
                        stop_loss=stop_loss,
                        target_1=target_1,
                        target_2=target_2,
                        confidence=confidence,
                        risk_reward=self.min_risk_reward,
                        reasoning=f"MACD crossover in {self.regime['regime']}"
                    )
        
        # SHORT: Only in strong downtrend
        if self.regime['regime'] == 'STRONG_DOWNTREND' and not self.prefer_long:
            if macd < macd_signal and macd_prev >= self.macd_signal.iloc[-2]:
                if price < ema_21 and 30 < rsi < 55:
                    stop_loss = self._calculate_stop_loss('SHORT')
                    target_1, target_2 = self._calculate_targets(price, stop_loss, 'SHORT')
                    
                    return OptimizedTradeSetup(
                        strategy=OptimizedStrategy.TREND_FOLLOWING,
                        direction='SHORT',
                        entry_price=price,
                        stop_loss=stop_loss,
                        target_1=target_1,
                        target_2=target_2,
                        confidence=75,
                        risk_reward=self.min_risk_reward,
                        reasoning="MACD crossover in STRONG_DOWNTREND"
                    )
        
        return None
    
    def vwap_bounce_strategy(self) -> Optional[OptimizedTradeSetup]:
        """
        Optimized VWAP Bounce Strategy
        
        LONG: Price bounces off VWAP from below in uptrend
        """
        if len(self.data) < 30:
            return None
        
        price = self.current_price
        vwap = self.vwap.iloc[-1]
        prev_price = self.data['close'].iloc[-2]
        rsi = self.rsi.iloc[-1]
        
        # Only LONG (better accuracy)
        if self.regime['regime'] in ['UPTREND', 'STRONG_UPTREND', 'RANGING']:
            # Price crossed above VWAP
            if prev_price < vwap and price > vwap:
                distance = abs(price - vwap) / price * 100
                if distance < 0.15 and 40 < rsi < 65:
                    stop_loss = self._calculate_stop_loss('LONG')
                    target_1, target_2 = self._calculate_targets(price, stop_loss, 'LONG')
                    
                    confidence = 78 if self.regime['trend_aligned'] else 75
                    
                    return OptimizedTradeSetup(
                        strategy=OptimizedStrategy.VWAP_BOUNCE,
                        direction='LONG',
                        entry_price=price,
                        stop_loss=stop_loss,
                        target_1=target_1,
                        target_2=target_2,
                        confidence=confidence,
                        risk_reward=self.min_risk_reward,
                        reasoning="VWAP bounce in favorable regime"
                    )
        
        return None
    
    def momentum_breakout_strategy(self) -> Optional[OptimizedTradeSetup]:
        """
        Optimized Momentum Breakout
        
        Enter on strong momentum with volume confirmation
        """
        if len(self.data) < 30:
            return None
        
        price = self.current_price
        bb_upper = self.bb_upper.iloc[-1]
        bb_lower = self.bb_lower.iloc[-1]
        rsi = self.rsi.iloc[-1]
        
        # Check for strong momentum candle
        current_candle = self.data.iloc[-1]
        candle_range = current_candle['high'] - current_candle['low']
        candle_body = abs(current_candle['close'] - current_candle['open'])
        
        # Strong bullish candle
        is_bullish_momentum = (
            candle_body > candle_range * 0.6 and
            current_candle['close'] > current_candle['open'] and
            candle_range > self.atr.iloc[-1] * 1.2
        )
        
        if is_bullish_momentum and self.regime['regime'] in ['UPTREND', 'STRONG_UPTREND']:
            if price > bb_upper * 0.998 and 55 < rsi < 75:
                stop_loss = self._calculate_stop_loss('LONG')
                target_1, target_2 = self._calculate_targets(price, stop_loss, 'LONG')
                
                return OptimizedTradeSetup(
                    strategy=OptimizedStrategy.MOMENTUM_BREAKOUT,
                    direction='LONG',
                    entry_price=price,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    confidence=77,
                    risk_reward=self.min_risk_reward,
                    reasoning="Momentum breakout above BB"
                )
        
        return None
    
    def run_all_strategies(self) -> List[OptimizedTradeSetup]:
        """Run all optimized strategies and return valid setups"""
        setups = []
        
        # Check time filter
        if not self._is_valid_time():
            return setups
        
        # Check market regime (avoid ranging/unknown)
        if not self.regime['trend_aligned'] and self.regime['regime'] != 'RANGING':
            return setups
        
        # Run strategies in order of historical performance
        strategies = [
            self.pullback_entry_strategy,     # Best: 50% WR
            self.trend_following_strategy,     # New optimized
            self.vwap_bounce_strategy,         # Optimized
            self.momentum_breakout_strategy,   # Optimized
        ]
        
        for strategy_func in strategies:
            try:
                setup = strategy_func()
                if setup and setup.confidence >= self.min_confidence:
                    setups.append(setup)
            except Exception as e:
                continue
        
        # Sort by confidence
        setups.sort(key=lambda x: x.confidence, reverse=True)
        
        return setups
    
    def get_market_analysis(self) -> dict:
        """Get current market analysis"""
        return {
            'price': self.current_price,
            'regime': self.regime,
            'rsi': round(self.rsi.iloc[-1], 1),
            'ema_9': round(self.ema_9.iloc[-1], 2),
            'ema_21': round(self.ema_21.iloc[-1], 2),
            'vwap': round(self.vwap.iloc[-1], 2),
            'atr': round(self.atr.iloc[-1], 2),
            'valid_time': self._is_valid_time()
        }
