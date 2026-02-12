"""Advanced Scalping Strategies for 5-minute Nifty 50 Trading"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum
import config


class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class StrategyType(Enum):
    VWAP_BOUNCE = "VWAP Bounce"
    EMA_CROSSOVER = "EMA Crossover"
    RSI_DIVERGENCE = "RSI Divergence"
    BOLLINGER_SQUEEZE = "Bollinger Squeeze Breakout"
    MOMENTUM_BREAKOUT = "Momentum Breakout"
    PULLBACK_ENTRY = "Pullback Entry"
    OPENING_RANGE = "Opening Range Breakout"
    SCALP_REVERSAL = "Scalp Reversal"


@dataclass
class TradeSetup:
    strategy: StrategyType
    signal: SignalType
    direction: str  # LONG or SHORT
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    confidence: float  # 0-100
    reasoning: str
    risk_reward: float
    timestamp: str
    indicators: Dict


class VWAPBounceStrategy:
    """
    VWAP Bounce Strategy - Trade bounces off VWAP in trending markets
    
    Rules:
    - Price pulls back to VWAP in an uptrend (EMA fast > EMA slow) = Long
    - Price rallies to VWAP in a downtrend = Short
    - Requires volume confirmation
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.name = StrategyType.VWAP_BOUNCE
    
    def analyze(self) -> Optional[TradeSetup]:
        if len(self.data) < 20:
            return None
        
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        
        # Check trend
        uptrend = latest["ema_fast"] > latest["ema_slow"]
        downtrend = latest["ema_fast"] < latest["ema_slow"]
        
        # VWAP touch conditions
        vwap_touch_tolerance = latest["atr"] * 0.3
        near_vwap = abs(latest["close"] - latest["vwap"]) < vwap_touch_tolerance
        
        # Volume confirmation
        volume_ok = latest["volume_ratio"] > 0.8
        
        # Long setup: Uptrend + Price bouncing off VWAP from below
        if uptrend and near_vwap and latest["close"] > latest["vwap"] and prev["close"] < prev["vwap"]:
            if volume_ok and latest["rsi"] < 60:
                return self._create_setup(
                    direction="LONG",
                    latest=latest,
                    confidence=75 if latest["volume_spike"] else 60,
                    reasoning=f"VWAP bounce in uptrend. Price crossed above VWAP ({latest['vwap']:.2f}) with RSI at {latest['rsi']:.1f}. EMA trend bullish."
                )
        
        # Short setup: Downtrend + Price rejecting VWAP from above
        if downtrend and near_vwap and latest["close"] < latest["vwap"] and prev["close"] > prev["vwap"]:
            if volume_ok and latest["rsi"] > 40:
                return self._create_setup(
                    direction="SHORT",
                    latest=latest,
                    confidence=75 if latest["volume_spike"] else 60,
                    reasoning=f"VWAP rejection in downtrend. Price crossed below VWAP ({latest['vwap']:.2f}) with RSI at {latest['rsi']:.1f}. EMA trend bearish."
                )
        
        return None
    
    def _create_setup(self, direction: str, latest: pd.Series, confidence: float, reasoning: str) -> TradeSetup:
        atr = latest["atr"]
        entry = latest["close"]
        
        if direction == "LONG":
            sl = entry - (atr * 1.5)
            t1 = entry + (atr * 2)
            t2 = entry + (atr * 3)
            signal = SignalType.BUY if confidence >= 70 else SignalType.WEAK_BUY
        else:
            sl = entry + (atr * 1.5)
            t1 = entry - (atr * 2)
            t2 = entry - (atr * 3)
            signal = SignalType.SELL if confidence >= 70 else SignalType.WEAK_SELL
        
        risk = abs(entry - sl)
        reward = abs(t1 - entry)
        
        return TradeSetup(
            strategy=self.name,
            signal=signal,
            direction=direction,
            entry_price=round(entry, 2),
            stop_loss=round(sl, 2),
            target_1=round(t1, 2),
            target_2=round(t2, 2),
            confidence=confidence,
            reasoning=reasoning,
            risk_reward=round(reward/risk, 2) if risk > 0 else 0,
            timestamp=str(self.data.index[-1]),
            indicators={
                "vwap": round(latest["vwap"], 2),
                "rsi": round(latest["rsi"], 1),
                "ema_fast": round(latest["ema_fast"], 2),
                "ema_slow": round(latest["ema_slow"], 2),
                "atr": round(atr, 2)
            }
        )


class EMACrossoverStrategy:
    """
    EMA Crossover Strategy with momentum confirmation
    
    Rules:
    - Fast EMA crosses above Slow EMA = Long
    - Fast EMA crosses below Slow EMA = Short
    - Requires MACD and RSI confirmation
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.name = StrategyType.EMA_CROSSOVER
    
    def analyze(self) -> Optional[TradeSetup]:
        if len(self.data) < 30:
            return None
        
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        
        # Bullish crossover
        if latest["ema_bullish"]:
            # Confirmation: MACD bullish or turning, RSI not overbought
            macd_ok = latest["macd_histogram"] > prev["macd_histogram"]
            rsi_ok = latest["rsi"] < 65
            
            if macd_ok and rsi_ok:
                confidence = 80 if latest["volume_spike"] else 65
                return self._create_setup(
                    direction="LONG",
                    latest=latest,
                    confidence=confidence,
                    reasoning=f"EMA{config.EMA_FAST}/{config.EMA_SLOW} bullish crossover. MACD histogram rising ({latest['macd_histogram']:.2f}). RSI at {latest['rsi']:.1f} (healthy)."
                )
        
        # Bearish crossover
        if latest["ema_bearish"]:
            macd_ok = latest["macd_histogram"] < prev["macd_histogram"]
            rsi_ok = latest["rsi"] > 35
            
            if macd_ok and rsi_ok:
                confidence = 80 if latest["volume_spike"] else 65
                return self._create_setup(
                    direction="SHORT",
                    latest=latest,
                    confidence=confidence,
                    reasoning=f"EMA{config.EMA_FAST}/{config.EMA_SLOW} bearish crossover. MACD histogram falling ({latest['macd_histogram']:.2f}). RSI at {latest['rsi']:.1f}."
                )
        
        return None
    
    def _create_setup(self, direction: str, latest: pd.Series, confidence: float, reasoning: str) -> TradeSetup:
        atr = latest["atr"]
        entry = latest["close"]
        
        if direction == "LONG":
            sl = min(latest["low"], latest["ema_slow"]) - (atr * 0.5)
            t1 = entry + (atr * 2.5)
            t2 = entry + (atr * 4)
            signal = SignalType.STRONG_BUY if confidence >= 75 else SignalType.BUY
        else:
            sl = max(latest["high"], latest["ema_slow"]) + (atr * 0.5)
            t1 = entry - (atr * 2.5)
            t2 = entry - (atr * 4)
            signal = SignalType.STRONG_SELL if confidence >= 75 else SignalType.SELL
        
        risk = abs(entry - sl)
        reward = abs(t1 - entry)
        
        return TradeSetup(
            strategy=self.name,
            signal=signal,
            direction=direction,
            entry_price=round(entry, 2),
            stop_loss=round(sl, 2),
            target_1=round(t1, 2),
            target_2=round(t2, 2),
            confidence=confidence,
            reasoning=reasoning,
            risk_reward=round(reward/risk, 2) if risk > 0 else 0,
            timestamp=str(self.data.index[-1]),
            indicators={
                "ema_fast": round(latest["ema_fast"], 2),
                "ema_slow": round(latest["ema_slow"], 2),
                "macd": round(latest["macd"], 2),
                "macd_signal": round(latest["macd_signal"], 2),
                "rsi": round(latest["rsi"], 1)
            }
        )


class RSIDivergenceStrategy:
    """
    RSI Divergence Strategy - Catch reversals using RSI divergence
    
    Rules:
    - Bullish divergence: Price makes lower low, RSI makes higher low
    - Bearish divergence: Price makes higher high, RSI makes lower high
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.name = StrategyType.RSI_DIVERGENCE
    
    def analyze(self) -> Optional[TradeSetup]:
        if len(self.data) < 20:
            return None
        
        latest = self.data.iloc[-1]
        
        # Check for divergence over last 10 candles
        lookback = self.data.tail(10)
        
        # Find price and RSI extremes
        price_low_idx = lookback["low"].idxmin()
        price_high_idx = lookback["high"].idxmax()
        rsi_low_idx = lookback["rsi"].idxmin()
        rsi_high_idx = lookback["rsi"].idxmax()
        
        # Bullish divergence: Recent low in price but RSI bottomed earlier
        if latest["rsi_bullish_div"] and latest["rsi"] < 40:
            # Confirm with candle pattern
            if latest["bullish_candle"] or latest["hammer"]:
                return self._create_setup(
                    direction="LONG",
                    latest=latest,
                    confidence=70,
                    reasoning=f"Bullish RSI divergence detected. Price making lower lows while RSI making higher lows. RSI at {latest['rsi']:.1f} showing momentum shift."
                )
        
        # Bearish divergence
        if latest["rsi_bearish_div"] and latest["rsi"] > 60:
            if latest["bearish_candle"] or latest["shooting_star"]:
                return self._create_setup(
                    direction="SHORT",
                    latest=latest,
                    confidence=70,
                    reasoning=f"Bearish RSI divergence detected. Price making higher highs while RSI making lower highs. RSI at {latest['rsi']:.1f} showing weakening momentum."
                )
        
        return None
    
    def _create_setup(self, direction: str, latest: pd.Series, confidence: float, reasoning: str) -> TradeSetup:
        atr = latest["atr"]
        entry = latest["close"]
        
        if direction == "LONG":
            sl = latest["low"] - (atr * 0.5)
            t1 = entry + (atr * 2)
            t2 = latest["ema_slow"]  # Target EMA as first resistance
            signal = SignalType.BUY
        else:
            sl = latest["high"] + (atr * 0.5)
            t1 = entry - (atr * 2)
            t2 = latest["ema_slow"]
            signal = SignalType.SELL
        
        risk = abs(entry - sl)
        reward = abs(t1 - entry)
        
        return TradeSetup(
            strategy=self.name,
            signal=signal,
            direction=direction,
            entry_price=round(entry, 2),
            stop_loss=round(sl, 2),
            target_1=round(t1, 2),
            target_2=round(t2, 2),
            confidence=confidence,
            reasoning=reasoning,
            risk_reward=round(reward/risk, 2) if risk > 0 else 0,
            timestamp=str(self.data.index[-1]),
            indicators={
                "rsi": round(latest["rsi"], 1),
                "price_low": round(latest["low"], 2),
                "price_high": round(latest["high"], 2)
            }
        )


class BollingerSqueezeStrategy:
    """
    Bollinger Squeeze Breakout Strategy
    
    Rules:
    - Identify squeeze (narrow BB width)
    - Enter on breakout with volume confirmation
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.name = StrategyType.BOLLINGER_SQUEEZE
    
    def analyze(self) -> Optional[TradeSetup]:
        if len(self.data) < 25:
            return None
        
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        
        # Check if we're coming out of a squeeze
        was_squeeze = prev["bb_squeeze"]
        not_squeeze_now = not latest["bb_squeeze"]
        
        if was_squeeze and not_squeeze_now and latest["volume_spike"]:
            # Breakout direction
            if latest["close"] > latest["bb_upper"]:
                return self._create_setup(
                    direction="LONG",
                    latest=latest,
                    confidence=80,
                    reasoning=f"Bollinger Band squeeze breakout to upside. Price broke above upper band ({latest['bb_upper']:.2f}) with volume spike. Momentum expansion expected."
                )
            elif latest["close"] < latest["bb_lower"]:
                return self._create_setup(
                    direction="SHORT",
                    latest=latest,
                    confidence=80,
                    reasoning=f"Bollinger Band squeeze breakout to downside. Price broke below lower band ({latest['bb_lower']:.2f}) with volume spike. Momentum expansion expected."
                )
        
        return None
    
    def _create_setup(self, direction: str, latest: pd.Series, confidence: float, reasoning: str) -> TradeSetup:
        atr = latest["atr"]
        entry = latest["close"]
        bb_width = latest["bb_upper"] - latest["bb_lower"]
        
        if direction == "LONG":
            sl = latest["bb_middle"] - (atr * 0.5)
            t1 = entry + bb_width
            t2 = entry + (bb_width * 1.5)
            signal = SignalType.STRONG_BUY
        else:
            sl = latest["bb_middle"] + (atr * 0.5)
            t1 = entry - bb_width
            t2 = entry - (bb_width * 1.5)
            signal = SignalType.STRONG_SELL
        
        risk = abs(entry - sl)
        reward = abs(t1 - entry)
        
        return TradeSetup(
            strategy=self.name,
            signal=signal,
            direction=direction,
            entry_price=round(entry, 2),
            stop_loss=round(sl, 2),
            target_1=round(t1, 2),
            target_2=round(t2, 2),
            confidence=confidence,
            reasoning=reasoning,
            risk_reward=round(reward/risk, 2) if risk > 0 else 0,
            timestamp=str(self.data.index[-1]),
            indicators={
                "bb_upper": round(latest["bb_upper"], 2),
                "bb_middle": round(latest["bb_middle"], 2),
                "bb_lower": round(latest["bb_lower"], 2),
                "bb_width": round(latest["bb_width"], 4)
            }
        )


class MomentumBreakoutStrategy:
    """
    Momentum Breakout Strategy - Trade strong directional moves
    
    Rules:
    - Strong candle with volume spike
    - Breaking recent high/low
    - MACD and RSI confirmation
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.name = StrategyType.MOMENTUM_BREAKOUT
    
    def analyze(self) -> Optional[TradeSetup]:
        if len(self.data) < 20:
            return None
        
        latest = self.data.iloc[-1]
        recent = self.data.tail(10)
        
        # Recent range
        recent_high = recent["high"].max()
        recent_low = recent["low"].min()
        
        # Strong bullish breakout
        if (latest["close"] > recent_high and 
            latest["strong_candle"] and 
            latest["bullish_candle"] and
            latest["volume_spike"] and
            latest["macd"] > latest["macd_signal"]):
            
            return self._create_setup(
                direction="LONG",
                latest=latest,
                confidence=85,
                reasoning=f"Strong momentum breakout above {recent_high:.2f}. Large bullish candle with {latest['volume_ratio']:.1f}x average volume. MACD bullish."
            )
        
        # Strong bearish breakout
        if (latest["close"] < recent_low and 
            latest["strong_candle"] and 
            latest["bearish_candle"] and
            latest["volume_spike"] and
            latest["macd"] < latest["macd_signal"]):
            
            return self._create_setup(
                direction="SHORT",
                latest=latest,
                confidence=85,
                reasoning=f"Strong momentum breakdown below {recent_low:.2f}. Large bearish candle with {latest['volume_ratio']:.1f}x average volume. MACD bearish."
            )
        
        return None
    
    def _create_setup(self, direction: str, latest: pd.Series, confidence: float, reasoning: str) -> TradeSetup:
        atr = latest["atr"]
        entry = latest["close"]
        
        if direction == "LONG":
            sl = latest["low"] - (atr * 0.3)
            t1 = entry + (atr * 2.5)
            t2 = entry + (atr * 4)
            signal = SignalType.STRONG_BUY
        else:
            sl = latest["high"] + (atr * 0.3)
            t1 = entry - (atr * 2.5)
            t2 = entry - (atr * 4)
            signal = SignalType.STRONG_SELL
        
        risk = abs(entry - sl)
        reward = abs(t1 - entry)
        
        return TradeSetup(
            strategy=self.name,
            signal=signal,
            direction=direction,
            entry_price=round(entry, 2),
            stop_loss=round(sl, 2),
            target_1=round(t1, 2),
            target_2=round(t2, 2),
            confidence=confidence,
            reasoning=reasoning,
            risk_reward=round(reward/risk, 2) if risk > 0 else 0,
            timestamp=str(self.data.index[-1]),
            indicators={
                "volume_ratio": round(latest["volume_ratio"], 2),
                "candle_range": round(latest["candle_range"], 2),
                "macd": round(latest["macd"], 2)
            }
        )


class PullbackEntryStrategy:
    """
    Pullback Entry Strategy - Enter on pullbacks in trending markets
    
    Rules:
    - Strong trend (price above/below EMA 50)
    - Pullback to EMA 9 or 21
    - Reversal candle pattern
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.name = StrategyType.PULLBACK_ENTRY
    
    def analyze(self) -> Optional[TradeSetup]:
        if len(self.data) < 50:
            return None
        
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        
        # Strong uptrend: Price above EMA 50, EMA 9 > EMA 21 > EMA 50
        strong_uptrend = (
            latest["close"] > latest["ema_signal"] and
            latest["ema_fast"] > latest["ema_slow"] > latest["ema_signal"]
        )
        
        # Strong downtrend
        strong_downtrend = (
            latest["close"] < latest["ema_signal"] and
            latest["ema_fast"] < latest["ema_slow"] < latest["ema_signal"]
        )
        
        # Pullback to EMA in uptrend
        if strong_uptrend:
            ema_touch = (
                latest["low"] <= latest["ema_fast"] * 1.002 or
                latest["low"] <= latest["ema_slow"] * 1.002
            )
            reversal = latest["bullish_candle"] and (latest["hammer"] or latest["close"] > latest["open"])
            
            if ema_touch and reversal and latest["rsi"] > 40:
                return self._create_setup(
                    direction="LONG",
                    latest=latest,
                    confidence=75,
                    reasoning=f"Pullback buy in uptrend. Price pulled back to EMA zone and showing reversal. Trend structure intact (EMA 9 > 21 > 50)."
                )
        
        # Pullback in downtrend
        if strong_downtrend:
            ema_touch = (
                latest["high"] >= latest["ema_fast"] * 0.998 or
                latest["high"] >= latest["ema_slow"] * 0.998
            )
            reversal = latest["bearish_candle"] and (latest["shooting_star"] or latest["close"] < latest["open"])
            
            if ema_touch and reversal and latest["rsi"] < 60:
                return self._create_setup(
                    direction="SHORT",
                    latest=latest,
                    confidence=75,
                    reasoning=f"Pullback sell in downtrend. Price rallied to EMA zone and showing rejection. Trend structure intact (EMA 9 < 21 < 50)."
                )
        
        return None
    
    def _create_setup(self, direction: str, latest: pd.Series, confidence: float, reasoning: str) -> TradeSetup:
        atr = latest["atr"]
        entry = latest["close"]
        
        if direction == "LONG":
            sl = min(latest["low"], latest["ema_slow"]) - (atr * 0.5)
            t1 = entry + (atr * 2)
            t2 = entry + (atr * 3.5)
            signal = SignalType.BUY
        else:
            sl = max(latest["high"], latest["ema_slow"]) + (atr * 0.5)
            t1 = entry - (atr * 2)
            t2 = entry - (atr * 3.5)
            signal = SignalType.SELL
        
        risk = abs(entry - sl)
        reward = abs(t1 - entry)
        
        return TradeSetup(
            strategy=self.name,
            signal=signal,
            direction=direction,
            entry_price=round(entry, 2),
            stop_loss=round(sl, 2),
            target_1=round(t1, 2),
            target_2=round(t2, 2),
            confidence=confidence,
            reasoning=reasoning,
            risk_reward=round(reward/risk, 2) if risk > 0 else 0,
            timestamp=str(self.data.index[-1]),
            indicators={
                "ema_fast": round(latest["ema_fast"], 2),
                "ema_slow": round(latest["ema_slow"], 2),
                "ema_signal": round(latest["ema_signal"], 2),
                "rsi": round(latest["rsi"], 1)
            }
        )


class ScalpReversalStrategy:
    """
    Scalp Reversal Strategy - Quick reversal trades at extremes
    
    Rules:
    - RSI extreme + Stochastic extreme
    - Price at Bollinger Band extreme
    - Reversal candle pattern
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.name = StrategyType.SCALP_REVERSAL
    
    def analyze(self) -> Optional[TradeSetup]:
        if len(self.data) < 20:
            return None
        
        latest = self.data.iloc[-1]
        
        # Oversold reversal
        oversold = (
            latest["rsi"] < 30 and
            latest["stoch_oversold"] and
            latest["price_at_lower_bb"]
        )
        
        if oversold and (latest["bullish_candle"] or latest["hammer"] or latest["doji"]):
            return self._create_setup(
                direction="LONG",
                latest=latest,
                confidence=70,
                reasoning=f"Oversold reversal setup. RSI={latest['rsi']:.1f}, Stochastic oversold, price at lower BB. Reversal candle forming."
            )
        
        # Overbought reversal
        overbought = (
            latest["rsi"] > 70 and
            latest["stoch_overbought"] and
            latest["price_at_upper_bb"]
        )
        
        if overbought and (latest["bearish_candle"] or latest["shooting_star"] or latest["doji"]):
            return self._create_setup(
                direction="SHORT",
                latest=latest,
                confidence=70,
                reasoning=f"Overbought reversal setup. RSI={latest['rsi']:.1f}, Stochastic overbought, price at upper BB. Reversal candle forming."
            )
        
        return None
    
    def _create_setup(self, direction: str, latest: pd.Series, confidence: float, reasoning: str) -> TradeSetup:
        atr = latest["atr"]
        entry = latest["close"]
        
        if direction == "LONG":
            sl = latest["bb_lower"] - (atr * 0.3)
            t1 = latest["bb_middle"]
            t2 = latest["vwap"]
            signal = SignalType.BUY
        else:
            sl = latest["bb_upper"] + (atr * 0.3)
            t1 = latest["bb_middle"]
            t2 = latest["vwap"]
            signal = SignalType.SELL
        
        risk = abs(entry - sl)
        reward = abs(t1 - entry)
        
        return TradeSetup(
            strategy=self.name,
            signal=signal,
            direction=direction,
            entry_price=round(entry, 2),
            stop_loss=round(sl, 2),
            target_1=round(t1, 2),
            target_2=round(t2, 2),
            confidence=confidence,
            reasoning=reasoning,
            risk_reward=round(reward/risk, 2) if risk > 0 else 0,
            timestamp=str(self.data.index[-1]),
            indicators={
                "rsi": round(latest["rsi"], 1),
                "stoch_k": round(latest["stoch_k"], 1),
                "bb_position": "LOWER" if direction == "LONG" else "UPPER"
            }
        )


class StrategyEngine:
    """Main engine to run all strategies and aggregate signals"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.strategies = [
            VWAPBounceStrategy(data),
            EMACrossoverStrategy(data),
            RSIDivergenceStrategy(data),
            BollingerSqueezeStrategy(data),
            MomentumBreakoutStrategy(data),
            PullbackEntryStrategy(data),
            ScalpReversalStrategy(data)
        ]
    
    def run_all_strategies(self) -> List[TradeSetup]:
        """Run all strategies and return valid setups"""
        setups = []
        
        for strategy in self.strategies:
            try:
                setup = strategy.analyze()
                if setup and setup.risk_reward >= config.MIN_RISK_REWARD:
                    setups.append(setup)
            except Exception as e:
                print(f"Error in {strategy.name}: {e}")
        
        # Sort by confidence
        setups.sort(key=lambda x: x.confidence, reverse=True)
        
        return setups
    
    def get_best_setup(self) -> Optional[TradeSetup]:
        """Get the highest confidence setup"""
        setups = self.run_all_strategies()
        return setups[0] if setups else None
    
    def get_market_bias(self) -> Dict:
        """Determine overall market bias"""
        if self.data.empty:
            return {"bias": "NEUTRAL", "strength": 0}
        
        latest = self.data.iloc[-1]
        
        bullish_factors = sum([
            latest["ema_fast"] > latest["ema_slow"],
            latest["close"] > latest["vwap"],
            latest["macd"] > latest["macd_signal"],
            latest["rsi"] > 50,
            latest["obv_rising"]
        ])
        
        bearish_factors = 5 - bullish_factors
        
        if bullish_factors >= 4:
            bias = "STRONGLY BULLISH"
            strength = bullish_factors * 20
        elif bullish_factors >= 3:
            bias = "BULLISH"
            strength = bullish_factors * 15
        elif bearish_factors >= 4:
            bias = "STRONGLY BEARISH"
            strength = bearish_factors * 20
        elif bearish_factors >= 3:
            bias = "BEARISH"
            strength = bearish_factors * 15
        else:
            bias = "NEUTRAL"
            strength = 50
        
        return {
            "bias": bias,
            "strength": strength,
            "bullish_factors": bullish_factors,
            "bearish_factors": bearish_factors,
            "details": {
                "ema_trend": "BULLISH" if latest["ema_fast"] > latest["ema_slow"] else "BEARISH",
                "vwap_position": "ABOVE" if latest["close"] > latest["vwap"] else "BELOW",
                "macd_trend": "BULLISH" if latest["macd"] > latest["macd_signal"] else "BEARISH",
                "rsi_position": "ABOVE 50" if latest["rsi"] > 50 else "BELOW 50",
                "obv_trend": "RISING" if latest["obv_rising"] else "FALLING"
            }
        }
