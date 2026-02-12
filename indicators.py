"""Technical indicators module for scalping analysis"""

import pandas as pd
import numpy as np
import ta
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
import config


class ScalpingIndicators:
    """Calculate technical indicators optimized for 5-minute scalping"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.signals = {}
    
    def calculate_all(self) -> pd.DataFrame:
        """Calculate all technical indicators"""
        self._calculate_emas()
        self._calculate_rsi()
        self._calculate_macd()
        self._calculate_bollinger()
        self._calculate_vwap()
        self._calculate_atr()
        self._calculate_stochastic()
        self._calculate_volume_analysis()
        self._calculate_price_action()
        self._generate_signals()
        
        return self.data
    
    def _calculate_emas(self):
        """Calculate Exponential Moving Averages"""
        self.data["ema_fast"] = EMAIndicator(
            close=self.data["close"], 
            window=config.EMA_FAST
        ).ema_indicator()
        
        self.data["ema_slow"] = EMAIndicator(
            close=self.data["close"], 
            window=config.EMA_SLOW
        ).ema_indicator()
        
        self.data["ema_signal"] = EMAIndicator(
            close=self.data["close"], 
            window=config.EMA_SIGNAL
        ).ema_indicator()
        
        # EMA crossover signals
        self.data["ema_bullish"] = (
            (self.data["ema_fast"] > self.data["ema_slow"]) & 
            (self.data["ema_fast"].shift(1) <= self.data["ema_slow"].shift(1))
        )
        self.data["ema_bearish"] = (
            (self.data["ema_fast"] < self.data["ema_slow"]) & 
            (self.data["ema_fast"].shift(1) >= self.data["ema_slow"].shift(1))
        )
    
    def _calculate_rsi(self):
        """Calculate RSI"""
        rsi = RSIIndicator(close=self.data["close"], window=config.RSI_PERIOD)
        self.data["rsi"] = rsi.rsi()
        
        # RSI conditions
        self.data["rsi_oversold"] = self.data["rsi"] < config.RSI_OVERSOLD
        self.data["rsi_overbought"] = self.data["rsi"] > config.RSI_OVERBOUGHT
        
        # RSI divergence (simplified)
        self.data["rsi_bullish_div"] = (
            (self.data["close"] < self.data["close"].shift(5)) & 
            (self.data["rsi"] > self.data["rsi"].shift(5))
        )
        self.data["rsi_bearish_div"] = (
            (self.data["close"] > self.data["close"].shift(5)) & 
            (self.data["rsi"] < self.data["rsi"].shift(5))
        )
    
    def _calculate_macd(self):
        """Calculate MACD"""
        macd = MACD(
            close=self.data["close"],
            window_fast=config.MACD_FAST,
            window_slow=config.MACD_SLOW,
            window_sign=config.MACD_SIGNAL
        )
        self.data["macd"] = macd.macd()
        self.data["macd_signal"] = macd.macd_signal()
        self.data["macd_histogram"] = macd.macd_diff()
        
        # MACD signals
        self.data["macd_bullish"] = (
            (self.data["macd"] > self.data["macd_signal"]) & 
            (self.data["macd"].shift(1) <= self.data["macd_signal"].shift(1))
        )
        self.data["macd_bearish"] = (
            (self.data["macd"] < self.data["macd_signal"]) & 
            (self.data["macd"].shift(1) >= self.data["macd_signal"].shift(1))
        )
    
    def _calculate_bollinger(self):
        """Calculate Bollinger Bands"""
        bb = BollingerBands(
            close=self.data["close"],
            window=config.BOLLINGER_PERIOD,
            window_dev=config.BOLLINGER_STD
        )
        self.data["bb_upper"] = bb.bollinger_hband()
        self.data["bb_middle"] = bb.bollinger_mavg()
        self.data["bb_lower"] = bb.bollinger_lband()
        self.data["bb_width"] = (self.data["bb_upper"] - self.data["bb_lower"]) / self.data["bb_middle"]
        
        # Bollinger Band signals
        self.data["bb_squeeze"] = self.data["bb_width"] < self.data["bb_width"].rolling(20).mean() * 0.8
        self.data["price_at_lower_bb"] = self.data["close"] <= self.data["bb_lower"]
        self.data["price_at_upper_bb"] = self.data["close"] >= self.data["bb_upper"]
    
    def _calculate_vwap(self):
        """Calculate VWAP"""
        try:
            vwap = VolumeWeightedAveragePrice(
                high=self.data["high"],
                low=self.data["low"],
                close=self.data["close"],
                volume=self.data["volume"]
            )
            self.data["vwap"] = vwap.volume_weighted_average_price()
        except:
            # Manual VWAP calculation
            self.data["typical_price"] = (
                self.data["high"] + self.data["low"] + self.data["close"]
            ) / 3
            self.data["vwap"] = (
                (self.data["typical_price"] * self.data["volume"]).cumsum() / 
                self.data["volume"].cumsum()
            )
        
        # VWAP deviation
        self.data["vwap_deviation"] = (
            (self.data["close"] - self.data["vwap"]) / self.data["vwap"] * 100
        )
        self.data["above_vwap"] = self.data["close"] > self.data["vwap"]
    
    def _calculate_atr(self):
        """Calculate Average True Range"""
        atr = AverageTrueRange(
            high=self.data["high"],
            low=self.data["low"],
            close=self.data["close"],
            window=config.ATR_PERIOD
        )
        self.data["atr"] = atr.average_true_range()
        self.data["atr_pct"] = self.data["atr"] / self.data["close"] * 100
    
    def _calculate_stochastic(self):
        """Calculate Stochastic Oscillator"""
        stoch = StochasticOscillator(
            high=self.data["high"],
            low=self.data["low"],
            close=self.data["close"],
            window=14,
            smooth_window=3
        )
        self.data["stoch_k"] = stoch.stoch()
        self.data["stoch_d"] = stoch.stoch_signal()
        
        # Stochastic signals
        self.data["stoch_oversold"] = (self.data["stoch_k"] < 20) & (self.data["stoch_d"] < 20)
        self.data["stoch_overbought"] = (self.data["stoch_k"] > 80) & (self.data["stoch_d"] > 80)
    
    def _calculate_volume_analysis(self):
        """Analyze volume patterns"""
        self.data["volume_ma"] = self.data["volume"].rolling(20).mean()
        self.data["volume_ratio"] = self.data["volume"] / self.data["volume_ma"]
        self.data["volume_spike"] = self.data["volume_ratio"] > config.VOLUME_SPIKE_THRESHOLD
        
        # On Balance Volume
        obv = OnBalanceVolumeIndicator(
            close=self.data["close"],
            volume=self.data["volume"]
        )
        self.data["obv"] = obv.on_balance_volume()
        self.data["obv_ma"] = self.data["obv"].rolling(20).mean()
        self.data["obv_rising"] = self.data["obv"] > self.data["obv_ma"]
    
    def _calculate_price_action(self):
        """Calculate price action patterns"""
        # Candle body and wick analysis
        self.data["body"] = abs(self.data["close"] - self.data["open"])
        self.data["upper_wick"] = self.data["high"] - self.data[["open", "close"]].max(axis=1)
        self.data["lower_wick"] = self.data[["open", "close"]].min(axis=1) - self.data["low"]
        self.data["candle_range"] = self.data["high"] - self.data["low"]
        
        # Bullish/Bearish candles
        self.data["bullish_candle"] = self.data["close"] > self.data["open"]
        self.data["bearish_candle"] = self.data["close"] < self.data["open"]
        
        # Strong candles (body > 60% of range)
        self.data["strong_candle"] = self.data["body"] > (self.data["candle_range"] * 0.6)
        
        # Doji detection (body < 10% of range)
        self.data["doji"] = self.data["body"] < (self.data["candle_range"] * 0.1)
        
        # Hammer/Shooting star patterns
        self.data["hammer"] = (
            (self.data["lower_wick"] > self.data["body"] * 2) & 
            (self.data["upper_wick"] < self.data["body"] * 0.5)
        )
        self.data["shooting_star"] = (
            (self.data["upper_wick"] > self.data["body"] * 2) & 
            (self.data["lower_wick"] < self.data["body"] * 0.5)
        )
        
        # Support/Resistance levels (recent swing highs/lows)
        self.data["swing_high"] = (
            (self.data["high"] > self.data["high"].shift(1)) & 
            (self.data["high"] > self.data["high"].shift(-1))
        )
        self.data["swing_low"] = (
            (self.data["low"] < self.data["low"].shift(1)) & 
            (self.data["low"] < self.data["low"].shift(-1))
        )
    
    def _generate_signals(self):
        """Generate composite trading signals"""
        # Bullish score (0-10)
        self.data["bullish_score"] = (
            self.data["rsi_oversold"].astype(int) * 2 +
            self.data["ema_bullish"].astype(int) * 2 +
            self.data["macd_bullish"].astype(int) * 2 +
            self.data["price_at_lower_bb"].astype(int) * 1 +
            self.data["above_vwap"].astype(int) * 1 +
            self.data["volume_spike"].astype(int) * 1 +
            self.data["obv_rising"].astype(int) * 1
        )
        
        # Bearish score (0-10)
        self.data["bearish_score"] = (
            self.data["rsi_overbought"].astype(int) * 2 +
            self.data["ema_bearish"].astype(int) * 2 +
            self.data["macd_bearish"].astype(int) * 2 +
            self.data["price_at_upper_bb"].astype(int) * 1 +
            (~self.data["above_vwap"]).astype(int) * 1 +
            self.data["volume_spike"].astype(int) * 1 +
            (~self.data["obv_rising"]).astype(int) * 1
        )
        
        # Overall signal
        self.data["signal_strength"] = self.data["bullish_score"] - self.data["bearish_score"]
    
    def get_current_analysis(self) -> dict:
        """Get analysis for the current candle"""
        if self.data.empty:
            return {}
        
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2] if len(self.data) > 1 else latest
        
        return {
            "timestamp": str(self.data.index[-1]),
            "price": {
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "close": float(latest["close"]),
                "change_pct": float((latest["close"] - prev["close"]) / prev["close"] * 100)
            },
            "ema": {
                "ema_fast": float(latest["ema_fast"]),
                "ema_slow": float(latest["ema_slow"]),
                "ema_signal": float(latest["ema_signal"]),
                "trend": "BULLISH" if latest["ema_fast"] > latest["ema_slow"] else "BEARISH"
            },
            "rsi": {
                "value": float(latest["rsi"]),
                "condition": "OVERSOLD" if latest["rsi_oversold"] else ("OVERBOUGHT" if latest["rsi_overbought"] else "NEUTRAL")
            },
            "macd": {
                "macd": float(latest["macd"]),
                "signal": float(latest["macd_signal"]),
                "histogram": float(latest["macd_histogram"]),
                "trend": "BULLISH" if latest["macd"] > latest["macd_signal"] else "BEARISH"
            },
            "bollinger": {
                "upper": float(latest["bb_upper"]),
                "middle": float(latest["bb_middle"]),
                "lower": float(latest["bb_lower"]),
                "squeeze": bool(latest["bb_squeeze"])
            },
            "vwap": {
                "value": float(latest["vwap"]),
                "deviation": float(latest["vwap_deviation"]),
                "position": "ABOVE" if latest["above_vwap"] else "BELOW"
            },
            "atr": {
                "value": float(latest["atr"]),
                "pct": float(latest["atr_pct"])
            },
            "volume": {
                "current": int(latest["volume"]),
                "ratio": float(latest["volume_ratio"]),
                "spike": bool(latest["volume_spike"])
            },
            "signals": {
                "bullish_score": int(latest["bullish_score"]),
                "bearish_score": int(latest["bearish_score"]),
                "signal_strength": int(latest["signal_strength"]),
                "recommendation": self._get_recommendation(latest)
            }
        }
    
    def _get_recommendation(self, row) -> str:
        """Get trading recommendation based on signals"""
        strength = row["signal_strength"]
        
        if strength >= 5:
            return "STRONG BUY"
        elif strength >= 3:
            return "BUY"
        elif strength >= 1:
            return "WEAK BUY"
        elif strength <= -5:
            return "STRONG SELL"
        elif strength <= -3:
            return "SELL"
        elif strength <= -1:
            return "WEAK SELL"
        else:
            return "NEUTRAL"
    
    def get_support_resistance(self, lookback: int = 50) -> dict:
        """Calculate dynamic support and resistance levels"""
        recent = self.data.tail(lookback)
        
        # Find recent swing highs and lows
        swing_highs = recent[recent["swing_high"]]["high"].tolist()
        swing_lows = recent[recent["swing_low"]]["low"].tolist()
        
        current_price = float(recent.iloc[-1]["close"])
        
        # Get nearest support and resistance
        resistances = sorted([h for h in swing_highs if h > current_price])
        supports = sorted([l for l in swing_lows if l < current_price], reverse=True)
        
        return {
            "current_price": current_price,
            "resistance_1": resistances[0] if len(resistances) > 0 else None,
            "resistance_2": resistances[1] if len(resistances) > 1 else None,
            "support_1": supports[0] if len(supports) > 0 else None,
            "support_2": supports[1] if len(supports) > 1 else None,
            "vwap": float(recent.iloc[-1]["vwap"]),
            "bb_upper": float(recent.iloc[-1]["bb_upper"]),
            "bb_lower": float(recent.iloc[-1]["bb_lower"])
        }


def calculate_scalp_targets(entry_price: float, atr: float, direction: str) -> dict:
    """Calculate stop loss and target levels for scalp trade"""
    sl_distance = atr * config.STOP_LOSS_ATR_MULTIPLIER
    target_distance = atr * config.TARGET_ATR_MULTIPLIER
    
    if direction.upper() == "LONG":
        stop_loss = entry_price - sl_distance
        target_1 = entry_price + target_distance
        target_2 = entry_price + (target_distance * 1.5)
    else:  # SHORT
        stop_loss = entry_price + sl_distance
        target_1 = entry_price - target_distance
        target_2 = entry_price - (target_distance * 1.5)
    
    risk = abs(entry_price - stop_loss)
    reward = abs(target_1 - entry_price)
    
    return {
        "entry": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "risk_reward_ratio": round(reward / risk, 2) if risk > 0 else 0
    }


def get_candle_pattern_analysis(data: pd.DataFrame) -> dict:
    """
    Get comprehensive candlestick pattern analysis.
    Returns detected patterns with confidence scores and trading suggestions.
    """
    try:
        from candle_patterns import CandlePatternAnalyzer
        analyzer = CandlePatternAnalyzer(data)
        return analyzer.get_pattern_summary()
    except Exception as e:
        print(f"Error in candle pattern analysis: {e}")
        return {
            "total_patterns": 0,
            "bullish_patterns": 0,
            "bearish_patterns": 0,
            "neutral_patterns": 0,
            "overall_bias": "NEUTRAL",
            "patterns": [],
            "error": str(e)
        }


if __name__ == "__main__":
    # Test indicators
    from data_fetcher import NiftyDataFetcher
    
    fetcher = NiftyDataFetcher()
    data = fetcher.fetch_data()
    
    if not data.empty:
        indicators = ScalpingIndicators(data)
        analyzed_data = indicators.calculate_all()
        
        print("\n--- Current Analysis ---")
        analysis = indicators.get_current_analysis()
        for key, value in analysis.items():
            print(f"{key}: {value}")
        
        print("\n--- Support/Resistance ---")
        sr = indicators.get_support_resistance()
        print(sr)
        
        print("\n--- Candlestick Patterns ---")
        patterns = get_candle_pattern_analysis(analyzed_data)
        print(f"Total Patterns: {patterns['total_patterns']}")
        print(f"Bullish: {patterns['bullish_patterns']}, Bearish: {patterns['bearish_patterns']}")
        print(f"Overall Bias: {patterns['overall_bias']}")
        for p in patterns.get('patterns', [])[:5]:
            print(f"  - {p['type']}: {p['signal']} ({p['confidence']}%)")
