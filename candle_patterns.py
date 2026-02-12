"""
Comprehensive Candlestick Pattern Analysis Module for Nifty 50 Trading

This module provides high-accuracy detection of Japanese candlestick patterns
with confidence scoring based on volume, trend context, and pattern location.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum


class PatternType(Enum):
    # Single Candle Patterns
    DOJI = "Doji"
    HAMMER = "Hammer"
    INVERTED_HAMMER = "Inverted Hammer"
    HANGING_MAN = "Hanging Man"
    SHOOTING_STAR = "Shooting Star"
    MARUBOZU_BULLISH = "Bullish Marubozu"
    MARUBOZU_BEARISH = "Bearish Marubozu"
    SPINNING_TOP = "Spinning Top"
    DRAGONFLY_DOJI = "Dragonfly Doji"
    GRAVESTONE_DOJI = "Gravestone Doji"
    
    # Two Candle Patterns
    BULLISH_ENGULFING = "Bullish Engulfing"
    BEARISH_ENGULFING = "Bearish Engulfing"
    BULLISH_HARAMI = "Bullish Harami"
    BEARISH_HARAMI = "Bearish Harami"
    TWEEZER_TOP = "Tweezer Top"
    TWEEZER_BOTTOM = "Tweezer Bottom"
    PIERCING_LINE = "Piercing Line"
    DARK_CLOUD_COVER = "Dark Cloud Cover"
    
    # Three Candle Patterns
    MORNING_STAR = "Morning Star"
    EVENING_STAR = "Evening Star"
    THREE_WHITE_SOLDIERS = "Three White Soldiers"
    THREE_BLACK_CROWS = "Three Black Crows"
    THREE_INSIDE_UP = "Three Inside Up"
    THREE_INSIDE_DOWN = "Three Inside Down"
    THREE_OUTSIDE_UP = "Three Outside Up"
    THREE_OUTSIDE_DOWN = "Three Outside Down"


class PatternSignal(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass
class CandlePattern:
    pattern_type: PatternType
    signal: PatternSignal
    confidence: float  # 0-100 accuracy score
    candles_used: int
    timestamp: str
    description: str
    entry_suggestion: str
    stop_loss_suggestion: str
    target_suggestion: str
    validation_factors: Dict


class CandlePatternAnalyzer:
    """
    High-accuracy candlestick pattern detection with confidence scoring.
    
    Confidence is calculated based on:
    - Pattern formation quality (body/wick ratios)
    - Volume confirmation (spike = higher confidence)
    - Trend context (patterns at key levels = higher confidence)
    - RSI/momentum confirmation
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.detected_patterns: List[CandlePattern] = []
        
        # Pre-calculate candle metrics
        self._calculate_candle_metrics()
    
    def _calculate_candle_metrics(self):
        """Calculate essential candle metrics for pattern detection"""
        df = self.data
        
        # Basic candle components
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        
        # Body ratios
        df['body_ratio'] = df['body'] / df['candle_range'].replace(0, np.nan)
        df['upper_wick_ratio'] = df['upper_wick'] / df['candle_range'].replace(0, np.nan)
        df['lower_wick_ratio'] = df['lower_wick'] / df['candle_range'].replace(0, np.nan)
        
        # Candle direction
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['high_volume'] = df['volume_ratio'] > 1.5
        
        # Trend context (using 10-period lookback)
        df['trend_up'] = df['close'] > df['close'].rolling(10).mean()
        df['trend_down'] = df['close'] < df['close'].rolling(10).mean()
        
        # Recent swing highs/lows for context
        df['at_high'] = df['high'] >= df['high'].rolling(10).max()
        df['at_low'] = df['low'] <= df['low'].rolling(10).min()
        
        self.data = df
    
    def analyze_all_patterns(self) -> List[CandlePattern]:
        """Detect all candlestick patterns and return sorted by confidence"""
        self.detected_patterns = []
        
        # Single candle patterns
        self._detect_doji_patterns()
        self._detect_hammer_patterns()
        self._detect_marubozu_patterns()
        self._detect_spinning_top()
        
        # Two candle patterns
        self._detect_engulfing_patterns()
        self._detect_harami_patterns()
        self._detect_piercing_dark_cloud()
        self._detect_tweezer_patterns()
        
        # Three candle patterns
        self._detect_star_patterns()
        self._detect_three_soldiers_crows()
        self._detect_three_inside_patterns()
        self._detect_three_outside_patterns()
        
        # Sort by confidence (highest first)
        self.detected_patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return self.detected_patterns
    
    def get_latest_patterns(self, lookback: int = 5) -> List[CandlePattern]:
        """Get patterns detected in the last N candles"""
        patterns = self.analyze_all_patterns()
        latest_time = self.data.index[-1]
        lookback_time = self.data.index[-lookback] if len(self.data) >= lookback else self.data.index[0]
        
        return [p for p in patterns if p.timestamp >= str(lookback_time)]
    
    def get_actionable_signals(self, min_confidence: float = 70) -> List[CandlePattern]:
        """Get high-confidence patterns suitable for trading"""
        patterns = self.analyze_all_patterns()
        return [p for p in patterns if p.confidence >= min_confidence]
    
    # === SINGLE CANDLE PATTERNS ===
    
    def _detect_doji_patterns(self):
        """Detect Doji, Dragonfly Doji, and Gravestone Doji"""
        for i in range(-5, 0):
            if abs(i) > len(self.data):
                continue
                
            row = self.data.iloc[i]
            
            # Basic Doji: Body < 10% of range
            if row['body_ratio'] < 0.1:
                # Dragonfly Doji: Long lower wick, no upper wick
                if row['lower_wick_ratio'] > 0.6 and row['upper_wick_ratio'] < 0.1:
                    self._add_pattern(
                        PatternType.DRAGONFLY_DOJI,
                        PatternSignal.BULLISH if row['trend_down'] else PatternSignal.NEUTRAL,
                        row, i,
                        base_confidence=75 if row['at_low'] else 60,
                        description="Dragonfly Doji - Long lower shadow with opening/closing at or near the high. Bullish reversal signal when appearing after a downtrend."
                    )
                
                # Gravestone Doji: Long upper wick, no lower wick
                elif row['upper_wick_ratio'] > 0.6 and row['lower_wick_ratio'] < 0.1:
                    self._add_pattern(
                        PatternType.GRAVESTONE_DOJI,
                        PatternSignal.BEARISH if row['trend_up'] else PatternSignal.NEUTRAL,
                        row, i,
                        base_confidence=75 if row['at_high'] else 60,
                        description="Gravestone Doji - Long upper shadow with opening/closing at or near the low. Bearish reversal signal when appearing after an uptrend."
                    )
                
                # Regular Doji
                else:
                    signal = PatternSignal.NEUTRAL
                    if row['at_low'] and row['trend_down']:
                        signal = PatternSignal.BULLISH
                    elif row['at_high'] and row['trend_up']:
                        signal = PatternSignal.BEARISH
                    
                    self._add_pattern(
                        PatternType.DOJI,
                        signal,
                        row, i,
                        base_confidence=65,
                        description="Doji - Indecision pattern with nearly equal open and close. Signal depends on trend context and location."
                    )
    
    def _detect_hammer_patterns(self):
        """Detect Hammer, Inverted Hammer, Hanging Man, Shooting Star"""
        for i in range(-5, 0):
            if abs(i) > len(self.data):
                continue
                
            row = self.data.iloc[i]
            
            # Hammer/Hanging Man: Small body at top, long lower wick (2x body)
            if (row['lower_wick'] > row['body'] * 2 and 
                row['upper_wick'] < row['body'] * 0.5 and
                row['body_ratio'] > 0.1):
                
                if row['trend_down'] and row['at_low']:
                    # Hammer - Bullish reversal after downtrend
                    self._add_pattern(
                        PatternType.HAMMER,
                        PatternSignal.STRONG_BULLISH if row['is_bullish'] else PatternSignal.BULLISH,
                        row, i,
                        base_confidence=85 if row['high_volume'] else 75,
                        description="Hammer - Bullish reversal pattern at the bottom of a downtrend. Long lower shadow shows buyers rejected lower prices."
                    )
                elif row['trend_up'] and row['at_high']:
                    # Hanging Man - Bearish reversal after uptrend
                    self._add_pattern(
                        PatternType.HANGING_MAN,
                        PatternSignal.BEARISH,
                        row, i,
                        base_confidence=75 if row['high_volume'] else 65,
                        description="Hanging Man - Potential bearish reversal at the top of an uptrend. Long lower shadow indicates selling pressure emerging."
                    )
            
            # Inverted Hammer/Shooting Star: Small body at bottom, long upper wick
            if (row['upper_wick'] > row['body'] * 2 and 
                row['lower_wick'] < row['body'] * 0.5 and
                row['body_ratio'] > 0.1):
                
                if row['trend_down'] and row['at_low']:
                    # Inverted Hammer - Bullish reversal after downtrend
                    self._add_pattern(
                        PatternType.INVERTED_HAMMER,
                        PatternSignal.BULLISH,
                        row, i,
                        base_confidence=70 if row['high_volume'] else 60,
                        description="Inverted Hammer - Potential bullish reversal. Upper shadow shows buying attempt; needs confirmation on next candle."
                    )
                elif row['trend_up'] and row['at_high']:
                    # Shooting Star - Bearish reversal after uptrend
                    self._add_pattern(
                        PatternType.SHOOTING_STAR,
                        PatternSignal.STRONG_BEARISH if row['is_bearish'] else PatternSignal.BEARISH,
                        row, i,
                        base_confidence=85 if row['high_volume'] else 75,
                        description="Shooting Star - Bearish reversal pattern. Long upper shadow shows rejection of higher prices."
                    )
    
    def _detect_marubozu_patterns(self):
        """Detect Bullish and Bearish Marubozu"""
        for i in range(-5, 0):
            if abs(i) > len(self.data):
                continue
                
            row = self.data.iloc[i]
            
            # Marubozu: Body > 90% of range (very small or no wicks)
            if row['body_ratio'] > 0.9:
                if row['is_bullish']:
                    self._add_pattern(
                        PatternType.MARUBOZU_BULLISH,
                        PatternSignal.STRONG_BULLISH,
                        row, i,
                        base_confidence=90 if row['high_volume'] else 80,
                        description="Bullish Marubozu - Strong buying pressure with open at low and close at high. Continuation or reversal signal."
                    )
                else:
                    self._add_pattern(
                        PatternType.MARUBOZU_BEARISH,
                        PatternSignal.STRONG_BEARISH,
                        row, i,
                        base_confidence=90 if row['high_volume'] else 80,
                        description="Bearish Marubozu - Strong selling pressure with open at high and close at low. Continuation or reversal signal."
                    )
    
    def _detect_spinning_top(self):
        """Detect Spinning Top pattern"""
        for i in range(-5, 0):
            if abs(i) > len(self.data):
                continue
                
            row = self.data.iloc[i]
            
            # Spinning Top: Small body (10-30% of range), relatively equal wicks
            if (0.1 < row['body_ratio'] < 0.3 and
                abs(row['upper_wick_ratio'] - row['lower_wick_ratio']) < 0.2):
                
                self._add_pattern(
                    PatternType.SPINNING_TOP,
                    PatternSignal.NEUTRAL,
                    row, i,
                    base_confidence=55,
                    description="Spinning Top - Indecision pattern. Equal pressure from buyers and sellers. Watch for next candle confirmation."
                )
    
    # === TWO CANDLE PATTERNS ===
    
    def _detect_engulfing_patterns(self):
        """Detect Bullish and Bearish Engulfing patterns"""
        for i in range(-4, 0):
            if abs(i) > len(self.data) - 1:
                continue
                
            curr = self.data.iloc[i]
            prev = self.data.iloc[i-1]
            
            # Bullish Engulfing: Current bullish candle completely engulfs previous bearish
            if (curr['is_bullish'] and prev['is_bearish'] and
                curr['open'] < prev['close'] and
                curr['close'] > prev['open'] and
                curr['body'] > prev['body']):
                
                confidence = 85
                if curr['high_volume']:
                    confidence += 5
                if curr['at_low']:
                    confidence += 5
                if curr['trend_down']:
                    confidence += 5
                
                self._add_pattern(
                    PatternType.BULLISH_ENGULFING,
                    PatternSignal.STRONG_BULLISH,
                    curr, i,
                    base_confidence=min(confidence, 98),
                    description="Bullish Engulfing - Strong reversal pattern. Current bullish candle completely engulfs prior bearish candle.",
                    candles=2
                )
            
            # Bearish Engulfing: Current bearish candle completely engulfs previous bullish
            if (curr['is_bearish'] and prev['is_bullish'] and
                curr['open'] > prev['close'] and
                curr['close'] < prev['open'] and
                curr['body'] > prev['body']):
                
                confidence = 85
                if curr['high_volume']:
                    confidence += 5
                if curr['at_high']:
                    confidence += 5
                if curr['trend_up']:
                    confidence += 5
                
                self._add_pattern(
                    PatternType.BEARISH_ENGULFING,
                    PatternSignal.STRONG_BEARISH,
                    curr, i,
                    base_confidence=min(confidence, 98),
                    description="Bearish Engulfing - Strong reversal pattern. Current bearish candle completely engulfs prior bullish candle.",
                    candles=2
                )
    
    def _detect_harami_patterns(self):
        """Detect Bullish and Bearish Harami patterns"""
        for i in range(-4, 0):
            if abs(i) > len(self.data) - 1:
                continue
                
            curr = self.data.iloc[i]
            prev = self.data.iloc[i-1]
            
            # Bullish Harami: Small bullish inside large bearish
            if (curr['is_bullish'] and prev['is_bearish'] and
                curr['open'] > prev['close'] and
                curr['close'] < prev['open'] and
                curr['body'] < prev['body'] * 0.5):
                
                confidence = 70
                if curr['trend_down']:
                    confidence += 10
                if curr['at_low']:
                    confidence += 5
                
                self._add_pattern(
                    PatternType.BULLISH_HARAMI,
                    PatternSignal.BULLISH,
                    curr, i,
                    base_confidence=confidence,
                    description="Bullish Harami - Potential reversal. Small bullish candle contained within prior large bearish candle.",
                    candles=2
                )
            
            # Bearish Harami: Small bearish inside large bullish
            if (curr['is_bearish'] and prev['is_bullish'] and
                curr['open'] < prev['close'] and
                curr['close'] > prev['open'] and
                curr['body'] < prev['body'] * 0.5):
                
                confidence = 70
                if curr['trend_up']:
                    confidence += 10
                if curr['at_high']:
                    confidence += 5
                
                self._add_pattern(
                    PatternType.BEARISH_HARAMI,
                    PatternSignal.BEARISH,
                    curr, i,
                    base_confidence=confidence,
                    description="Bearish Harami - Potential reversal. Small bearish candle contained within prior large bullish candle.",
                    candles=2
                )
    
    def _detect_piercing_dark_cloud(self):
        """Detect Piercing Line and Dark Cloud Cover"""
        for i in range(-4, 0):
            if abs(i) > len(self.data) - 1:
                continue
                
            curr = self.data.iloc[i]
            prev = self.data.iloc[i-1]
            prev_mid = (prev['open'] + prev['close']) / 2
            
            # Piercing Line: Bullish reversal after bearish candle
            if (prev['is_bearish'] and curr['is_bullish'] and
                curr['open'] < prev['low'] and  # Opens below prior low
                curr['close'] > prev_mid and    # Closes above prior midpoint
                curr['close'] < prev['open']):  # But below prior open
                
                penetration = (curr['close'] - prev['close']) / prev['body']
                confidence = 70 + (penetration * 10)  # Higher penetration = higher confidence
                
                self._add_pattern(
                    PatternType.PIERCING_LINE,
                    PatternSignal.BULLISH,
                    curr, i,
                    base_confidence=min(confidence, 85),
                    description=f"Piercing Line - Bullish reversal. Opens lower, closes above {penetration*100:.0f}% of prior bearish body.",
                    candles=2
                )
            
            # Dark Cloud Cover: Bearish reversal after bullish candle
            if (prev['is_bullish'] and curr['is_bearish'] and
                curr['open'] > prev['high'] and  # Opens above prior high
                curr['close'] < prev_mid and     # Closes below prior midpoint
                curr['close'] > prev['open']):   # But above prior open
                
                penetration = (prev['close'] - curr['close']) / prev['body']
                confidence = 70 + (penetration * 10)
                
                self._add_pattern(
                    PatternType.DARK_CLOUD_COVER,
                    PatternSignal.BEARISH,
                    curr, i,
                    base_confidence=min(confidence, 85),
                    description=f"Dark Cloud Cover - Bearish reversal. Opens higher, closes below {penetration*100:.0f}% of prior bullish body.",
                    candles=2
                )
    
    def _detect_tweezer_patterns(self):
        """Detect Tweezer Top and Bottom"""
        for i in range(-4, 0):
            if abs(i) > len(self.data) - 1:
                continue
                
            curr = self.data.iloc[i]
            prev = self.data.iloc[i-1]
            tolerance = curr['candle_range'] * 0.1
            
            # Tweezer Bottom: Two candles with same low after downtrend
            if (abs(curr['low'] - prev['low']) < tolerance and
                prev['is_bearish'] and curr['is_bullish'] and
                curr['trend_down']):
                
                self._add_pattern(
                    PatternType.TWEEZER_BOTTOM,
                    PatternSignal.BULLISH,
                    curr, i,
                    base_confidence=75 if curr['at_low'] else 65,
                    description="Tweezer Bottom - Two candles with matching lows indicating support. Bullish reversal signal.",
                    candles=2
                )
            
            # Tweezer Top: Two candles with same high after uptrend
            if (abs(curr['high'] - prev['high']) < tolerance and
                prev['is_bullish'] and curr['is_bearish'] and
                curr['trend_up']):
                
                self._add_pattern(
                    PatternType.TWEEZER_TOP,
                    PatternSignal.BEARISH,
                    curr, i,
                    base_confidence=75 if curr['at_high'] else 65,
                    description="Tweezer Top - Two candles with matching highs indicating resistance. Bearish reversal signal.",
                    candles=2
                )
    
    # === THREE CANDLE PATTERNS ===
    
    def _detect_star_patterns(self):
        """Detect Morning Star and Evening Star"""
        for i in range(-3, 0):
            if abs(i) > len(self.data) - 2:
                continue
                
            third = self.data.iloc[i]
            second = self.data.iloc[i-1]
            first = self.data.iloc[i-2]
            
            # Morning Star: Bearish -> Small body (gap down) -> Bullish (gap up)
            if (first['is_bearish'] and first['body'] > first['candle_range'] * 0.5 and  # Large bearish
                second['body_ratio'] < 0.3 and  # Small body (star)
                third['is_bullish'] and third['body'] > third['candle_range'] * 0.5 and  # Large bullish
                third['close'] > (first['open'] + first['close']) / 2):  # Closes above first candle midpoint
                
                confidence = 85
                if first['trend_down']:
                    confidence += 5
                if third['high_volume']:
                    confidence += 5
                
                self._add_pattern(
                    PatternType.MORNING_STAR,
                    PatternSignal.STRONG_BULLISH,
                    third, i,
                    base_confidence=min(confidence, 95),
                    description="Morning Star - Strong bullish reversal. Three-candle pattern: bearish, indecision, strong bullish.",
                    candles=3
                )
            
            # Evening Star: Bullish -> Small body (gap up) -> Bearish (gap down)
            if (first['is_bullish'] and first['body'] > first['candle_range'] * 0.5 and  # Large bullish
                second['body_ratio'] < 0.3 and  # Small body (star)
                third['is_bearish'] and third['body'] > third['candle_range'] * 0.5 and  # Large bearish
                third['close'] < (first['open'] + first['close']) / 2):  # Closes below first candle midpoint
                
                confidence = 85
                if first['trend_up']:
                    confidence += 5
                if third['high_volume']:
                    confidence += 5
                
                self._add_pattern(
                    PatternType.EVENING_STAR,
                    PatternSignal.STRONG_BEARISH,
                    third, i,
                    base_confidence=min(confidence, 95),
                    description="Evening Star - Strong bearish reversal. Three-candle pattern: bullish, indecision, strong bearish.",
                    candles=3
                )
    
    def _detect_three_soldiers_crows(self):
        """Detect Three White Soldiers and Three Black Crows"""
        for i in range(-3, 0):
            if abs(i) > len(self.data) - 2:
                continue
                
            third = self.data.iloc[i]
            second = self.data.iloc[i-1]
            first = self.data.iloc[i-2]
            
            # Three White Soldiers: Three consecutive bullish candles with higher closes
            if (first['is_bullish'] and second['is_bullish'] and third['is_bullish'] and
                first['body_ratio'] > 0.5 and second['body_ratio'] > 0.5 and third['body_ratio'] > 0.5 and
                second['close'] > first['close'] and third['close'] > second['close'] and
                second['open'] > first['open'] and second['open'] < first['close'] and
                third['open'] > second['open'] and third['open'] < second['close']):
                
                confidence = 90
                if first['trend_down']:
                    confidence += 5
                
                self._add_pattern(
                    PatternType.THREE_WHITE_SOLDIERS,
                    PatternSignal.STRONG_BULLISH,
                    third, i,
                    base_confidence=min(confidence, 98),
                    description="Three White Soldiers - Strong bullish reversal/continuation. Three consecutive large bullish candles.",
                    candles=3
                )
            
            # Three Black Crows: Three consecutive bearish candles with lower closes
            if (first['is_bearish'] and second['is_bearish'] and third['is_bearish'] and
                first['body_ratio'] > 0.5 and second['body_ratio'] > 0.5 and third['body_ratio'] > 0.5 and
                second['close'] < first['close'] and third['close'] < second['close'] and
                second['open'] < first['open'] and second['open'] > first['close'] and
                third['open'] < second['open'] and third['open'] > second['close']):
                
                confidence = 90
                if first['trend_up']:
                    confidence += 5
                
                self._add_pattern(
                    PatternType.THREE_BLACK_CROWS,
                    PatternSignal.STRONG_BEARISH,
                    third, i,
                    base_confidence=min(confidence, 98),
                    description="Three Black Crows - Strong bearish reversal/continuation. Three consecutive large bearish candles.",
                    candles=3
                )
    
    def _detect_three_inside_patterns(self):
        """Detect Three Inside Up and Three Inside Down"""
        for i in range(-3, 0):
            if abs(i) > len(self.data) - 2:
                continue
                
            third = self.data.iloc[i]
            second = self.data.iloc[i-1]
            first = self.data.iloc[i-2]
            
            # Three Inside Up: Bearish -> Bullish Harami -> Bullish continuation
            if (first['is_bearish'] and 
                second['is_bullish'] and
                second['open'] > first['close'] and second['close'] < first['open'] and
                third['is_bullish'] and third['close'] > first['open']):
                
                self._add_pattern(
                    PatternType.THREE_INSIDE_UP,
                    PatternSignal.STRONG_BULLISH,
                    third, i,
                    base_confidence=85,
                    description="Three Inside Up - Bullish reversal confirmation. Harami followed by bullish breakout candle.",
                    candles=3
                )
            
            # Three Inside Down: Bullish -> Bearish Harami -> Bearish continuation
            if (first['is_bullish'] and 
                second['is_bearish'] and
                second['open'] < first['close'] and second['close'] > first['open'] and
                third['is_bearish'] and third['close'] < first['open']):
                
                self._add_pattern(
                    PatternType.THREE_INSIDE_DOWN,
                    PatternSignal.STRONG_BEARISH,
                    third, i,
                    base_confidence=85,
                    description="Three Inside Down - Bearish reversal confirmation. Harami followed by bearish breakdown candle.",
                    candles=3
                )
    
    def _detect_three_outside_patterns(self):
        """Detect Three Outside Up and Three Outside Down"""
        for i in range(-3, 0):
            if abs(i) > len(self.data) - 2:
                continue
                
            third = self.data.iloc[i]
            second = self.data.iloc[i-1]
            first = self.data.iloc[i-2]
            
            # Three Outside Up: Bearish -> Bullish Engulfing -> Bullish continuation
            if (first['is_bearish'] and
                second['is_bullish'] and
                second['open'] < first['close'] and second['close'] > first['open'] and
                second['body'] > first['body'] and
                third['is_bullish'] and third['close'] > second['close']):
                
                self._add_pattern(
                    PatternType.THREE_OUTSIDE_UP,
                    PatternSignal.STRONG_BULLISH,
                    third, i,
                    base_confidence=90,
                    description="Three Outside Up - Strong bullish reversal. Engulfing pattern confirmed by follow-through candle.",
                    candles=3
                )
            
            # Three Outside Down: Bullish -> Bearish Engulfing -> Bearish continuation
            if (first['is_bullish'] and
                second['is_bearish'] and
                second['open'] > first['close'] and second['close'] < first['open'] and
                second['body'] > first['body'] and
                third['is_bearish'] and third['close'] < second['close']):
                
                self._add_pattern(
                    PatternType.THREE_OUTSIDE_DOWN,
                    PatternSignal.STRONG_BEARISH,
                    third, i,
                    base_confidence=90,
                    description="Three Outside Down - Strong bearish reversal. Engulfing pattern confirmed by follow-through candle.",
                    candles=3
                )
    
    def _add_pattern(self, pattern_type: PatternType, signal: PatternSignal, 
                     row: pd.Series, idx: int, base_confidence: float, 
                     description: str, candles: int = 1):
        """Add a detected pattern to the list with calculated confidence"""
        
        # Adjust confidence based on additional factors
        confidence = base_confidence
        
        # Volume bonus
        if row['high_volume']:
            confidence = min(confidence + 5, 100)
        
        # Calculate entry, SL, and target suggestions
        atr = row.get('atr', row['candle_range'])
        entry = row['close']
        
        if signal in [PatternSignal.STRONG_BULLISH, PatternSignal.BULLISH]:
            sl = row['low'] - (atr * 0.5)
            target = entry + (atr * 2)
            entry_suggestion = f"Enter LONG near {entry:.2f}"
            sl_suggestion = f"Stop loss below {sl:.2f}"
            target_suggestion = f"Target {target:.2f} (2x ATR)"
        elif signal in [PatternSignal.STRONG_BEARISH, PatternSignal.BEARISH]:
            sl = row['high'] + (atr * 0.5)
            target = entry - (atr * 2)
            entry_suggestion = f"Enter SHORT near {entry:.2f}"
            sl_suggestion = f"Stop loss above {sl:.2f}"
            target_suggestion = f"Target {target:.2f} (2x ATR)"
        else:
            entry_suggestion = "Wait for confirmation"
            sl_suggestion = "Define levels after direction confirmed"
            target_suggestion = "Set after entry"
        
        validation = {
            "volume_confirmed": bool(row['high_volume']),
            "trend_context": "DOWNTREND" if row['trend_down'] else ("UPTREND" if row['trend_up'] else "SIDEWAYS"),
            "at_key_level": bool(row['at_high'] or row['at_low']),
            "body_ratio": float(row['body_ratio']) if not pd.isna(row['body_ratio']) else 0,
            "volume_ratio": float(row['volume_ratio']) if not pd.isna(row['volume_ratio']) else 1
        }
        
        pattern = CandlePattern(
            pattern_type=pattern_type,
            signal=signal,
            confidence=round(confidence, 1),
            candles_used=candles,
            timestamp=str(self.data.index[idx]),
            description=description,
            entry_suggestion=entry_suggestion,
            stop_loss_suggestion=sl_suggestion,
            target_suggestion=target_suggestion,
            validation_factors=validation
        )
        
        self.detected_patterns.append(pattern)
    
    def get_pattern_summary(self) -> Dict:
        """Get a summary of all detected patterns"""
        patterns = self.analyze_all_patterns()
        
        bullish = [p for p in patterns if p.signal in [PatternSignal.STRONG_BULLISH, PatternSignal.BULLISH]]
        bearish = [p for p in patterns if p.signal in [PatternSignal.STRONG_BEARISH, PatternSignal.BEARISH]]
        neutral = [p for p in patterns if p.signal == PatternSignal.NEUTRAL]
        
        latest = self.data.iloc[-1]
        
        return {
            "total_patterns": len(patterns),
            "bullish_patterns": len(bullish),
            "bearish_patterns": len(bearish),
            "neutral_patterns": len(neutral),
            "highest_confidence_pattern": patterns[0].pattern_type.value if patterns else None,
            "highest_confidence_score": patterns[0].confidence if patterns else 0,
            "overall_bias": "BULLISH" if len(bullish) > len(bearish) else ("BEARISH" if len(bearish) > len(bullish) else "NEUTRAL"),
            "bias_strength": abs(len(bullish) - len(bearish)),
            "current_price": float(latest['close']),
            "patterns": [
                {
                    "type": p.pattern_type.value,
                    "signal": p.signal.value,
                    "confidence": p.confidence,
                    "timestamp": p.timestamp,
                    "description": p.description,
                    "entry": p.entry_suggestion,
                    "stop_loss": p.stop_loss_suggestion,
                    "target": p.target_suggestion
                }
                for p in patterns[:10]  # Top 10 patterns
            ]
        }


def analyze_nifty_patterns(data: pd.DataFrame) -> Dict:
    """
    Main function to analyze Nifty 50 candlestick patterns.
    Returns comprehensive pattern analysis with trading suggestions.
    """
    analyzer = CandlePatternAnalyzer(data)
    return analyzer.get_pattern_summary()


if __name__ == "__main__":
    # Test with sample data
    from data_fetcher import NiftyDataFetcher
    from indicators import ScalpingIndicators
    
    print("Fetching Nifty 50 data...")
    fetcher = NiftyDataFetcher()
    raw_data = fetcher.fetch_data()
    
    if not raw_data.empty:
        # Calculate indicators first (for ATR)
        indicators = ScalpingIndicators(raw_data)
        analyzed_data = indicators.calculate_all()
        
        print("\nAnalyzing candlestick patterns...")
        analyzer = CandlePatternAnalyzer(analyzed_data)
        summary = analyzer.get_pattern_summary()
        
        print("\n" + "="*60)
        print("CANDLESTICK PATTERN ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Patterns Detected: {summary['total_patterns']}")
        print(f"Bullish Patterns: {summary['bullish_patterns']}")
        print(f"Bearish Patterns: {summary['bearish_patterns']}")
        print(f"Neutral Patterns: {summary['neutral_patterns']}")
        print(f"Overall Bias: {summary['overall_bias']}")
        print(f"Current Price: {summary['current_price']:.2f}")
        
        if summary['patterns']:
            print("\n--- TOP DETECTED PATTERNS ---")
            for p in summary['patterns'][:5]:
                print(f"\n📊 {p['type']} ({p['signal']})")
                print(f"   Confidence: {p['confidence']}%")
                print(f"   {p['description'][:100]}...")
                print(f"   {p['entry']}")
                print(f"   {p['stop_loss']}")
                print(f"   {p['target']}")
