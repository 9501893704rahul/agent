"""Configuration settings for Nifty 50 Scalping Agent"""

import os
from dotenv import load_dotenv

load_dotenv()

# Nifty 50 Symbol (Yahoo Finance format)
NIFTY_SYMBOL = "^NSEI"  # Nifty 50 Index
NIFTY_FUTURES = "NIFTY24JANFUT.NS"  # Update this based on current expiry

# Alternative symbols for components
NIFTY_ETF = "NIFTYBEES.NS"  # Nifty BeES ETF

# Timeframe settings
INTERVAL = "5m"  # 5-minute candles
LOOKBACK_PERIOD = "5d"  # Maximum allowed for 5m data on yfinance

# Technical Indicator Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

EMA_FAST = 9
EMA_SLOW = 21
EMA_SIGNAL = 50

VWAP_DEVIATION_THRESHOLD = 0.5  # Percentage deviation from VWAP

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

ATR_PERIOD = 14

# Scalping Parameters
MIN_RISK_REWARD = 1.5
MAX_TRADE_DURATION_CANDLES = 12  # 60 minutes max for scalp
STOP_LOSS_ATR_MULTIPLIER = 1.5
TARGET_ATR_MULTIPLIER = 2.0

# Volume thresholds
VOLUME_SPIKE_THRESHOLD = 1.5  # 1.5x average volume

# LLM Configuration - OpenRouter
LLM_PROVIDER = "openrouter"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Using cost-effective model for scalping research
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")  # Free tier model

# Server Configuration
HOST = "0.0.0.0"
PORT = 12000
