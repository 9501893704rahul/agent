# 🎯 Nifty 50 5-Minute Scalping Agent

An advanced AI-powered scalping research agent for Nifty 50 index trading on the 5-minute timeframe. Built with OpenRouter API integration for intelligent trade signal generation and market analysis.

![Dashboard](https://img.shields.io/badge/Dashboard-Live-green)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-purple)

## 🚀 Features

### Real-Time Market Analysis
- **Live Nifty 50 Data**: Fetches 5-minute OHLCV data via Yahoo Finance
- **Technical Indicators**: RSI, MACD, EMA (9/21/50), Bollinger Bands, VWAP, ATR, Stochastic
- **Support/Resistance Levels**: Dynamic swing high/low detection

### AI-Powered Scalping Strategies
1. **VWAP Bounce** - Trade pullbacks to VWAP in trending markets
2. **EMA Crossover** - Momentum shifts using EMA 9/21 crossovers
3. **RSI Divergence** - Catch reversals at overbought/oversold extremes
4. **Bollinger Squeeze** - Volatility expansion breakout trades
5. **Momentum Breakout** - Strong directional moves with volume
6. **Pullback Entry** - Trend continuation on EMA pullbacks
7. **Scalp Reversal** - Quick trades at technical extremes

### LLM Integration (OpenRouter)
- **Market Commentary**: AI-generated analysis of current conditions
- **Trade Recommendations**: Confidence-rated entry/exit levels
- **Risk Assessment**: Dynamic position sizing recommendations
- **Interactive Chat**: Ask questions about market conditions

## 📊 Dashboard Preview

The web dashboard provides:
- Current price and market bias
- Active trade setups with entry, stop-loss, and targets
- Technical indicator status
- Key support/resistance levels
- AI analysis panel
- Interactive chat with the agent

## 🛠️ Installation

```bash
# Clone the repository
cd /workspace/project

# Install dependencies
pip install -r requirements.txt

# Set your OpenRouter API key (optional - has offline mode)
export OPENROUTER_API_KEY="your-api-key"

# Run the application
python app.py
```

## 📁 Project Structure

```
project/
├── app.py                 # Main Flask application & web dashboard
├── config.py              # Configuration settings
├── data_fetcher.py        # Yahoo Finance data fetching
├── indicators.py          # Technical indicators calculation
├── scalping_strategies.py # 7 scalping strategy implementations
├── openai_agent.py        # OpenRouter LLM agent
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Technical Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_FAST = 9
EMA_SLOW = 21
EMA_SIGNAL = 50

# Scalping Parameters
MIN_RISK_REWARD = 1.5
STOP_LOSS_ATR_MULTIPLIER = 1.5
TARGET_ATR_MULTIPLIER = 2.0

# LLM Model
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/api/health` | GET | Health check |
| `/api/analysis` | GET | Full market analysis |
| `/api/ai-analysis` | POST | AI-powered analysis |
| `/api/chat` | POST | Chat with agent |
| `/api/scalp-ideas` | GET | Generate trade ideas |
| `/api/price` | GET | Current price |

## 📈 Trade Setup Example

```json
{
  "strategy": "Pullback Entry",
  "direction": "SHORT",
  "entry_price": 25796.15,
  "stop_loss": 25821.53,
  "target_1": 25753.96,
  "target_2": 25711.79,
  "confidence": 75,
  "risk_reward": 1.66,
  "reasoning": "Pullback sell in downtrend. Price rallied to EMA zone and showing rejection."
}
```

## 🤖 Supported LLM Models

Via OpenRouter:
- `meta-llama/llama-3.1-8b-instruct:free` (Default - Free)
- `anthropic/claude-3.5-sonnet` (Premium)
- `openai/gpt-4o` (Premium)
- `google/gemini-pro` (Premium)

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. 

- Not financial advice
- Past performance does not guarantee future results
- Always do your own research before trading
- Use proper risk management
- Paper trade before using real money

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for market data
- [OpenRouter](https://openrouter.ai/) for LLM API access
- [TA Library](https://github.com/bukosabino/ta) for technical indicators
