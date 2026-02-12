"""
Nifty 50 5-Minute Scalping Research Agent - Main Application

Flask-based web application providing real-time scalping analysis
powered by OpenAI GPT models with PAPER TRADING and REAL-TIME MONITORING.
"""

from flask import Flask, jsonify, request, render_template_string, Response
from flask_cors import CORS
import json
import os
from datetime import datetime
import traceback
import numpy as np
import threading
import time

import config


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


from data_fetcher import NiftyDataFetcher
from indicators import ScalpingIndicators, get_candle_pattern_analysis
from scalping_strategies import StrategyEngine
from openai_agent import OpenAIScalpingAgent, create_openai_agent
from realtime_agent import get_realtime_agent, get_paper_engine, RealTimeScalpingAgent
from paper_trading import PaperTradingEngine
from candle_patterns import CandlePatternAnalyzer, analyze_nifty_patterns

app = Flask(__name__)
app.json_encoder = NumpyEncoder
CORS(app)

# Global instances
data_fetcher = NiftyDataFetcher()
agent = None
realtime_agent = None
paper_engine = None


def get_agent():
    """Get or create the OpenAI agent"""
    global agent
    if agent is None:
        agent = create_openai_agent()
    return agent


def get_rt_agent() -> RealTimeScalpingAgent:
    """Get or create the real-time agent"""
    global realtime_agent, paper_engine
    if realtime_agent is None:
        paper_engine = get_paper_engine()
        realtime_agent = get_realtime_agent()
    return realtime_agent


def convert_numpy(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        val = float(obj)
        # Handle NaN and Inf
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy(x) for x in obj.tolist()]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def get_full_analysis():
    """Get complete market analysis with candlestick patterns"""
    # Fetch fresh data
    raw_data = data_fetcher.fetch_data()
    
    if raw_data.empty:
        return None, "Failed to fetch market data"
    
    # Calculate indicators
    indicators = ScalpingIndicators(raw_data)
    analyzed_data = indicators.calculate_all()
    
    # Get current analysis
    current_analysis = indicators.get_current_analysis()
    market_data = data_fetcher.get_current_price()
    
    # Run strategy engine
    strategy_engine = StrategyEngine(analyzed_data)
    setups = strategy_engine.run_all_strategies()
    market_bias = strategy_engine.get_market_bias()
    
    # Get support/resistance
    sr_levels = indicators.get_support_resistance()
    
    # === CANDLESTICK PATTERN ANALYSIS ===
    pattern_analyzer = CandlePatternAnalyzer(analyzed_data)
    pattern_summary = pattern_analyzer.get_pattern_summary()
    actionable_patterns = pattern_analyzer.get_actionable_signals(70)  # 70%+ confidence
    
    result = {
        "market_data": market_data,
        "analysis": current_analysis,
        "setups": [
            {
                "strategy": s.strategy.value,
                "signal": s.signal.value,
                "direction": s.direction,
                "entry_price": s.entry_price,
                "stop_loss": s.stop_loss,
                "target_1": s.target_1,
                "target_2": s.target_2,
                "confidence": s.confidence,
                "risk_reward": s.risk_reward,
                "reasoning": s.reasoning,
                "indicators": s.indicators
            }
            for s in setups
        ],
        "market_bias": market_bias,
        "support_resistance": sr_levels,
        "candle_patterns": pattern_summary,
        "actionable_patterns": [
            {
                "type": p.pattern_type.value,
                "signal": p.signal.value,
                "confidence": p.confidence,
                "description": p.description,
                "entry": p.entry_suggestion,
                "stop_loss": p.stop_loss_suggestion,
                "target": p.target_suggestion,
                "validation": p.validation_factors
            }
            for p in actionable_patterns[:10]
        ],
        "raw_data": analyzed_data.tail(20).to_dict(orient="records")
    }
    
    # Convert all numpy types
    return convert_numpy(result), None


# HTML Template for Dashboard
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nifty 50 Scalping Agent</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .gradient-bg { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
        .card { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .bullish { color: #10b981; }
        .bearish { color: #ef4444; }
        .neutral { color: #f59e0b; }
    </style>
</head>
<body class="gradient-bg min-h-screen text-white">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="flex justify-between items-center mb-8">
            <div>
                <h1 class="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                    🎯 Nifty 50 Scalping Agent
                </h1>
                <p class="text-gray-400 mt-1">5-Minute AI-Powered Research</p>
            </div>
            <div class="flex items-center gap-4">
                <span id="lastUpdate" class="text-sm text-gray-400"></span>
                <button onclick="refreshData()" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg flex items-center gap-2">
                    <span id="refreshIcon">🔄</span> Refresh
                </button>
            </div>
        </div>

        <!-- Market Overview -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div class="card rounded-xl p-4">
                <div class="text-gray-400 text-sm">Current Price</div>
                <div id="currentPrice" class="text-2xl font-bold">--</div>
                <div id="priceChange" class="text-sm">--</div>
            </div>
            <div class="card rounded-xl p-4">
                <div class="text-gray-400 text-sm">Market Bias</div>
                <div id="marketBias" class="text-2xl font-bold">--</div>
                <div id="biasStrength" class="text-sm text-gray-400">--</div>
            </div>
            <div class="card rounded-xl p-4">
                <div class="text-gray-400 text-sm">RSI</div>
                <div id="rsiValue" class="text-2xl font-bold">--</div>
                <div id="rsiCondition" class="text-sm">--</div>
            </div>
            <div class="card rounded-xl p-4">
                <div class="text-gray-400 text-sm">VWAP Position</div>
                <div id="vwapPosition" class="text-2xl font-bold">--</div>
                <div id="vwapDeviation" class="text-sm text-gray-400">--</div>
            </div>
        </div>

        <!-- Main Content Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Trade Setups -->
            <div class="lg:col-span-2 card rounded-xl p-6">
                <h2 class="text-xl font-semibold mb-4 flex items-center gap-2">
                    📊 Active Trade Setups
                </h2>
                <div id="tradeSetups" class="space-y-4">
                    <div class="text-gray-400 text-center py-8">Loading...</div>
                </div>
            </div>

            <!-- AI Analysis -->
            <div class="card rounded-xl p-6">
                <h2 class="text-xl font-semibold mb-4 flex items-center gap-2">
                    🤖 AI Analysis
                </h2>
                <div id="aiAnalysis" class="text-gray-300 text-sm">
                    <div class="text-gray-400 text-center py-8">Loading AI insights...</div>
                </div>
            </div>
        </div>

        <!-- Technical Indicators -->
        <div class="mt-6 card rounded-xl p-6">
            <h2 class="text-xl font-semibold mb-4">📈 Technical Indicators</h2>
            <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4" id="indicators">
                <div class="text-gray-400 text-center py-4 col-span-full">Loading...</div>
            </div>
        </div>

        <!-- Key Levels -->
        <div class="mt-6 card rounded-xl p-6">
            <h2 class="text-xl font-semibold mb-4">🎯 Key Levels</h2>
            <div class="grid grid-cols-2 md:grid-cols-5 gap-4" id="keyLevels">
                <div class="text-gray-400 text-center py-4 col-span-full">Loading...</div>
            </div>
        </div>

        <!-- Chat Interface -->
        <div class="mt-6 card rounded-xl p-6">
            <h2 class="text-xl font-semibold mb-4">💬 Ask the Agent</h2>
            <div class="flex gap-2">
                <input type="text" id="chatInput" 
                    class="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white"
                    placeholder="Ask about market conditions, trade ideas, or strategies..."
                    onkeypress="if(event.key==='Enter')sendChat()">
                <button onclick="sendChat()" class="bg-purple-600 hover:bg-purple-700 px-6 py-2 rounded-lg">
                    Send
                </button>
            </div>
            <div id="chatResponse" class="mt-4 text-gray-300 text-sm hidden">
            </div>
        </div>
    </div>

    <script>
        let currentData = null;

        async function refreshData() {
            document.getElementById('refreshIcon').classList.add('pulse');
            try {
                const response = await fetch('/api/analysis');
                const data = await response.json();
                
                if (data.error) {
                    console.error(data.error);
                    return;
                }
                
                currentData = data;
                updateUI(data);
                
                // Get AI analysis
                const aiResponse = await fetch('/api/ai-analysis', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const aiData = await aiResponse.json();
                updateAIAnalysis(aiData);
                
            } catch (error) {
                console.error('Error:', error);
            }
            document.getElementById('refreshIcon').classList.remove('pulse');
            document.getElementById('lastUpdate').textContent = 'Updated: ' + new Date().toLocaleTimeString();
        }

        function updateUI(data) {
            // Price
            const price = data.market_data?.close || '--';
            const change = data.market_data?.change_pct || 0;
            document.getElementById('currentPrice').textContent = typeof price === 'number' ? price.toFixed(2) : price;
            const changeEl = document.getElementById('priceChange');
            changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
            changeEl.className = `text-sm ${change >= 0 ? 'bullish' : 'bearish'}`;

            // Bias
            const bias = data.market_bias?.bias || '--';
            const biasEl = document.getElementById('marketBias');
            biasEl.textContent = bias;
            biasEl.className = `text-2xl font-bold ${bias.includes('BULLISH') ? 'bullish' : bias.includes('BEARISH') ? 'bearish' : 'neutral'}`;
            document.getElementById('biasStrength').textContent = `Strength: ${data.market_bias?.strength || 0}%`;

            // RSI
            const rsi = data.analysis?.rsi?.value || '--';
            document.getElementById('rsiValue').textContent = typeof rsi === 'number' ? rsi.toFixed(1) : rsi;
            const rsiCond = data.analysis?.rsi?.condition || 'NEUTRAL';
            const rsiCondEl = document.getElementById('rsiCondition');
            rsiCondEl.textContent = rsiCond;
            rsiCondEl.className = `text-sm ${rsiCond === 'OVERSOLD' ? 'bullish' : rsiCond === 'OVERBOUGHT' ? 'bearish' : 'neutral'}`;

            // VWAP
            const vwapPos = data.analysis?.vwap?.position || '--';
            const vwapEl = document.getElementById('vwapPosition');
            vwapEl.textContent = vwapPos;
            vwapEl.className = `text-2xl font-bold ${vwapPos === 'ABOVE' ? 'bullish' : vwapPos === 'BELOW' ? 'bearish' : 'neutral'}`;
            document.getElementById('vwapDeviation').textContent = `Deviation: ${(data.analysis?.vwap?.deviation || 0).toFixed(2)}%`;

            // Trade Setups
            const setupsContainer = document.getElementById('tradeSetups');
            if (data.setups && data.setups.length > 0) {
                setupsContainer.innerHTML = data.setups.map((setup, i) => `
                    <div class="bg-gray-800/50 rounded-lg p-4 border-l-4 ${setup.direction === 'LONG' ? 'border-green-500' : 'border-red-500'}">
                        <div class="flex justify-between items-start mb-2">
                            <div>
                                <span class="font-semibold ${setup.direction === 'LONG' ? 'bullish' : 'bearish'}">${setup.direction}</span>
                                <span class="text-gray-400 ml-2">${setup.strategy}</span>
                            </div>
                            <span class="text-sm px-2 py-1 rounded ${setup.confidence >= 75 ? 'bg-green-600' : setup.confidence >= 60 ? 'bg-yellow-600' : 'bg-gray-600'}">
                                ${setup.confidence}% Confidence
                            </span>
                        </div>
                        <div class="grid grid-cols-4 gap-2 text-sm mb-2">
                            <div><span class="text-gray-400">Entry:</span> <span class="font-mono">${setup.entry_price}</span></div>
                            <div><span class="text-gray-400">SL:</span> <span class="font-mono text-red-400">${setup.stop_loss}</span></div>
                            <div><span class="text-gray-400">T1:</span> <span class="font-mono text-green-400">${setup.target_1}</span></div>
                            <div><span class="text-gray-400">R:R:</span> <span class="font-mono">${setup.risk_reward}</span></div>
                        </div>
                        <p class="text-gray-400 text-sm">${setup.reasoning}</p>
                    </div>
                `).join('');
            } else {
                setupsContainer.innerHTML = '<div class="text-gray-400 text-center py-8">No active trade setups. Wait for valid signals.</div>';
            }

            // Indicators
            const analysis = data.analysis || {};
            document.getElementById('indicators').innerHTML = `
                <div class="bg-gray-800/30 rounded-lg p-3 text-center">
                    <div class="text-gray-400 text-xs">EMA Trend</div>
                    <div class="font-semibold ${analysis.ema?.trend === 'BULLISH' ? 'bullish' : 'bearish'}">${analysis.ema?.trend || '--'}</div>
                </div>
                <div class="bg-gray-800/30 rounded-lg p-3 text-center">
                    <div class="text-gray-400 text-xs">MACD</div>
                    <div class="font-semibold ${analysis.macd?.trend === 'BULLISH' ? 'bullish' : 'bearish'}">${analysis.macd?.trend || '--'}</div>
                </div>
                <div class="bg-gray-800/30 rounded-lg p-3 text-center">
                    <div class="text-gray-400 text-xs">BB Squeeze</div>
                    <div class="font-semibold ${analysis.bollinger?.squeeze ? 'text-yellow-400' : 'text-gray-400'}">${analysis.bollinger?.squeeze ? 'YES ⚠️' : 'NO'}</div>
                </div>
                <div class="bg-gray-800/30 rounded-lg p-3 text-center">
                    <div class="text-gray-400 text-xs">Volume</div>
                    <div class="font-semibold ${analysis.volume?.spike ? 'text-yellow-400' : 'text-gray-400'}">${analysis.volume?.spike ? 'SPIKE 📊' : 'Normal'}</div>
                </div>
                <div class="bg-gray-800/30 rounded-lg p-3 text-center">
                    <div class="text-gray-400 text-xs">ATR</div>
                    <div class="font-semibold">${(analysis.atr?.value || 0).toFixed(2)}</div>
                </div>
                <div class="bg-gray-800/30 rounded-lg p-3 text-center">
                    <div class="text-gray-400 text-xs">Signal</div>
                    <div class="font-semibold">${analysis.signals?.recommendation || '--'}</div>
                </div>
            `;

            // Key Levels
            const sr = data.support_resistance || {};
            document.getElementById('keyLevels').innerHTML = `
                <div class="bg-red-900/30 rounded-lg p-3 text-center">
                    <div class="text-gray-400 text-xs">Resistance 2</div>
                    <div class="font-mono font-semibold text-red-400">${sr.resistance_2?.toFixed(2) || '--'}</div>
                </div>
                <div class="bg-red-900/30 rounded-lg p-3 text-center">
                    <div class="text-gray-400 text-xs">Resistance 1</div>
                    <div class="font-mono font-semibold text-red-400">${sr.resistance_1?.toFixed(2) || '--'}</div>
                </div>
                <div class="bg-blue-900/30 rounded-lg p-3 text-center">
                    <div class="text-gray-400 text-xs">VWAP</div>
                    <div class="font-mono font-semibold text-blue-400">${sr.vwap?.toFixed(2) || '--'}</div>
                </div>
                <div class="bg-green-900/30 rounded-lg p-3 text-center">
                    <div class="text-gray-400 text-xs">Support 1</div>
                    <div class="font-mono font-semibold text-green-400">${sr.support_1?.toFixed(2) || '--'}</div>
                </div>
                <div class="bg-green-900/30 rounded-lg p-3 text-center">
                    <div class="text-gray-400 text-xs">Support 2</div>
                    <div class="font-mono font-semibold text-green-400">${sr.support_2?.toFixed(2) || '--'}</div>
                </div>
            `;
        }

        function updateAIAnalysis(data) {
            const container = document.getElementById('aiAnalysis');
            
            if (data.error) {
                container.innerHTML = `<div class="text-red-400">${data.error}</div>`;
                return;
            }

            let html = '';
            
            if (data.commentary) {
                html += `<div class="mb-4 p-3 bg-purple-900/30 rounded-lg">${data.commentary}</div>`;
            }
            
            if (data.market_analysis) {
                html += `
                    <div class="mb-3">
                        <div class="text-gray-400 text-xs mb-1">Market Condition</div>
                        <div class="flex flex-wrap gap-2">
                            <span class="px-2 py-1 rounded text-xs bg-gray-700">${data.market_analysis.trend}</span>
                            <span class="px-2 py-1 rounded text-xs bg-gray-700">${data.market_analysis.momentum}</span>
                            <span class="px-2 py-1 rounded text-xs bg-gray-700">${data.market_analysis.volatility} Vol</span>
                        </div>
                    </div>
                `;
            }
            
            if (data.risk_assessment) {
                const riskColor = data.risk_assessment.risk_level === 'LOW' ? 'green' : 
                                  data.risk_assessment.risk_level === 'MEDIUM' ? 'yellow' : 'red';
                html += `
                    <div class="mb-3">
                        <div class="text-gray-400 text-xs mb-1">Risk Level</div>
                        <span class="px-2 py-1 rounded text-xs bg-${riskColor}-900 text-${riskColor}-300">
                            ${data.risk_assessment.risk_level}
                        </span>
                        <div class="text-xs text-gray-500 mt-1">
                            Position Size: ${(data.risk_assessment.position_size_adjustment * 100).toFixed(0)}%
                        </div>
                    </div>
                `;
            }
            
            if (data.trade_idea) {
                html += `
                    <div class="mt-4 p-3 border border-gray-700 rounded-lg">
                        <div class="text-gray-400 text-xs mb-2">AI Trade Idea</div>
                        <div class="font-semibold ${data.trade_idea.direction === 'LONG' ? 'bullish' : data.trade_idea.direction === 'SHORT' ? 'bearish' : 'neutral'}">
                            ${data.trade_idea.direction} - ${data.trade_idea.strategy}
                        </div>
                        <div class="text-xs mt-2 grid grid-cols-2 gap-1">
                            <div>Entry: ${data.trade_idea.entry_price}</div>
                            <div>SL: ${data.trade_idea.stop_loss}</div>
                            <div>Target: ${data.trade_idea.target_1}</div>
                            <div>Conf: ${data.trade_idea.confidence}%</div>
                        </div>
                    </div>
                `;
            }
            
            container.innerHTML = html || '<div class="text-gray-400">No AI analysis available</div>';
        }

        async function sendChat() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            
            const responseDiv = document.getElementById('chatResponse');
            responseDiv.classList.remove('hidden');
            responseDiv.innerHTML = '<div class="text-gray-400">Thinking...</div>';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: message,
                        context: currentData
                    })
                });
                const data = await response.json();
                responseDiv.innerHTML = `<div class="bg-gray-800/50 rounded-lg p-4">${data.response || data.error}</div>`;
            } catch (error) {
                responseDiv.innerHTML = `<div class="text-red-400">Error: ${error.message}</div>`;
            }
            
            input.value = '';
        }

        // Auto-refresh every 60 seconds
        setInterval(refreshData, 60000);
        
        // Initial load
        refreshData();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Serve the dashboard"""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/analysis')
def api_analysis():
    """Get full market analysis"""
    try:
        data, error = get_full_analysis()
        if error:
            return jsonify({"error": error}), 500
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/ai-analysis', methods=['POST'])
def api_ai_analysis():
    """Get AI-powered analysis"""
    try:
        data = request.json
        ai_agent = get_agent()
        
        result = ai_agent.analyze(
            market_data=data.get('market_data', {}),
            technical_analysis=data.get('analysis', {}),
            trade_setups=data.get('setups', [])
        )
        
        # Add commentary
        if data.get('market_data') and data.get('analysis'):
            result['commentary'] = ai_agent.get_market_commentary(
                data['market_data'],
                data['analysis']
            )
        
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat with the AI agent"""
    try:
        data = request.json
        message = data.get('message', '')
        context = data.get('context')
        
        ai_agent = get_agent()
        response = ai_agent.chat(message, context)
        
        return jsonify({"response": response})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/scalp-ideas')
def api_scalp_ideas():
    """Get quick scalp ideas"""
    try:
        data, error = get_full_analysis()
        if error:
            return jsonify({"error": error}), 500
        
        ai_agent = get_agent()
        ideas = ai_agent.generate_scalp_ideas(
            market_data=data['market_data'],
            technical_analysis=data['analysis'],
            num_ideas=5
        )
        
        return jsonify({
            "ideas": [
                {
                    "direction": i.direction,
                    "strategy": i.strategy,
                    "entry_price": i.entry_price,
                    "stop_loss": i.stop_loss,
                    "target_1": i.target_1,
                    "target_2": i.target_2,
                    "confidence": i.confidence,
                    "risk_reward": i.risk_reward,
                    "reasoning": i.reasoning
                }
                for i in ideas
            ]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/price')
def api_price():
    """Get current price"""
    try:
        return jsonify(data_fetcher.get_current_price())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openrouter_configured": bool(config.OPENROUTER_API_KEY),
        "model": config.OPENROUTER_MODEL
    })


# ============================================================
# CANDLESTICK PATTERN ANALYSIS ENDPOINTS
# ============================================================

@app.route('/api/patterns')
def api_patterns():
    """Get comprehensive candlestick pattern analysis"""
    try:
        raw_data = data_fetcher.fetch_data()
        
        if raw_data.empty:
            return jsonify({"error": "Failed to fetch market data"}), 500
        
        indicators = ScalpingIndicators(raw_data)
        analyzed_data = indicators.calculate_all()
        
        pattern_analyzer = CandlePatternAnalyzer(analyzed_data)
        summary = pattern_analyzer.get_pattern_summary()
        actionable = pattern_analyzer.get_actionable_signals(70)
        
        result = {
            "summary": summary,
            "actionable_patterns": [
                {
                    "type": p.pattern_type.value,
                    "signal": p.signal.value,
                    "confidence": p.confidence,
                    "candles_used": p.candles_used,
                    "timestamp": p.timestamp,
                    "description": p.description,
                    "entry": p.entry_suggestion,
                    "stop_loss": p.stop_loss_suggestion,
                    "target": p.target_suggestion,
                    "validation": p.validation_factors
                }
                for p in actionable
            ],
            "current_price": float(analyzed_data.iloc[-1]['close']),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(convert_numpy(result))
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/patterns/high-accuracy')
def api_patterns_high_accuracy():
    """Get only high-accuracy patterns (85%+ confidence)"""
    try:
        raw_data = data_fetcher.fetch_data()
        
        if raw_data.empty:
            return jsonify({"error": "Failed to fetch market data"}), 500
        
        indicators = ScalpingIndicators(raw_data)
        analyzed_data = indicators.calculate_all()
        
        pattern_analyzer = CandlePatternAnalyzer(analyzed_data)
        high_accuracy_patterns = pattern_analyzer.get_actionable_signals(85)  # 85%+ only
        
        result = {
            "high_accuracy_patterns": [
                {
                    "type": p.pattern_type.value,
                    "signal": p.signal.value,
                    "confidence": p.confidence,
                    "description": p.description,
                    "entry": p.entry_suggestion,
                    "stop_loss": p.stop_loss_suggestion,
                    "target": p.target_suggestion,
                    "validation": p.validation_factors,
                    "action": "TAKE TRADE" if p.confidence >= 90 else "CONSIDER TRADE"
                }
                for p in high_accuracy_patterns
            ],
            "total_high_accuracy": len(high_accuracy_patterns),
            "recommendation": "TRADE" if any(p.confidence >= 90 for p in high_accuracy_patterns) else "WAIT",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(convert_numpy(result))
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# PAPER TRADING & REAL-TIME AGENT ENDPOINTS
# ============================================================

@app.route('/api/paper/status')
def paper_status():
    """Get paper trading and agent status"""
    try:
        rt_agent = get_rt_agent()
        status = rt_agent.get_status()
        return jsonify(convert_numpy(status))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/start', methods=['POST'])
def paper_start():
    """Start the real-time scalping agent"""
    try:
        rt_agent = get_rt_agent()
        data = request.json or {}
        
        if 'auto_trade' in data:
            rt_agent.auto_trade = data['auto_trade']
        if 'min_confidence' in data:
            rt_agent.min_confidence = data['min_confidence']
        if 'update_interval' in data:
            rt_agent.update_interval = data['update_interval']
        
        rt_agent.start()
        return jsonify({"status": "started", "message": "Real-time agent started"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/stop', methods=['POST'])
def paper_stop():
    """Stop the real-time scalping agent"""
    try:
        rt_agent = get_rt_agent()
        rt_agent.stop()
        return jsonify({"status": "stopped", "message": "Real-time agent stopped"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/pause', methods=['POST'])
def paper_pause():
    """Pause auto-trading"""
    try:
        rt_agent = get_rt_agent()
        rt_agent.pause()
        return jsonify({"status": "paused", "message": "Auto-trading paused"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/resume', methods=['POST'])
def paper_resume():
    """Resume auto-trading"""
    try:
        rt_agent = get_rt_agent()
        rt_agent.resume()
        return jsonify({"status": "running", "message": "Auto-trading resumed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/positions')
def paper_positions():
    """Get open positions"""
    try:
        rt_agent = get_rt_agent()
        positions = rt_agent.paper_engine.get_open_positions()
        return jsonify({"positions": [p.to_dict() for p in positions], "count": len(positions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/statistics')
def paper_statistics():
    """Get paper trading statistics"""
    try:
        rt_agent = get_rt_agent()
        stats = rt_agent.paper_engine.get_statistics()
        return jsonify(convert_numpy(stats))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/history')
def paper_history():
    """Get trade history"""
    try:
        rt_agent = get_rt_agent()
        limit = request.args.get('limit', 20, type=int)
        history = rt_agent.paper_engine.get_trade_history(limit)
        return jsonify({"trades": history, "count": len(history)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/events')
def paper_events():
    """Get recent trade events"""
    try:
        rt_agent = get_rt_agent()
        limit = request.args.get('limit', 50, type=int)
        events = rt_agent.get_events(limit)
        return jsonify({"events": events, "count": len(events)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/entry', methods=['POST'])
def paper_manual_entry():
    """Manual trade entry"""
    try:
        rt_agent = get_rt_agent()
        data = request.json
        
        direction = data.get('direction', 'LONG').upper()
        entry_price = data.get('entry_price')
        stop_loss = data.get('stop_loss')
        target = data.get('target')
        
        success = rt_agent.manual_entry(direction, entry_price, stop_loss, target)
        
        if success:
            return jsonify({"status": "success", "message": f"Opened {direction} position"})
        else:
            return jsonify({"status": "error", "message": "Failed to open position"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/exit', methods=['POST'])
def paper_manual_exit():
    """Manual trade exit"""
    try:
        rt_agent = get_rt_agent()
        data = request.json
        
        position_id = data.get('position_id')
        reason = data.get('reason', 'MANUAL')
        
        if not position_id:
            return jsonify({"error": "position_id required"}), 400
        
        success = rt_agent.manual_exit(position_id, reason)
        
        if success:
            return jsonify({"status": "success", "message": "Position closed"})
        else:
            return jsonify({"status": "error", "message": "Failed to close position"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/close-all', methods=['POST'])
def paper_close_all():
    """Close all open positions"""
    try:
        rt_agent = get_rt_agent()
        rt_agent.close_all_positions("MANUAL_CLOSE_ALL")
        return jsonify({"status": "success", "message": "All positions closed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/reset', methods=['POST'])
def paper_reset():
    """Reset paper trading account"""
    try:
        rt_agent = get_rt_agent()
        rt_agent.paper_engine.reset()
        return jsonify({"status": "success", "message": "Paper trading account reset"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/stream')
def paper_stream():
    """Server-Sent Events stream for real-time updates"""
    def generate():
        rt_agent = get_rt_agent()
        last_event_count = 0
        
        while True:
            try:
                status = rt_agent.get_status()
                events = rt_agent.get_events(10)
                
                data = {
                    'price': status['current_price'],
                    'state': status['state'],
                    'positions': status['positions'],
                    'statistics': status['statistics'],
                    'new_events': events[last_event_count:] if len(events) > last_event_count else []
                }
                last_event_count = len(events)
                
                yield f"data: {json.dumps(convert_numpy(data))}\n\n"
                time.sleep(2)
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                time.sleep(5)
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/paper')
def paper_dashboard():
    """Paper Trading Dashboard"""
    return render_template_string(PAPER_TRADING_HTML)


PAPER_TRADING_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper Trading - Nifty 50 Scalping</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gradient-bg { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); }
        .card { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }
        .long { color: #10b981; background: rgba(16,185,129,0.1); }
        .short { color: #ef4444; background: rgba(239,68,68,0.1); }
        .win { color: #10b981; }
        .loss { color: #ef4444; }
    </style>
</head>
<body class="gradient-bg min-h-screen text-white">
    <div class="container mx-auto px-4 py-6">
        <div class="flex flex-wrap justify-between items-center mb-6 gap-4">
            <div>
                <h1 class="text-2xl font-bold bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">
                    💰 Paper Trading Dashboard
                </h1>
                <p class="text-gray-400 text-sm">Real-Time Scalping Agent • ₹50,000 Virtual Capital</p>
            </div>
            <div class="flex items-center gap-2">
                <span id="agentState" class="px-3 py-1 rounded-full text-sm bg-gray-700">STOPPED</span>
                <button onclick="startAgent()" class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg text-sm">▶ Start</button>
                <button onclick="stopAgent()" class="bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg text-sm">⏹ Stop</button>
                <button onclick="pauseAgent()" class="bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded-lg text-sm">⏸ Pause</button>
                <a href="/" class="bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded-lg text-sm">📊 Analysis</a>
            </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-6 gap-3 mb-6">
            <div class="card rounded-xl p-4 text-center">
                <div class="text-gray-400 text-xs">Current Price</div>
                <div id="currentPrice" class="text-xl font-bold font-mono">--</div>
            </div>
            <div class="card rounded-xl p-4 text-center">
                <div class="text-gray-400 text-xs">Capital</div>
                <div id="capital" class="text-xl font-bold font-mono">₹50,000</div>
            </div>
            <div class="card rounded-xl p-4 text-center">
                <div class="text-gray-400 text-xs">Total P&L</div>
                <div id="totalPnl" class="text-xl font-bold font-mono">₹0</div>
            </div>
            <div class="card rounded-xl p-4 text-center">
                <div class="text-gray-400 text-xs">Win Rate</div>
                <div id="winRate" class="text-xl font-bold">0%</div>
            </div>
            <div class="card rounded-xl p-4 text-center">
                <div class="text-gray-400 text-xs">Total Trades</div>
                <div id="totalTrades" class="text-xl font-bold">0</div>
            </div>
            <div class="card rounded-xl p-4 text-center">
                <div class="text-gray-400 text-xs">Open Positions</div>
                <div id="openPositions" class="text-xl font-bold">0</div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-2 card rounded-xl p-5">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-lg font-semibold">📈 Open Positions</h2>
                    <button onclick="closeAllPositions()" class="text-xs bg-red-600 hover:bg-red-700 px-3 py-1 rounded">Close All</button>
                </div>
                <div id="positionsContainer" class="space-y-3">
                    <div class="text-gray-500 text-center py-6">No open positions</div>
                </div>
            </div>

            <div class="card rounded-xl p-5">
                <h2 class="text-lg font-semibold mb-4">🎮 Manual Trade</h2>
                <div class="space-y-3">
                    <div>
                        <label class="text-xs text-gray-400">Direction</label>
                        <div class="flex gap-2 mt-1">
                            <button onclick="setDirection('LONG')" id="btnLong" class="flex-1 py-2 rounded bg-green-600/20 border border-green-600 text-green-400">LONG</button>
                            <button onclick="setDirection('SHORT')" id="btnShort" class="flex-1 py-2 rounded bg-gray-700 text-gray-400">SHORT</button>
                        </div>
                    </div>
                    <div>
                        <label class="text-xs text-gray-400">Entry Price (blank = market)</label>
                        <input type="number" id="entryPrice" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 mt-1" placeholder="Market price">
                    </div>
                    <div>
                        <label class="text-xs text-gray-400">Stop Loss</label>
                        <input type="number" id="stopLoss" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 mt-1" placeholder="Auto-calculate">
                    </div>
                    <div>
                        <label class="text-xs text-gray-400">Target</label>
                        <input type="number" id="target" class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 mt-1" placeholder="Auto-calculate">
                    </div>
                    <button onclick="placeManualTrade()" class="w-full bg-blue-600 hover:bg-blue-700 py-3 rounded-lg font-semibold">Place Trade</button>
                </div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <div class="card rounded-xl p-5">
                <h2 class="text-lg font-semibold mb-4">📡 Live Events</h2>
                <div id="eventsContainer" class="space-y-2 max-h-80 overflow-y-auto">
                    <div class="text-gray-500 text-center py-4">Waiting for events...</div>
                </div>
            </div>

            <div class="card rounded-xl p-5">
                <h2 class="text-lg font-semibold mb-4">📜 Trade History</h2>
                <div id="historyContainer" class="space-y-2 max-h-80 overflow-y-auto">
                    <div class="text-gray-500 text-center py-4">No trades yet</div>
                </div>
            </div>
        </div>

        <div class="mt-6 text-center">
            <button onclick="resetAccount()" class="text-sm text-gray-500 hover:text-red-400">🔄 Reset Paper Trading Account</button>
        </div>
    </div>

    <script>
        let currentDirection = 'LONG';
        let eventSource = null;

        function setDirection(dir) {
            currentDirection = dir;
            document.getElementById('btnLong').className = dir === 'LONG' 
                ? 'flex-1 py-2 rounded bg-green-600/20 border border-green-600 text-green-400'
                : 'flex-1 py-2 rounded bg-gray-700 text-gray-400';
            document.getElementById('btnShort').className = dir === 'SHORT'
                ? 'flex-1 py-2 rounded bg-red-600/20 border border-red-600 text-red-400'
                : 'flex-1 py-2 rounded bg-gray-700 text-gray-400';
        }

        async function startAgent() {
            await fetch('/api/paper/start', { method: 'POST' });
            refreshStatus();
            startEventStream();
        }

        async function stopAgent() {
            await fetch('/api/paper/stop', { method: 'POST' });
            refreshStatus();
            if (eventSource) eventSource.close();
        }

        async function pauseAgent() {
            await fetch('/api/paper/pause', { method: 'POST' });
            refreshStatus();
        }

        async function closeAllPositions() {
            if (confirm('Close all open positions?')) {
                await fetch('/api/paper/close-all', { method: 'POST' });
                refreshStatus();
            }
        }

        async function resetAccount() {
            if (confirm('Reset paper trading account? All history will be lost.')) {
                await fetch('/api/paper/reset', { method: 'POST' });
                refreshStatus();
            }
        }

        async function placeManualTrade() {
            const data = {
                direction: currentDirection,
                entry_price: document.getElementById('entryPrice').value || null,
                stop_loss: document.getElementById('stopLoss').value || null,
                target: document.getElementById('target').value || null
            };
            const res = await fetch('/api/paper/entry', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await res.json();
            alert(result.message || result.error);
            refreshStatus();
        }

        async function closePosition(id) {
            await fetch('/api/paper/exit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ position_id: id })
            });
            refreshStatus();
        }

        async function refreshStatus() {
            try {
                const res = await fetch('/api/paper/status');
                const data = await res.json();
                
                const stateEl = document.getElementById('agentState');
                stateEl.textContent = data.state;
                stateEl.className = `px-3 py-1 rounded-full text-sm ${
                    data.state === 'RUNNING' ? 'bg-green-600' : 
                    data.state === 'PAUSED' ? 'bg-yellow-600' : 'bg-gray-700'
                }`;
                
                document.getElementById('currentPrice').textContent = data.current_price?.toFixed(2) || '--';
                
                const stats = data.statistics || {};
                document.getElementById('capital').textContent = `₹${(stats.current_capital || 50000).toLocaleString()}`;
                
                const pnl = stats.total_pnl || 0;
                const pnlEl = document.getElementById('totalPnl');
                pnlEl.textContent = `₹${pnl >= 0 ? '+' : ''}${pnl.toLocaleString()}`;
                pnlEl.className = `text-xl font-bold font-mono ${pnl >= 0 ? 'win' : 'loss'}`;
                
                document.getElementById('winRate').textContent = `${stats.win_rate || 0}%`;
                document.getElementById('totalTrades').textContent = stats.total_trades || 0;
                document.getElementById('openPositions').textContent = stats.open_positions || 0;
                
                updatePositions(data.positions || []);
                if (data.recent_events) updateEvents(data.recent_events);
            } catch (e) { console.error('Status refresh error:', e); }
        }

        function updatePositions(positions) {
            const container = document.getElementById('positionsContainer');
            if (!positions.length) {
                container.innerHTML = '<div class="text-gray-500 text-center py-6">No open positions</div>';
                return;
            }
            container.innerHTML = positions.map(p => `
                <div class="p-4 rounded-lg ${p.side === 'LONG' ? 'long' : 'short'}">
                    <div class="flex justify-between items-start">
                        <div><span class="font-bold">${p.side}</span><span class="text-xs ml-2 text-gray-400">${p.strategy}</span></div>
                        <button onclick="closePosition('${p.id}')" class="text-xs bg-red-600 hover:bg-red-700 px-2 py-1 rounded">Exit</button>
                    </div>
                    <div class="grid grid-cols-4 gap-2 mt-2 text-sm">
                        <div><span class="text-gray-400">Entry:</span> ${p.entry_price?.toFixed(2)}</div>
                        <div><span class="text-gray-400">Current:</span> ${p.current_price?.toFixed(2)}</div>
                        <div><span class="text-gray-400">SL:</span> ${p.stop_loss?.toFixed(2)}</div>
                        <div><span class="text-gray-400">Target:</span> ${p.target_1?.toFixed(2)}</div>
                    </div>
                    <div class="mt-2 text-lg font-bold ${p.unrealized_pnl >= 0 ? 'win' : 'loss'}">
                        P&L: ₹${p.unrealized_pnl >= 0 ? '+' : ''}${p.unrealized_pnl?.toFixed(2)}
                    </div>
                </div>
            `).join('');
        }

        function updateEvents(events) {
            const container = document.getElementById('eventsContainer');
            if (!events.length) {
                container.innerHTML = '<div class="text-gray-500 text-center py-4">Waiting for events...</div>';
                return;
            }
            container.innerHTML = events.reverse().map(e => {
                const time = new Date(e.timestamp).toLocaleTimeString();
                const icon = e.event_type === 'ENTRY' ? '🟢' : e.event_type === 'EXIT' ? '🔴' : e.event_type === 'SIGNAL' ? '📡' : '📊';
                return `<div class="p-2 bg-gray-800/50 rounded text-sm">
                    <span class="text-gray-500">${time}</span>
                    <span class="ml-2">${icon} ${e.event_type}</span>
                    ${e.side ? `<span class="ml-2 ${e.side === 'LONG' ? 'text-green-400' : 'text-red-400'}">${e.side}</span>` : ''}
                    <span class="ml-2">@ ${e.price?.toFixed(2)}</span>
                </div>`;
            }).join('');
        }

        async function loadHistory() {
            try {
                const res = await fetch('/api/paper/history?limit=20');
                const data = await res.json();
                const container = document.getElementById('historyContainer');
                if (!data.trades?.length) {
                    container.innerHTML = '<div class="text-gray-500 text-center py-4">No trades yet</div>';
                    return;
                }
                container.innerHTML = data.trades.reverse().map(t => `
                    <div class="p-3 bg-gray-800/50 rounded text-sm">
                        <div class="flex justify-between">
                            <span class="${t.side === 'LONG' ? 'text-green-400' : 'text-red-400'}">${t.side}</span>
                            <span class="${t.result === 'WIN' ? 'win' : 'loss'}">₹${t.pnl >= 0 ? '+' : ''}${t.pnl?.toFixed(2)}</span>
                        </div>
                        <div class="text-xs text-gray-500 mt-1">${t.strategy} • ${t.exit_reason} • ${t.duration_minutes?.toFixed(0)}min</div>
                    </div>
                `).join('');
            } catch (e) { console.error('History load error:', e); }
        }

        function startEventStream() {
            if (eventSource) eventSource.close();
            eventSource = new EventSource('/api/paper/stream');
            eventSource.onmessage = (e) => {
                try {
                    const data = JSON.parse(e.data);
                    if (data.price) document.getElementById('currentPrice').textContent = data.price.toFixed(2);
                    if (data.positions) updatePositions(data.positions);
                    if (data.statistics) {
                        const stats = data.statistics;
                        document.getElementById('capital').textContent = `₹${(stats.current_capital || 50000).toLocaleString()}`;
                        const pnl = stats.total_pnl || 0;
                        const pnlEl = document.getElementById('totalPnl');
                        pnlEl.textContent = `₹${pnl >= 0 ? '+' : ''}${pnl.toLocaleString()}`;
                        pnlEl.className = `text-xl font-bold font-mono ${pnl >= 0 ? 'win' : 'loss'}`;
                        document.getElementById('openPositions').textContent = stats.open_positions || 0;
                    }
                    if (data.new_events?.length) loadHistory();
                } catch (err) { console.error('SSE parse error:', err); }
            };
        }

        refreshStatus();
        loadHistory();
        setInterval(refreshStatus, 5000);
        setInterval(loadHistory, 10000);
    </script>
</body>
</html>
'''


if __name__ == '__main__':
    print("=" * 60)
    print("🎯 Nifty 50 5-Minute Scalping Research Agent")
    print("=" * 60)
    print(f"Provider: OpenRouter")
    print(f"Model: {config.OPENROUTER_MODEL}")
    print(f"API Key: {'Configured ✅' if config.OPENROUTER_API_KEY else 'Not Set ⚠️'}")
    print(f"Server: http://{config.HOST}:{config.PORT}")
    print(f"Paper Trading: http://{config.HOST}:{config.PORT}/paper")
    print("=" * 60)
    
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
