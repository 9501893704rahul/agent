"""
Nifty 50 5-Minute Scalping Research Agent - Main Application

Flask-based web application providing real-time scalping analysis
powered by OpenAI GPT models.
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import json
import os
from datetime import datetime
import traceback
import numpy as np

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
from indicators import ScalpingIndicators
from scalping_strategies import StrategyEngine
from openai_agent import OpenAIScalpingAgent, create_openai_agent

app = Flask(__name__)
app.json_encoder = NumpyEncoder
CORS(app)

# Global instances
data_fetcher = NiftyDataFetcher()
agent = None


def get_agent():
    """Get or create the OpenAI agent"""
    global agent
    if agent is None:
        agent = create_openai_agent()
    return agent


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
    """Get complete market analysis"""
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


if __name__ == '__main__':
    print("=" * 60)
    print("🎯 Nifty 50 5-Minute Scalping Research Agent")
    print("=" * 60)
    print(f"Provider: OpenRouter")
    print(f"Model: {config.OPENROUTER_MODEL}")
    print(f"API Key: {'Configured ✅' if config.OPENROUTER_API_KEY else 'Not Set ⚠️'}")
    print(f"Server: http://{config.HOST}:{config.PORT}")
    print("=" * 60)
    
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
