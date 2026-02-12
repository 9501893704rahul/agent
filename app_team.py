"""
Nifty 50 5-Minute Scalping - Multi-Agent Trading Desk Application

Advanced Flask application with team of AI agents collaborating
for comprehensive scalping research.
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import json
import os
from datetime import datetime
import traceback
import numpy as np

import config
from data_fetcher import NiftyDataFetcher
from indicators import ScalpingIndicators
from scalping_strategies import StrategyEngine
from agents import create_trading_desk, TradingDeskCoordinator

app = Flask(__name__)
CORS(app)

# Global instances
data_fetcher = NiftyDataFetcher()
trading_desk: TradingDeskCoordinator = None


def get_trading_desk():
    """Get or create the trading desk"""
    global trading_desk
    if trading_desk is None:
        trading_desk = create_trading_desk(capital=1000000)  # 10 Lakh capital
    return trading_desk


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


def get_market_data():
    """Get processed market data for agents"""
    raw_data = data_fetcher.fetch_data()
    
    if raw_data.empty:
        return None, "Failed to fetch market data"
    
    indicators = ScalpingIndicators(raw_data)
    analyzed_data = indicators.calculate_all()
    
    current_analysis = indicators.get_current_analysis()
    market_data = data_fetcher.get_current_price()
    sr_levels = indicators.get_support_resistance()
    
    # Combine into format expected by agents
    agent_data = {
        "price": current_analysis.get("price", {}),
        "ema": current_analysis.get("ema", {}),
        "rsi": current_analysis.get("rsi", {}),
        "macd": current_analysis.get("macd", {}),
        "bollinger": current_analysis.get("bollinger", {}),
        "vwap": current_analysis.get("vwap", {}),
        "atr": current_analysis.get("atr", {}),
        "volume": current_analysis.get("volume", {}),
        "signals": current_analysis.get("signals", {}),
        "support_resistance": sr_levels,
        "day_high": market_data.get("high"),
        "day_low": market_data.get("low"),
        "timestamp": current_analysis.get("timestamp")
    }
    
    return convert_numpy(agent_data), None


# Advanced Dashboard HTML with Team View
TEAM_DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nifty 50 Trading Desk - Multi-Agent</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gradient-bg { background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0f2460 100%); }
        .card { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }
        .agent-card { transition: all 0.3s ease; }
        .agent-card:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .bullish { color: #10b981; }
        .bearish { color: #ef4444; }
        .neutral { color: #f59e0b; }
        .thinking { animation: thinking 1.5s infinite; }
        @keyframes thinking { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }
        .workflow-line { border-left: 2px dashed rgba(255,255,255,0.2); }
    </style>
</head>
<body class="gradient-bg min-h-screen text-white">
    <div class="container mx-auto px-4 py-6">
        <!-- Header -->
        <div class="flex justify-between items-center mb-6">
            <div>
                <h1 class="text-3xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
                    🏢 Nifty 50 Trading Desk
                </h1>
                <p class="text-gray-400 mt-1">Multi-Agent AI Research Team | 5-Minute Scalping</p>
            </div>
            <div class="flex items-center gap-4">
                <span id="lastUpdate" class="text-sm text-gray-400"></span>
                <button onclick="runFullAnalysis()" class="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 px-6 py-2 rounded-lg flex items-center gap-2 font-semibold">
                    <span id="analysisIcon">🚀</span> Run Team Analysis
                </button>
            </div>
        </div>

        <!-- Market Overview Bar -->
        <div class="grid grid-cols-5 gap-3 mb-6">
            <div class="card rounded-lg p-3 text-center">
                <div class="text-gray-400 text-xs">NIFTY 50</div>
                <div id="currentPrice" class="text-xl font-bold">--</div>
                <div id="priceChange" class="text-sm">--</div>
            </div>
            <div class="card rounded-lg p-3 text-center">
                <div class="text-gray-400 text-xs">TREND</div>
                <div id="trend" class="text-xl font-bold">--</div>
            </div>
            <div class="card rounded-lg p-3 text-center">
                <div class="text-gray-400 text-xs">RSI</div>
                <div id="rsi" class="text-xl font-bold">--</div>
            </div>
            <div class="card rounded-lg p-3 text-center">
                <div class="text-gray-400 text-xs">RISK LEVEL</div>
                <div id="riskLevel" class="text-xl font-bold">--</div>
            </div>
            <div class="card rounded-lg p-3 text-center">
                <div class="text-gray-400 text-xs">SIGNAL</div>
                <div id="signal" class="text-xl font-bold">--</div>
            </div>
        </div>

        <!-- Team Section -->
        <div class="mb-6">
            <h2 class="text-xl font-semibold mb-4 flex items-center gap-2">
                👥 Trading Desk Team
            </h2>
            <div class="grid grid-cols-5 gap-4" id="teamCards">
                <!-- Market Analyst -->
                <div class="agent-card card rounded-xl p-4 border-t-4 border-blue-500">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="text-2xl">📊</span>
                        <div>
                            <div class="font-semibold">Arjun</div>
                            <div class="text-xs text-gray-400">Market Analyst</div>
                        </div>
                    </div>
                    <div id="analystStatus" class="text-xs text-gray-500">Ready</div>
                    <div id="analystOutput" class="mt-2 text-sm text-gray-300 min-h-[60px]">--</div>
                </div>
                
                <!-- Strategist -->
                <div class="agent-card card rounded-xl p-4 border-t-4 border-purple-500">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="text-2xl">🎯</span>
                        <div>
                            <div class="font-semibold">Priya</div>
                            <div class="text-xs text-gray-400">Strategist</div>
                        </div>
                    </div>
                    <div id="strategistStatus" class="text-xs text-gray-500">Ready</div>
                    <div id="strategistOutput" class="mt-2 text-sm text-gray-300 min-h-[60px]">--</div>
                </div>
                
                <!-- Risk Manager -->
                <div class="agent-card card rounded-xl p-4 border-t-4 border-yellow-500">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="text-2xl">🛡️</span>
                        <div>
                            <div class="font-semibold">Vikram</div>
                            <div class="text-xs text-gray-400">Risk Manager</div>
                        </div>
                    </div>
                    <div id="riskStatus" class="text-xs text-gray-500">Ready</div>
                    <div id="riskOutput" class="mt-2 text-sm text-gray-300 min-h-[60px]">--</div>
                </div>
                
                <!-- Execution -->
                <div class="agent-card card rounded-xl p-4 border-t-4 border-green-500">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="text-2xl">⚡</span>
                        <div>
                            <div class="font-semibold">Ravi</div>
                            <div class="text-xs text-gray-400">Execution</div>
                        </div>
                    </div>
                    <div id="executionStatus" class="text-xs text-gray-500">Ready</div>
                    <div id="executionOutput" class="mt-2 text-sm text-gray-300 min-h-[60px]">--</div>
                </div>
                
                <!-- Head Trader -->
                <div class="agent-card card rounded-xl p-4 border-t-4 border-red-500">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="text-2xl">👔</span>
                        <div>
                            <div class="font-semibold">Anand</div>
                            <div class="text-xs text-gray-400">Head Trader</div>
                        </div>
                    </div>
                    <div id="headTraderStatus" class="text-xs text-gray-500">Ready</div>
                    <div id="headTraderOutput" class="mt-2 text-sm text-gray-300 min-h-[60px]">--</div>
                </div>
            </div>
        </div>

        <!-- Main Content Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Team Discussion / Workflow -->
            <div class="lg:col-span-2 card rounded-xl p-6">
                <h2 class="text-xl font-semibold mb-4 flex items-center gap-2">
                    💬 Team Discussion
                </h2>
                <div id="teamDiscussion" class="space-y-3 max-h-[400px] overflow-y-auto">
                    <div class="text-gray-400 text-center py-8">
                        Click "Run Team Analysis" to start the discussion...
                    </div>
                </div>
            </div>

            <!-- Final Decision -->
            <div class="card rounded-xl p-6">
                <h2 class="text-xl font-semibold mb-4 flex items-center gap-2">
                    ✅ Final Decision
                </h2>
                <div id="finalDecision" class="space-y-4">
                    <div class="text-gray-400 text-center py-8">
                        Awaiting team analysis...
                    </div>
                </div>
            </div>
        </div>

        <!-- Trade Setup Details -->
        <div class="mt-6 card rounded-xl p-6" id="tradeSetupSection" style="display:none;">
            <h2 class="text-xl font-semibold mb-4">📋 Trade Setup Details</h2>
            <div id="tradeSetupDetails" class="grid grid-cols-2 md:grid-cols-4 gap-4">
            </div>
        </div>

        <!-- Chat with Team -->
        <div class="mt-6 card rounded-xl p-6">
            <h2 class="text-xl font-semibold mb-4">🗣️ Chat with Team</h2>
            <div class="flex gap-2">
                <input type="text" id="chatInput" 
                    class="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white"
                    placeholder="Ask the team about market, strategies, risk, or execution..."
                    onkeypress="if(event.key==='Enter')sendTeamChat()">
                <button onclick="sendTeamChat()" class="bg-purple-600 hover:bg-purple-700 px-6 py-2 rounded-lg">
                    Ask Team
                </button>
            </div>
            <div id="chatResponse" class="mt-4 space-y-3 hidden">
            </div>
        </div>
    </div>

    <script>
        let analysisRunning = false;

        async function runFullAnalysis() {
            if (analysisRunning) return;
            analysisRunning = true;
            
            document.getElementById('analysisIcon').textContent = '⏳';
            document.getElementById('teamDiscussion').innerHTML = '<div class="text-center py-4"><span class="thinking">🔄 Team analyzing market conditions...</span></div>';
            
            // Update agent statuses
            updateAgentStatus('analyst', 'Analyzing...', 'thinking');
            updateAgentStatus('strategist', 'Waiting...', '');
            updateAgentStatus('risk', 'Waiting...', '');
            updateAgentStatus('execution', 'Waiting...', '');
            updateAgentStatus('headTrader', 'Waiting...', '');
            
            try {
                const response = await fetch('/api/team/full-analysis');
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('teamDiscussion').innerHTML = `<div class="text-red-400 p-4">${data.error}</div>`;
                    return;
                }
                
                updateDashboard(data);
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('teamDiscussion').innerHTML = `<div class="text-red-400 p-4">Error: ${error.message}</div>`;
            }
            
            document.getElementById('analysisIcon').textContent = '🚀';
            document.getElementById('lastUpdate').textContent = 'Updated: ' + new Date().toLocaleTimeString();
            analysisRunning = false;
        }

        function updateAgentStatus(agent, status, className) {
            const el = document.getElementById(agent + 'Status');
            if (el) {
                el.textContent = status;
                el.className = 'text-xs ' + (className || 'text-gray-500');
            }
        }

        function updateDashboard(data) {
            // Update market overview
            const stages = data.stages || {};
            const analysis = stages.market_analysis || {};
            const structured = analysis.structured || {};
            
            // Price
            const price = structured.key_resistance || '--';
            document.getElementById('currentPrice').textContent = data.market_data?.price?.close?.toFixed(2) || '--';
            
            // Trend
            const trend = structured.trend || '--';
            const trendEl = document.getElementById('trend');
            trendEl.textContent = trend;
            trendEl.className = `text-xl font-bold ${trend === 'BULLISH' ? 'bullish' : trend === 'BEARISH' ? 'bearish' : 'neutral'}`;
            
            // RSI
            document.getElementById('rsi').textContent = data.market_data?.rsi?.value?.toFixed(1) || '--';
            
            // Risk Level
            const riskAssessment = stages.risk_assessment || {};
            const riskLevel = riskAssessment.rating || '--';
            const riskEl = document.getElementById('riskLevel');
            riskEl.textContent = riskLevel;
            riskEl.className = `text-xl font-bold ${riskLevel === 'GREEN' ? 'bullish' : riskLevel === 'RED' ? 'bearish' : 'neutral'}`;
            
            // Signal
            const decision = data.final_decision || {};
            const signal = decision.action || '--';
            const signalEl = document.getElementById('signal');
            signalEl.textContent = signal;
            signalEl.className = `text-xl font-bold ${signal.includes('EXECUTE') ? 'bullish' : signal === 'NO_TRADE' ? 'bearish' : 'neutral'}`;
            
            // Update agent cards
            updateAgentStatus('analyst', '✅ Complete', 'text-green-400');
            document.getElementById('analystOutput').textContent = `Trend: ${structured.trend || 'N/A'} | Momentum: ${structured.momentum || 'N/A'}`;
            
            const strategyRec = stages.strategy_recommendation || {};
            updateAgentStatus('strategist', '✅ Complete', 'text-green-400');
            document.getElementById('strategistOutput').textContent = `${strategyRec.strategy_name || 'N/A'} | Score: ${strategyRec.score || 0}`;
            
            updateAgentStatus('risk', '✅ Complete', 'text-green-400');
            document.getElementById('riskOutput').textContent = `${riskAssessment.rating || 'N/A'} | ${riskAssessment.recommendation?.substring(0, 30) || ''}...`;
            
            const execSignal = stages.execution_signal || {};
            updateAgentStatus('execution', '✅ Complete', 'text-green-400');
            document.getElementById('executionOutput').textContent = `${execSignal.action || 'N/A'} @ ${execSignal.entry_price || '--'}`;
            
            updateAgentStatus('headTrader', '✅ Complete', 'text-green-400');
            document.getElementById('headTraderOutput').textContent = `${decision.decision || 'N/A'} - ${decision.action || ''}`;
            
            // Update team discussion
            const discussion = data.team_discussion || [];
            let discussionHtml = '';
            discussion.forEach((item, i) => {
                const isLast = i === discussion.length - 1;
                const bgColor = isLast ? 'bg-purple-900/30' : 'bg-gray-800/30';
                discussionHtml += `
                    <div class="${bgColor} rounded-lg p-3 ${i > 0 ? 'ml-4 workflow-line pl-6' : ''}">
                        <div class="flex items-center gap-2 mb-1">
                            <span class="font-semibold text-sm">${item.agent}</span>
                            <span class="text-xs text-gray-500">${item.stage}</span>
                        </div>
                        <div class="text-sm text-gray-300">${item.summary}</div>
                    </div>
                `;
            });
            document.getElementById('teamDiscussion').innerHTML = discussionHtml || '<div class="text-gray-400">No discussion yet</div>';
            
            // Update final decision
            const decisionHtml = `
                <div class="text-center p-4 rounded-lg ${decision.decision === 'APPROVED' ? 'bg-green-900/30 border border-green-500' : decision.decision === 'REJECTED' ? 'bg-red-900/30 border border-red-500' : 'bg-yellow-900/30 border border-yellow-500'}">
                    <div class="text-3xl mb-2">${decision.decision === 'APPROVED' ? '✅' : decision.decision === 'REJECTED' ? '❌' : '⏳'}</div>
                    <div class="text-xl font-bold">${decision.decision || 'PENDING'}</div>
                    <div class="text-sm text-gray-400">${decision.action || ''}</div>
                </div>
                ${decision.rationale ? `<div class="mt-4 p-3 bg-gray-800/50 rounded-lg text-sm">${decision.rationale}</div>` : ''}
                ${decision.approval_score ? `<div class="mt-2 text-center text-sm text-gray-400">Approval Score: ${decision.approval_score}/100</div>` : ''}
            `;
            document.getElementById('finalDecision').innerHTML = decisionHtml;
            
            // Update trade setup if available
            const setup = stages.trade_setup || decision.setup;
            if (setup && setup.entry) {
                document.getElementById('tradeSetupSection').style.display = 'block';
                document.getElementById('tradeSetupDetails').innerHTML = `
                    <div class="bg-gray-800/50 rounded-lg p-3">
                        <div class="text-gray-400 text-xs">Direction</div>
                        <div class="font-bold ${setup.direction === 'LONG' ? 'bullish' : 'bearish'}">${setup.direction || '--'}</div>
                    </div>
                    <div class="bg-gray-800/50 rounded-lg p-3">
                        <div class="text-gray-400 text-xs">Entry</div>
                        <div class="font-bold font-mono">${setup.entry || '--'}</div>
                    </div>
                    <div class="bg-gray-800/50 rounded-lg p-3">
                        <div class="text-gray-400 text-xs">Stop Loss</div>
                        <div class="font-bold font-mono text-red-400">${setup.stop_loss || '--'}</div>
                    </div>
                    <div class="bg-gray-800/50 rounded-lg p-3">
                        <div class="text-gray-400 text-xs">Target</div>
                        <div class="font-bold font-mono text-green-400">${setup.target || setup.target_1 || '--'}</div>
                    </div>
                `;
            } else {
                document.getElementById('tradeSetupSection').style.display = 'none';
            }
        }

        async function sendTeamChat() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            
            const responseDiv = document.getElementById('chatResponse');
            responseDiv.classList.remove('hidden');
            responseDiv.innerHTML = '<div class="text-gray-400">Team is discussing...</div>';
            
            try {
                const response = await fetch('/api/team/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ message: message })
                });
                const data = await response.json();
                
                let html = '';
                (data.responses || []).forEach(resp => {
                    html += `
                        <div class="bg-gray-800/50 rounded-lg p-4">
                            <div class="font-semibold text-purple-400 mb-2">${resp.agent}</div>
                            <div class="text-gray-300">${resp.response}</div>
                        </div>
                    `;
                });
                responseDiv.innerHTML = html || '<div class="text-gray-400">No response</div>';
                
            } catch (error) {
                responseDiv.innerHTML = `<div class="text-red-400">Error: ${error.message}</div>`;
            }
            
            input.value = '';
        }

        // Load initial market data
        async function loadMarketData() {
            try {
                const response = await fetch('/api/market-data');
                const data = await response.json();
                
                if (data.price) {
                    document.getElementById('currentPrice').textContent = data.price.close?.toFixed(2) || '--';
                    const change = data.price.change_pct || 0;
                    const changeEl = document.getElementById('priceChange');
                    changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                    changeEl.className = `text-sm ${change >= 0 ? 'bullish' : 'bearish'}`;
                    
                    const trendEl = document.getElementById('trend');
                    trendEl.textContent = data.ema?.trend || '--';
                    trendEl.className = `text-xl font-bold ${data.ema?.trend === 'BULLISH' ? 'bullish' : data.ema?.trend === 'BEARISH' ? 'bearish' : 'neutral'}`;
                    
                    document.getElementById('rsi').textContent = data.rsi?.value?.toFixed(1) || '--';
                }
            } catch (error) {
                console.error('Error loading market data:', error);
            }
        }

        // Initial load
        loadMarketData();
        
        // Auto refresh market data every 30 seconds
        setInterval(loadMarketData, 30000);
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Serve the team dashboard"""
    return render_template_string(TEAM_DASHBOARD_HTML)


@app.route('/api/market-data')
def api_market_data():
    """Get current market data"""
    try:
        data, error = get_market_data()
        if error:
            return jsonify({"error": error}), 500
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/team/full-analysis')
def api_team_full_analysis():
    """Run full team analysis"""
    try:
        # Get market data
        market_data, error = get_market_data()
        if error:
            return jsonify({"error": error}), 500
        
        # Get trading desk
        desk = get_trading_desk()
        
        # Run full analysis
        result = desk.run_full_analysis(market_data)
        result['market_data'] = market_data
        
        return jsonify(convert_numpy(result))
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/team/quick-signal')
def api_team_quick_signal():
    """Get quick signal without full analysis"""
    try:
        market_data, error = get_market_data()
        if error:
            return jsonify({"error": error}), 500
        
        desk = get_trading_desk()
        signal = desk.get_quick_signal(market_data)
        
        return jsonify(convert_numpy(signal))
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/team/brief')
def api_team_brief():
    """Get market brief from team"""
    try:
        market_data, error = get_market_data()
        if error:
            return jsonify({"error": error}), 500
        
        desk = get_trading_desk()
        brief = desk.get_market_brief(market_data)
        
        return jsonify(convert_numpy(brief))
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/team/chat', methods=['POST'])
def api_team_chat():
    """Chat with the trading team"""
    try:
        data = request.json
        message = data.get('message', '')
        
        market_data, _ = get_market_data()
        desk = get_trading_desk()
        
        response = desk.chat_with_team(message, market_data)
        
        return jsonify(response)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/team/status')
def api_team_status():
    """Get team status"""
    try:
        desk = get_trading_desk()
        status = desk.get_team_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "mode": "multi-agent",
        "timestamp": datetime.now().isoformat(),
        "model": config.OPENROUTER_MODEL
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🏢 NIFTY 50 MULTI-AGENT TRADING DESK")
    print("=" * 60)
    print(f"Mode: Multi-Agent Team")
    print(f"Model: {config.OPENROUTER_MODEL}")
    print(f"API Key: {'Configured ✅' if config.OPENROUTER_API_KEY else 'Not Set ⚠️'}")
    print(f"Server: http://{config.HOST}:{config.PORT}")
    print("=" * 60)
    
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
