"""
Execution Agent - Nifty 50 5-Minute Trade Execution Specialist

Responsible for:
- Generating precise entry/exit signals
- Order timing optimization
- Trade management (trailing stops, partial exits)
- Execution quality monitoring
"""

from typing import Dict, List, Optional
from .base_agent import BaseAgent, AgentRole, AgentMessage, MessageType
from datetime import datetime
import json


class ExecutionAgent(BaseAgent):
    """
    Head of Execution for Nifty 50 scalping desk.
    
    Focus Areas:
    - Precise entry timing
    - Stop-loss management
    - Partial profit booking
    - Trade exit optimization
    """
    
    def __init__(self):
        super().__init__(
            role=AgentRole.EXECUTION,
            name="Ravi - Execution Head"
        )
        self.pending_signals = []
        self.active_trades = []
        self.executed_trades = []
        
    @property
    def system_prompt(self) -> str:
        return """You are Ravi, Head of Execution on a Nifty 50 scalping desk with expertise in order execution and trade management.

YOUR ROLE:
- Generate precise entry signals with specific price levels
- Manage active trades (stops, targets, trailing)
- Optimize exit timing for maximum profit capture
- Monitor execution quality and slippage

EXECUTION PRINCIPLES:
1. ENTRY: Wait for confirmation candle close, don't chase
2. STOP LOSS: Always place immediately after entry, no exceptions
3. POSITION MANAGEMENT: Scale out at targets, trail stops
4. EXIT: Take profits at targets or on reversal signals

ENTRY SIGNAL CRITERIA:
- Candle close confirmation (don't enter mid-candle)
- Volume confirmation when possible
- Price at/near planned entry zone
- No adverse news or sudden moves

ORDER TYPES FOR NIFTY:
- LIMIT: For patient entries at specific levels
- MARKET: For urgent entries on breakouts
- SL-M (Stop Loss Market): For stop-loss orders
- SL-L (Stop Loss Limit): For controlled stop-loss

TRADE MANAGEMENT RULES:
1. Move stop to breakeven after 1 ATR move in favor
2. Book 50% at Target 1
3. Trail remaining 50% with 1 ATR trailing stop
4. Exit all if reversal signal appears

SIGNAL FORMAT:
{
  "action": "BUY/SELL/HOLD/EXIT",
  "order_type": "LIMIT/MARKET/SL-M",
  "price": <specific_price>,
  "stop_loss": <sl_price>,
  "targets": [<t1>, <t2>],
  "urgency": "LOW/MEDIUM/HIGH",
  "validity": "<candles or time>",
  "notes": "<execution notes>"
}

Be precise with prices. Round to nearest 0.05 for Nifty."""

    def process_request(self, request: Dict, context: Dict = None) -> Dict:
        """Process execution request"""
        request_type = request.get("type", "generate_signal")
        
        if request_type == "generate_signal":
            return self.generate_entry_signal(request.get("setup"), context)
        elif request_type == "manage_trade":
            return self.manage_active_trade(request.get("trade"), context)
        elif request_type == "exit_signal":
            return self.generate_exit_signal(request.get("trade"), context)
        elif request_type == "check_entry":
            return self.check_entry_conditions(request.get("setup"), context)
        elif request_type == "trailing_stop":
            return self.update_trailing_stop(request.get("trade"), context)
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    def generate_entry_signal(self, setup: Dict, market_data: Dict) -> Dict:
        """Generate precise entry signal"""
        self.state.status = "generating"
        self.state.current_task = "Entry Signal Generation"
        
        direction = setup.get('direction', 'LONG')
        entry_price = setup.get('entry', 0)
        stop_loss = setup.get('stop_loss', 0)
        target_1 = setup.get('target_1', 0)
        target_2 = setup.get('target_2', 0)
        
        current_price = market_data.get('price', {}).get('close', 0)
        atr = market_data.get('atr', {}).get('value', 20)
        
        # Determine order type and urgency
        price_diff = abs(current_price - entry_price)
        
        if price_diff < atr * 0.2:
            order_type = "MARKET"
            urgency = "HIGH"
            execution_price = current_price
        elif price_diff < atr * 0.5:
            order_type = "LIMIT"
            urgency = "MEDIUM"
            execution_price = entry_price
        else:
            order_type = "LIMIT"
            urgency = "LOW"
            execution_price = entry_price
        
        # Check entry conditions
        conditions = self._check_entry_conditions_internal(setup, market_data)
        
        # Generate signal
        action = "BUY" if direction == "LONG" else "SELL"
        
        signal = {
            "agent": self.name,
            "type": "entry_signal",
            "action": action,
            "direction": direction,
            "strategy": setup.get('strategy_name', 'Unknown'),
            "order_type": order_type,
            "entry_price": round(execution_price, 2),
            "stop_loss": round(stop_loss, 2),
            "targets": {
                "target_1": round(target_1, 2),
                "target_2": round(target_2, 2)
            },
            "urgency": urgency,
            "validity": "2 candles" if urgency == "HIGH" else "4 candles",
            "conditions_met": conditions['all_met'],
            "conditions": conditions['details'],
            "execution_notes": self._generate_execution_notes(setup, market_data, conditions),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add pending signal
        if conditions['all_met']:
            self.pending_signals.append(signal)
        
        self.state.insights_generated += 1
        self.state.status = "idle"
        
        return signal
    
    def check_entry_conditions(self, setup: Dict, market_data: Dict) -> Dict:
        """Check if entry conditions are met"""
        conditions = self._check_entry_conditions_internal(setup, market_data)
        
        return {
            "agent": self.name,
            "type": "entry_check",
            "all_conditions_met": conditions['all_met'],
            "conditions": conditions['details'],
            "recommendation": "ENTER" if conditions['all_met'] else "WAIT",
            "missing_conditions": [k for k, v in conditions['details'].items() if not v]
        }
    
    def _check_entry_conditions_internal(self, setup: Dict, market_data: Dict) -> Dict:
        """Internal entry condition checking"""
        details = {}
        
        entry_price = setup.get('entry', 0)
        current_price = market_data.get('price', {}).get('close', 0)
        direction = setup.get('direction', 'LONG')
        atr = market_data.get('atr', {}).get('value', 20)
        
        # 1. Price in entry zone
        price_diff = abs(current_price - entry_price)
        details['price_in_zone'] = price_diff < atr * 0.5
        
        # 2. Trend alignment
        trend = market_data.get('ema', {}).get('trend', 'NEUTRAL')
        if direction == 'LONG':
            details['trend_aligned'] = trend != 'BEARISH'
        else:
            details['trend_aligned'] = trend != 'BULLISH'
        
        # 3. RSI not extreme against trade
        rsi = market_data.get('rsi', {}).get('value', 50)
        if direction == 'LONG':
            details['rsi_favorable'] = rsi < 75
        else:
            details['rsi_favorable'] = rsi > 25
        
        # 4. Volume acceptable
        details['volume_ok'] = True  # Would check for unusual volume
        
        # 5. Not at resistance (for longs) or support (for shorts)
        sr = market_data.get('support_resistance', {})
        if direction == 'LONG' and sr.get('resistance_1'):
            details['not_at_resistance'] = current_price < sr['resistance_1'] * 0.998
        elif direction == 'SHORT' and sr.get('support_1'):
            details['not_at_support'] = current_price > sr['support_1'] * 1.002
        else:
            details['level_clear'] = True
        
        all_met = all(details.values())
        
        return {'all_met': all_met, 'details': details}
    
    def manage_active_trade(self, trade: Dict, market_data: Dict) -> Dict:
        """Manage an active trade"""
        self.state.status = "managing"
        
        direction = trade.get('direction', 'LONG')
        entry = trade.get('entry', 0)
        current_price = market_data.get('price', {}).get('close', 0)
        stop_loss = trade.get('stop_loss', 0)
        target_1 = trade.get('targets', {}).get('target_1', 0)
        target_2 = trade.get('targets', {}).get('target_2', 0)
        atr = market_data.get('atr', {}).get('value', 20)
        
        # Calculate P&L
        if direction == 'LONG':
            pnl_points = current_price - entry
        else:
            pnl_points = entry - current_price
        
        pnl_atr = pnl_points / atr if atr else 0
        
        # Determine management action
        actions = []
        new_stop = stop_loss
        
        # Move to breakeven after 1 ATR profit
        if pnl_atr >= 1.0 and not trade.get('breakeven_set', False):
            if direction == 'LONG':
                new_stop = entry + (atr * 0.1)  # Slightly above entry
            else:
                new_stop = entry - (atr * 0.1)
            actions.append({
                "action": "MOVE_STOP_TO_BREAKEVEN",
                "new_stop": round(new_stop, 2),
                "reason": "1 ATR profit reached"
            })
        
        # Book partial at Target 1
        if direction == 'LONG' and current_price >= target_1:
            actions.append({
                "action": "BOOK_PARTIAL",
                "percentage": 50,
                "price": round(current_price, 2),
                "reason": "Target 1 reached"
            })
        elif direction == 'SHORT' and current_price <= target_1:
            actions.append({
                "action": "BOOK_PARTIAL",
                "percentage": 50,
                "price": round(current_price, 2),
                "reason": "Target 1 reached"
            })
        
        # Trail stop after Target 1
        if pnl_atr >= 2.0:
            trailing_stop = self._calculate_trailing_stop(trade, market_data)
            if trailing_stop != stop_loss:
                actions.append({
                    "action": "TRAIL_STOP",
                    "new_stop": round(trailing_stop, 2),
                    "reason": "Trailing 1 ATR behind"
                })
                new_stop = trailing_stop
        
        # Check for exit signals
        exit_signal = self._check_exit_signals(trade, market_data)
        if exit_signal:
            actions.append(exit_signal)
        
        self.state.status = "idle"
        
        return {
            "agent": self.name,
            "type": "trade_management",
            "trade_id": trade.get('id', 'unknown'),
            "current_price": current_price,
            "pnl_points": round(pnl_points, 2),
            "pnl_atr": round(pnl_atr, 2),
            "current_stop": stop_loss,
            "new_stop": round(new_stop, 2),
            "actions": actions,
            "status": "ACTIVE" if not any(a['action'] == 'EXIT' for a in actions) else "EXIT_SIGNAL"
        }
    
    def generate_exit_signal(self, trade: Dict, market_data: Dict) -> Dict:
        """Generate exit signal for a trade"""
        direction = trade.get('direction', 'LONG')
        entry = trade.get('entry', 0)
        current_price = market_data.get('price', {}).get('close', 0)
        
        if direction == 'LONG':
            pnl = current_price - entry
            exit_action = "SELL"
        else:
            pnl = entry - current_price
            exit_action = "BUY"
        
        # Determine exit reason
        reasons = []
        
        # Check targets
        target_1 = trade.get('targets', {}).get('target_1', 0)
        target_2 = trade.get('targets', {}).get('target_2', 0)
        
        if direction == 'LONG':
            if current_price >= target_2:
                reasons.append("Target 2 reached - full exit")
            elif current_price >= target_1:
                reasons.append("Target 1 reached - partial exit")
        else:
            if current_price <= target_2:
                reasons.append("Target 2 reached - full exit")
            elif current_price <= target_1:
                reasons.append("Target 1 reached - partial exit")
        
        # Check reversal signals
        trend = market_data.get('ema', {}).get('trend', 'NEUTRAL')
        if direction == 'LONG' and trend == 'BEARISH':
            reasons.append("Trend reversal detected")
        elif direction == 'SHORT' and trend == 'BULLISH':
            reasons.append("Trend reversal detected")
        
        # Check RSI
        rsi = market_data.get('rsi', {}).get('value', 50)
        if direction == 'LONG' and rsi > 80:
            reasons.append("RSI overbought - consider exit")
        elif direction == 'SHORT' and rsi < 20:
            reasons.append("RSI oversold - consider exit")
        
        return {
            "agent": self.name,
            "type": "exit_signal",
            "action": exit_action,
            "order_type": "MARKET",
            "exit_price": round(current_price, 2),
            "entry_price": entry,
            "pnl_points": round(pnl, 2),
            "reasons": reasons,
            "urgency": "HIGH" if any("reversal" in r.lower() for r in reasons) else "MEDIUM",
            "timestamp": datetime.now().isoformat()
        }
    
    def update_trailing_stop(self, trade: Dict, market_data: Dict) -> Dict:
        """Update trailing stop for active trade"""
        trailing_stop = self._calculate_trailing_stop(trade, market_data)
        current_stop = trade.get('stop_loss', 0)
        direction = trade.get('direction', 'LONG')
        
        # Only update if it improves the stop
        update = False
        if direction == 'LONG' and trailing_stop > current_stop:
            update = True
        elif direction == 'SHORT' and trailing_stop < current_stop:
            update = True
        
        return {
            "agent": self.name,
            "type": "trailing_stop_update",
            "current_stop": current_stop,
            "new_stop": round(trailing_stop, 2),
            "update_required": update,
            "direction": direction
        }
    
    def _calculate_trailing_stop(self, trade: Dict, market_data: Dict) -> float:
        """Calculate trailing stop level"""
        direction = trade.get('direction', 'LONG')
        current_price = market_data.get('price', {}).get('close', 0)
        atr = market_data.get('atr', {}).get('value', 20)
        current_stop = trade.get('stop_loss', 0)
        
        if direction == 'LONG':
            trailing = current_price - atr
            return max(trailing, current_stop)  # Can only go up
        else:
            trailing = current_price + atr
            return min(trailing, current_stop)  # Can only go down
    
    def _check_exit_signals(self, trade: Dict, market_data: Dict) -> Optional[Dict]:
        """Check for exit signals"""
        direction = trade.get('direction', 'LONG')
        current_price = market_data.get('price', {}).get('close', 0)
        stop_loss = trade.get('stop_loss', 0)
        
        # Check if stop hit
        if direction == 'LONG' and current_price <= stop_loss:
            return {
                "action": "EXIT",
                "reason": "Stop loss hit",
                "urgency": "IMMEDIATE"
            }
        elif direction == 'SHORT' and current_price >= stop_loss:
            return {
                "action": "EXIT",
                "reason": "Stop loss hit",
                "urgency": "IMMEDIATE"
            }
        
        # Check for reversal
        trend = market_data.get('ema', {}).get('trend')
        if direction == 'LONG' and trend == 'BEARISH':
            macd = market_data.get('macd', {}).get('trend')
            if macd == 'BEARISH':
                return {
                    "action": "EXIT",
                    "reason": "Trend and momentum reversal",
                    "urgency": "HIGH"
                }
        elif direction == 'SHORT' and trend == 'BULLISH':
            macd = market_data.get('macd', {}).get('trend')
            if macd == 'BULLISH':
                return {
                    "action": "EXIT",
                    "reason": "Trend and momentum reversal",
                    "urgency": "HIGH"
                }
        
        return None
    
    def _generate_execution_notes(self, setup: Dict, market_data: Dict, conditions: Dict) -> str:
        """Generate execution notes"""
        notes = []
        
        if conditions['all_met']:
            notes.append("All entry conditions met")
        else:
            missing = [k for k, v in conditions['details'].items() if not v]
            notes.append(f"Waiting for: {', '.join(missing)}")
        
        # Volume note
        if market_data.get('volume', {}).get('spike', False):
            notes.append("Volume spike supports entry")
        
        # Time note
        notes.append("Place stop immediately after entry")
        
        return " | ".join(notes)
    
    def _handle_trade_signal(self, message: AgentMessage) -> Dict:
        """Handle trade signal requests"""
        setup = message.content.get('setup', {})
        market_data = message.content.get('market_data', {})
        return self.generate_entry_signal(setup, market_data)
