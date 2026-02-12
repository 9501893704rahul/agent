"""
Paper Trading Engine for Nifty 50 Scalping Agent

Simulates real trading with virtual capital:
- Track positions (long/short)
- Automatic stop-loss and target execution
- P&L tracking
- Trade history
"""

import threading
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable
from enum import Enum
import json
import os

import config


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"


class TradeResult(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float
    status: str = "PENDING"
    filled_price: Optional[float] = None
    filled_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        return {
            **asdict(self),
            'side': self.side.value,
            'order_type': self.order_type.value
        }


@dataclass
class Position:
    id: str
    symbol: str
    side: str  # LONG or SHORT
    quantity: int
    entry_price: float
    current_price: float
    stop_loss: float
    target_1: float
    target_2: Optional[float]
    strategy: str
    status: PositionStatus = PositionStatus.OPEN
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_time: str = field(default_factory=lambda: datetime.now().isoformat())
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    max_profit: float = 0.0
    max_loss: float = 0.0
    
    def to_dict(self):
        return {
            **asdict(self),
            'status': self.status.value
        }


@dataclass
class TradeRecord:
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    strategy: str
    entry_time: str
    exit_time: str
    exit_reason: str
    duration_minutes: float
    result: TradeResult
    
    def to_dict(self):
        return {
            **asdict(self),
            'result': self.result.value
        }


class PaperTradingEngine:
    """
    Paper Trading Engine - Simulates real trading with virtual capital
    """
    
    def __init__(self, initial_capital: float = 50000.0, max_positions: int = 3, 
                 risk_per_trade: float = 0.02, data_file: str = "paper_trades.json"):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade  # 2% per trade
        self.data_file = data_file
        
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trade_history: List[TradeRecord] = []
        self.current_price: float = 0.0
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        
        # Callbacks
        self.on_position_opened: Optional[Callable] = None
        self.on_position_closed: Optional[Callable] = None
        self.on_price_update: Optional[Callable] = None
        
        # Load saved state
        self._load_state()
    
    def _save_state(self):
        """Save trading state to file"""
        state = {
            'current_capital': self.current_capital,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'trade_history': [t.to_dict() for t in self.trade_history[-100:]],  # Keep last 100
            'statistics': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'total_pnl': self.total_pnl,
                'max_drawdown': self.max_drawdown,
                'peak_capital': self.peak_capital
            }
        }
        try:
            with open(self.data_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def _load_state(self):
        """Load trading state from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    state = json.load(f)
                self.current_capital = state.get('current_capital', self.initial_capital)
                self.total_trades = state.get('statistics', {}).get('total_trades', 0)
                self.winning_trades = state.get('statistics', {}).get('winning_trades', 0)
                self.losing_trades = state.get('statistics', {}).get('losing_trades', 0)
                self.total_pnl = state.get('statistics', {}).get('total_pnl', 0.0)
                self.max_drawdown = state.get('statistics', {}).get('max_drawdown', 0.0)
                self.peak_capital = state.get('statistics', {}).get('peak_capital', self.initial_capital)
                print(f"📂 Loaded paper trading state: Capital={self.current_capital:.2f}")
            except Exception as e:
                print(f"Error loading state: {e}")
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk management"""
        risk_amount = self.current_capital * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit <= 0:
            return 0
        
        # For Nifty 50, lot size is typically 25
        lot_size = 25
        units = int(risk_amount / risk_per_unit)
        lots = max(1, units // lot_size)
        
        return lots * lot_size
    
    def open_position(self, side: str, entry_price: float, stop_loss: float, 
                      target_1: float, target_2: Optional[float], 
                      strategy: str, quantity: Optional[int] = None) -> Optional[Position]:
        """Open a new paper trading position"""
        
        # Check max positions
        if len(self.positions) >= self.max_positions:
            print(f"⚠️ Max positions ({self.max_positions}) reached")
            return None
        
        # Check for existing position in same direction
        for pos in self.positions.values():
            if pos.side == side and pos.status == PositionStatus.OPEN:
                print(f"⚠️ Already have an open {side} position")
                return None
        
        # Calculate position size if not provided
        if quantity is None:
            quantity = self.calculate_position_size(entry_price, stop_loss)
        
        if quantity <= 0:
            print("⚠️ Invalid position size")
            return None
        
        position_id = str(uuid.uuid4())[:8]
        
        position = Position(
            id=position_id,
            symbol=config.NIFTY_SYMBOL,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            strategy=strategy
        )
        
        self.positions[position_id] = position
        self._save_state()
        
        print(f"✅ OPENED {side} position: Entry={entry_price:.2f}, SL={stop_loss:.2f}, T1={target_1:.2f}, Qty={quantity}")
        
        if self.on_position_opened:
            self.on_position_opened(position)
        
        return position
    
    def close_position(self, position_id: str, exit_price: float, reason: str) -> Optional[TradeRecord]:
        """Close an open position"""
        
        if position_id not in self.positions:
            print(f"⚠️ Position {position_id} not found")
            return None
        
        position = self.positions[position_id]
        
        if position.status != PositionStatus.OPEN:
            print(f"⚠️ Position {position_id} already closed")
            return None
        
        # Calculate P&L
        if position.side == "LONG":
            pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.quantity
        
        pnl_percent = (pnl / (position.entry_price * position.quantity)) * 100
        
        # Determine result
        if pnl > 0:
            result = TradeResult.WIN
            self.winning_trades += 1
        elif pnl < 0:
            result = TradeResult.LOSS
            self.losing_trades += 1
        else:
            result = TradeResult.BREAKEVEN
        
        # Update position
        position.status = PositionStatus.CLOSED
        position.exit_price = exit_price
        position.exit_time = datetime.now().isoformat()
        position.exit_reason = reason
        position.realized_pnl = pnl
        
        # Calculate duration
        entry_dt = datetime.fromisoformat(position.entry_time)
        exit_dt = datetime.now()
        duration_minutes = (exit_dt - entry_dt).total_seconds() / 60
        
        # Create trade record
        trade = TradeRecord(
            id=position.id,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=pnl,
            pnl_percent=pnl_percent,
            strategy=position.strategy,
            entry_time=position.entry_time,
            exit_time=position.exit_time,
            exit_reason=reason,
            duration_minutes=duration_minutes,
            result=result
        )
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += pnl
        self.current_capital += pnl
        
        # Update peak and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        else:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        # Add to history and remove from active
        self.trade_history.append(trade)
        del self.positions[position_id]
        
        self._save_state()
        
        emoji = "🟢" if result == TradeResult.WIN else "🔴" if result == TradeResult.LOSS else "⚪"
        print(f"{emoji} CLOSED {position.side}: Exit={exit_price:.2f}, P&L={pnl:+.2f} ({pnl_percent:+.2f}%), Reason={reason}")
        
        if self.on_position_closed:
            self.on_position_closed(trade)
        
        return trade
    
    def update_price(self, price: float):
        """Update current price and check for stop-loss/target hits"""
        self.current_price = price
        
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            if position.status != PositionStatus.OPEN:
                continue
            
            position.current_price = price
            
            # Calculate unrealized P&L
            if position.side == "LONG":
                position.unrealized_pnl = (price - position.entry_price) * position.quantity
                
                # Track max profit/loss
                if position.unrealized_pnl > position.max_profit:
                    position.max_profit = position.unrealized_pnl
                if position.unrealized_pnl < position.max_loss:
                    position.max_loss = position.unrealized_pnl
                
                # Check stop-loss
                if price <= position.stop_loss:
                    positions_to_close.append((position_id, position.stop_loss, "STOP_LOSS"))
                # Check target 1
                elif price >= position.target_1:
                    positions_to_close.append((position_id, position.target_1, "TARGET_1"))
            
            else:  # SHORT
                position.unrealized_pnl = (position.entry_price - price) * position.quantity
                
                # Track max profit/loss
                if position.unrealized_pnl > position.max_profit:
                    position.max_profit = position.unrealized_pnl
                if position.unrealized_pnl < position.max_loss:
                    position.max_loss = position.unrealized_pnl
                
                # Check stop-loss
                if price >= position.stop_loss:
                    positions_to_close.append((position_id, position.stop_loss, "STOP_LOSS"))
                # Check target 1
                elif price <= position.target_1:
                    positions_to_close.append((position_id, position.target_1, "TARGET_1"))
        
        # Close triggered positions
        for pos_id, exit_price, reason in positions_to_close:
            self.close_position(pos_id, exit_price, reason)
        
        if self.on_price_update:
            self.on_price_update(price, list(self.positions.values()))
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
    
    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Calculate average win/loss
        wins = [t for t in self.trade_history if t.result == TradeResult.WIN]
        losses = [t for t in self.trade_history if t.result == TradeResult.LOSS]
        
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': round(self.current_capital, 2),
            'total_pnl': round(self.total_pnl, 2),
            'total_pnl_percent': round((self.total_pnl / self.initial_capital) * 100, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(win_rate, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'open_positions': len(self.get_open_positions())
        }
    
    def get_trade_history(self, limit: int = 20) -> List[Dict]:
        """Get recent trade history"""
        return [t.to_dict() for t in self.trade_history[-limit:]]
    
    def reset(self):
        """Reset paper trading account"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = self.initial_capital
        
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        
        print("🔄 Paper trading account reset")
