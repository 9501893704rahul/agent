"""Data fetching module for Nifty 50 5-minute data"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import config


class NiftyDataFetcher:
    """Fetch and manage Nifty 50 5-minute OHLCV data"""
    
    def __init__(self, symbol: str = None):
        self.symbol = symbol or config.NIFTY_SYMBOL
        self.data = None
        self.last_fetch = None
    
    def fetch_data(self, period: str = None, interval: str = None) -> pd.DataFrame:
        """
        Fetch 5-minute OHLCV data for Nifty 50
        
        Args:
            period: Lookback period (e.g., '5d', '1mo')
            interval: Candle interval (e.g., '5m', '15m')
            
        Returns:
            DataFrame with OHLCV data
        """
        period = period or config.LOOKBACK_PERIOD
        interval = interval or config.INTERVAL
        
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=period, interval=interval)
            
            if self.data.empty:
                print(f"Warning: No data returned for {self.symbol}")
                return pd.DataFrame()
            
            # Clean column names
            self.data.columns = [col.lower() for col in self.data.columns]
            
            # Ensure proper datetime index
            self.data.index = pd.to_datetime(self.data.index)
            
            # Remove timezone info for easier handling
            if self.data.index.tz is not None:
                self.data.index = self.data.index.tz_localize(None)
            
            self.last_fetch = datetime.now()
            
            print(f"Fetched {len(self.data)} candles for {self.symbol}")
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def get_latest_candles(self, n: int = 100) -> pd.DataFrame:
        """Get the latest n candles"""
        if self.data is None or self.data.empty:
            self.fetch_data()
        
        return self.data.tail(n) if not self.data.empty else pd.DataFrame()
    
    def get_current_price(self) -> dict:
        """Get current market snapshot"""
        if self.data is None or self.data.empty:
            self.fetch_data()
        
        if self.data.empty:
            return {}
        
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2] if len(self.data) > 1 else latest
        
        return {
            "symbol": self.symbol,
            "timestamp": str(self.data.index[-1]),
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "volume": int(latest["volume"]),
            "change": float(latest["close"] - prev["close"]),
            "change_pct": float((latest["close"] - prev["close"]) / prev["close"] * 100)
        }
    
    def get_intraday_stats(self) -> dict:
        """Calculate intraday statistics"""
        if self.data is None or self.data.empty:
            self.fetch_data()
        
        if self.data.empty:
            return {}
        
        # Filter today's data
        today = datetime.now().date()
        today_data = self.data[self.data.index.date == today]
        
        if today_data.empty:
            today_data = self.data.tail(78)  # ~6.5 hours of 5-min candles
        
        return {
            "day_high": float(today_data["high"].max()),
            "day_low": float(today_data["low"].min()),
            "day_open": float(today_data.iloc[0]["open"]) if len(today_data) > 0 else 0,
            "current": float(today_data.iloc[-1]["close"]) if len(today_data) > 0 else 0,
            "day_range": float(today_data["high"].max() - today_data["low"].min()),
            "avg_volume": float(today_data["volume"].mean()),
            "total_volume": int(today_data["volume"].sum()),
            "candle_count": len(today_data)
        }


def fetch_nifty_components() -> list:
    """Fetch list of Nifty 50 component stocks"""
    # Top Nifty 50 components (by weight)
    nifty_50_stocks = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS",
        "NESTLEIND.NS", "TATAMOTORS.NS", "POWERGRID.NS", "M&M.NS", "NTPC.NS",
        "TATASTEEL.NS", "JSWSTEEL.NS", "TECHM.NS", "ADANIENT.NS", "ADANIPORTS.NS",
        "ONGC.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "GRASIM.NS", "INDUSINDBK.NS",
        "BPCL.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "APOLLOHOSP.NS",
        "TATACONSUM.NS", "DIVISLAB.NS", "BRITANNIA.NS", "SBILIFE.NS", "HEROMOTOCO.NS",
        "HINDALCO.NS", "LTIM.NS", "HDFCLIFE.NS", "BAJAJ-AUTO.NS", "UPL.NS"
    ]
    return nifty_50_stocks


if __name__ == "__main__":
    # Test data fetching
    fetcher = NiftyDataFetcher()
    data = fetcher.fetch_data()
    
    if not data.empty:
        print("\n--- Latest Data ---")
        print(data.tail())
        
        print("\n--- Current Price ---")
        print(fetcher.get_current_price())
        
        print("\n--- Intraday Stats ---")
        print(fetcher.get_intraday_stats())
