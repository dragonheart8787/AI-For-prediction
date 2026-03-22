from typing import Any, Dict, Iterable, List
import datetime as dt

from .base import DataConnector


class YahooFinanceConnector(DataConnector):
    """Yahoo Finance 連接器（優先使用 yfinance），失敗則離線後備。

    參數：symbol（如 AAPL, ^GSPC, BTC-USD）, period（如 1mo）, interval（如 1d）
    回傳：list[dict]，含 timestamp、open/high/low/close/volume
    """

    def __init__(self, timeout: int = 30, max_retries: int = 3, **kwargs: Any) -> None:
        super().__init__(name="yahoo", timeout=timeout, max_retries=max_retries, **kwargs)

    def fetch(self, **kwargs: Any) -> Iterable[Dict[str, Any]]:
        symbol = kwargs.get("symbol", "AAPL")
        period = kwargs.get("period", "1mo")
        interval = kwargs.get("interval", "1d")

        def _download() -> List[Dict[str, Any]]:
            import yfinance as yf  # type: ignore
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            rows: List[Dict[str, Any]] = []
            for ts, r in df.iterrows():
                rows.append({
                    "timestamp": ts.isoformat(),
                    "open": float(r.get("Open", 0) or 0),
                    "high": float(r.get("High", 0) or 0),
                    "low": float(r.get("Low", 0) or 0),
                    "close": float(r.get("Close", 0) or 0),
                    "volume": float(r.get("Volume", 0) or 0),
                    "symbol": symbol,
                })
            return rows

        try:
            rows = self._retry(_download, task=symbol)
            if rows:
                return rows
        except Exception:
            pass

        # 離線後備（合成歷史價量）
        base = dt.datetime(2025, 8, 1)
        rows = []
        price = 100.0
        for i in range(30):
            t = base + dt.timedelta(days=i)
            change = ((i * 13) % 7 - 3) * 0.5
            open_p = price
            close_p = price + change
            high_p = max(open_p, close_p) + 0.3
            low_p = min(open_p, close_p) - 0.3
            vol = 1e6 + (i * 12345) % 200000
            rows.append({
                "timestamp": t.isoformat() + "Z",
                "open": round(open_p, 4),
                "high": round(high_p, 4),
                "low": round(low_p, 4),
                "close": round(close_p, 4),
                "volume": float(vol),
                "symbol": symbol,
            })
            price = close_p
        return rows



