from typing import Any, Dict, Iterable, List, Union

import datetime as dt
import numpy as np

from .base import DataConnector


def _flatten_yfinance_columns(df: Any) -> Any:
    """新版 yfinance 單一 symbol 仍可能回傳 MultiIndex 欄位，需攤平以免 iterrows 得到 Series。"""
    try:
        import pandas as pd
    except ImportError:
        return df
    if df is None or getattr(df, "empty", True):
        return df
    if isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        # 通常為 (Open, USDJPY=X) — 取第一層 OHLCV 名稱
        out.columns = out.columns.get_level_values(0)
        return out
    return df


def _row_float(row: Any, *keys: str, default: float = 0.0) -> float:
    """從 Series 列安全取純量 float，避免 `series or 0` 觸發 ambiguous truth。"""
    for k in keys:
        if k not in row.index:
            continue
        v: Any = row[k]
        if hasattr(v, "iloc") and not isinstance(v, str):
            try:
                v = v.iloc[0] if len(v) > 0 else default
            except Exception:
                continue
        try:
            x = float(v)
            if np.isnan(x):
                return default
            return x
        except (TypeError, ValueError):
            continue
    return default


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

            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            df = _flatten_yfinance_columns(df)
            if getattr(df, "empty", True):
                return []
            rows: List[Dict[str, Any]] = []
            for ts, r in df.iterrows():
                tss: Union[dt.datetime, Any] = ts
                ts_out = tss.isoformat() if hasattr(tss, "isoformat") else str(tss)
                rows.append(
                    {
                        "timestamp": ts_out,
                        "open": _row_float(r, "Open", "open"),
                        "high": _row_float(r, "High", "high"),
                        "low": _row_float(r, "Low", "low"),
                        "close": _row_float(r, "Close", "close", "Adj Close", "adj close"),
                        "volume": _row_float(r, "Volume", "volume"),
                        "symbol": symbol,
                    }
                )
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



