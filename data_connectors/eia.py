from typing import Any, Dict, Iterable, List
import datetime as dt

from .base import DataConnector


class EIAConnector(DataConnector):
    """EIA 能源資料連接器（簡化）：若無 API key 或無網路，使用離線後備。

    參數：series_id（如 PET.WCESTUS1.W），api_key（可選）
    回傳：list[dict]，含 timestamp 與 value
    """

    BASE = "https://api.eia.gov/series/"

    def __init__(self, timeout: int = 15, max_retries: int = 3, **kwargs: Any) -> None:
        super().__init__(name="eia", timeout=timeout, max_retries=max_retries, **kwargs)

    def fetch(self, **kwargs: Any) -> Iterable[Dict[str, Any]]:
        series_id = kwargs.get("series_id", "PET.WCESTUS1.W")
        api_key = kwargs.get("api_key")
        if api_key:
            try:
                data = self._request_json(
                    self.BASE,
                    params={"api_key": api_key, "series_id": series_id},
                    task=series_id,
                )
                series = data.get("series", [{}])[0]
                rows: List[Dict[str, Any]] = []
                for d, v in series.get("data", [])[:200]:
                    ts = str(d)
                    if len(ts) == 8:
                        ts_iso = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
                    elif len(ts) == 6:
                        ts_iso = f"{ts[:4]}-{ts[4:6]}-01"
                    else:
                        ts_iso = ts
                    rows.append({"timestamp": ts_iso, "value": float(v), "series_id": series_id})
                if rows:
                    return rows
            except Exception:
                pass

        # 離線後備（合成週期性需求）
        base = dt.datetime(2025, 1, 1)
        rows: List[Dict[str, Any]] = []
        for i in range(52):
            t = base + dt.timedelta(weeks=i)
            val = 100 + 10 * ((i % 13) - 6) + (i * 3) % 5
            rows.append({"timestamp": t.date().isoformat(), "value": float(val), "series_id": series_id})
        return rows



