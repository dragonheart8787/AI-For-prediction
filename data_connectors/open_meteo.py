import datetime as dt
from typing import Any, Dict, Iterable, List
import urllib.parse

from .base import DataConnector


class OpenMeteoConnector(DataConnector):
    """Open-Meteo 簡易連接器：抓取指定經緯度的即時/歷史天氣變數。

    參數：latitude, longitude, hourly(list)
    回傳：list[dict]，每筆包含 timestamp 與 features
    """

    BASE = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout: int = 20, max_retries: int = 3, **kwargs: Any) -> None:
        super().__init__(name="open_meteo", timeout=timeout, max_retries=max_retries, **kwargs)

    def fetch(self, **kwargs: Any) -> Iterable[Dict[str, Any]]:
        latitude = float(kwargs.get("latitude", 25.04))
        longitude = float(kwargs.get("longitude", 121.56))
        hourly: List[str] = kwargs.get("hourly", ["temperature_2m", "relative_humidity_2m"])  # type: ignore
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(hourly),
            "past_days": kwargs.get("past_days", 1),
            "forecast_days": kwargs.get("forecast_days", 1),
            "timezone": kwargs.get("timezone", "UTC"),
        }
        data = self._request_json(
            self.BASE,
            params=params,
            task=f"{latitude},{longitude}",
        )
        times = data.get("hourly", {}).get("time", [])
        rows: List[Dict[str, Any]] = []
        for i, t in enumerate(times):
            feats: Dict[str, Any] = {k: data["hourly"].get(k, [None])[i] for k in hourly}
            rows.append({
                "timestamp": t,
                **feats,
                "lat": latitude,
                "lon": longitude,
            })
        return rows


