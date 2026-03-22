from typing import Any, Dict, Iterable, List
import datetime as dt

from .base import DataConnector


class OWIDConnector(DataConnector):
    """Our World in Data（範例：COVID）連接器。

    參數：country_code（如 TW、US），fields(list)
    無網路或 API 失敗時使用合成離線後備。
    """

    URL = "https://covid.ourworldindata.org/data/owid-covid-data.json"

    def __init__(self, timeout: int = 30, max_retries: int = 3, **kwargs: Any) -> None:
        super().__init__(name="owid", timeout=timeout, max_retries=max_retries, **kwargs)

    def _offline_rows(self, country: str, fields: List[str]) -> List[Dict[str, Any]]:
        base = dt.date.today() - dt.timedelta(days=100)
        rows: List[Dict[str, Any]] = []
        for i in range(100):
            d = base + dt.timedelta(days=i)
            nc = max(0.0, float(50 + (i * 17) % 40 - (i * 3) % 15))
            nd = max(0.0, float((i * 2) % 5))
            rec: Dict[str, Any] = {"timestamp": d.isoformat(), "country": country}
            for f in fields:
                if f == "new_cases":
                    rec[f] = nc
                elif f == "new_deaths":
                    rec[f] = nd
                else:
                    rec[f] = float((hash(f) + i) % 100) / 10.0
            rows.append(rec)
        return rows

    def fetch(self, **kwargs: Any) -> Iterable[Dict[str, Any]]:
        country = (kwargs.get("country_code") or "TW").upper()
        fields: List[str] = kwargs.get("fields", ["new_cases", "new_deaths"])  # type: ignore
        try:
            data = self._request_json(self.URL, task=country)
            node = data.get(country, {})
            series: List[Dict[str, Any]] = node.get("data", [])
            rows: List[Dict[str, Any]] = []
            for rec in series[-100:]:
                feats = {k: rec.get(k) for k in fields}
                rows.append({"timestamp": rec.get("date"), **feats, "country": country})
            if rows:
                return rows
        except Exception:
            pass
        return self._offline_rows(country, fields)


