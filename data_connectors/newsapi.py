from typing import Any, Dict, Iterable, List
import datetime as dt

from .base import DataConnector


class NewsAPIConnector(DataConnector):
    """NewsAPI 連接器（簡化）：若無 API key 或無網路，使用離線後備。

    參數：api_key、q（關鍵字）、from_param、to
    回傳：list[dict]：timestamp、title、source、sentiment（簡單詞彙分數）
    """

    BASE = "https://newsapi.org/v2/everything"

    POS = {"good", "gain", "rise", "beat", "strong", "growth"}
    NEG = {"bad", "loss", "fall", "miss", "weak", "decline"}

    def __init__(self, timeout: int = 15, max_retries: int = 3, **kwargs: Any) -> None:
        super().__init__(name="newsapi", timeout=timeout, max_retries=max_retries, **kwargs)

    def _sentiment(self, text: str) -> float:
        s = text.lower()
        score = 0
        for w in self.POS:
            score += s.count(w)
        for w in self.NEG:
            score -= s.count(w)
        return float(score)

    def fetch(self, **kwargs: Any) -> Iterable[Dict[str, Any]]:
        api_key = kwargs.get("api_key")
        q = kwargs.get("q", "market")
        if api_key:
            try:
                data = self._request_json(
                    self.BASE,
                    params={
                        "q": q,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 50,
                        "apiKey": api_key,
                    },
                    task=q,
                )
                rows: List[Dict[str, Any]] = []
                for art in data.get("articles", []):
                    title = art.get("title") or ""
                    rows.append({
                        "timestamp": art.get("publishedAt"),
                        "title": title,
                        "source": (art.get("source") or {}).get("name"),
                        "sentiment": self._sentiment(title),
                    })
                if rows:
                    return rows
            except Exception:
                pass

        # 離線後備（合成新聞）
        base = dt.datetime(2025, 9, 1)
        titles = [
            "Market shows strong growth as demand rise",
            "Weak outlook causes fall in tech stocks",
            "Energy sector gains on positive forecast",
            "Bad weather leads to decline in sales",
        ]
        rows: List[Dict[str, Any]] = []
        for i, t in enumerate(titles * 8):
            ts = base + dt.timedelta(hours=i)
            rows.append({
                "timestamp": ts.isoformat() + "Z",
                "title": t,
                "source": "offline",
                "sentiment": self._sentiment(t),
            })
        return rows



