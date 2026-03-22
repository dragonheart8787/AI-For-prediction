import json
import numpy as np

from data_connectors import OpenMeteoConnector, OWIDConnector
from schema_infer import infer_timestamp_key, rows_to_features
from unified_predict import UnifiedPredictor


def main() -> None:
    # 抓兩個來源
    # 嘗試抓取資料，失敗時使用離線後備樣本
    try:
        meteo_rows = OpenMeteoConnector().fetch(latitude=25.04, longitude=121.56, past_days=1, forecast_days=0)
    except Exception:
        meteo_rows = [
            {"timestamp": f"2025-09-01T{str(h).zfill(2)}:00:00Z", "temperature_2m": 25.0 + (h % 5), "relative_humidity_2m": 60 + (h % 10), "lat": 25.04, "lon": 121.56}
            for h in range(24)
        ]
    try:
        owid_rows = OWIDConnector().fetch(country_code="TW")
    except Exception:
        owid_rows = [
            {"timestamp": f"2025-08-{str(d).zfill(2)}", "new_cases": max(0, 100 + d*2), "new_deaths": max(0, 3 + d % 5), "country": "TW"}
            for d in range(1, 31)
        ]

    # 合併與自動 schema
    rows = list(meteo_rows)[:200] + list(owid_rows)[:200]
    ts_key = infer_timestamp_key(rows)
    rows = rows_to_features(rows, ts_key)

    # 構建樣本 X/y（示範：以部分欄位線性組合當 y）
    if not rows:
        print(json.dumps({"ok": False, "error": "no_rows"}, ensure_ascii=False))
        return
    keys = sorted({k for r in rows for k in r.keys() if k != "timestamp"})
    # 建立 X
    def enc(v):
        try:
            return float(v)
        except Exception:
            return float(hash(str(v)) % 1000) / 1000.0
    X = np.asarray([[enc(r.get(k)) for k in keys] for r in rows], dtype=float)
    # 合成 y
    w = np.random.randn(len(keys))
    y = X @ w + np.random.randn(X.shape[0]) * 0.1

    # 訓練 + 自動 ONNX 切換（預設開啟）
    p = UnifiedPredictor()
    p.fit(X, y, model="linear")

    out = p.predict_many(X[:4096] if X.shape[0] >= 4096 else X, domain="custom", batch_size=2048)
    print(json.dumps({"ok": True, "model": out.get("model"), "rows": len(out.get("prediction", []))}, ensure_ascii=False))


if __name__ == "__main__":
    main()


