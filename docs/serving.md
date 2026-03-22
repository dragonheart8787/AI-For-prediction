# HTTP 推論服務

啟動：

```bash
python launch_predict_service.py --model-path models/task_stock_price_next.pkl --port 8765
```

端點：

- `GET /health`
- `GET /v1/model/info`
- `POST /v1/predict` — body 見 `examples/sample_predict_payload.json`
- `POST /v1/predict_many` — 可帶 `batch_size`（有上限）

可選 header：`X-Request-ID` 或 `X-Trace-ID`。

詳見 [`model_serving/unified_http_service.py`](../model_serving/unified_http_service.py)。
