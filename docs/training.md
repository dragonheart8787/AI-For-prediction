# 訓練說明

- **如何把模型練強**（資料與特徵優先）：[`強模型訓練要點.md`](強模型訓練要點.md)
- **爬蟲管線**：`python crawler_train_pipeline.py --help`
- **產物**：模型預設在 `models/`（可用 `--models-dir` 或 `PREDICT_AI_MODELS_DIR`）；每次 run 摘要於 `artifacts/<task>/<run_id>/`（見 README「輸出產物」）。

選配 AutoML／深度模型：`pip install -r requirements-automl.txt`。
