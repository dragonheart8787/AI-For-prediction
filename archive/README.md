# Archive（非主線產物）

本目錄收納**歷史腳本、舊版 AGI／增強系統、論文素材、範例資料與長篇報告**，避免與主線預測管線混淆。

## 主線（留在 repo 根目錄）

- `crawler_train_pipeline.py`、`unified_predict.py`、`launch_predict_service.py`、`demo_backtest_all.py`
- `data_connectors/`、`validation/`、`model_serving/`、`config/`、`automl/`、`nn_models/`
- `tests/`、`docs/`、`examples/`、`scripts/`
- 主線共用模組：`runtime_paths.py`、`schema_infer.py`、`feature_expansion.py`、`prediction_*.py`、`experiment_log.py`、`artifact_registry.py` 等

## 本目錄結構（概覽）

| 路徑 | 內容 |
|------|------|
| `legacy_root_py/` | 自根目錄移入的舊版 `.py`（AGI、demo、啟動腳本等） |
| `documentation/` | 自根目錄移入的舊版 `.md`、`.txt` 說明（非 README／CHANGELOG） |
| `paper/` | LaTeX／論文相關 |
| `legacy/serving/` | 舊版 vLLM／TensorRT 等 shell（`run_vllm.sh` 等；非主線 `model_serving/`） |
| `legacy/guides/` | 舊版指南 |
| `legacy/configs/` | 舊版 `train_enhanced` 用 YAML（主線設定在 `config/`） |
| `legacy/packages/` | 實驗性子套件（NAS、KD、GPU 等） |
| `sample_data/` | 範例 CSV、測試用模型目錄等（主線訓練可改指向自有路徑） |
| `experimental/` | 原 `experimental/` 目錄 |

若需還原某腳本行為，請以 `git log` 追蹤移動前的路徑，或將檔案複回根目錄後自行調整 `import` 路徑。
