# config／主線設定

| 檔案 | 說明 |
|------|------|
| **`prediction_schema.yaml`** | 任務、資料源、管線預設（walk-forward、實驗紀錄路徑等）。**主線訓練以此為準。** |
| **`tasks.yaml`**、`schema.json` 等 | 任務／schema 輔助設定（依專案引用為準）。 |
| **`train_enhanced.yaml`** | 歸檔版 **train_enhanced** 管線用的大型 YAML（編譯器／VLM 等），**非** `crawler_train_pipeline` 主線必要檔；與舊檔對照見 `archive/legacy/configs/train_enhanced_legacy.yaml`。 |

歷史 `configs/` 目錄已取消；舊路徑見 `archive/legacy/configs/`。
