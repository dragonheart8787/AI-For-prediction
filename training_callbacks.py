#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
訓練回呼系統（Training Callbacks）。

提供通用回呼架構，供 TorchRegressorWrapper 或自定訓練迴圈使用。

回呼類別：
- CallbackBase          — 基礎類別（繼承此類實作自定義回呼）
- EarlyStoppingCallback — 驗證集無改善時提前停止
- ModelCheckpointCallback — 儲存最佳模型
- LRScheduleCallback    — 手動 LR 調整（無需 PyTorch Scheduler）
- MetricsLoggerCallback — 將逐 epoch 指標寫入 experiment_log
- CallbackList          — 管理多個回呼的容器

用法（範例，整合進自定訓練迴圈）：
    cb = CallbackList([
        EarlyStoppingCallback(patience=15),
        ModelCheckpointCallback("checkpoints/best.pkl"),
        MetricsLoggerCallback(run_id="run_001"),
    ])
    cb.on_train_begin(config={...})
    for ep in range(epochs):
        ...
        cb.on_epoch_end(epoch=ep, logs={"train_loss": ..., "val_loss": ...})
        if cb.stop_training:
            break
    cb.on_train_end(logs={"final_epoch": ep})
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 基礎類別
# ============================================================================

class CallbackBase:
    """所有回呼的基礎類別。"""

    def __init__(self) -> None:
        self.stop_training: bool = False
        self._model: Any = None

    def set_model(self, model: Any) -> None:
        self._model = model

    def on_train_begin(self, config: Optional[Dict[str, Any]] = None) -> None:
        """訓練開始時呼叫。"""

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """每個 epoch 開始時呼叫。"""

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """每個 epoch 結束時呼叫。"""

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """訓練結束時呼叫。"""


# ============================================================================
# 早停回呼
# ============================================================================

class EarlyStoppingCallback(CallbackBase):
    """
    驗證集無改善時提前停止訓練。

    Parameters
    ----------
    patience    : 容忍無改善的 epoch 數
    monitor     : 監控的 log 鍵（"val_loss" 或 "val_rmse"）
    min_delta   : 視為改善的最小變化量
    mode        : "min"（越小越好）或 "max"（越大越好）
    restore_best: 訓練結束時恢復最佳 epoch 的模型（僅支援有 set_model 的場景）
    """

    def __init__(
        self,
        patience: int = 20,
        monitor: str = "val_loss",
        min_delta: float = 1e-6,
        mode: str = "min",
        restore_best: bool = False,
    ) -> None:
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self._best_value: Optional[float] = None
        self._best_weights: Any = None
        self._no_improve = 0
        self.stopped_epoch: Optional[int] = None

    def on_train_begin(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._best_value = None
        self._no_improve = 0
        self.stopped_epoch = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        val = logs.get(self.monitor)
        if val is None:
            return

        val = float(val)
        if self._best_value is None:
            self._best_value = val
            self._save_best_weights()
            return

        improved = (val < self._best_value - self.min_delta) if self.mode == "min" else (val > self._best_value + self.min_delta)
        if improved:
            self._best_value = val
            self._no_improve = 0
            self._save_best_weights()
        else:
            self._no_improve += 1
            if self._no_improve >= self.patience:
                self.stop_training = True
                self.stopped_epoch = epoch
                logger.info("EarlyStopping：epoch %d，%s=%.6f，已停止", epoch, self.monitor, val)
                if self.restore_best and self._best_weights is not None:
                    self._restore_best_weights()

    def _save_best_weights(self) -> None:
        if self._model is not None and hasattr(self._model, "net"):
            try:
                import torch
                self._best_weights = {k: v.cpu().clone() for k, v in self._model.net.state_dict().items()}
            except Exception:
                pass

    def _restore_best_weights(self) -> None:
        if self._model is not None and hasattr(self._model, "net") and self._best_weights is not None:
            try:
                self._model.net.load_state_dict(self._best_weights)
                logger.info("EarlyStopping：已還原最佳 epoch 權重")
            except Exception as ex:
                logger.warning("還原權重失敗：%s", ex)


# ============================================================================
# Checkpoint 回呼
# ============================================================================

class ModelCheckpointCallback(CallbackBase):
    """
    儲存最佳模型（依監控指標）或每個 epoch 的模型。

    Parameters
    ----------
    filepath    : 儲存路徑（支援 {epoch} / {val_loss} 佔位符）
    monitor     : 監控指標
    save_best_only: True 時只儲存改善的 checkpoint
    mode        : "min" 或 "max"
    use_pickle  : True 用 pickle，False 用 torch.save（如有 net 屬性）
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
        use_pickle: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.use_pickle = use_pickle
        self.verbose = verbose
        self._best_value: Optional[float] = None

    def on_train_begin(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._best_value = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        val = logs.get(self.monitor)

        if self.save_best_only:
            if val is None:
                return
            val = float(val)
            improved = (self._best_value is None) or (
                val < self._best_value if self.mode == "min" else val > self._best_value
            )
            if not improved:
                return
            self._best_value = val

        # 解析路徑佔位符
        path = self.filepath.format(epoch=epoch, **{k: f"{v:.4f}" for k, v in (logs or {}).items() if isinstance(v, (int, float))})
        self._save(path)
        if self.verbose:
            logger.info("ModelCheckpoint：已儲存至 %s（epoch %d）", path, epoch)

    def _save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        model = self._model
        if model is None:
            return
        try:
            if not self.use_pickle and hasattr(model, "save_checkpoint"):
                model.save_checkpoint(path)
            else:
                with open(path, "wb") as f:
                    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            logger.warning("ModelCheckpoint 儲存失敗：%s", ex)


# ============================================================================
# LR 調整回呼（無需 PyTorch Scheduler）
# ============================================================================

class LRScheduleCallback(CallbackBase):
    """
    手動 LR 調整回呼（用於不原生支援 Scheduler 的訓練迴圈）。

    schedule_fn: callable(epoch, current_lr) → new_lr
    """

    def __init__(
        self,
        schedule_fn: Any,
        optimizer: Any = None,
    ) -> None:
        super().__init__()
        self.schedule_fn = schedule_fn
        self.optimizer = optimizer
        self._current_lr: float = 1e-3

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        new_lr = self.schedule_fn(epoch, self._current_lr)
        self._current_lr = float(new_lr)
        if self.optimizer is not None:
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr
        if logs is not None:
            logs["lr"] = new_lr


# ============================================================================
# 指標紀錄回呼
# ============================================================================

class MetricsLoggerCallback(CallbackBase):
    """
    將逐 epoch 指標寫入 experiment_log.log_epoch()。

    Parameters
    ----------
    run_id     : 與 log_training_event 的 run_id 對應
    log_every  : 每 N 個 epoch 記錄一次（減少 I/O）
    log_path   : epoch 日誌檔路徑
    """

    def __init__(
        self,
        run_id: str,
        log_every: int = 1,
        log_path: str = "data/epoch_logs.jsonl",
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.log_every = log_every
        self.log_path = log_path
        self._epoch_count = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._epoch_count += 1
        if self._epoch_count % self.log_every != 0:
            return
        logs = logs or {}
        try:
            from experiment_log import log_epoch
            log_epoch(
                run_id=self.run_id,
                epoch=epoch,
                train_loss=float(logs.get("train_loss", -1.0)),
                val_loss=float(logs["val_loss"]) if "val_loss" in logs else None,
                lr=float(logs["lr"]) if "lr" in logs else None,
                path=self.log_path,
            )
        except Exception as ex:
            logger.debug("MetricsLoggerCallback 寫入失敗：%s", ex)


# ============================================================================
# CallbackList（容器）
# ============================================================================

class CallbackList:
    """
    管理多個回呼的容器。

    用法：
        cbs = CallbackList([EarlyStoppingCallback(20), ModelCheckpointCallback("best.pkl")])
        cbs.set_model(my_model)
        cbs.on_train_begin(config={"epochs": 100})
        for ep in range(100):
            cbs.on_epoch_end(ep, {"train_loss": 0.1, "val_loss": 0.2})
            if cbs.stop_training:
                break
        cbs.on_train_end()
    """

    def __init__(self, callbacks: Optional[List[CallbackBase]] = None) -> None:
        self.callbacks: List[CallbackBase] = list(callbacks or [])

    @property
    def stop_training(self) -> bool:
        return any(cb.stop_training for cb in self.callbacks)

    def set_model(self, model: Any) -> None:
        for cb in self.callbacks:
            cb.set_model(model)

    def append(self, callback: CallbackBase) -> None:
        self.callbacks.append(callback)

    def on_train_begin(self, config: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(config)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_end(logs)
