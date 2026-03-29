#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch 迴歸包裝：MLP、LSTM、Transformer、TCN（滑動視窗序列）。

新功能（v0.9.5+）：
- LR Scheduler：CosineAnnealingLR / ReduceLROnPlateau / StepLR
- Automatic Mixed Precision（AMP）：GPU 自動啟用
- 驗證集早停（patience）
- 梯度裁切（clip_grad_norm）
- Checkpoint 儲存／載入（best val loss）
- Huber Loss 選項
- TCN（Temporal Convolutional Network）
- 多輸出頭（n_outputs > 1 時輸出多地平線）
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def make_sliding_windows(X: np.ndarray, seq_len: int) -> np.ndarray:
    """將 (N, F) 轉為 (N, seq_len, F)，時間由列順序定義；不足處前方填 0。"""
    X = np.asarray(X, dtype=np.float32)
    n, f = X.shape
    seq_len = max(1, int(seq_len))
    out = np.zeros((n, seq_len, f), dtype=np.float32)
    for i in range(n):
        start = max(0, i - seq_len + 1)
        chunk = X[start : i + 1]
        k = chunk.shape[0]
        if k < seq_len:
            out[i, seq_len - k :] = chunk.astype(np.float32)
        else:
            out[i] = chunk[-seq_len:].astype(np.float32)
    return out


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    **kw: Any,
) -> Optional[Any]:
    """建立 LR Scheduler。scheduler_type: cosine | plateau | step | none。"""
    t = scheduler_type.lower()
    if t == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=kw.get("eta_min", 1e-6)
        )
    if t == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kw.get("factor", 0.5),
            patience=max(1, kw.get("scheduler_patience", 10)),
        )
    if t == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, kw.get("step_size", epochs // 3)),
            gamma=kw.get("gamma", 0.5),
        )
    return None  # "none"


# ---------------------------------------------------------------------------
# 網路架構
# ---------------------------------------------------------------------------

class _LSTMHead(nn.Module):
    def __init__(
        self, n_features: int, hidden: int, num_layers: int, n_outputs: int, dropout: float
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class _TransformerHead(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        dim_ff: int,
        n_outputs: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(dim_ff, d_model * 2),
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.fc = nn.Linear(d_model, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.size(1), :]
        m = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        out = self.enc(h, src_key_padding_mask=m)
        return self.fc(out[:, -1, :])


class _TemporalBlock(nn.Module):
    """TCN 單層：因果膨脹卷積 + Residual。"""

    def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size, dilation=dilation, padding=padding)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self._init()

    def _init(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.relu(self.conv1(x)[..., : x.size(-1)])
        out = self.drop(out)
        out = self.relu(self.conv2(out)[..., : x.size(-1)])
        out = self.drop(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TCNHead(nn.Module):
    """Temporal Convolutional Network（因果、膨脹、殘差）。"""

    def __init__(
        self,
        n_features: int,
        hidden: int,
        n_layers: int,
        kernel_size: int,
        n_outputs: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = n_features
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(_TemporalBlock(in_ch, hidden, kernel_size, dilation, dropout))
            in_ch = hidden
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        out = self.network(x)  # (B, H, T)
        return self.fc(out[:, :, -1])  # last timestep


def _build_module(
    kind: str,
    n_features: int,
    seq_len: int,
    hidden: int,
    n_outputs: int,
    dropout: float,
    **kw: Any,
) -> nn.Module:
    if kind == "mlp_torch":
        h2 = max(8, hidden // 2)
        return nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, h2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(h2, n_outputs),
        )

    if kind == "lstm":
        nl = int(kw.get("num_layers", 2))
        return _LSTMHead(n_features, hidden, nl, n_outputs, dropout)

    if kind == "transformer":
        nhead = int(kw.get("nhead", 4))
        nlayers = int(kw.get("transformer_layers", 2))
        d_model = max(nhead, int(kw.get("d_model", 64)))
        while d_model % nhead != 0:
            d_model += 1
        return _TransformerHead(n_features, seq_len, d_model, nhead, nlayers, hidden, n_outputs, dropout)

    if kind == "tcn":
        n_layers = int(kw.get("tcn_layers", 4))
        kernel_size = int(kw.get("tcn_kernel_size", 3))
        return _TCNHead(n_features, hidden, n_layers, kernel_size, n_outputs, dropout)

    raise ValueError(f"未知 kind: {kind}，支援：mlp_torch, lstm, transformer, tcn")


# ---------------------------------------------------------------------------
# 主包裝類別
# ---------------------------------------------------------------------------

class TorchRegressorWrapper:
    """
    可 pickle（__getstate__/__setstate__）的 torch 迴歸包裝。

    新功能（v0.9.5+）:
    - scheduler_type: "cosine" | "plateau" | "step" | "none"
    - use_amp: 自動混合精度（僅 CUDA）
    - patience: 驗證集早停（0 = 停用）
    - clip_grad_norm: 梯度裁切（0.0 = 停用）
    - loss_fn: "mse" | "huber" | "mae"
    - n_outputs: 多地平線輸出數
    - dropout: dropout 比例
    """

    def __init__(
        self,
        kind: str,
        n_features: int,
        seq_len: int = 14,
        hidden: int = 64,
        device: Optional[str] = None,
        n_outputs: int = 1,
        dropout: float = 0.1,
        **arch_kw: Any,
    ) -> None:
        self.kind = kind
        self.n_features = int(n_features)
        self.seq_len = max(1, int(seq_len))
        self.hidden = int(hidden)
        self.arch_kw = dict(arch_kw)
        self.n_outputs = max(1, int(n_outputs))
        self.dropout = float(dropout)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._best_checkpoint: Optional[Dict[str, Any]] = None
        self.net = _build_module(
            kind, self.n_features, self.seq_len, self.hidden, self.n_outputs, self.dropout, **self.arch_kw
        )
        self.net = self.net.to(self.device)

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------

    def _input_tensor(self, X: np.ndarray) -> torch.Tensor:
        X = np.asarray(X, dtype=np.float32)
        if self.kind == "mlp_torch":
            return torch.from_numpy(X).to(self.device)
        Sw = make_sliding_windows(X, self.seq_len)
        return torch.from_numpy(Sw).to(self.device)

    def _loss_fn(self, name: str) -> nn.Module:
        name = name.lower()
        if name == "huber":
            return nn.HuberLoss()
        if name == "mae":
            return nn.L1Loss()
        return nn.MSELoss()

    def _val_loss(self, Xi_va: torch.Tensor, y_va_t: torch.Tensor, criterion: nn.Module) -> float:
        self.net.eval()
        with torch.no_grad():
            pred = self.net(Xi_va)
            loss = criterion(pred, y_va_t)
        self.net.train()
        return float(loss.item())

    # ------------------------------------------------------------------
    # 訓練
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 120,
        lr: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        # 新參數
        scheduler_type: str = "cosine",
        use_amp: bool = True,
        patience: int = 20,
        clip_grad_norm: float = 1.0,
        loss_fn: str = "mse",
        val_ratio: float = 0.15,
        checkpoint_path: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        訓練模型。

        Returns:
            history: {"train_loss": [...], "val_loss": [...], "best_epoch": int, "lr_history": [...]}
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[1] != self.n_outputs:
            y = y[:, :self.n_outputs]

        n = min(X.shape[0], len(y))
        X, y = X[:n], y[:n]

        if n < 4:
            logger.warning("樣本過少（%d），torch 模型僅初始化", n)
            return {"train_loss": [], "val_loss": [], "best_epoch": 0, "lr_history": []}

        # 驗證集切分
        n_val = max(2, int(n * val_ratio)) if (patience > 0 and n >= 16) else 0
        n_train = n - n_val
        X_tr, y_tr = X[:n_train], y[:n_train]
        X_va, y_va = (X[n_train:], y[n_train:]) if n_val > 0 else (None, None)

        Xi_tr = self._input_tensor(X_tr)
        y_tr_t = torch.from_numpy(y_tr).to(self.device)

        Xi_va: Optional[torch.Tensor] = None
        y_va_t: Optional[torch.Tensor] = None
        if X_va is not None:
            Xi_va = self._input_tensor(X_va)
            y_va_t = torch.from_numpy(y_va).to(self.device)

        criterion = self._loss_fn(loss_fn)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = _build_scheduler(opt, scheduler_type, epochs)

        # AMP：僅 CUDA 可用且使用者開啟
        amp_enabled = use_amp and self.device.startswith("cuda") and torch.cuda.is_available()
        scaler: Optional[torch.cuda.amp.GradScaler] = (
            torch.cuda.amp.GradScaler() if amp_enabled else None
        )

        self.net.train()
        bs = max(1, min(batch_size, n_train))
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr_history": []}
        best_val = float("inf")
        best_epoch = 0
        no_improve = 0

        for ep in range(int(epochs)):
            perm = torch.randperm(n_train, device=self.device)
            ep_loss = 0.0
            steps = 0
            for start in range(0, n_train, bs):
                idx = perm[start : start + bs]
                xb = Xi_tr[idx]
                yb = y_tr_t[idx]
                opt.zero_grad()

                if amp_enabled and scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred = self.net(xb)
                        loss = criterion(pred, yb)
                    scaler.scale(loss).backward()
                    if clip_grad_norm > 0:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    pred = self.net(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    if clip_grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.net.parameters(), clip_grad_norm)
                    opt.step()

                ep_loss += float(loss.item())
                steps += 1

            avg_loss = ep_loss / max(steps, 1)
            history["train_loss"].append(avg_loss)
            current_lr = opt.param_groups[0]["lr"]
            history["lr_history"].append(current_lr)

            # 驗證
            if Xi_va is not None and y_va_t is not None:
                val_loss = self._val_loss(Xi_va, y_va_t, criterion)
                history["val_loss"].append(val_loss)

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = ep
                    no_improve = 0
                    self._best_checkpoint = {
                        k: v.cpu().clone() for k, v in self.net.state_dict().items()
                    }
                else:
                    no_improve += 1

                if patience > 0 and no_improve >= patience:
                    logger.debug("早停於 epoch %d（patience=%d）", ep, patience)
                    break
            else:
                if scheduler is not None and not isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()
                best_epoch = ep

            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                vl_str = f"val={history['val_loss'][-1]:.5f}" if history["val_loss"] else ""
                logger.info("ep %d/%d  train=%.5f  %s  lr=%.2e", ep, epochs, avg_loss, vl_str, current_lr)

        # 恢復最佳 checkpoint
        if self._best_checkpoint is not None and Xi_va is not None:
            self.net.load_state_dict(self._best_checkpoint)
            logger.debug("已恢復 best checkpoint（epoch %d val=%.5f）", best_epoch, best_val)

        # 外部 checkpoint 儲存
        if checkpoint_path:
            self.save_checkpoint(checkpoint_path)

        self.net.eval()
        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val if Xi_va is not None else None
        return history

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X = np.asarray(X, dtype=np.float32)
        Xi = self._input_tensor(X)
        with torch.no_grad():
            out = self.net(Xi)
        arr = np.asarray(out.cpu().numpy(), dtype=np.float64)
        # 單輸出保持 (N,1)，多輸出保持 (N, n_outputs)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """儲存網路權重與超參到檔案（.pt）。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        state = {
            "kind": self.kind,
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "hidden": self.hidden,
            "arch_kw": self.arch_kw,
            "n_outputs": self.n_outputs,
            "dropout": self.dropout,
            "device": self.device,
            "net_state": {k: v.cpu() for k, v in self.net.state_dict().items()},
        }
        torch.save(state, path)
        logger.info("Checkpoint 已儲存至 %s", path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> "TorchRegressorWrapper":
        """從 checkpoint 還原包裝器。"""
        state = torch.load(path, map_location="cpu")
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        wrapper = cls(
            kind=state["kind"],
            n_features=state["n_features"],
            seq_len=state["seq_len"],
            hidden=state["hidden"],
            device=dev,
            n_outputs=state.get("n_outputs", 1),
            dropout=state.get("dropout", 0.1),
            **state.get("arch_kw", {}),
        )
        wrapper.net.load_state_dict(state["net_state"])
        wrapper.net = wrapper.net.to(dev)
        wrapper.net.eval()
        logger.info("Checkpoint 已從 %s 載入", path)
        return wrapper

    # ------------------------------------------------------------------
    # Pickle 支援
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "hidden": self.hidden,
            "arch_kw": self.arch_kw,
            "n_outputs": self.n_outputs,
            "dropout": self.dropout,
            "device": self.device,
            "sd": {k: v.cpu() for k, v in self.net.state_dict().items()},
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.kind = state["kind"]
        self.n_features = state["n_features"]
        self.seq_len = state["seq_len"]
        self.hidden = state["hidden"]
        self.arch_kw = state.get("arch_kw", {})
        self.n_outputs = state.get("n_outputs", 1)
        self.dropout = state.get("dropout", 0.1)
        self.device = state.get("device", "cpu")
        self._best_checkpoint = None
        self.net = _build_module(
            self.kind, self.n_features, self.seq_len, self.hidden,
            self.n_outputs, self.dropout, **self.arch_kw,
        )
        self.net.load_state_dict(state["sd"])
        self.net = self.net.to(self.device)
        self.net.eval()


# ---------------------------------------------------------------------------
# 可解釋性：輸入梯度特徵重要性
# ---------------------------------------------------------------------------

def compute_input_gradient_importance(
    wrapper: "TorchRegressorWrapper",
    X: np.ndarray,
    *,
    max_samples: int = 64,
) -> np.ndarray:
    """
    以輸出對輸入的梯度絕對值（再對 batch／時間維度平均）當特徵重要性，已正規化為合 1。
    僅供解釋用，非因果或 Shapley。
    """
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        return np.array([], dtype=float)
    n_feat = X.shape[1]
    n = min(X.shape[0], max(1, int(max_samples)))
    Xb = X[:n]
    wrapper.net.eval()
    t = wrapper._input_tensor(Xb)
    t = t.detach().clone().float().requires_grad_(True)
    out = wrapper.net(t)
    loss = out.sum()
    loss.backward()
    g = t.grad
    if g is None:
        return np.ones(n_feat, dtype=float) / max(1, n_feat)
    ag = g.detach().abs().float().cpu().numpy()
    if ag.ndim == 2:
        imp = ag.mean(axis=0)
    elif ag.ndim == 3:
        imp = ag.mean(axis=(0, 1))
    else:
        imp = np.ones(n_feat, dtype=float) / max(1, n_feat)
    imp = np.asarray(imp, dtype=float).ravel()
    if imp.size != n_feat:
        imp = np.ones(n_feat, dtype=float) / max(1, n_feat)
    imp = np.maximum(imp, 0.0)
    s = float(imp.sum())
    if s <= 0:
        return np.ones(n_feat, dtype=float) / max(1, n_feat)
    return imp / s


# ---------------------------------------------------------------------------
# ONNX 匯出
# ---------------------------------------------------------------------------

def try_export_torch_onnx(
    wrapper: "TorchRegressorWrapper",
    path: str,
    *,
    opset_version: int = 17,
) -> bool:
    """
    嘗試將目前 net 匯出為 ONNX（僅供進階部署）；失敗時回傳 False 不拋錯。
    序列模型輸入為 (batch, seq, features)。
    """
    try:
        wrapper.net.eval()
        dummy = np.zeros((1, wrapper.n_features), dtype=np.float32)
        xi = wrapper._input_tensor(dummy)
        dynamic_axes = {"x": {0: "batch"}}
        if xi.dim() == 3:
            dynamic_axes["x"][1] = "seq"
        torch.onnx.export(
            wrapper.net,
            xi,
            path,
            input_names=["x"],
            output_names=["y"],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )
        return True
    except Exception as ex:
        logger.info("torch ONNX 匯出略過或失敗: %s", ex)
        return False
