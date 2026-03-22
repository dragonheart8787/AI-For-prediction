#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch 迴歸包裝：MLP（逐列）、LSTM / Transformer（滑動視窗序列）。
需安裝 torch。輸入 X 為 (N, F)；序列模型內部為 (N, seq_len, F)。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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


class _LSTMHead(nn.Module):
    def __init__(self, n_features: int, hidden: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
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
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(dim_ff, d_model * 2),
            dropout=0.1,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x) + self.pos[:, : x.size(1), :]
        m = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        out = self.enc(h, src_key_padding_mask=m)
        return self.fc(out[:, -1, :])


def _build_module(kind: str, n_features: int, seq_len: int, hidden: int, **kw: Any) -> nn.Module:
    if kind == "mlp_torch":
        h2 = max(8, hidden // 2)
        return nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    if kind == "lstm":
        nl = int(kw.get("num_layers", 1))
        return _LSTMHead(n_features, hidden, nl)

    if kind == "transformer":
        nhead = int(kw.get("nhead", 4))
        nlayers = int(kw.get("transformer_layers", 2))
        d_model = max(nhead, int(kw.get("d_model", 64)))
        while d_model % nhead != 0:
            d_model += 1
        return _TransformerHead(n_features, seq_len, d_model, nhead, nlayers, hidden)

    raise ValueError(f"未知 kind: {kind}")


class TorchRegressorWrapper:
    """可 pickle（__getstate__/__setstate__）的 torch 迴歸包裝。"""

    def __init__(
        self,
        kind: str,
        n_features: int,
        seq_len: int = 14,
        hidden: int = 64,
        device: Optional[str] = None,
        **arch_kw: Any,
    ) -> None:
        self.kind = kind
        self.n_features = int(n_features)
        self.seq_len = max(1, int(seq_len))
        self.hidden = int(hidden)
        self.arch_kw = dict(arch_kw)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _build_module(
            kind, self.n_features, self.seq_len, self.hidden, **self.arch_kw
        )
        self.net = self.net.to(self.device)

    def _input_tensor(self, X: np.ndarray) -> torch.Tensor:
        X = np.asarray(X, dtype=np.float32)
        if self.kind == "mlp_torch":
            t = torch.from_numpy(X).to(self.device)
        else:
            Sw = make_sliding_windows(X, self.seq_len)
            t = torch.from_numpy(Sw).to(self.device)
        return t

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 120,
        lr: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
    ) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n = min(X.shape[0], len(y))
        X, y = X[:n], y[:n]
        if n < 2:
            logger.warning("樣本過少，torch 模型僅初始化")
            return

        Xi = self._input_tensor(X)
        y_t = torch.from_numpy(y.reshape(-1, 1)).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        self.net.train()
        bs = max(1, min(batch_size, n))
        for ep in range(int(epochs)):
            perm = torch.randperm(n, device=self.device)
            tot_loss = 0.0
            steps = 0
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                xb = Xi[idx]
                yb = y_t[idx]
                opt.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                tot_loss += float(loss.item())
                steps += 1
            if ep == 0 or ep == epochs - 1:
                logger.debug("torch ep %d loss=%.6f", ep, tot_loss / max(steps, 1))
        self.net.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X = np.asarray(X, dtype=np.float32)
        Xi = self._input_tensor(X)
        with torch.no_grad():
            out = self.net(Xi)
        return np.asarray(out.cpu().numpy(), dtype=np.float64)

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "hidden": self.hidden,
            "arch_kw": self.arch_kw,
            "device": self.device,
            "sd": {k: v.cpu() for k, v in self.net.state_dict().items()},
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.kind = state["kind"]
        self.n_features = state["n_features"]
        self.seq_len = state["seq_len"]
        self.hidden = state["hidden"]
        self.arch_kw = state.get("arch_kw", {})
        self.device = state.get("device", "cpu")
        self.net = _build_module(
            self.kind,
            self.n_features,
            self.seq_len,
            self.hidden,
            **self.arch_kw,
        )
        self.net.load_state_dict(state["sd"])
        self.net = self.net.to(self.device)
        self.net.eval()


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
        # (batch, seq, feat) → 對 batch、seq 平均
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
