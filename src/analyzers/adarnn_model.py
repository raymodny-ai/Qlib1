"""
AdaRNN 自适应循环神经网络模型

基于 "AdaRNN: Adaptive Learning and Forecasting of Time Series" (CIKM 2021)
实现时序分布匹配 (Temporal Distribution Matching) 与 GRU 编码器。

核心思想:
- 将非平稳时间序列分割为多个时期 (periods)
- 通过 MMD (Maximum Mean Discrepancy) 最小化不同时期间的分布差异
- 使用 GRU 编码器学习时序依赖 + 全连接层输出预测

架构:
  Input → GRU Encoder → Distribution Matching (MMD) → FC Output

适用场景:
- 基本面因子时间序列 (季报/年报)
- 存在 regime shift 的金融市场数据
- 需要 temporal adaptation 的非平稳序列

设计原则:
- 继承 BaseForecastModel，遵循 fit/predict 统一接口
- 支持可配置的 period 分割和 MMD 正则化强度
- GPU 加速 (自动检测 CUDA)

使用示例:
    from src.analyzers.adarnn_model import AdaRNNModel

    model = AdaRNNModel(
        input_dim=64,
        hidden_dim=128,
        num_layers=2,
        n_periods=5,
        lambda_mmd=0.1,
        learning_rate=1e-3,
    )
    model.fit(X_train, y_train, X_valid, y_valid)
    predictions = model.predict(X_test)
"""

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.analyzers.ml_pipeline import BaseForecastModel, TrainingResult, PredictionResult
from src.utils.logger import get_logger


class AdaRNNModel(BaseForecastModel):
    """
    AdaRNN: 自适应循环神经网络

    用 GRU 编码器 + MMD 分布匹配损失实现时序自适应预测。
    自动检测 CUDA/CPU 设备。

    参数:
        input_dim: 输入特征维度
        hidden_dim: GRU 隐藏层维度 (默认 128)
        num_layers: GRU 层数 (默认 2)
        n_periods: 时间序列分割期数 (默认 5)
        lambda_mmd: MMD 损失权重 (默认 0.1)
        dropout: Dropout 比例 (默认 0.2)
        learning_rate: 学习率 (默认 1e-3)
        n_epochs: 最大训练轮数 (默认 200)
        batch_size: 批次大小 (默认 64)
        early_stopping_patience: 早停耐心轮数 (默认 20)
        seq_len: 输入序列长度 (默认 10, 自动推断)
    """

    def __init__(self, **params):
        defaults = {
            "input_dim": 64,
            "hidden_dim": 128,
            "num_layers": 2,
            "n_periods": 5,
            "lambda_mmd": 0.1,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "n_epochs": 200,
            "batch_size": 64,
            "early_stopping_patience": 20,
            "seq_len": 10,
            "random_state": 42,
        }
        defaults.update(params)
        super().__init__(**defaults)
        self._model = None  # PyTorch model
        self._device = "cuda" if self._has_cuda() else "cpu"
        self.logger = get_logger()

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ================================================================
    #  PyTorch 模型定义 (内部类 — 支持 pickle)
    # ================================================================

    class _GRUMMDNet:
        """GRU + MMD 核心网络 (独立类以确保可序列化)"""

        def __init__(self, input_dim, hidden_dim, num_layers, dropout, device="cpu"):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout = dropout
            self.device = device

            import torch
            import torch.nn as nn

            self.torch = torch
            self.nn = nn

            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        def to(self, device):
            self.gru.to(device)
            self.fc.to(device)
            self.device = device
            return self

        def forward(self, x):
            """x: (batch, seq_len, input_dim) → (batch, 1)"""
            import torch
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
            out, _ = self.gru(x, h0)
            last_hidden = out[:, -1, :]  # 取最后时间步
            return self.fc(last_hidden)

        def state_dict(self):
            return {
                "gru": {k: v.cpu() for k, v in self.gru.state_dict().items()},
                "fc": {k: v.cpu() for k, v in self.fc.state_dict().items()},
            }

        def load_state_dict(self, sd):
            self.gru.load_state_dict(sd["gru"])
            self.fc.load_state_dict(sd["fc"])

        def train(self):
            self.gru.train()
            self.fc.train()

        def eval(self):
            self.gru.eval()
            self.fc.eval()

    # ================================================================
    #  MMD 损失计算
    # ================================================================

    @staticmethod
    def _gaussian_kernel(x, y, sigma_list=None):
        """高斯核函数 (多带宽)"""
        import torch
        if sigma_list is None:
            sigma_list = [1.0, 2.0, 4.0, 8.0, 16.0]

        xx = torch.sum(x * x, dim=1, keepdim=True)  # (n, 1)
        yy = torch.sum(y * y, dim=1, keepdim=True)   # (m, 1)
        xy = torch.mm(x, y.t())                       # (n, m)

        d2 = xx + yy.t() - 2 * xy  # (n, m)

        total = torch.zeros(1, device=x.device)
        for sigma in sigma_list:
            k_val = torch.exp(-d2 / (2.0 * sigma * sigma))
            total = total + k_val.mean()

        return total / len(sigma_list)

    @staticmethod
    def _mmd_loss(source, target):
        """MMD^2 距离 (无偏估计)"""
        import torch
        n_s = source.size(0)
        n_t = target.size(0)
        n = min(n_s, n_t)

        source = source[:n]
        target = target[:n]

        k_ss = AdaRNNModel._gaussian_kernel(source, source)
        k_tt = AdaRNNModel._gaussian_kernel(target, target)
        k_st = AdaRNNModel._gaussian_kernel(source, target)

        return k_ss + k_tt - 2 * k_st

    # ================================================================
    #  时期分割
    # ================================================================

    def _split_periods(self, X: np.ndarray) -> List[np.ndarray]:
        """将时间序列按时间顺序等分为 n_periods 段"""
        n = len(X)
        n_periods = min(self.params["n_periods"], n // self.params["seq_len"])
        if n_periods < 2:
            return [X]

        period_size = n // n_periods
        periods = []
        for i in range(n_periods):
            start = i * period_size
            end = (i + 1) * period_size if i < n_periods - 1 else n
            periods.append(X[start:end])
        return periods

    # ================================================================
    #  序列化输入
    # ================================================================

    def _to_sequences(self, X: np.ndarray, seq_len: int) -> np.ndarray:
        """将 (n_samples, n_features) 转为 (n_sequences, seq_len, n_features)"""
        n = X.shape[0]
        n_features = X.shape[1]
        n_seqs = n - seq_len + 1
        if n_seqs <= 0:
            # 样本不足时 padding
            pad = np.zeros((seq_len - n, n_features), dtype=X.dtype)
            X = np.concatenate([pad, X], axis=0)
            return X.reshape(1, seq_len, n_features)

        seqs = np.zeros((n_seqs, seq_len, n_features), dtype=X.dtype)
        for i in range(n_seqs):
            seqs[i] = X[i:i + seq_len]
        return seqs

    # ================================================================
    #  fit
    # ================================================================

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> TrainingResult:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch 未安装。请执行: pip install torch>=2.0.0")

        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        input_dim = X_train.shape[1]
        params = self.params
        hidden_dim = params["hidden_dim"]
        num_layers = params["num_layers"]
        dropout = params["dropout"]
        seq_len = params["seq_len"]
        n_epochs = params["n_epochs"]
        batch_size = params["batch_size"]
        lambda_mmd = params["lambda_mmd"]
        patience = params["early_stopping_patience"]
        lr = params["learning_rate"]
        n_periods = params["n_periods"]

        # 构建网络
        net = AdaRNNModel._GRUMMDNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            device=self._device,
        ).to(self._device)

        optimizer = optim.Adam(
            list(net.gru.parameters()) + list(net.fc.parameters()),
            lr=lr,
        )
        mse_loss = nn.MSELoss()

        # 转换为序列
        X_seq = self._to_sequences(X_train, seq_len)
        y_seq = y_train[seq_len - 1:] if len(y_train) >= seq_len else y_train
        n_samples = min(len(X_seq), len(y_seq))
        X_seq = X_seq[:n_samples]
        y_seq = y_seq[:n_samples]

        # 验证集
        X_valid_seq = None
        y_valid_seq = None
        if X_valid is not None and y_valid is not None:
            X_valid_seq = self._to_sequences(X_valid, seq_len)
            y_valid_seq = y_valid[seq_len - 1:] if len(y_valid) >= seq_len else y_valid

        # 时期分割
        periods = self._split_periods(X_seq)

        start_time = time.time()
        train_losses = []
        valid_losses = []
        best_valid_loss = float("inf")
        best_state = None
        patience_counter = 0
        n_batches = max(1, n_samples // batch_size)

        for epoch in range(n_epochs):
            net.train()
            epoch_loss = 0.0

            # 随机采样批次
            indices = np.random.permutation(n_samples)
            for b in range(n_batches):
                batch_idx = indices[b * batch_size:(b + 1) * batch_size]
                if len(batch_idx) == 0:
                    continue

                x_batch = torch.tensor(X_seq[batch_idx], dtype=torch.float32).to(self._device)
                y_batch = torch.tensor(y_seq[batch_idx], dtype=torch.float32).to(self._device).view(-1, 1)

                # 前向
                preds = net.forward(x_batch)
                task_loss = mse_loss(preds, y_batch)

                # MMD 分布匹配损失
                mmd_total = torch.tensor(0.0, device=self._device)
                if len(periods) >= 2 and lambda_mmd > 0:
                    for i in range(len(periods) - 1):
                        src, tgt = periods[i], periods[(i + 1) % len(periods)]
                        n_p = min(len(src), len(tgt))
                        if n_p < 2:
                            continue
                        x_s = net.gru(
                            torch.tensor(src[:n_p], dtype=torch.float32).to(self._device),
                            torch.zeros(num_layers, n_p, hidden_dim).to(self._device),
                        )[0][:, -1, :]
                        x_t = net.gru(
                            torch.tensor(tgt[:n_p], dtype=torch.float32).to(self._device),
                            torch.zeros(num_layers, n_p, hidden_dim).to(self._device),
                        )[0][:, -1, :]
                        mmd_total = mmd_total + AdaRNNModel._mmd_loss(x_s, x_t)

                loss = task_loss + lambda_mmd * mmd_total / max(len(periods) - 1, 1)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(net.gru.parameters()) + list(net.fc.parameters()), max_norm=5.0
                )
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # 验证
            if X_valid_seq is not None and y_valid_seq is not None:
                net.eval()
                with torch.no_grad():
                    n_valid = min(len(X_valid_seq), len(y_valid_seq))
                    x_val = torch.tensor(X_valid_seq[:n_valid], dtype=torch.float32).to(self._device)
                    y_val = torch.tensor(y_valid_seq[:n_valid], dtype=torch.float32).to(self._device).view(-1, 1)
                    val_preds = net.forward(x_val)
                    val_loss = mse_loss(val_preds, y_val).item()
                valid_losses.append(val_loss)

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    best_state = net.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.logger.info(f"AdaRNN 早停 @ epoch {epoch + 1}, best_valid={best_valid_loss:.6f}")
                    break
            else:
                if avg_train_loss < best_valid_loss:
                    best_valid_loss = avg_train_loss
                    best_state = net.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.logger.info(f"AdaRNN 早停 (训练) @ epoch {epoch + 1}")
                    break

        train_time_ms = (time.time() - start_time) * 1000

        # 恢复最优模型
        if best_state is not None:
            net.load_state_dict(best_state)

        net.eval()
        self._model = net
        self._fitted = True

        result = TrainingResult(
            model_name="AdaRNN",
            train_loss=train_losses,
            valid_loss=valid_losses,
            best_iteration=epoch + 1 - patience_counter,
            best_score=best_valid_loss,
            train_time_ms=train_time_ms,
            n_features=input_dim,
            n_samples=n_samples,
        )

        self.logger.info(
            "AdaRNN 训练完成",
            epoch=len(train_losses),
            best_loss=round(best_valid_loss, 6),
            train_ms=round(train_time_ms, 0),
        )

        return result

    # ================================================================
    #  predict
    # ================================================================

    def predict(self, X: np.ndarray) -> PredictionResult:
        if not self._fitted or self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        import torch

        seq_len = self.params["seq_len"]
        net = self._model
        net.eval()

        # 转换为序列
        X_seq = self._to_sequences(X, seq_len)
        n_samples = len(X_seq)

        # 批次预测
        batch_size = 256
        preds_list = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                x_batch = torch.tensor(X_seq[i:end], dtype=torch.float32).to(self._device)
                preds = net.forward(x_batch).cpu().numpy().flatten()
                preds_list.append(preds)

        predictions = np.concatenate(preds_list) if preds_list else np.array([])

        # 补齐到原始长度
        if len(predictions) < X.shape[0]:
            pad_len = X.shape[0] - len(predictions)
            predictions = np.concatenate([np.full(pad_len, predictions[0]), predictions])

        return PredictionResult(predictions=predictions)

    # ================================================================
    #  保存 / 加载 (覆盖以支持 PyTorch state_dict)
    # ================================================================

    def save(self, path: str):
        """保存 PyTorch 模型权重和配置"""
        import pickle
        import torch

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        data = {
            "state_dict": self._model.state_dict() if self._model else None,
            "params": self.params,
            "feature_names": self._feature_names,
            "config": {
                "input_dim": self.params["input_dim"],
                "hidden_dim": self.params["hidden_dim"],
                "num_layers": self.params["num_layers"],
                "dropout": self.params["dropout"],
                "device": self._device,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        self.logger.info("AdaRNN 模型已保存", path=path)

    def load(self, path: str):
        """加载 PyTorch 模型权重和配置"""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.params = data.get("params", self.params)
        self._feature_names = data.get("feature_names", [])
        self._device = data.get("config", {}).get("device", "cpu")

        cfg = data.get("config", {})
        input_dim = cfg.get("input_dim", self.params.get("input_dim", 64))
        hidden_dim = cfg.get("hidden_dim", self.params.get("hidden_dim", 128))
        num_layers = cfg.get("num_layers", self.params.get("num_layers", 2))
        dropout = cfg.get("dropout", self.params.get("dropout", 0.2))

        self._model = AdaRNNModel._GRUMMDNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            device=self._device,
        ).to(self._device)

        if data.get("state_dict"):
            self._model.load_state_dict(data["state_dict"])

        self._fitted = True
        self.logger.info("AdaRNN 模型已加载", path=path)
