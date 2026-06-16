"""
LSTM regressor for temperature forecasting.
Ported from smart-wallet-ml/models/neural.py (classifier → regressor).
1-layer LSTM with seq_len=20 days over feature sequences.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

SEQ_LEN = 20
HIDDEN = 64
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 32


class _LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    seqs, labels = [], []
    for i in range(seq_len, len(X)):
        seqs.append(X[i - seq_len:i])
        labels.append(y[i])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)


def train_lstm(X: pd.DataFrame, y: pd.Series) -> dict:
    if not _HAS_TORCH:
        return {"type": "lstm", "unavailable": True}

    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X).astype(np.float32)
    y_np = y.values.astype(np.float32)

    seqs, labels = _make_sequences(X_sc, y_np, SEQ_LEN)
    if len(seqs) < 10:
        return {"type": "lstm", "unavailable": True, "reason": "insufficient data"}

    model = _LSTMRegressor(X_sc.shape[1], HIDDEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(seqs), torch.tensor(labels)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    return {"type": "lstm", "model": model, "scaler": scaler}


def predict_lstm(state: dict, X: pd.DataFrame) -> np.ndarray:
    if state.get("unavailable"):
        return np.full(len(X), np.nan)

    model = state["model"]
    scaler = state["scaler"]
    X_sc = scaler.transform(X).astype(np.float32)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X_sc)):
            start = max(0, i - SEQ_LEN + 1)
            window = X_sc[start:i + 1]
            if len(window) < SEQ_LEN:
                preds.append(np.nan)
            else:
                tensor = torch.tensor(window[np.newaxis])
                preds.append(model(tensor).item())

    return np.array(preds, dtype=np.float32)
