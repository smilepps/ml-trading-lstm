#!/usr/bin/env python3
"""
train_gpu.py — Обучение LSTM на PyTorch CUDA с мультитаймфреймным входом (900 свечей)
Экспорт в ONNX. Порог настраивается через --threshold или env THRESHOLD.

Иерархия интервалов (для добавления двух старших):
1m -> 10m -> 60m -> 4h -> 1D -> (нет старшего)

usage:
  python train_gpu.py [--intervals 10m,60m,4h,1D] [--ticker IMOEX]
                      [--window 300] [--horizon 10] [--threshold 0.003]
                      [--epochs 100] [--intervals-map '{"10m":["60m","4h"]}']
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Конфигурация по умолчанию ──
WINDOW_SIZE = 300
HORIZON = 10
TICKER = "IMOEX"
INTERVALS = ["60m", "10m", "4h", "1D", "2h"]
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.001
PATIENCE = 10
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# Иерархия интервалов: для каждого — два старших (если есть)
HIGHER_MAP_DEFAULT = {
    "1m": ["10m", "60m"],
    "10m": ["60m", "4h"],
    "15m": ["60m", "4h"],
    "30m": ["60m", "4h"],
    "60m": ["4h", "1D"],
    "2h": ["4h", "1D"],
    "4h": ["1D", None],
    "1D": [None, None],
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_db_conn():
    db_url = os.environ.get("DB", "postgresql://smile:5ty765@127.0.0.1:5432/main")
    return psycopg2.connect(db_url)


def load_candles(conn, interval, ticker):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT begin, open, high, low, close, volume
        FROM candles
        WHERE ticker = %s AND interval = %s
        ORDER BY begin ASC
        """,
        (ticker, interval),
    )
    rows = cur.fetchall()
    cur.close()
    return [
        {
            "begin": r[0],
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": int(r[5]) or 0,
        }
        for r in rows
    ]


def normalize_candles(candles):
    """Возвращает (normalized_array [n-1, 5], scaler dict)"""
    n = len(candles)
    log_returns = np.zeros(n - 1, dtype=np.float32)
    volumes = np.zeros(n, dtype=np.float32)

    for i in range(n):
        volumes[i] = candles[i]["volume"]

    for i in range(1, n):
        log_returns[i - 1] = math.log(candles[i]["close"] / candles[i - 1]["close"])

    price_mean = float(np.mean(log_returns))
    price_std = float(np.std(log_returns)) or 1.0
    vol_mean = float(np.mean(volumes))
    vol_std = float(np.std(volumes)) or 1.0

    normalized = np.zeros((n - 1, 5), dtype=np.float32)
    for i in range(1, n):
        c = candles[i]
        o = candles[i - 1]["close"]
        normalized[i - 1] = [
            (log_returns[i - 1] - price_mean) / price_std,
            (c["high"] - o) / o,
            (c["low"] - o) / o,
            (c["close"] - o) / o,
            (volumes[i] - vol_mean) / vol_std,
        ]

    scaler = {
        "price_mean": price_mean,
        "price_std": price_std,
        "vol_mean": vol_mean,
        "vol_std": vol_std,
    }
    return normalized, scaler


def label_candle(candles, idx, horizon, threshold):
    """
    Возвращает 1/0/-1 для свечи с индексом idx (последняя в окне текущего интервала).
    Смотрит на следующие `horizon` свечей этого же интервала.
    """
    if idx + horizon >= len(candles):
        return -1
    current_close = candles[idx]["close"]
    max_high = -float("inf")
    min_low = float("inf")
    for j in range(1, horizon + 1):
        c = candles[idx + j]
        max_high = max(max_high, c["high"])
        min_low = min(min_low, c["low"])

    up_target = current_close * (1.0 + threshold)
    down_target = current_close * (1.0 - threshold)

    if max_high >= up_target and min_low <= down_target:
        return -1  # оба достигли — исключаем
    if max_high >= up_target:
        return 1
    if min_low <= down_target:
        return 0
    return -1  # flat


def build_multi_tf_input(curr_norm, higher1_norm, higher2_norm, window_size):
    """
    Склеивает три окна по 300 свечей в одно: [900, 5].
    Если старшего нет — использует только доступные.
    """
    parts = []
    if curr_norm is not None and len(curr_norm) >= window_size:
        parts.append(curr_norm[-window_size:])
    if higher1_norm is not None and len(higher1_norm) >= window_size:
        parts.append(higher1_norm[-window_size:])
    if higher2_norm is not None and len(higher2_norm) >= window_size:
        parts.append(higher2_norm[-window_size:])

    if not parts:
        return None
    return np.concatenate(parts, axis=0)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=5, hidden1=64, hidden2=32, seq_len=900):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # последний таймстеп
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


def train_interval(interval, ticker, window_size, horizon, threshold, epochs, higher_map):
    print(f"\n{'='*60}")
    print(f"Interval: {interval} | threshold={threshold} | window={window_size}")
    print(f"{'='*60}")

    higher1, higher2 = higher_map.get(interval, [None, None])
    print(f"  Higher TFs: {higher1}, {higher2}")

    conn = get_db_conn()
    candles_curr = load_candles(conn, interval, ticker)
    conn.close()
    print(f"  Loaded {len(candles_curr)} candles @ {interval}")

    if len(candles_curr) < window_size + horizon + 100:
        print(f"  Not enough data, skipping")
        return None

    # Нормализация текущего интервала
    norm_curr, scaler_curr = normalize_candles(candles_curr)

    # Старшие интервалы (только для контекста, без меток)
    norm_h1, scaler_h1 = None, None
    norm_h2, scaler_h2 = None, None

    if higher1:
        conn = get_db_conn()
        candles_h1 = load_candles(conn, higher1, ticker)
        conn.close()
        if len(candles_h1) >= window_size + 10:
            norm_h1, scaler_h1 = normalize_candles(candles_h1)
            print(f"  Loaded {len(candles_h1)} candles @ {higher1}")
        else:
            print(f"  Not enough data for {higher1}, skipping")

    if higher2:
        conn = get_db_conn()
        candles_h2 = load_candles(conn, higher2, ticker)
        conn.close()
        if len(candles_h2) >= window_size + 10:
            norm_h2, scaler_h2 = normalize_candles(candles_h2)
            print(f"  Loaded {len(candles_h2)} candles @ {higher2}")
        else:
            print(f"  Not enough data for {higher2}, skipping")

    # Строим датасет: X = [900, 5], y = 0/1
    X = []
    y = []
    dates = []

    max_i = len(norm_curr) - window_size - horizon
    for i in range(max_i + 1):
        # Окно текущего интервала
        curr_win = norm_curr[i : i + window_size]
        # Окна старших интервалов — берём последние window_size перед текущей точкой
        # Привязка по времени: ищем ближайшие свечи старших ТФ к моменту i+window_size-1
        target_time = candles_curr[i + window_size - 1]["begin"]

        h1_win = None
        h2_win = None

        if norm_h1 is not None and candles_h1:
            # Находим индекс в h1, correspondiente target_time (или ближайшая предыдущая)
            h1_idx = None
            for j, c in enumerate(candles_h1):
                if c["begin"] <= target_time:
                    h1_idx = j
                else:
                    break
            if h1_idx is not None and h1_idx >= window_size - 1:
                h1_win = norm_h1[h1_idx - window_size + 1 : h1_idx + 1]

        if norm_h2 is not None and candles_h2:
            h2_idx = None
            for j, c in enumerate(candles_h2):
                if c["begin"] <= target_time:
                    h2_idx = j
                else:
                    break
            if h2_idx is not None and h2_idx >= window_size - 1:
                h2_win = norm_h2[h2_idx - window_size + 1 : h2_idx + 1]

        # Метка по текущему интервалу
        label = label_candle(candles_curr, i + window_size - 1, horizon, threshold)
        if label == -1:
            continue

        # Собираем вход
        parts = [curr_win]
        if h1_win is not None:
            parts.append(h1_win)
        if h2_win is not None:
            parts.append(h2_win)

        if len(parts) < 1:
            continue

        x_sample = np.concatenate(parts, axis=0)  # [900, 5] или меньше
        X.append(x_sample)
        y.append(label)
        dates.append(str(candles_curr[i + window_size - 1]["begin"]))

    if len(X) < 100:
        print(f"  Too few labeled samples: {len(X)}")
        return None

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"  Dataset: {len(X)} samples (up: {int((y==1).sum())}, down: {int((y==0).sum())})")
    print(f"  Input shape per sample: {X.shape[1]} time steps x 5 features")

    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    seq_len = X.shape[1]
    model = LSTMClassifier(input_size=5, hidden1=64, hidden2=32, seq_len=seq_len).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            val_acc = ((val_preds > 0.5).float() == y_val_t).float().mean().item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Тест
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_loss = criterion(test_preds, y_test_t).item()
        test_acc = ((test_preds > 0.5).float() == y_test_t).float().mean().item()
        tp = int(((test_preds > 0.5).float() == 1) & (y_test_t == 1)).item()
        fp = int(((test_preds > 0.5).float() == 1) & (y_test_t == 0)).item()
        tn = int(((test_preds > 0.5).float() == 0) & (y_test_t == 0)).item()
        fn = int(((test_preds > 0.5).float() == 0) & (y_test_t == 1)).item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n  Test Metrics:")
    print(f"    Accuracy:  {test_acc*100:.2f}%")
    print(f"    Precision: {precision*100:.2f}%")
    print(f"    Recall:    {recall*100:.2f}%")
    print(f"    F1:        {f1*100:.2f}%")
    print(f"    TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Сохраняем скейлеры (все использованные)
    os.makedirs(OUT_DIR, exist_ok=True)
    interval_safe = interval.replace("m", "m").replace("h", "h").replace("D", "D")
    scaler_data = {
        "curr": scaler_curr,
        "higher1": scaler_h1,
        "higher2": scaler_h2,
        "higher1_name": higher1,
        "higher2_name": higher2,
        "threshold": threshold,
        "window_size": window_size,
        "horizon": horizon,
        "input_steps": X.shape[1],
    }
    scaler_path = os.path.join(OUT_DIR, f"scaler_{interval_safe}.json")
    with open(scaler_path, "w") as f:
        json.dump(scaler_data, f, indent=2)
    print(f"  Scaler saved: {scaler_path}")

    # Экспорт в ONNX
    model.eval()
    dummy_input = torch.randn(1, seq_len, 5, dtype=torch.float32).to(DEVICE)
    onnx_path = os.path.join(OUT_DIR, f"model_{interval_safe}.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 1: "time"}, "output": {0: "batch"}},
    )
    print(f"  Model saved: {onnx_path}")

    return {
        "interval": interval,
        "threshold": threshold,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "input_steps": seq_len,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", default=",".join(INTERVALS))
    parser.add_argument("--ticker", default=TICKER)
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    global WINDOW_SIZE, HORIZON, TICKER, INTERVALS, EPOCHS
    WINDOW_SIZE = args.window
    HORIZON = args.horizon
    TICKER = args.ticker
    INTERVALS = [s.strip() for s in args.intervals.split(",")]
    EPOCHS = args.epochs

    # Порог: аргумент > env > default
    threshold = args.threshold if args.threshold is not None else float(os.environ.get("THRESHOLD", "0.003"))

    # Иерархия старших интервалов
    higher_map = HIGHER_MAP_DEFAULT.copy()
    # Ограничиваем только теми интервалами, которые есть в списке обучения
    interval_set = set(INTERVALS)
    for k in list(higher_map.keys()):
        higher_map[k] = [h for h in higher_map[k] if h is None or h in interval_set]

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print(f"Threshold: {threshold}")
    print(f"Intervals: {INTERVALS}")
    print(f"Window: {WINDOW_SIZE}, Horizon: {HORIZON}, Epochs: {EPOCHS}")

    results = []
    for interval in INTERVALS:
        try:
            r = train_interval(
                interval, TICKER, WINDOW_SIZE, HORIZON, threshold, EPOCHS, higher_map
            )
            if r:
                results.append(r)
        except Exception as e:
            print(f"  Error training {interval}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Training complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
