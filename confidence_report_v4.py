#!/usr/bin/env python3
"""Отчёт по уверенности новых многотаймфреймовых моделей (scaler v4)."""
import json
import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import psycopg2

import train_gpu as T

TICKER = "IMOEX"
INTERVALS = ["5m", "10m", "15m", "30m", "60m", "1D"]
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
TMP = os.path.join(tempfile.gettempdir(), "opencode")
os.makedirs(TMP, exist_ok=True)


def make_dynamic(src, dst):
    m = onnx.load(src)
    for inp in m.graph.input:
        inp.type.tensor_type.shape.dim[0].dim_param = "batch"
    for out in m.graph.output:
        out.type.tensor_type.shape.dim[0].dim_param = "batch"
    onnx.save(m, dst)
    return dst


def split_ranges(n, horizon=T.HORIZON):
    gap = min(horizon, max(0, n // 10))
    train_limit = int(n * T.TRAIN_RATIO)
    val_limit = int(n * (T.TRAIN_RATIO + T.VAL_RATIO))
    train_end = max(1, train_limit - gap)
    val_start = min(n, train_limit)
    val_end = max(val_start, val_limit - gap)
    test_start = min(n, val_limit)
    return {
        "train": (0, train_end),
        "val": (val_start, val_end),
        "test": (test_start, n),
    }


def load_scaler(interval):
    path = os.path.join(MODELS, f"scaler_{TICKER}_{interval}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate(interval, conn):
    candles = T.load_candles(conn, interval, TICKER, quiet=True)
    scaler = load_scaler(interval)
    base_norm, _ = T.normalize_candles(candles, scaler)

    contexts = []
    for ci in T.CONTEXT_MAP.get(interval, []):
        ctx_candles = T.load_candles(conn, ci, TICKER, quiet=True)
        ctx_scaler = load_scaler(ci)
        ctx_norm, _ = T.normalize_candles(ctx_candles, ctx_scaler)
        contexts.append({
            "interval": ci,
            "candles": ctx_candles,
            "normalized": ctx_norm,
            "begins": [c["begin"] for c in ctx_candles],
            "scaler": ctx_scaler,
        })

    Xs, y, dates, skipped = T.create_dataset(
        base_norm, candles, T.WINDOW_SIZE, T.HORIZON, T.THRESHOLD, contexts, interval
    )
    n = len(y)
    src = os.path.join(MODELS, f"model_{TICKER}_{interval}.onnx")
    dst = os.path.join(TMP, f"model_{TICKER}_{interval}_dyn.onnx")
    sess = ort.InferenceSession(make_dynamic(src, dst), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]

    def run(a, b):
        preds = []
        for start in range(a, b, 256):
            feeds = {
                name: Xs[k][start:min(b, start + 256)]
                for k, name in enumerate(names)
            }
            preds.append(sess.run(None, feeds)[0].reshape(-1))
        return np.concatenate(preds) if preds else np.zeros(0)

    splits = split_ranges(n)
    report = {
        "interval": interval,
        "candles": len(candles),
        "labeled": n,
        "skipped_context": skipped,
        "inputs": 1 + len(contexts),
        "contexts": T.CONTEXT_MAP.get(interval, []),
        "splits": {},
    }
    for name, (a, b) in splits.items():
        p = run(a, b)
        yy = y[a:b]
        conf = np.maximum(p, 1.0 - p)
        mask = conf > 0.60
        item = {
            "n": int(len(p)),
            "max_confidence": float(conf.max()) if len(conf) else None,
            "p50_confidence": float(np.percentile(conf, 50)) if len(conf) else None,
            "p90_confidence": float(np.percentile(conf, 90)) if len(conf) else None,
            "p99_confidence": float(np.percentile(conf, 99)) if len(conf) else None,
            "accuracy": float(((p >= 0.5).astype(int) == yy).mean()) if len(yy) else None,
            "n_conf_gt_60": int(mask.sum()),
            "share_conf_gt_60": float(mask.mean()) if len(mask) else None,
            "acc_conf_gt_60": float(((p[mask] >= 0.5).astype(int) == yy[mask]).mean()) if mask.sum() else None,
            "n_conf_ge_90": int((conf >= 0.90).sum()),
            "share_conf_ge_90": float((conf >= 0.90).mean()) if len(mask) else None,
            "buckets": {
                "50-55": int(((conf > 0.50) & (conf <= 0.55)).sum()),
                "55-60": int(((conf > 0.55) & (conf <= 0.60)).sum()),
                "60-65": int(((conf > 0.60) & (conf <= 0.65)).sum()),
                "65-70": int(((conf > 0.65) & (conf <= 0.70)).sum()),
                "70+": int((conf > 0.70).sum()),
            },
        }
        report["splits"][name] = item
    return report


def load_env(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main():
    load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    conn = psycopg2.connect(os.environ["DB"])
    conn.autocommit = True
    reports = []
    try:
        for interval in INTERVALS:
            r = evaluate(interval, conn)
            reports.append(r)
            print(json.dumps(r, ensure_ascii=True), flush=True)
    finally:
        conn.close()
    out = os.path.join(MODELS, "confidence_report_v4.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
