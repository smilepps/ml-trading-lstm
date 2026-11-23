#!/usr/bin/env python3
"""
train_gpu.py — обучение моделей (LSTM / linear) и экспорт в ONNX (scaler v4).

На вход модели идёт несколько окон по 300 свечей:
  • base — свечи обучаемого интервала;
  • h1, h2 — по 300 свечей следующих старших интервалов.

Для 10m это 300×10m + 300×60m + 300×1D, для 60m — 300×60m + 300×1D,
для 1D — только base. Окна выровнены по времени закрытия текущей base-свечи:
старшие свечи берутся только полностью закрытые, без заглядывания в будущее.
Внутри LSTM каждое окно разворачивается в хронологический порядок.
Даты в выборку не входят — только 7 чисел на свечу (9 для тикеров с OI).

Признаки (7):
  ret_z      — лог-доходность close к предыдущему close, z-score по train
  high_dev   — (high - prev_close) / prev_close
  low_dev    — (low - prev_close) / prev_close
  close_dev  — (close - prev_close) / prev_close
  value_z    — log1p(value) (оборот), z-score по train
  dow_z      — день недели (ISODOW 1..7), z-score по train
  hour_z     — час свечи (0..23, московское время), z-score по train

Для тикеров с futoi-маппингом (IMOEX → IMOEXF) и данными в open_interest
добавляются ещё 2 признака открытых позиций фьючерса на базовый актив:
  yur_net_z  — нетто-позиция юрлиц (FIZ/YUR-разрез MOEX ISS futoi), z-score
  fiz_net_z  — нетто-позиция физлиц, z-score
OI привязывается as-of join к закрытию свечи (последний снимок ts <= close,
5-минутные снимки, forward-fill через паузы). Свечи без OI (до 2023-11-16,
embargo-окно последних 14 дней) получают нейтральный 0; статистики скейлера
считаются только по свечам, где OI реально есть.

1m-свечи обучаются как обычный интервал; контексты — 10m/60m.

Граница скейлеров причинная: mean/std считаются только по свечам, строго
предшествующим первому удержанному (val/test) примеру.
  • base-скейлер: candles[:анкор val_start];
  • скейлер контекста: свечи контекста, полностью закрытые к началу той же
    base-свечи (свои dow/hour-статистики).
Поэтому val/test не содержат заглядывания в будущее, а scaler_*.json ровно
такой, какой читает инференс: индекс границы — поле train_cut, её дата —
scaler_end.

usage:
  python train_gpu.py [--intervals "1D,60m,10m,1m"] [--ticker IMOEX]
                      [--window 300] [--horizon 10] [--threshold 0.003]
                      [--epochs 100] [--hidden1 64] [--hidden2 32]
                      [--dropout 0.2] [--lr 0.001] [--weight-decay 0]
                      [--patience 10] [--balance | --no-balance]
                      [--select auto|loss|bal]
                      [--reuse-scaler] [--arch auto|lstm|linear]

Порядок интервалов важен: старшие интервалы должны обучаться раньше младших,
чтобы их обновлённые скейлеры использовались как контекст.

--balance        — сэмплировать train с равным весом классов (борется с
                   вырождением «всегда UP»); по умолчанию auto: включается
                   для маленьких train-выборок (< 8000 окон), отключается
                   флагом --no-balance;
--select bal     — выбирать лучшую эпоху по balanced accuracy на val, а не
                   по val loss; по умолчанию auto (bal при включённом
                   balance, иначе loss);
--reuse-scaler   — не пересчитывать и не перезаписывать scaler_*.json
                   (нужно при точечном переобучении: контекстные модели
                   сохраняют совместимость);
--arch linear    — линейная модель по 19 pooled-признакам вместо LSTM
                   (для маленьких выборок). По умолчанию auto: linear для
                   интервала без контекстов с train < 5000 окон (1D).
"""

import argparse
import bisect
import json
import math
import os
import random
import time
from datetime import datetime, timedelta

import numpy as np
import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def load_env_file():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


load_env_file()

# ── Конфигурация ──
def load_threshold_config():
    """Читает thresholds.json, чтобы JS и Python считали порог одинаково."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds.json")
    defaults = {
        "intervalMultipliers": {"1m": 1.0, "10m": 2.0, "60m": 4.0, "1D": 8.0},
        "fallbackMultipliers": [
            {"maxMinutes": 1, "multiplier": 1.0},
            {"maxMinutes": 10, "multiplier": 2.0},
            {"maxMinutes": 60, "multiplier": 4.0},
            {"maxMinutes": 1000000, "multiplier": 8.0},
        ],
    }
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return defaults

    multipliers = {}
    source = raw.get("intervalMultipliers") if isinstance(raw, dict) else None
    if isinstance(source, dict):
        for interval, value in source.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number > 0:
                multipliers[str(interval)] = number

    fallback = []
    source = raw.get("fallbackMultipliers") if isinstance(raw, dict) else None
    if isinstance(source, list):
        for item in source:
            if not isinstance(item, dict):
                continue
            try:
                max_minutes = float(item.get("maxMinutes"))
                multiplier = float(item.get("multiplier"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(max_minutes) and math.isfinite(multiplier) and max_minutes > 0 and multiplier > 0:
                fallback.append({"maxMinutes": max_minutes, "multiplier": multiplier})

    return {
        "intervalMultipliers": multipliers or defaults["intervalMultipliers"],
        "fallbackMultipliers": sorted(fallback, key=lambda x: x["maxMinutes"]) or defaults["fallbackMultipliers"],
    }


THRESHOLD_CONFIG = load_threshold_config()

WINDOW_SIZE = 300
HORIZON = 10
HORIZON_MIN = 5
HORIZON_MAX = 25
THRESHOLD = 0.003
# Базовый порог 0.3% для 1m. На каждом следующем интервале множитель ×2.
# Горизонт задаёт число свечей прогноза, но не меняет порог.
THRESHOLD_PROGRESSIVE = True
THRESHOLD_INTERVAL_MULTIPLIER = THRESHOLD_CONFIG["intervalMultipliers"]
THRESHOLD_FALLBACK_MULTIPLIERS = THRESHOLD_CONFIG["fallbackMultipliers"]

# Минуты на одну свечу интервала.
INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "2h": 120,
    "4h": 240,
    "1D": 24 * 60,
}
TICKER = "IMOEX"
INTERVALS = ["1D", "60m", "10m", "1m"]
EPOCHS = 100
GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_DATAPARALLEL = GPU_COUNT > 1
BATCH_SIZE = 64 * max(1, GPU_COUNT)
LR = 0.001
PATIENCE = 10
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
SEED = 42

FEATURE_NAMES = [
    "ret_z",
    "high_dev",
    "low_dev",
    "close_dev",
    "value_z",
    "dow_z",
    "hour_z",
]
BASE_FEATURE_NAMES = list(FEATURE_NAMES)
# Открытые позиции фьючерса на базовый актив (MOEX ISS futoi, группы FIZ/YUR).
# Добавляются только для тикеров с futoi-маппингом и данными в open_interest.
OI_FEATURE_NAMES = ["yur_net_z", "fiz_net_z"]
FUTOI_SECID = {"IMOEX": "IMOEXF", "SBER": "SBRF"}
INPUT_SIZE = len(FEATURE_NAMES)


def oi_enabled():
    return len(FEATURE_NAMES) > len(BASE_FEATURE_NAMES)


def stat_keys():
    keys = [
        "ret_mean", "ret_std", "value_mean", "value_std",
        "dow_mean", "dow_std", "hour_mean", "hour_std",
    ]
    if oi_enabled():
        keys += ["yur_net_mean", "yur_net_std", "fiz_net_mean", "fiz_net_std"]
    return tuple(keys)

CONTEXT_MAP = {
    "1m": ["10m", "60m"],
    "10m": ["60m", "1D"],
    "60m": ["1D"],
    "1D": [],
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def interval_delta(interval):
    minutes = INTERVAL_MINUTES.get(str(interval))
    return timedelta(minutes=minutes) if minutes else timedelta(minutes=1)


def interval_minutes(interval):
    return int(INTERVAL_MINUTES.get(str(interval), 1))


def horizon_multiplier(interval, horizon=None):
    """Множитель порога по интервалу; горизонт порог не меняет."""
    key = str(interval)
    if key in THRESHOLD_INTERVAL_MULTIPLIER:
        return float(THRESHOLD_INTERVAL_MULTIPLIER[key])
    minutes = INTERVAL_MINUTES.get(key)
    if minutes:
        for item in THRESHOLD_FALLBACK_MULTIPLIERS:
            if minutes <= item["maxMinutes"]:
                return float(item["multiplier"])
    raise ValueError(f"Неподдерживаемый интервал для порога: {interval}")


def effective_threshold(threshold, interval, horizon):
    """Реальный порог движения (базовый × прогрессивный множитель)."""
    return float(threshold) * horizon_multiplier(interval, horizon)


def threshold_record(threshold, interval, horizon):
    """Пороговая часть метаданных скейлера/сэмпла (единая для всего приложения)."""
    mult = horizon_multiplier(interval, horizon)
    eff = effective_threshold(threshold, interval, horizon)
    return {
        # backward-compatible поле: эффективный порог для данного интервала/горизонта
        "threshold": round(eff, 8),
        "threshold_base": round(float(threshold), 8),
        "threshold_horizon": int(horizon),
        "threshold_multiplier": round(mult, 6),
        "threshold_progressive": True,
        "threshold_interval_minutes": interval_minutes(interval),
    }


def safe_name(value):
    return str(value).replace("/", "_").replace("\\", "_")


def write_json_atomic(path, payload):
    """Пишем через временный файл, чтобы сервер не прочитал половину JSON."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def get_db_conn():
    return psycopg2.connect(
        os.environ.get("DB", "postgresql://smile:5ty765@127.0.0.1:5432/main")
    )


def load_candles(conn, interval, ticker, quiet=False):
    """Свечи строго по возрастанию begin; dow/hour считаются в SQL (московское время)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT begin, open, high, low, close, value,
                   EXTRACT(ISODOW FROM begin)::int AS dow,
                   EXTRACT(HOUR FROM begin)::int AS hour
            FROM candles
            WHERE ticker = %s AND interval = %s
            ORDER BY begin ASC
            """,
            (ticker, interval),
        )
        rows = cur.fetchall()

    candles = []
    prev = None
    for r in rows:
        item = {
            "begin": r[0],
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "value": float(r[5]) if r[5] is not None else 0.0,
            "dow": int(r[6]),
            "hour": int(r[7]),
        }
        if prev is not None and item["begin"] <= prev:
            raise ValueError(
                f"{interval}: нарушен порядок дат: {prev} → {item['begin']}"
            )
        prev = item["begin"]
        candles.append(item)
    if not quiet:
        log(f"  {interval}: свечей {len(candles)}")
    return candles


def oi_has_data(conn, ticker):
    """Есть ли в open_interest данные для тикера (таблицы может ещё не быть)."""
    if ticker not in FUTOI_SECID:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM open_interest WHERE ticker = %s)",
                (ticker,),
            )
            return bool(cur.fetchone()[0])
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        return False


def load_oi(conn, ticker):
    """
    Снимки открытых позиций futoi: (sorted ts, нетто YUR, нетто FIZ).
    None — если данных нет. Значения pos — нетто-позиция группы (long + short,
    short отрицательный), как отдаёт MOEX ISS.
    """
    if ticker not in FUTOI_SECID:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ts, clgroup, pos FROM open_interest
                WHERE ticker = %s ORDER BY ts ASC
                """,
                (ticker,),
            )
            rows = cur.fetchall()
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        return None
    if not rows:
        return None

    by_ts = {}
    for ts, clgroup, pos in rows:
        if clgroup not in ("FIZ", "YUR") or pos is None:
            continue
        by_ts.setdefault(ts, {})[clgroup] = float(pos)
    ts_list = sorted(by_ts)
    return {
        "ts": ts_list,
        "yur": [by_ts[ts].get("YUR") for ts in ts_list],
        "fiz": [by_ts[ts].get("FIZ") for ts in ts_list],
    }


def attach_oi(candles, interval, oi):
    """
    As-of join без заглядывания вперёд: к закрытию свечи (begin + длительность
    интервала) берётся последний снимок OI с ts <= close. Паузы/выходные внутри
    периода с данными закрываются forward-fill (bisect сам берёт последний
    известный снимок). До первого снимка (история futoi с 2023-11-16) и в
    embargo-окне последних 14 дней значения None → нейтральный 0 после
    нормализации.
    """
    if oi is None:
        for c in candles:
            c["yur_net"] = None
            c["fiz_net"] = None
        return
    ts_list = oi["ts"]
    delta = interval_delta(interval)
    filled = 0
    for c in candles:
        i = bisect.bisect_right(ts_list, c["begin"] + delta) - 1
        if i >= 0 and oi["yur"][i] is not None and oi["fiz"][i] is not None:
            c["yur_net"] = oi["yur"][i]
            c["fiz_net"] = oi["fiz"][i]
            filled += 1
        else:
            c["yur_net"] = None
            c["fiz_net"] = None
    return filled


def _oi_values(candles):
    """Пары (yur_net, fiz_net) только по свечам, где OI реально есть."""
    yur, fiz = [], []
    for c in candles:
        y = c.get("yur_net")
        f = c.get("fiz_net")
        if y is not None and f is not None:
            yur.append(y)
            fiz.append(f)
    return yur, fiz


def _stats(values):
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not math.isfinite(std) or std < 1e-12:
        std = 1.0
    return mean, std


def compute_scaler(candles, oi_fallback=None):
    """Параметры нормализации по переданному срезу свечей (обычно train).

    OI-статистики считаются только по свечам, где OI реально есть. Если в
    срезе таких свечей нет (например, train-срез 1D целиком до 2023-11-16),
    используется oi_fallback — весь доступный ряд OI (с предупреждением).
    """
    n = len(candles)
    if n < 2:
        raise ValueError("Для скейлера нужно минимум 2 свечи")

    rets = np.empty(n - 1, dtype=np.float64)
    values = np.empty(n - 1, dtype=np.float64)
    dows = np.empty(n - 1, dtype=np.float64)
    hours = np.empty(n - 1, dtype=np.float64)

    for i in range(1, n):
        c = candles[i]
        prev_close = candles[i - 1]["close"]
        if prev_close <= 0 or c["close"] <= 0:
            raise ValueError(f"Некорректная цена в свече {c['begin']}")
        rets[i - 1] = math.log(c["close"] / prev_close)
        values[i - 1] = math.log1p(max(0.0, c["value"]))
        dows[i - 1] = c["dow"]
        hours[i - 1] = c["hour"]

    ret_mean, ret_std = _stats(rets)
    value_mean, value_std = _stats(values)
    dow_mean, dow_std = _stats(dows)
    hour_mean, hour_std = _stats(hours)

    stats = {
        "ret_mean": ret_mean,
        "ret_std": ret_std,
        "value_mean": value_mean,
        "value_std": value_std,
        "dow_mean": dow_mean,
        "dow_std": dow_std,
        "hour_mean": hour_mean,
        "hour_std": hour_std,
    }

    if oi_enabled():
        yur, fiz = _oi_values(candles[1:])
        if not yur and oi_fallback is not None:
            yur, fiz = _oi_values(oi_fallback[1:])
            if yur:
                log(
                    f"  OI-статистика: в train-срезе OI нет, считаем по всему "
                    f"доступному периоду ({len(yur)} свечей) — нейтральные "
                    "константы нормализации могут выходить за причинную границу"
                )
        if yur:
            yur_mean, yur_std = _stats(yur)
            fiz_mean, fiz_std = _stats(fiz)
        else:
            # OI нет вовсе — нейтральная нормализация, признаки будут нулями.
            yur_mean, yur_std = 0.0, 1.0
            fiz_mean, fiz_std = 0.0, 1.0
        stats.update({
            "yur_net_mean": yur_mean,
            "yur_net_std": yur_std,
            "fiz_net_mean": fiz_mean,
            "fiz_net_std": fiz_std,
        })

    return stats


def normalize_candles(candles, scaler):
    """
    Возвращает массив [len(candles)-1, F]: строка k соответствует свече k+1.
    F = 7 базовых признаков + 2 OI-признака, если скейлер содержит OI-статистики.
    Свечи без OI (до 2023-11-16, embargo-окно, тикеры без futoi) получают
    нейтральный 0 в OI-колонках.
    """
    n = len(candles)
    with_oi = "yur_net_mean" in scaler and "fiz_net_mean" in scaler
    width = len(BASE_FEATURE_NAMES) + (len(OI_FEATURE_NAMES) if with_oi else 0)
    out = np.empty((max(0, n - 1), width), dtype=np.float64)

    ret_mean = scaler["ret_mean"]
    ret_std = scaler.get("ret_std") or 1.0
    value_mean = scaler["value_mean"]
    value_std = scaler.get("value_std") or 1.0
    dow_mean = scaler["dow_mean"]
    dow_std = scaler.get("dow_std") or 1.0
    hour_mean = scaler["hour_mean"]
    hour_std = scaler.get("hour_std") or 1.0
    if with_oi:
        yur_mean = scaler["yur_net_mean"]
        yur_std = scaler.get("yur_net_std") or 1.0
        fiz_mean = scaler["fiz_net_mean"]
        fiz_std = scaler.get("fiz_net_std") or 1.0

    for i in range(1, n):
        c = candles[i]
        prev_close = candles[i - 1]["close"]
        if prev_close <= 0 or c["close"] <= 0:
            raise ValueError(f"Некорректная цена в свече {c['begin']}")

        out[i - 1, 0] = (math.log(c["close"] / prev_close) - ret_mean) / ret_std
        out[i - 1, 1] = (c["high"] - prev_close) / prev_close
        out[i - 1, 2] = (c["low"] - prev_close) / prev_close
        out[i - 1, 3] = (c["close"] - prev_close) / prev_close
        out[i - 1, 4] = (math.log1p(max(0.0, c["value"])) - value_mean) / value_std
        out[i - 1, 5] = (c["dow"] - dow_mean) / dow_std
        out[i - 1, 6] = (c["hour"] - hour_mean) / hour_std
        if with_oi:
            yur = c.get("yur_net")
            fiz = c.get("fiz_net")
            out[i - 1, 7] = (yur - yur_mean) / yur_std if yur is not None else 0.0
            out[i - 1, 8] = (fiz - fiz_mean) / fiz_std if fiz is not None else 0.0

    return out


def label_candle(candles, idx, horizon, threshold, interval):
    """1 — достигнут up-таргет, 0 — down-таргет, -1 — flat/оба/нет будущего.

    Порог прогрессивный: эффективное целевое движение = базовый порог ×
    horizon_multiplier(interval, horizon).
    """
    if idx + horizon >= len(candles):
        return -1

    current_close = candles[idx]["close"]
    move = effective_threshold(threshold, interval, horizon)
    max_high = -float("inf")
    min_low = float("inf")
    for j in range(1, horizon + 1):
        c = candles[idx + j]
        max_high = max(max_high, c["high"])
        min_low = min(min_low, c["low"])

    up_target = current_close * (1.0 + move)
    down_target = current_close * (1.0 - move)

    if max_high >= up_target and min_low <= down_target:
        return -1
    if max_high >= up_target:
        return 1
    if min_low <= down_target:
        return 0
    return -1


def base_window(base_norm, idx, window_size):
    """300 строк по возрастанию даты, последняя — свеча idx."""
    if idx < window_size:
        return None
    return base_norm[idx - window_size: idx]


def context_begins(candles_ctx):
    """Список начал свечей контекста (для bisect; считаем один раз на интервал)."""
    return [c["begin"] for c in candles_ctx]


def count_closed(begins, ctx_interval, deadline):
    """Сколько свечей контекста полностью закрыты к моменту deadline."""
    return bisect.bisect_right(begins, deadline - interval_delta(ctx_interval))


def context_window(candles_ctx, norm_ctx, ctx_interval,
                   base_candle, base_interval, window_size, begins=None):
    """
    Окно старшего интервала, выровненное по закрытию base-свечи.
    Берутся только полностью закрытые свечи.
    """
    if begins is None:
        begins = context_begins(candles_ctx)
    base_end = base_candle["begin"] + interval_delta(base_interval)
    count = count_closed(begins, ctx_interval, base_end)
    idx = count - 1
    if idx < window_size:
        return None
    return norm_ctx[idx - window_size: idx]


def scaler_is_valid(record, interval):
    """
    Можно ли переиспользовать сохранённый скейлер: версия, число и имена
    признаков, список контекстов (должен совпадать с CONTEXT_MAP интервала)
    и все статистики конечные. Легаси-файлы (15m/30m, v3) отвергаются.
    """
    if not isinstance(record, dict):
        return False, "не словарь"
    if int(record.get("version") or 0) != 4:
        return False, f"version={record.get('version')}"
    if int(record.get("input_size") or 0) != INPUT_SIZE:
        return False, f"input_size={record.get('input_size')}"
    if list(record.get("feature_names") or []) != FEATURE_NAMES:
        return False, "feature_names"
    expected = list(CONTEXT_MAP.get(interval, []))
    actual = list(record.get("contexts") or [])
    if actual != expected:
        return False, f"contexts={actual} вместо ожидаемых {expected}"
    for key in stat_keys():
        try:
            value = float(record[key])
        except (KeyError, TypeError, ValueError):
            return False, f"нет статистики {key}"
        if not math.isfinite(value):
            return False, f"{key} не конечное"
    if float(record["ret_std"]) <= 0 or float(record["value_std"]) <= 0:
        return False, "std <= 0"
    if oi_enabled() and (
        float(record["yur_net_std"]) <= 0 or float(record["fiz_net_std"]) <= 0
    ):
        return False, "oi std <= 0"
    return True, ""


def stored_fit_end(record, candles):
    """Дата последней свечи, попавшей в статистику сохранённого скейлера."""
    raw = record.get("scaler_end")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    cut = int(record.get("train_cut") or 0)
    if 1 <= cut <= len(candles):
        return candles[cut - 1]["begin"]
    return None


def load_context_scaler(ticker, interval, ctx_candles, cut_candle, base_interval):
    """
    Для контекста берём уже существующий скейлер (его же читает инференс), если
    он совместим с текущим контрактом и причинный для данного разбиения. Иначе
    считаем по причинной границе — только по свечам контекста, полностью
    закрытым к началу первой удержанной base-свечи — и сохраняем.

    Возвращает (запись, пересчитан_ли).
    """
    path = os.path.join(OUT_DIR, f"scaler_{safe_name(ticker)}_{safe_name(interval)}.json")
    if os.path.exists(path):
        reason = ""
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, ValueError) as exc:
            record = None
            reason = f"файл не читается: {exc}"
        if record is not None:
            ok, reason = scaler_is_valid(record, interval)
            if ok:
                fit_end = stored_fit_end(record, ctx_candles)
                if fit_end is None:
                    ok, reason = False, "в файле нет даты конца фитова (scaler_end/train_cut)"
                elif fit_end >= cut_candle["begin"]:
                    ok, reason = False, (
                        f"скейлер фитован по свечи включительно до {fit_end}, "
                        f"а первая удержанная base-свеча началась {cut_candle['begin']} "
                        f"(было бы заглядывание в будущее)"
                    )
            if ok:
                log(f"  Скалер контекста {interval} переиспользован: {path}")
                return record, False
        log(
            f"  Скалер контекста {interval} отброшен ({reason}), "
            "пересчёт по причинной границе"
        )

    begins = context_begins(ctx_candles)
    cut = count_closed(begins, interval, cut_candle["begin"])
    cut = max(2, min(cut, len(ctx_candles)))
    if cut < 2:
        raise ValueError(
            f"Контекст {interval}: нет свечей, закрытых до {cut_candle['begin']}"
        )
    stats = compute_scaler(ctx_candles[:cut], oi_fallback=ctx_candles)
    contexts = CONTEXT_MAP.get(interval, [])
    record = scaler_record(
        ticker, interval, stats, contexts, len(ctx_candles), cut,
        fit_end=ctx_candles[cut - 1]["begin"],
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    write_json_atomic(path, record)
    log(
        f"  Скейлер контекста сохранён: {path} "
        f"(статистика по {cut} свечам, граница {cut_candle['begin']})"
    )
    return record, True


def scaler_record(ticker, interval, stats, contexts, candles_count, train_cut,
                  fit_end=None):
    """
    train_cut — число свечей, попавших в статистику: все они строго раньше
    первой удержанной (val/test) свечи, т.е. скейлер причинный.
    """
    record = {key: float(value) for key, value in stats.items()}
    record.update({
        "version": 4,
        "input_size": INPUT_SIZE,
        "inputs": 1 + len(contexts),
        "contexts": list(contexts),
        "context_windows": WINDOW_SIZE,
        "feature_names": FEATURE_NAMES,
        "window": WINDOW_SIZE,
        "horizon": HORIZON,
        **threshold_record(THRESHOLD, interval, HORIZON),
        "order": "current_to_history",
        "ticker": ticker,
        "interval": interval,
        "candles": int(candles_count),
        "train_cut": int(train_cut),
        "scaler_end": (
            fit_end.isoformat() if fit_end is not None else None
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    return record


def scan_samples(interval, base, ctx_begins_map, window_size=WINDOW_SIZE,
                 horizon=HORIZON, threshold=THRESHOLD):
    """
    Анкоры размеченных примеров (индексы base-свечи), метки и даты.

    Метка и проверка истории контекста зависят только от OHLC и дат, поэтому
    выборку можно собрать до того, как посчитаны скейлеры: границы скейлера
    определяются по уже готовому разбиению (см. train_interval).
    """
    anchors = []
    labels = []
    dates = []
    skipped_context = 0

    last_idx = len(base) - horizon
    for idx in range(window_size, last_idx):
        label = label_candle(base, idx, horizon, threshold, interval)
        if label == -1:
            continue

        base_end = base[idx]["begin"] + interval_delta(interval)
        ok = True
        for ctx, begins in ctx_begins_map.items():
            if count_closed(begins, ctx, base_end) - 1 < window_size:
                ok = False
                break
        if not ok:
            skipped_context += 1
            continue

        anchors.append(idx)
        labels.append(label)
        dates.append(base[idx]["begin"])

    return anchors, labels, dates, skipped_context


def build_windows(interval, base, base_norm, ctx_items, anchors, window_size=WINDOW_SIZE):
    """Окна (newest→history) для анкоры из scan_samples(), по одному на анкор."""
    samples = []
    for idx in anchors:
        bw = base_window(base_norm, idx, window_size)
        if bw is None:
            raise AssertionError(f"анкор {idx}: не хватает истории base")
        windows = [bw[::-1].copy()]
        for ctx, (candles_ctx, norm_ctx, begins) in ctx_items.items():
            cw = context_window(
                candles_ctx, norm_ctx, ctx, base[idx], interval, window_size, begins
            )
            if cw is None:
                raise AssertionError(f"анкор {idx}: не хватает истории {ctx}")
            windows.append(cw[::-1].copy())
        samples.append(windows)

    return samples


class Branch(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden1=64, hidden2=32):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)

    def forward(self, x):
        x = torch.flip(x, [1])
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        return out[:, -1, :]


class LSTMClassifier(nn.Module):
    """Несколько независимых LSTM-веток (base + старшие интервалы)."""

    def __init__(self, branches=1, input_size=INPUT_SIZE, hidden1=64, hidden2=32,
                 dropout=0.2):
        super().__init__()
        self.branches = nn.ModuleList([
            Branch(input_size, hidden1, hidden2) for _ in range(branches)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2 * branches, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs):
        feats = []
        for branch, x in zip(self.branches, inputs):
            feats.append(branch(x))
        out = torch.cat(feats, dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)


class LinearClassifier(nn.Module):
    """
    Линейная модель по сводным признакам окна.

    Нужна для маленьких выборок (1D), где LSTM переобучается/вырождается.
    Вход и выход те же, что у LSTM: [batch, 300, F] → [batch, 1].
    При F = 9 (с OI) добавляются 4 pooled-признака открытых позиций.
    """

    FEATURES = 19
    OI_FEATURES = 4

    def __init__(self, feature_mean=None, feature_std=None):
        super().__init__()
        if feature_mean is None:
            feature_mean = np.zeros(self.FEATURES, dtype=np.float32)
        if feature_std is None:
            feature_std = np.ones(len(feature_mean), dtype=np.float32)
        self.features = len(feature_mean)
        self.register_buffer(
            "feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "feature_std", torch.as_tensor(feature_std, dtype=torch.float32)
        )
        self.fc = nn.Linear(self.features, 1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def pool(x):
        ret = x[:, :, 0]
        high = x[:, :, 1]
        low = x[:, :, 2]
        close = x[:, :, 3]
        value = x[:, :, 4]

        def mean(a, n):
            return a[:, :n].mean(dim=1)

        def std(a, n):
            return a[:, :n].std(dim=1, unbiased=False)

        rng = high - low
        feats = [
            mean(close, 1), mean(close, 5), mean(close, 20), mean(close, 60),
            mean(ret, 1), mean(ret, 5), mean(ret, 20), mean(ret, 60),
            std(ret, 20), std(ret, 60), std(ret, 300),
            mean(value, 20), mean(value, 60),
            mean(rng, 20), mean(rng, 60),
            (ret[:, :20] > 0).float().mean(dim=1),
            (ret[:, :60] > 0).float().mean(dim=1),
            x[:, 0, 5], x[:, 0, 6],
        ]
        if x.shape[2] >= len(BASE_FEATURE_NAMES) + len(OI_FEATURE_NAMES):
            yur = x[:, :, 7]
            fiz = x[:, :, 8]
            feats += [mean(yur, 20), mean(yur, 60), mean(fiz, 20), mean(fiz, 60)]
        return torch.stack(feats, dim=1)

    def forward(self, x):
        pooled = self.pool(x)
        pooled = (pooled - self.feature_mean) / self.feature_std
        return self.sigmoid(self.fc(pooled))


def evaluate_batches(model, tensors, y, criterion, batch_size=BATCH_SIZE):
    model.eval()
    loss_sum = 0.0
    correct = 0
    preds = []
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            xb = [t[start:start + batch_size].to(DEVICE) for t in tensors]
            yb = y[start:start + batch_size].to(DEVICE)
            out = model(*xb)
            loss_sum += criterion(out, yb).item() * xb[0].size(0)
            correct += int(((out > 0.5).float() == yb).sum().item())
            preds.append(out.cpu())

    total = max(1, len(y))
    preds_all = torch.cat(preds) if preds else torch.zeros((0, 1))
    return loss_sum / total, correct / total, preds_all


def write_sample(ticker, interval, base, base_norm, ctx_items, scalers):
    contexts = list(ctx_items.keys())
    idx = len(base) - 1
    bw = base_window(base_norm, idx, WINDOW_SIZE)
    if bw is None:
        log("  ПРОПУСК примера входа: недостаточно истории base")
        return

    blocks = {"base": bw[::-1]}
    for number, ctx in enumerate(contexts, start=1):
        candles_ctx, norm_ctx, begins = ctx_items[ctx]
        cw = context_window(
            candles_ctx, norm_ctx, ctx, base[idx], interval, WINDOW_SIZE, begins
        )
        if cw is None:
            log(f"  ПРОПУСК примера входа: не хватает истории {ctx}")
            return
        blocks[f"h{number}"] = cw[::-1]

    payload = {
        "interval": interval,
        "ticker": ticker,
        "shape": {key: [int(v.shape[0]), int(v.shape[1])] for key, v in blocks.items()},
        "contexts": contexts,
        "window": WINDOW_SIZE,
        "horizon": HORIZON,
        **threshold_record(THRESHOLD, interval, HORIZON),
        "order": "current_to_history",
        "note": (
            "Строка 0 — текущая свеча; даты и OHLC в выборку не входят. "
            "h1/h2 — старшие интервалы, выровненные по закрытию base-свечи. "
            "LSTM разворачивает каждое окно в хронологический порядок внутри forward()."
        ),
        "feature_names": FEATURE_NAMES,
        "scalers": {name: {k: v for k, v in scalers[name].items()
                           if k in stat_keys()}
                    for name in scalers},
    }
    for key, block in blocks.items():
        payload[key] = np.round(block, 6).tolist()

    path = os.path.join(OUT_DIR, f"sample_input_{safe_name(ticker)}_{safe_name(interval)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    log(f"  Пример входа сохранён: {path}")


def train_interval(interval, ticker, window_size, horizon, threshold, epochs,
                   hidden1=64, hidden2=32, dropout=0.2, lr=LR,
                   weight_decay=0.0, patience=PATIENCE, balance=None,
                   select="auto", reuse_scaler=False, arch="auto"):
    started = time.time()
    log("=" * 70)
    log(f"ИНТЕРВАЛ {interval}: старт")
    log("=" * 70)

    contexts = CONTEXT_MAP.get(interval, [])
    log(f"  Контексты: {', '.join(contexts) if contexts else 'нет'}")

    conn = get_db_conn()
    try:
        candles = load_candles(conn, interval, ticker)
        ctx_candles_all = {}
        for ctx in contexts:
            ctx_candles = load_candles(conn, ctx, ticker)
            ctx_candles_all[ctx] = ctx_candles
        oi = load_oi(conn, ticker) if oi_enabled() else None
    finally:
        conn.close()

    if oi_enabled():
        filled = attach_oi(candles, interval, oi)
        for ctx in contexts:
            attach_oi(ctx_candles_all[ctx], ctx, oi)
        if oi is None:
            log("  OI: данных в open_interest нет — признаки будут нейтральными нулями")
        else:
            log(
                f"  OI ({FUTOI_SECID.get(ticker)}): снимков {len(oi['ts'])}, "
                f"{oi['ts'][0]} → {oi['ts'][-1]}; свечей с OI: "
                f"{filled}/{len(candles)} ({filled / max(1, len(candles)) * 100:.1f}%)"
            )

    if not candles:
        log(f"  ПРОПУСК: нет свечей {interval}")
        return None

    log(f"  Диапазон: {candles[0]['begin']} → {candles[-1]['begin']}")

    min_candles = window_size + horizon + 100
    if len(candles) < min_candles:
        log(f"  ПРОПУСК: недостаточно данных ({len(candles)} < {min_candles})")
        return None

    ctx_begins_map = {ctx: context_begins(ctx_candles_all[ctx]) for ctx in contexts}
    anchors, labels, dates, skipped_context = scan_samples(
        interval, candles, ctx_begins_map, window_size, horizon, threshold
    )
    if len(labels) < 100:
        log(f"  ПРОПУСК: слишком мало размеченных примеров ({len(labels)} < 100)")
        return None

    up = int(sum(labels))
    down = len(labels) - up
    log(
        f"  Датасет: размечено={len(labels)} (UP={up}, DOWN={down}), "
        f"не хватило истории контекста={skipped_context}"
    )
    log(f"  Баланс классов: UP={up / len(labels) * 100:.1f}%, DOWN={down / len(labels) * 100:.1f}%")

    for i in range(1, len(dates)):
        if dates[i] <= dates[i - 1]:
            raise ValueError(f"{interval}: окна датасета идут не по возрастанию даты")

    n = len(labels)
    gap = min(horizon, max(0, n // 10))
    train_limit = int(n * TRAIN_RATIO)
    val_limit = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_end = max(1, train_limit - gap)
    val_start = train_limit
    val_end = max(val_start, val_limit - gap)
    test_start = val_limit

    if train_end < 50 or val_end - val_start < 20 or n - test_start < 20:
        log("  ПРОПУСК: слишком маленькие train/val/test после разбиения")
        return None

    # Граница скейлера — анкор первого удержанного примера. Тогда в mean/std
    # не попадает ни одна свеча val/test (ни входная, ни label-свеча).
    scaler_cut = max(2, min(anchors[val_start], len(candles) - 1))
    if scaler_cut < window_size + 2:
        log("  ПРОПУСК: слишком короткая история для причинной границы скейлера")
        return None

    log(
        f"  Split по времени (purge={gap}): train={train_end}, "
        f"val={val_end - val_start}, test={n - test_start}"
    )
    log(f"  train: {dates[0]} → {dates[train_end - 1]}")
    log(f"  val:   {dates[val_start]} → {dates[val_end - 1]}")
    log(f"  test:  {dates[test_start]} → {dates[-1]}")
    log(
        f"  Граница скейлера: свеча {scaler_cut} ({candles[scaler_cut]['begin']}) — "
        f"первый удержанный пример, {scaler_cut / len(candles) * 100:.1f}% истории"
    )

    scaler_path = os.path.join(
        OUT_DIR, f"scaler_{safe_name(ticker)}_{safe_name(interval)}.json"
    )
    scaler_reused = False
    scaler_base = None
    if reuse_scaler and os.path.exists(scaler_path):
        reason = ""
        try:
            with open(scaler_path, encoding="utf-8") as f:
                saved = json.load(f)
        except (OSError, ValueError) as exc:
            saved = None
            reason = f"файл не читается: {exc}"
        if saved is not None:
            ok, reason = scaler_is_valid(saved, interval)
            stored_cut = int(saved.get("train_cut") or 0)
            if ok and 2 <= stored_cut <= scaler_cut:
                scaler_base = saved
                scaler_reused = True
            elif ok:
                reason = (
                    f"train_cut={stored_cut} позже причинной границы {scaler_cut}"
                )
        if scaler_base is None:
            log(f"  Скалер base переиспользовать нельзя ({reason}): пересчёт")

    if scaler_base is None:
        scaler_base = compute_scaler(candles[:scaler_cut], oi_fallback=candles)
        train_cut = scaler_cut
    else:
        train_cut = int(scaler_base.get("train_cut") or scaler_cut)

    log(
        f"  Скалер base ({'переиспользован' if scaler_reused else 'пересчитан'}, "
        f"свечи 0..{train_cut - 1}): "
        f"ret {scaler_base['ret_mean']:.6g}/{scaler_base['ret_std']:.6g}, "
        f"value {scaler_base['value_mean']:.6g}/{scaler_base['value_std']:.6g}, "
        f"dow {scaler_base['dow_mean']:.6g}/{scaler_base['dow_std']:.6g}, "
        f"hour {scaler_base['hour_mean']:.6g}/{scaler_base['hour_std']:.6g}"
        + (
            f", yur_net {scaler_base['yur_net_mean']:.6g}/{scaler_base['yur_net_std']:.6g}, "
            f"fiz_net {scaler_base['fiz_net_mean']:.6g}/{scaler_base['fiz_net_std']:.6g}"
            if oi_enabled() else ""
        )
    )

    base_norm = normalize_candles(candles, scaler_base)
    log(f"  Нормализовано base-точек: {len(base_norm)}")

    scalers = {interval: scaler_base}
    ctx_items = {}
    ctx_recomputed = [] if not scaler_reused else []
    for ctx in contexts:
        ctx_candles = ctx_candles_all[ctx]
        ctx_scaler, recomputed = load_context_scaler(
            ticker, ctx, ctx_candles, candles[scaler_cut], interval
        )
        ctx_scaler = dict(ctx_scaler)
        if recomputed:
            ctx_recomputed.append(ctx)
        scalers[ctx] = ctx_scaler
        norm_ctx = normalize_candles(ctx_candles, ctx_scaler)
        ctx_items[ctx] = (ctx_candles, norm_ctx, ctx_begins_map[ctx])
        log(
            f"  Контекст {ctx}: {len(ctx_candles)} свечей, норма {len(norm_ctx)}, "
            f"скейлер {'пересчитан' if recomputed else 'переиспользован'} "
            f"(fitted до {stored_fit_end(ctx_scaler, ctx_candles)})"
        )

    samples = build_windows(interval, candles, base_norm, ctx_items, anchors, window_size)
    if len(samples) != n:
        raise AssertionError(f"окон {len(samples)}, а размечено {n}")

    branches = 1 + len(contexts)
    X = [np.stack([sample[b] for sample in samples]).astype(np.float32)
         for b in range(branches)]
    y = np.asarray(labels, dtype=np.float32)

    tensors = [torch.from_numpy(arr) for arr in X]
    y_tensor = torch.from_numpy(y).unsqueeze(1)

    train_tensors = [t[:train_end] for t in tensors]
    val_tensors = [t[val_start:val_end] for t in tensors]
    test_tensors = [t[test_start:] for t in tensors]
    y_train = y_tensor[:train_end]
    y_val = y_tensor[val_start:val_end]
    y_test = y_tensor[test_start:]

    if balance is None:
        balance = train_end < 8000
    if select == "auto":
        select = "bal" if balance else "loss"
    log(f"  Режим: balance={'да' if balance else 'нет'}, select={select}")

    train_ds = TensorDataset(*train_tensors, y_train)
    if balance:
        y_train_np = y_train.view(-1).cpu().numpy().astype(np.int64)
        counts = np.bincount(y_train_np, minlength=2).astype(np.float64)
        class_weights = 1.0 / np.maximum(counts, 1.0)
        sample_weights = class_weights[y_train_np]
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
        log(
            f"  Балансировка классов: UP={class_weights[1]:.4g}, "
            f"DOWN={class_weights[0]:.4g} (сэмплов на эпоху={len(train_ds)})"
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    if arch == "auto":
        arch = "linear" if (not contexts and len(samples) < 5000) else "lstm"
        log(f"  Архитектура (auto): {arch}")
    if arch == "linear":
        if branches != 1:
            raise ValueError(
                "linear-архитектура поддерживает только интервал без контекстов"
            )
        with torch.no_grad():
            pooled = LinearClassifier.pool(train_tensors[0]).cpu().numpy()
        feature_mean = pooled.mean(axis=0)
        feature_std = pooled.std(axis=0)
        feature_std[feature_std < 1e-6] = 1.0
        model = LinearClassifier(feature_mean, feature_std).to(DEVICE)
        log(f"  Модель: linear ({model.features} pooled-признаков)")
    else:
        model = LSTMClassifier(
            branches=branches, input_size=INPUT_SIZE,
            hidden1=hidden1, hidden2=hidden2, dropout=dropout,
        ).to(DEVICE)
        if USE_DATAPARALLEL:
            model = nn.DataParallel(model)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_metric = float("-inf") if select == "bal" else float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            *xb, yb = batch
            xb = [x.to(DEVICE) for x in xb]
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(*xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb[0].size(0)
        train_loss /= len(train_loader.dataset)

        val_loss, val_acc, val_preds = evaluate_batches(
            model, val_tensors, y_val, criterion
        )

        if select == "bal":
            val_bin = (val_preds > 0.5).view(-1).cpu().numpy().astype(np.int64)
            val_true = y_val.view(-1).cpu().numpy().astype(np.int64)
            tp = int(((val_bin == 1) & (val_true == 1)).sum())
            fn = int(((val_bin == 0) & (val_true == 1)).sum())
            tn = int(((val_bin == 0) & (val_true == 0)).sum())
            fp = int(((val_bin == 1) & (val_true == 0)).sum())
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            tnr = tn / (tn + fp) if (tn + fp) else 0.0
            val_metric = 0.5 * (tpr + tnr)
            improved = val_metric > best_metric
        else:
            val_metric = val_loss
            improved = val_metric < best_metric

        if epoch % 5 == 0 or epoch == 1:
            extra = f" val_bal={val_metric:.4f}" if select == "bal" else ""
            log(
                f"  Эпоха {epoch:3d}/{epochs}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.1f}%{extra}"
            )

        if improved:
            best_metric = val_metric
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(
                    f"  Early stopping на эпохе {epoch} "
                    f"(best {select}={best_metric:.4f}, эпоха {best_epoch})"
                )
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    test_loss, test_acc, test_preds = evaluate_batches(
        model, test_tensors, y_test, criterion
    )
    binary = (test_preds > 0.5).float()
    tp = int(((binary == 1) & (y_test == 1)).sum().item())
    fp = int(((binary == 1) & (y_test == 0)).sum().item())
    tn = int(((binary == 0) & (y_test == 0)).sum().item())
    fn = int(((binary == 0) & (y_test == 1)).sum().item())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced = 0.5 * (recall + specificity)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    log(
        f"  Тест: acc={test_acc * 100:.2f}% precision={precision * 100:.2f}% "
        f"recall={recall * 100:.2f}% spec={specificity * 100:.2f}% "
        f"bal_acc={balanced * 100:.2f}% F1={f1 * 100:.2f}% "
        f"(TP={tp}, FP={fp}, TN={tn}, FN={fn}) loss={test_loss:.4f}"
    )

    # ── Сохранение скейлера ──
    os.makedirs(OUT_DIR, exist_ok=True)
    if scaler_reused:
        log(f"  Скалер оставлен без изменений: {scaler_path}")
    else:
        record = scaler_record(
            ticker, interval, scaler_base, contexts, len(candles), train_cut,
            fit_end=candles[train_cut - 1]["begin"],
        )
        write_json_atomic(scaler_path, record)
        log(f"  Скалер сохранён: {scaler_path}")

    # ── Экспорт в ONNX ──
    export_model = model.module if isinstance(model, nn.DataParallel) else model
    dummy = tuple(
        torch.zeros(1, window_size, INPUT_SIZE, dtype=torch.float32, device=DEVICE)
        for _ in range(branches)
    )
    input_names = ["input_base", "input_h1", "input_h2"][:branches]
    dynamic_axes = {name: {0: "batch", 1: "time"} for name in input_names}
    dynamic_axes["output"] = {0: "batch"}

    onnx_path = os.path.join(
        OUT_DIR, f"model_{safe_name(ticker)}_{safe_name(interval)}.onnx"
    )
    onnx_tmp = f"{onnx_path}.tmp"
    torch.onnx.export(
        export_model,
        dummy,
        onnx_tmp,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    os.replace(onnx_tmp, onnx_path)
    log(f"  Модель сохранена: {onnx_path} ({os.path.getsize(onnx_path)} байт)")

    # ── Пример входа ──
    write_sample(ticker, interval, candles, base_norm, ctx_items, scalers)

    log(f"  Готово за {time.time() - started:.1f} с")
    return {
        "interval": interval,
        "candles": len(candles),
        "samples": len(samples),
        "inputs": branches,
        "contexts": contexts,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "balanced": balanced,
        "f1": f1,
        "threshold": threshold,
        "threshold_effective": effective_threshold(threshold, interval, horizon),
        "horizon": horizon,
        "window": window_size,
        "scaler_cut": train_cut,
        "scaler_end": candles[train_cut - 1]["begin"],
        "scaler_reused": scaler_reused,
        "ctx_scalers_recomputed": ctx_recomputed,
    }


def order_intervals(intervals):
    """
    Старшие интервалы должны обучаться раньше младших: их скейлеры используются
    как контекст. Сортируем по зависимостям CONTEXT_MAP, порядок аргумента —
    только разбор ничьих.
    """
    remaining = list(intervals)
    ordered = []
    while remaining:
        picked = None
        for number, interval in enumerate(remaining):
            deps = [c for c in CONTEXT_MAP.get(interval, []) if c in intervals]
            if all(d not in remaining for d in deps):
                picked = number
                break
        if picked is None:  # цикл зависимостей — доверяем заданный порядок
            ordered.extend(remaining)
            break
        ordered.append(remaining.pop(picked))
    return ordered


def main():
    global WINDOW_SIZE, HORIZON, THRESHOLD, TICKER, INTERVALS
    global FEATURE_NAMES, INPUT_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", default=",".join(INTERVALS))
    parser.add_argument("--ticker", default=os.environ.get("TICKER", TICKER))
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--balance", dest="balance", action="store_true", default=None)
    parser.add_argument("--no-balance", dest="balance", action="store_false")
    parser.add_argument("--select", choices=["auto", "loss", "bal"], default="auto")
    parser.add_argument("--reuse-scaler", action="store_true")
    parser.add_argument("--arch", choices=["auto", "lstm", "linear"], default="auto")
    args = parser.parse_args()

    WINDOW_SIZE = args.window
    HORIZON = max(HORIZON_MIN, min(HORIZON_MAX, args.horizon))
    TICKER = str(args.ticker).strip().upper()

    # OI-признаки включаются только для тикеров с futoi-маппингом и реальными
    # данными в open_interest; остальные тикеры обучаются на базовых 7 признаках.
    use_oi = False
    if TICKER in FUTOI_SECID:
        conn = get_db_conn()
        try:
            use_oi = oi_has_data(conn, TICKER)
        finally:
            conn.close()
    FEATURE_NAMES = (
        BASE_FEATURE_NAMES + OI_FEATURE_NAMES if use_oi else list(BASE_FEATURE_NAMES)
    )
    INPUT_SIZE = len(FEATURE_NAMES)

    INTERVALS = [s.strip() for s in args.intervals.split(",") if s.strip()]
    if not INTERVALS:
        raise SystemExit("Не указаны интервалы для обучения")
    requested_intervals = list(INTERVALS)
    INTERVALS = order_intervals(INTERVALS)
    requested_intervals = list(INTERVALS)
    INTERVALS = order_intervals(INTERVALS)

    THRESHOLD = (
        args.threshold
        if args.threshold is not None
        else float(os.environ.get("THRESHOLD", "0.003"))
    )
    if not math.isfinite(THRESHOLD) or THRESHOLD <= 0:
        raise SystemExit("--threshold must be a positive number (e.g. 0.003)")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    log("=" * 70)
    log("ОБУЧЕНИЕ LSTM: PyTorch → ONNX (scaler v4)")
    log("=" * 70)
    log(f"Python: {'.'.join(map(str, __import__('sys').version_info[:3]))} | "
        f"PyTorch: {torch.__version__} | Device: {DEVICE}")
    if torch.cuda.is_available():
        gpu_names = ", ".join(
            f"{i}:{torch.cuda.get_device_name(i)}" for i in range(GPU_COUNT)
        )
        log(f"CUDA: {torch.version.cuda} | GPU ({GPU_COUNT}): {gpu_names} | DataParallel: {USE_DATAPARALLEL}")
    log(f"Тикер: {TICKER} | Интервалы: {', '.join(INTERVALS)}")
    log(f"Фич: {INPUT_SIZE} ({', '.join(FEATURE_NAMES)})")
    if use_oi:
        log(f"OI: futoi {FUTOI_SECID[TICKER]}, нетто YUR/FIZ; без OI — нейтральный 0")
    elif TICKER in FUTOI_SECID:
        log("OI: данных в open_interest нет — обучение на базовых признаках")
    log(f"Окно: {WINDOW_SIZE} (текущая свеча → история) | горизонт: {HORIZON} | "
        f"базовый порог: {THRESHOLD * 100:.2f}% (прогрессивный по интервалу)")
    log("Контексты: " + ", ".join(
        f"{i} → {', '.join(CONTEXT_MAP.get(i, [])) or 'нет'} "
        f"(эфф. порог {effective_threshold(THRESHOLD, i, HORIZON) * 100:.2f}%)"
        for i in INTERVALS
    ))
    log(
        f"Эпох: {args.epochs} | batch: {BATCH_SIZE} | lr: {args.lr} | "
        f"patience: {args.patience} | hidden: {args.hidden1}/{args.hidden2} | "
        f"dropout: {args.dropout} | arch: {args.arch} | "
        f"balance: {'auto' if args.balance is None else ('да' if args.balance else 'нет')} | "
        f"select: {args.select} | reuse-scaler: {'да' if args.reuse_scaler else 'нет'}"
    )
    log(f"Каталог моделей: {OUT_DIR}")
    if INTERVALS != requested_intervals:
        log(
            f"Порядок интервалов изменён по зависимостям контекста: "
            f"{', '.join(requested_intervals)} → {', '.join(INTERVALS)}"
        )
    log("")

    results = []
    for number, interval in enumerate(INTERVALS, start=1):
        try:
            result = train_interval(
                interval, TICKER, WINDOW_SIZE, HORIZON, THRESHOLD, args.epochs,
                hidden1=args.hidden1, hidden2=args.hidden2, dropout=args.dropout,
                lr=args.lr, weight_decay=args.weight_decay,
                patience=args.patience, balance=args.balance,
                select=args.select, reuse_scaler=args.reuse_scaler,
                arch=args.arch,
            )
            if result:
                results.append(result)
        except Exception as exc:  # noqa: BLE001
            log(f"  Ошибка обучения {interval}: {exc}")
            import traceback
            traceback.print_exc()

    log("")
    log("=" * 70)
    log(f"ГОТОВО: обучено {len(results)} из {len(INTERVALS)}")
    for result in results:
        log(
            f"  {result['interval']}: вход={result['inputs']}×{result['window']}×{INPUT_SIZE}, "
            f"контексты={result['contexts'] or 'нет'}, samples={result['samples']}, "
            f"test_acc={result['test_acc'] * 100:.2f}%, "
            f"bal_acc={result['balanced'] * 100:.2f}%, F1={result['f1'] * 100:.2f}%"
        )
    trained = {result["interval"] for result in results}
    stale = []
    for result in results:
        if result["scaler_reused"]:
            continue
        for other, ctxs in CONTEXT_MAP.items():
            if result["interval"] not in ctxs or other in trained:
                continue
            model_file = os.path.join(
                OUT_DIR, f"model_{safe_name(TICKER)}_{safe_name(other)}.onnx"
            )
            if os.path.exists(model_file):
                stale.append((other, result["interval"]))
    if stale:
        log(
            "  ВНИМАНИЕ: скейлер обновлён, но модели, для которых этот интервал — "
            "контекст, в этом прогоне не переобучены (нормализация контекста в них "
            "осталась со старым скалером):"
        )
        for other, ctx in sorted(set(stale)):
            log(f"    {other} (контекст {ctx}) → переобучите {other}")

    log("=" * 70)


if __name__ == "__main__":
    main()
