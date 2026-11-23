/**
 * server.js — Express-сервер с графической оболочкой для ONNX LSTM прогнозов.
 *
 * Модели scaler v4: 7 признаков на свечу, отдельные входы input_base/h1/h2,
 * окно 300 свечей (текущая → история).
 *
 * Все интервалы включая 1m участвуют в обучении, прогнозах и интерфейсе:
 * они загружаются с MOEX и доступны для работы.
 */

require('dotenv').config();
const express = require('express');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn, execFileSync } = require('child_process');
const { Client, Pool } = require('pg');

const { ensureSchema, listTickers, ensureTicker, removeTicker, validateTicker } = require('./tickers');
const { refreshTicker } = require('./refresh_data');
const inf = require('./inference_v4');
const thresholds = require('./thresholds');

const app = express();
const server = http.createServer(app);

function positiveIntEnv(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} должен быть положительным целым числом, получено: ${raw}`);
  }
  return value;
}

const PORT = positiveIntEnv('PORT', 3000);
const TICKER = validateTicker(process.env.TICKER || 'IMOEX');
const WINDOW_SIZE = inf.WINDOW_SIZE; // 300
const FEATURES = 7;
const HORIZON = 10;
const HORIZON_MIN = 5;
const HORIZON_MAX = 25;
const THRESHOLD_BASE = parseFloat(process.env.THRESHOLD || '0.003');
if (!Number.isFinite(THRESHOLD_BASE) || THRESHOLD_BASE <= 0) {
  throw new Error('THRESHOLD must be a positive number (e.g. 0.003)');
}

// Единая логика прогрессивного порога — thresholds.js.
// 0.3% для 1m, ×2 на каждом следующем интервале; горизонт порог не меняет.
function thresholdFor(interval) {
  return thresholds.safeEffectiveThreshold(THRESHOLD_BASE, interval);
}

const SLEEP_MS = positiveIntEnv('SLEEP_MS', 60000);
const MODEL_DIR = path.join(__dirname, 'models');

// Все интервалы загружаются напрямую с MOEX ISS: 1m, 10m, 60m, 1D.
const INTERVAL_ORDER = [...thresholds.INTERVAL_ORDER];

const FEATURE_DESCRIPTIONS = {
  ret_z: 'log(close/prev_close), z-score по train',
  high_dev: '(high − prev_close) / prev_close',
  low_dev: '(low − prev_close) / prev_close',
  close_dev: '(close − prev_close) / prev_close',
  value_z: 'log1p(value) (оборот), z-score по train',
  dow_z: 'день недели ISODOW (1..7), z-score по train',
  hour_z: 'час свечи (московское время), z-score по train',
  yur_net_z: 'нетто-позиция юрлиц по фьючерсу на актив (futoi), z-score; 0 если OI нет',
  fiz_net_z: 'нетто-позиция физлиц по фьючерсу на актив (futoi), z-score; 0 если OI нет',
};

// ── Состояние ──
const LOG_LIMIT = 1200;
const LOG_FILE = path.join(__dirname, 'server.log');
const LOG_MAX_BYTES = 5 * 1024 * 1024;
const logs = [];
const models = new Map();      // "TICKER|interval" → модель
const predictions = new Map(); // "TICKER|interval" → прогноз
let sseClients = [];
let sseHeartbeat = null;
const SSE_HEARTBEAT_MS = 25000;

const pool = new Pool({
  connectionString: process.env.DB,
  max: 8,
  query_timeout: 30000,
  connectionTimeoutMillis: 15000,
});
// Idle-клиенты пула могут получить FATAL от PostgreSQL (рестарт/админ-команда).
// Без этого обработчика EventEmitter роняет процесс.
pool.on('error', (e) => {
  pushLog('warn', `PostgreSQL pool: ${e.message}`);
});
const intervalCache = new Map(); // ticker → { ts, rows }
const INTERVAL_CACHE_MS = 15 * 1000;

const refreshState = { running: false, current: null, queue: [], last: null };
const trainState = { running: false, code: null, pid: null, ticker: null, progress: null, startedAt: null };

let monitorTimer = null;
let monitorClient = null;
let monitorBusy = false;
let monitorLastTick = null;
let monitorSleep = SLEEP_MS;
const monitorRefreshCooldown = new Map(); // ticker -> last refresh timestamp
const MONITOR_REFRESH_MIN_MS = 5 * 60 * 1000; // не чаще раза в 5 минут

// ── Утилиты ──
function logKey(ticker, interval) {
  return `${ticker}|${interval}`;
}

function intervalRank(interval) {
  const idx = INTERVAL_ORDER.indexOf(interval);
  return idx === -1 ? INTERVAL_ORDER.length : idx;
}

function normalizeTicker(value) {
  return String(value || '').trim().toUpperCase();
}

/**
 * Тикер из query/body. Валидация обязательна: тикер попадает в имена файлов
 * моделей, поэтому «../» и прочие символы недопустимы.
 */
function requestTicker(value) {
  const raw = String(value == null ? '' : value).trim();
  if (!raw) return TICKER;
  try {
    return validateTicker(raw);
  } catch (e) {
    e.status = 400;
    throw e;
  }
}

function pad2(n) {
  return String(n).padStart(2, '0');
}

// Даты в БД — timestamp without time zone (московское время).
// Форматируем локальные компоненты Date, чтобы не сдвигать время.
function formatBegin(value) {
  if (!value) return null;
  const d = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ` +
    `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}

function nowStr() {
  return formatBegin(new Date());
}

function modelPaths(ticker, interval) {
  return {
    model: path.join(MODEL_DIR, `model_${ticker}_${interval}.onnx`),
    scaler: path.join(MODEL_DIR, `scaler_${ticker}_${interval}.json`),
  };
}

const modelIssues = new Map(); // logKey -> { reason, error, signature, recordedAt }
// Записанную ошибку загрузки держим недолго: после этого монитор снова
// пытается загрузить модель, иначе разовый сбой намертво пометил бы её сломанной.
const MODEL_ISSUE_TTL_MS = 60000;

function ctxScalerPath(ticker, ctx) {
  return path.join(MODEL_DIR, `scaler_${ticker}_${ctx}.json`);
}

function modelFiles(ticker, interval) {
  const p = modelPaths(ticker, interval);
  const files = [p.model, p.scaler];
  try {
    const scaler = JSON.parse(fs.readFileSync(p.scaler, 'utf8'));
    for (const ctx of Array.isArray(scaler.contexts) ? scaler.contexts : []) {
      files.push(ctxScalerPath(ticker, ctx));
    }
  } catch (e) {
    /* скейлер не читается — об этом скажет modelStatus */
  }
  return files;
}

// Подпись файлов модели: по ней понимаем, что проблема относится к текущей
// версии файлов, а к переобученной модели старая запись об ошибке не относится.
function modelSignature(ticker, interval) {
  return modelFiles(ticker, interval)
    .map((file) => {
      try {
        const st = fs.statSync(file);
        return `${st.mtimeMs}|${st.size}`;
      } catch (e) {
        return 'missing';
      }
    })
    .join(';');
}

/**
 * Состояние модели с различимой причиной. hasModel() возвращал false и для
 * «модели никогда не обучали», и для «модель есть, но сломана» — монитор и
 * интерфейс молча прятали вторую ситуацию.
 *
 * @returns {{ok: boolean, reason: string, error: string|null, contexts: string[]}}
 */
function modelStatus(ticker, interval) {
  const p = modelPaths(ticker, interval);
  const fail = (reason, error) => ({ ok: false, reason, error, contexts: [] });

  if (!fs.existsSync(p.scaler)) {
    return fail('missing_scaler', `Скейлер не найден: ${path.basename(p.scaler)}`);
  }
  let scaler;
  try {
    scaler = JSON.parse(fs.readFileSync(p.scaler, 'utf8'));
  } catch (e) {
    return fail('broken_scaler', `Скейлер ${path.basename(p.scaler)} не читается: ${e.message}`);
  }
  if (!Number.isInteger(scaler.window) || scaler.window <= 0) {
    return fail('broken_scaler', `Скейлер ${path.basename(p.scaler)}: нет корректного window`);
  }

  const contexts = Array.isArray(scaler.contexts) ? scaler.contexts : [];
  for (const ctx of contexts) {
    if (!thresholds.isKnownInterval(ctx)) {
      return fail(
        'broken_scaler',
        `Скейлер ${path.basename(p.scaler)} ждёт контекст ${ctx}, которого нет в ${INTERVAL_ORDER.join('/')}: нужен перенатренинг`
      );
    }
    if (!fs.existsSync(ctxScalerPath(ticker, ctx))) {
      return fail('missing_context_scaler', `Скейлер контекста не найден: ${path.basename(ctxScalerPath(ticker, ctx))}`);
    }
  }

  if (!fs.existsSync(p.model)) {
    return fail('missing_model', `Модель не обучена: ${path.basename(p.model)}`);
  }

  // Модель и скейлеры на месте: остаётся проверить, не падала ли их загрузка.
  const issue = modelIssues.get(logKey(ticker, interval));
  if (
    issue &&
    issue.signature === modelSignature(ticker, interval) &&
    Date.now() - issue.recordedAt < MODEL_ISSUE_TTL_MS
  ) {
    return { ok: false, reason: issue.reason, error: issue.error, contexts };
  }
  return { ok: true, reason: 'ok', error: null, contexts };
}

function clearModelIssue(ticker, interval) {
  modelIssues.delete(logKey(ticker, interval));
}

/**
 * Запоминаем только проблемы самой модели (файлы/версии/несовместимые входы).
 * Сбои БД и «мало свечей» состоянием модели не являются и не должны
 * превращать /api/intervals в «модель сломана».
 */
function recordModelIssue(ticker, interval, error, isLoadFailure) {
  const status = modelStatus(ticker, interval);
  let reason = status.reason;
  let message = status.error;
  if (status.ok) {
    reason = isLoadFailure ? 'broken_model' : 'predict_failed';
    message = error.message;
  }
  if (reason === 'predict_failed') return { reason, error: message };
  modelIssues.set(logKey(ticker, interval), {
    reason,
    error: message,
    signature: modelSignature(ticker, interval),
    recordedAt: Date.now(),
  });
  return { reason, error: message };
}

function clearTickerModelIssues(ticker) {
  for (const key of [...modelIssues.keys()]) {
    if (key.startsWith(`${ticker}|`)) modelIssues.delete(key);
  }
}

async function getModel(ticker, interval) {
  const key = logKey(ticker, interval);
  const cached = models.get(key);
  if (cached && inf.isModelFresh(cached)) return cached;
  const status = modelStatus(ticker, interval);
  if (!status.ok) {
    models.delete(key);
    const e = new Error(status.error);
    e.modelStatus = status;
    throw e;
  }
  let model;
  try {
    model = await inf.loadModel(ticker, interval);
  } catch (e) {
    // Файлы есть, но ONNX/скейлер несовместимы: это «сломанная модель»,
    // а не «модели нет».
    models.delete(key);
    e.modelLoadFailure = true;
    throw e;
  }
  models.set(key, model);
  clearModelIssue(ticker, interval);
  return model;
}

function readScaler(ticker, interval) {
  const p = modelPaths(ticker, interval);
  try {
    return JSON.parse(fs.readFileSync(p.scaler, 'utf8'));
  } catch (e) {
    return null;
  }
}

/**
 * Порог конкретной модели. Источник истины — scaler_*.json:
 * модель обучалась с этим порогом, поэтому .env не должен его молча менять.
 * Если модели ещё нет — прогрессивное значение от THRESHOLD_BASE.
 */
function modelThresholdMeta(ticker, interval) {
  const scaler = readScaler(ticker, interval);
  if (!scaler) {
    return thresholds.thresholdMeta(THRESHOLD_BASE, interval, { horizon: HORIZON });
  }
  return thresholds.resolveModelThreshold(scaler, interval, THRESHOLD_BASE);
}

function thresholdSummary(ticker) {
  return Object.fromEntries(
    INTERVAL_ORDER.map((interval) => [interval, modelThresholdMeta(ticker, interval)])
  );
}

async function withClient(fn) {
  const client = await pool.connect();
  let released = false;
  try {
    return await fn(client);
  } catch (e) {
    released = true;
    client.release(e); // битый клиент не возвращаем в пул
    throw e;
  } finally {
    if (!released) client.release();
  }
}

// ── Логи и SSE ──
function broadcast(msg) {
  const payload = `data: ${JSON.stringify(msg)}\n\n`;
  for (const res of sseClients) {
    try {
      res.write(payload);
    } catch (e) {
      /* клиент отвалился */
    }
  }
}

let logWrites = 0;
function rotateLogIfNeeded() {
  try {
    if (fs.statSync(LOG_FILE).size <= LOG_MAX_BYTES) return;
    const backup = `${LOG_FILE}.1`;
    if (fs.existsSync(backup)) fs.unlinkSync(backup);
    fs.renameSync(LOG_FILE, backup);
  } catch (e) {
    /* файла ещё нет или он занят */
  }
}

function pushLog(level, msg) {
  const entry = { ts: new Date().toISOString(), level, msg: String(msg) };
  logs.push(entry);
  if (logs.length > LOG_LIMIT) logs.shift();
  broadcast({ type: 'log', line: entry });
  try {
    if (logWrites++ % 50 === 0) rotateLogIfNeeded();
    fs.appendFileSync(LOG_FILE, `[${nowStr()}] ${entry.msg}\n`, 'utf8');
  } catch (e) {
    /* лог-файл не критичен */
  }
}

function invalidateIntervalCache(ticker) {
  if (ticker) intervalCache.delete(ticker);
  else intervalCache.clear();
}

// ── Интервалы и прогнозы ──
async function listIntervals(client, ticker) {
  let rows;
  const cached = intervalCache.get(ticker);
  if (cached && Date.now() - cached.ts < INTERVAL_CACHE_MS) {
    rows = cached.rows;
  } else {
    const r = await client.query(
      `SELECT interval, COUNT(*)::int AS cnt, MIN(begin) AS first, MAX(begin) AS last
       FROM candles
       WHERE ticker = $1
       GROUP BY interval`,
      [ticker]
    );
    rows = r.rows.filter((row) => INTERVAL_ORDER.includes(row.interval));
    intervalCache.set(ticker, { ts: Date.now(), rows });
  }

  return [...rows]
    .sort((a, b) => intervalRank(a.interval) - intervalRank(b.interval))
    .map((row) => {
      const status = modelStatus(ticker, row.interval);
      return {
        interval: row.interval,
        candles: Number(row.cnt),
        first: formatBegin(row.first),
        last: formatBegin(row.last),
        hasModel: status.ok,
        // hasModel — просто «можно ли предсказывать»; modelStatus объясняет,
        // почему нельзя: нет файлов или файлы есть, но они несовместимы.
        modelStatus: { ok: status.ok, reason: status.reason, error: status.error },
        prediction: predictions.get(logKey(ticker, row.interval)) || null,
      };
    });
}

async function predictOne(client, ticker, interval) {
  const requested = String(interval || '').trim();
  try {
    thresholds.validateInterval(requested);
    const model = await getModel(ticker, requested);
    const r = await inf.predictWithModel(client, model, ticker, requested, {
      fallbackBaseThreshold: THRESHOLD_BASE,
    });

    // inference_v4 уже разрешил порог по scaler: не пересчитываем его здесь,
    // иначе .env мог бы изменить цель обученной модели.
    const result = {
      interval: requested,
      ticker,
      pUp: r.pUp,
      pDown: r.pDown,
      direction: r.direction,
      confidence: r.confidence,
      lastClose: r.lastClose,
      lastBegin: formatBegin(r.lastBegin),
      threshold: r.threshold,
      thresholdBase: r.thresholdBase,
      thresholdHorizon: r.thresholdHorizon,
      thresholdMultiplier: r.thresholdMultiplier,
      thresholdProgressive: r.thresholdProgressive,
      thresholdSource: r.thresholdSource,
      upTarget: r.lastClose * (1 + r.threshold),
      downTarget: r.lastClose * (1 - r.threshold),
      inputs: r.steps,
      contexts: r.contexts,
      window: r.window || WINDOW_SIZE,
      features: Array.isArray(r.inputShape) ? r.inputShape[2] : FEATURES,
      updatedAt: new Date().toISOString(),
    };
    predictions.set(logKey(ticker, requested), result);
    return result;
  } catch (e) {
    // Ошибку в кэш прогнозов не кладём. Раньше строка {error} оставалась в
    // predictions и возвращалась из /api/intervals как «прогноз», поэтому
    // временный сбой вытеснял валидный прогноз и путался с настоящим
    // состоянием модели.
    const key = logKey(ticker, requested);
    predictions.delete(key);

    // Даже для неизвестного интервала отвечаем строкой ошибки, а не 500.
    // Если модель существует, показываем её реальный порог из скейлера.
    const known = thresholds.isSupportedInterval(requested);
    const meta = known
      ? modelThresholdMeta(ticker, requested)
      : thresholds.thresholdMeta(THRESHOLD_BASE, requested, { horizon: HORIZON });
    let issue;
    if (!known) {
      issue = { reason: 'unsupported_interval', error: e.message };
    } else if (e.modelStatus) {
      issue = { reason: e.modelStatus.reason, error: e.modelStatus.error };
    } else {
      issue = recordModelIssue(ticker, requested, e, !!e.modelLoadFailure);
    }
    const result = {
      interval: requested,
      ticker,
      error: issue.error || e.message,
      errorReason: issue.reason,
      threshold: meta.effective,
      thresholdBase: meta.base,
      thresholdHorizon: meta.horizon,
      thresholdMultiplier: meta.multiplier,
      thresholdProgressive: meta.progressive,
      thresholdSource: meta.source,
      window: WINDOW_SIZE,
      features: FEATURES,
      updatedAt: new Date().toISOString(),
    };
    return result;
  }
}

// ── Пример входа модели ──
async function buildSample(client, ticker, interval) {
  const safeInterval = thresholds.validateInterval(interval);
  const status = modelStatus(ticker, safeInterval);
  if (!status.ok) {
    // Состояние модели, а не сбой сервера: отвечаем 409 с причиной,
    // чтобы интерфейс показывал «модель сломана», а не «500 Internal Server Error».
    const err = new Error(`${ticker} ${safeInterval}: ${status.error || 'модель недоступна'}`);
    err.status = 409;
    err.errorReason = status.reason;
    throw err;
  }
  const model = await getModel(ticker, safeInterval);
  const { scaler, contexts, contextScalers } = model;
  const windowSize = Number.isInteger(model.window) && model.window > 0 ? model.window : WINDOW_SIZE;
  const historyLimit = inf.historyLimitFor(windowSize);

  const baseCandles = await inf.loadCandles(client, ticker, safeInterval, historyLimit);
  if (baseCandles.length < windowSize + 1) {
    throw new Error(`Недостаточно свечей ${safeInterval}: ${baseCandles.length}, нужно ${windowSize + 1}`);
  }

  const needsOi = [scaler, ...contextScalers].some(
    (s) => Number.isFinite(Number(s.yur_net_mean)) && Number.isFinite(Number(s.fiz_net_mean))
  );
  const oi = needsOi ? await inf.loadOiSeries(client, ticker) : null;
  if (needsOi) inf.attachOi(baseCandles, safeInterval, oi);

  const baseNormalized = inf.normalizeCandles(baseCandles, scaler);
  const blocks = { base: baseNormalized.slice(-windowSize).reverse() };

  for (let i = 0; i < contexts.length; i++) {
    const ctx = contexts[i];
    const ctxCandles = await inf.loadCandles(client, ticker, ctx, historyLimit);
    if (ctxCandles.length < windowSize + 1) {
      throw new Error(`Недостаточно свечей контекста ${ctx}: ${ctxCandles.length}`);
    }
    if (needsOi) inf.attachOi(ctxCandles, ctx, oi);
    const ctxNormalized = inf.normalizeCandles(ctxCandles, contextScalers[i]);
    const win = inf.contextWindow(baseCandles, safeInterval, ctxCandles, ctxNormalized, ctx, windowSize);
    if (!win) throw new Error(`Недостаточно истории контекста ${ctx} для ${safeInterval}`);
    blocks[i === 0 ? 'h1' : 'h2'] = win;
  }

  const shape = {};
  const rows = {};
  for (const [key, block] of Object.entries(blocks)) {
    shape[key] = [block.length, block[0].length];
    rows[key] = block.slice(0, 3);
  }

  const featureNames = Array.isArray(scaler.feature_names) ? scaler.feature_names : Object.keys(FEATURE_DESCRIPTIONS);

  const meta = thresholds.resolveModelThreshold(scaler, safeInterval, THRESHOLD_BASE);
  return {
    ticker,
    interval: safeInterval,
    window: windowSize,
    horizon: meta.horizon,
    threshold: meta.effective,
    thresholdBase: meta.base,
    thresholdMultiplier: meta.multiplier,
    thresholdProgressive: meta.progressive,
    thresholdSource: meta.source,
    order: scaler.order || 'current_to_history',
    shape,
    contexts,
    featureNames,
    featureDescriptions: Object.fromEntries(
      featureNames.map((name) => [name, FEATURE_DESCRIPTIONS[name] || ''])
    ),
    rows,
    blocks,
    download: `/api/sample/download?ticker=${encodeURIComponent(ticker)}&interval=${encodeURIComponent(safeInterval)}`,
  };
}

// ── Обновление свечей ──
function refreshStatus() {
  return {
    running: refreshState.running,
    current: refreshState.current,
    queue: [...refreshState.queue],
    last: refreshState.last,
  };
}

function broadcastRefresh() {
  broadcast({ type: 'refresh_status', data: refreshStatus() });
}

function enqueueRefresh(ticker) {
  const t = normalizeTicker(ticker);
  if (!t) return false;
  if (refreshState.running) {
    const current = refreshState.current && refreshState.current.currentTicker;
    // Не ставим в очередь тикер, который уже качается.
    if (t !== current && !refreshState.queue.includes(t)) refreshState.queue.push(t);
    broadcastRefresh();
    return true;
  }
  runRefresh(t).catch((e) => pushLog('error', `Обновление ${t}: ${e.message}`));
  return false;
}

let refreshChild = null;
function spawnStep(command, args, onLine) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { cwd: __dirname });
    refreshChild = child;
    const handle = (chunk) => {
      const text = chunk.toString();
      for (const line of text.split(/\r?\n/)) {
        const clean = line.trim();
        if (clean) onLine(clean);
      }
    };
    child.stdout.on('data', handle);
    child.stderr.on('data', handle);
    child.on('error', (e) => {
      onLine(`Ошибка запуска: ${e.message}`);
      if (refreshChild === child) refreshChild = null;
      resolve(1);
    });
    child.on('close', (code) => {
      if (refreshChild === child) refreshChild = null;
      resolve(code == null ? 1 : code);
    });
  });
}

async function runRefresh(ticker) {
  refreshState.running = true;
  refreshState.current = { step: 'Загрузка свечей', currentTicker: ticker };
  broadcastRefresh();
  pushLog('info', `Обновление свечей ${ticker}: инкрементальная загрузка`);

  try {
    const code = await spawnStep(
      process.execPath,
      ['refresh_data.js', '--ticker', ticker, '--incremental'],
      (line) => {
        const page = line.match(/страниц=\d+, свечей=(\d+)/);
        if (page) {
          refreshState.current = {
            step: `Загрузка свечей: ${Number(page[1]).toLocaleString('ru-RU')}`,
            currentTicker: ticker,
          };
          broadcastRefresh();
        }
        if (/готово/.test(line) || /Ошибка|ошибка/.test(line)) pushLog(/Ошибка|ошибка/.test(line) ? 'error' : 'info', line);
      }
    );

    refreshState.last = { ticker, code, finishedAt: new Date().toISOString() };
    broadcastsRefreshDone(ticker, code);
  } catch (e) {
    refreshState.last = { ticker, code: 1, error: e.message, finishedAt: new Date().toISOString() };
    pushLog('error', `Обновление ${ticker}: ${e.message}`);
    broadcast({ type: 'refresh_done', tickers: [ticker], code: 1 });
  } finally {
    // Сброс состояния обязателен при любом исходе, иначе очередь залипнет.
    invalidateIntervalCache(ticker);
    refreshState.running = false;
    refreshState.current = null;

    // При остановке сервера не запускаем следующие тикеры из очереди.
    if (shuttingDown) {
      broadcastRefresh();
      return;
    }

    const next = refreshState.queue.shift();
    if (next) {
      broadcastRefresh();
      runRefresh(next).catch((e) => pushLog('error', `Обновление ${next}: ${e.message}`));
    } else {
      broadcastRefresh();
    }
  }
}

function broadcastsRefreshDone(ticker, code) {
  pushLog(code === 0 ? 'info' : 'error', `Загрузка свечей ${ticker} завершена (code=${code})`);
  broadcast({ type: 'refresh_done', tickers: [ticker], code });
}

// ── Python для обучения ──
let cachedPython;
let cachedPythonAt = 0;
const PYTHON_MISS_CACHE_MS = 60 * 1000;
function findPython() {
  if (cachedPython !== undefined) {
    if (cachedPython !== null) return cachedPython;
    // Не кешируем отсутствие навсегда: пользователь мог установить Python.
    if (Date.now() - cachedPythonAt < PYTHON_MISS_CACHE_MS) return null;
  }

  const candidates = [
    process.env.PYTHON,
    'C:\\ProgramData\\miniconda3\\python.exe',
    'C:\\Users\\memor\\anaconda3\\python.exe',
    'C:\\Users\\memor\\miniconda3\\python.exe',
    'C:\\Python311\\python.exe',
    'C:\\Python310\\python.exe',
    'C:\\Python312\\python.exe',
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      cachedPython = candidate;
      return candidate;
    }
  }

  try {
    const cmd = process.platform === 'win32' ? 'where' : 'which';
    const out = execFileSync(cmd, ['python'], { encoding: 'utf8', timeout: 3000 });
    const real = out
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean)
      .find((p) => !p.toLowerCase().includes('windowsapps') && fs.existsSync(p));
    if (real) {
      cachedPython = real;
      return real;
    }
  } catch (e) {
    /* python не найден */
  }

  cachedPython = null;
  cachedPythonAt = Date.now();
  return null;
}

function trainStatus() {
  return {
    running: trainState.running,
    code: trainState.code,
    pid: trainState.pid,
    ticker: trainState.ticker,
    progress: trainState.progress,
    startedAt: trainState.startedAt,
  };
}

// ── Express ──
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

const wrap = (fn) => (req, res) =>
  Promise.resolve(fn(req, res)).catch((e) => {
    const status = Number.isInteger(e.status) && e.status >= 400 && e.status < 600 ? e.status : 500;
    pushLog(status >= 500 ? 'error' : 'warn', e.message);
    if (!res.headersSent) {
      res.status(status).json({ error: e.message, ...(e.errorReason ? { errorReason: e.errorReason } : {}) });
    }
  });

// Тикеры
app.get('/api/tickers', wrap(async (req, res) => {
  const data = await withClient((client) => listTickers(client));
  res.json(data);
}));

app.post('/api/tickers', wrap(async (req, res) => {
  const rawTicker = String((req.body && req.body.ticker) || '').trim();
  if (!rawTicker) return res.status(400).json({ error: 'Укажите тикер' });
  const ticker = requestTicker(rawTicker);

  const info = await withClient(async (client) => {
    await ensureSchema(client);
    return ensureTicker(client, ticker);
  });

  pushLog('info', `Добавлен тикер ${info.ticker} — «${info.title || ''}» (${info.engine}/${info.market})`);
  invalidateIntervalCache(info.ticker);
  broadcast({ type: 'tickers_changed' });
  enqueueRefresh(info.ticker);
  res.json(info);
}));

app.delete('/api/tickers/:ticker', wrap(async (req, res) => {
  const ticker = requestTicker(req.params.ticker);
  const deleted = await withClient(async (client) => {
    const removed = await removeTicker(client, ticker);
    if (!removed) return false;
    await client.query(`DELETE FROM candles WHERE ticker = $1`, [ticker]);
    await client.query(`DELETE FROM open_interest WHERE ticker = $1`, [ticker]);
    return true;
  });

  if (!deleted) return res.status(404).json({ error: `Тикер ${ticker} не найден` });

  for (const [key] of models) {
    if (key.startsWith(`${ticker}|`)) models.delete(key);
  }
  for (const [key] of predictions) {
    if (key.startsWith(`${ticker}|`)) predictions.delete(key);
  }
  clearTickerModelIssues(ticker);
  try {
    for (const name of fs.readdirSync(MODEL_DIR)) {
      // Точное совпадение по тикеру: model_SBER_* не должен задеть model_SBERP_*.
      if (name.startsWith(`model_${ticker}_`) || name.startsWith(`scaler_${ticker}_`)) {
        fs.unlinkSync(path.join(MODEL_DIR, name));
      }
    }
  } catch (e) {
    /* файлы моделей могли не существовать */
  }

  pushLog('warn', `Тикер ${ticker} удалён вместе со свечами и моделями`);
  invalidateIntervalCache(ticker);
  broadcast({ type: 'tickers_changed' });
  res.json({ deleted: true, ticker });
}));

// Конфигурация
app.get('/api/config', wrap(async (req, res) => {
  const ticker = requestTicker(req.query.ticker);
  const details = thresholdSummary(ticker);
  const effective = Object.fromEntries(
    Object.entries(details).map(([interval, meta]) => [interval, meta.effective])
  );
  res.json({
    ticker,
    windowSize: WINDOW_SIZE,
    features: FEATURES,
    horizon: HORIZON,
    horizonMin: HORIZON_MIN,
    horizonMax: HORIZON_MAX,
    threshold: effective['1m'],
    baseThreshold: THRESHOLD_BASE,
    thresholds: effective,
    thresholdDetails: details,
    thresholdMultipliers: thresholds.multipliersForIntervals(),
    thresholdProgressive: true,
    thresholdStepMultiplier: 2,
    intervals: INTERVAL_ORDER,
  });
}));

// Интервалы
app.get('/api/intervals', wrap(async (req, res) => {
  const ticker = requestTicker(req.query.ticker);
  const data = await withClient((client) => listIntervals(client, ticker));
  res.json(data);
}));

// Прогнозы
app.get('/api/predict', wrap(async (req, res) => {
  const ticker = requestTicker(req.query.ticker);
  const requested = [...new Set(String(req.query.intervals || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean))];

  const result = await withClient(async (client) => {
    const available = await listIntervals(client, ticker);
    const intervals = (requested.length ? requested : available.map((i) => i.interval));

    const out = [];
    for (const interval of intervals) {
      out.push(await predictOne(client, ticker, interval));
    }
    return out;
  });

  res.json(result);
}));

app.get('/api/predict/:interval', wrap(async (req, res) => {
  const ticker = requestTicker(req.query.ticker);
  const result = await withClient((client) => predictOne(client, ticker, req.params.interval));
  res.json(result);
}));

// Логи
app.get('/api/logs', wrap(async (req, res) => {
  const raw = parseInt(req.query.limit, 10);
  const limit = Number.isFinite(raw) ? Math.max(1, Math.min(LOG_LIMIT, raw)) : 500;
  res.json({ logs: logs.slice(-limit) });
}));

// Пример входа
app.get('/api/sample', wrap(async (req, res) => {
  const ticker = requestTicker(req.query.ticker);
  const interval = String(req.query.interval || '').trim();
  if (!interval) return res.status(400).json({ error: 'Укажите interval' });
  try {
    thresholds.validateInterval(interval);
  } catch (e) {
    return res.status(400).json({ error: e.message });
  }

  const sample = await withClient((client) => buildSample(client, ticker, interval));
  const { blocks, ...payload } = sample;
  res.json(payload);
}));

app.get('/api/sample/download', wrap(async (req, res) => {
  const ticker = requestTicker(req.query.ticker);
  const interval = String(req.query.interval || '').trim();
  if (!interval) return res.status(400).json({ error: 'Укажите interval' });
  try {
    thresholds.validateInterval(interval);
  } catch (e) {
    return res.status(400).json({ error: e.message });
  }

  const sample = await withClient((client) => buildSample(client, ticker, interval));
  const payload = {
    interval,
    ticker,
    shape: sample.shape,
    contexts: sample.contexts,
    window: sample.window,
    horizon: sample.horizon,
    threshold: sample.threshold,
    order: sample.order,
    note: 'Строка 0 — текущая свеча; даты и OHLC в выборку не входят. ' +
      'h1/h2 — старшие интервалы, выровненные по закрытию base-свечи. ' +
      'LSTM разворачивает каждое окно в хронологический порядок внутри forward().',
    feature_names: sample.featureNames,
    scalers: Object.fromEntries(
      [interval, ...sample.contexts].map((name) => {
        const file = path.join(MODEL_DIR, `scaler_${ticker}_${name}.json`);
        try {
          return [name, JSON.parse(fs.readFileSync(file, 'utf8'))];
        } catch (e) {
          const err = new Error(`Не удалось прочитать скейлер: ${path.basename(file)}`);
          err.status = 409;
          err.errorReason = name === interval ? 'missing_scaler' : 'missing_context_scaler';
          throw err;
        }
      })
    ),
    ...sample.blocks,
  };

  res.setHeader('Content-Disposition', `attachment; filename="sample_input_${ticker}_${interval}.json"`);
  res.json(payload);
}));

// Обновление свечей
app.post('/api/refresh', wrap(async (req, res) => {
  const ticker = requestTicker(req.body && req.body.ticker);
  const queued = refreshState.running;
  enqueueRefresh(ticker);
  res.json({ queued, mode: 'incremental', ticker });
}));

app.get('/api/refresh/status', wrap(async (req, res) => {
  res.json(refreshStatus());
}));

// Обучение
app.post('/api/train', wrap(async (req, res) => {
  if (trainState.running) {
    return res.status(409).json({ started: false, error: 'Обучение уже выполняется' });
  }

  const ticker = requestTicker((req.body && req.body.ticker) || TICKER);
  const intervals = [...new Set(String((req.body && req.body.intervals) || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean))];

  if (!intervals.length) {
    return res.status(400).json({
      started: false,
      error: 'Выберите хотя бы один интервал',
    });
  }

  const invalidIntervals = intervals.filter((interval) => !thresholds.isSupportedInterval(interval));
  if (invalidIntervals.length) {
    return res.status(400).json({
      started: false,
      error: `Неизвестные интервалы: ${invalidIntervals.join(', ')}. Допустимы: ${INTERVAL_ORDER.join(', ')}`,
    });
  }

  let horizon = parseInt((req.body && req.body.horizon) || HORIZON, 10);
  if (!Number.isInteger(horizon)) horizon = HORIZON;
  horizon = Math.min(HORIZON_MAX, Math.max(HORIZON_MIN, horizon));

  const python = findPython();
  if (!python) {
    return res.status(400).json({
      started: false,
      error: 'Python с PyTorch не найден. Укажите путь в переменной PYTHON или установите Python/CUDA-сборку PyTorch.',
    });
  }

  const args = [
    'train_gpu.py',
    '--ticker', ticker,
    '--intervals', intervals.join(','),
    '--horizon', String(horizon),
    '--threshold', String(THRESHOLD_BASE),
  ];

  const child = spawn(python, args, { cwd: __dirname });

  trainState.running = true;
  trainState.code = null;
  trainState.pid = child.pid;
  trainState.ticker = ticker;
  trainState.progress = null;
  trainState.startedAt = new Date().toISOString();

  pushLog('train', `Обучение ${ticker}: ${intervals.join(', ')} | горизонт ${horizon} свечей | пороги ${intervals.map((i) => `${i}=${(thresholdFor(i) * 100).toFixed(2)}%`).join(', ')}`);

  const handle = (chunk) => {
    for (const rawLine of chunk.toString().split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line) continue;
      const m = line.match(/(?:Эпоха|epoch)\s+(\d+)\s*\/\s*(\d+)/i);
      if (m) trainState.progress = { epoch: Number(m[1]), epochs: Number(m[2]) };
      pushLog('train', line);
    }
  };
  child.stdout.on('data', handle);
  child.stderr.on('data', handle);

  let finished = false;
  const finish = (code, error) => {
    if (finished) return;
    finished = true;
    trainState.running = false;
    trainState.code = code == null ? 1 : code;
    trainState.progress = null;

    for (const interval of intervals) {
      models.delete(logKey(ticker, interval));
    }
    // Новая модель — старые записи «модель сломана» больше не действуют.
    clearTickerModelIssues(ticker);
    // Чистим прогнозы только обученного тикера, остальные не трогаем.
    for (const key of [...predictions.keys()]) {
      if (key.startsWith(`${ticker}|`)) predictions.delete(key);
    }
    invalidateIntervalCache(ticker);

    if (error) pushLog('error', `Обучение ${ticker}: ${error}`);
    pushLog(code === 0 ? 'info' : 'error', `Обучение ${ticker} завершено (code=${code})`);
    broadcast({ type: 'train_done', ticker, code });
    broadcast({ type: 'tickers_changed' });
  };

  child.on('close', (code) => finish(code));
  child.on('error', (e) => finish(1, e.message));

  res.json({
    started: true, pid: child.pid, ticker, intervals, horizon,
    thresholds: Object.fromEntries(intervals.map((interval) => [interval, thresholdFor(interval)])),
  });
}));

app.get('/api/train/status', wrap(async (req, res) => {
  res.json(trainStatus());
}));

// Монитор
function monitorStatus() {
  return {
    enabled: !!monitorTimer,
    running: !!monitorTimer,
    busy: monitorBusy,
    lastTick: monitorLastTick,
    sleepMs: monitorSleep,
    threshold: thresholdFor('1m'),
    thresholdEffective: thresholdFor('1m'),
    baseThreshold: THRESHOLD_BASE,
    thresholds: Object.fromEntries(INTERVAL_ORDER.map((interval) => [
      interval, thresholdFor(interval),
    ])),
    thresholdMultipliers: thresholds.multipliersForIntervals(),
    tickers: [...new Set([...predictions.keys()].map((k) => k.split('|')[0]))],
  };
}

const monitorIssueLog = new Map(); // logKey -> { signature, ts }
const MONITOR_ISSUE_REPEAT_MS = 10 * 60 * 1000;

/**
 * Монитор раньше молча делал continue для интервала без модели, поэтому
 * сломанная модель (например, SBER 10m со скейлером контекста 15m) не
 * появлялась в логе никогда. Пишем причину, но не чаще раза в 10 минут.
 */
function warnModelIssue(ticker, interval, status) {
  if (!status || status.ok) return;
  const key = logKey(ticker, interval);
  const signature = `${status.reason}|${status.error}`;
  const prev = monitorIssueLog.get(key);
  if (prev && prev.signature === signature && Date.now() - prev.ts < MONITOR_ISSUE_REPEAT_MS) return;
  monitorIssueLog.set(key, { signature, ts: Date.now() });
  pushLog('warn', `${ticker} ${interval}: прогноз недоступен (${status.reason}): ${status.error}`);
}

async function monitorTick() {
  if (monitorBusy) return;
  monitorBusy = true;
  monitorLastTick = Date.now();
  try {
    if (!monitorClient) {
      const client = new Client({
        connectionString: process.env.DB,
        query_timeout: 30000,
        connectionTimeoutMillis: 15000,
      });
      // При обрыве соединения PostgreSQL клиент шлёт 'error'; без слушателя
      // процесс падает. Следующий тик переподключится.
      client.on('error', (e) => {
        pushLog('warn', `Монитор: PostgreSQL: ${e.message}`);
        if (monitorClient === client) {
          monitorClient = null;
          client.end().catch(() => {});
        }
      });
      monitorClient = client;
      await client.connect();
    }

    const tickers = await listTickers(monitorClient);
    const now = Date.now();
    for (const item of tickers) {
      const lastRefresh = monitorRefreshCooldown.get(item.ticker) || 0;
      const manualBusy = refreshState.running &&
        (refreshState.current?.currentTicker === item.ticker ||
          refreshState.queue.includes(item.ticker));
      if (!manualBusy && now - lastRefresh >= MONITOR_REFRESH_MIN_MS) {
        // Ставим cooldown до запуска: тик может длиться дольше интервала таймера.
        monitorRefreshCooldown.set(item.ticker, now);
        try {
          await refreshTicker(monitorClient, item, {
            incremental: true,
            // В дашборд попадают только проблемы, иначе лог шумит каждые 5 минут.
            logger: (msg, level) => {
              if (level === 'error' || level === 'warn') {
                pushLog(level, `Монитор ${item.ticker}: ${msg}`);
              }
            },
          });
          invalidateIntervalCache(item.ticker);
        } catch (e) {
          pushLog('warn', `Монитор: не удалось обновить свечи ${item.ticker}: ${e.message}`);
        }
      }
      const intervals = await listIntervals(monitorClient, item.ticker);
      for (const info of intervals) {
        if (!info.hasModel) {
          warnModelIssue(item.ticker, info.interval, info.modelStatus);
          continue;
        }
        const prediction = await predictOne(monitorClient, item.ticker, info.interval);
        if (prediction.error) {
          warnModelIssue(item.ticker, info.interval, {
            ok: false,
            reason: prediction.errorReason || 'predict_failed',
            error: prediction.error,
          });
        } else {
          broadcast({
            type: 'prediction',
            ticker: item.ticker,
            interval: info.interval,
            data: prediction,
          });
        }
      }
    }
  } catch (e) {
    pushLog('error', `Монитор: ${e.message}`);
    if (monitorClient) {
      await monitorClient.end().catch(() => {});
      monitorClient = null;
    }
  } finally {
    monitorBusy = false;
  }
}

function startMonitor() {
  if (monitorTimer) return;
  monitorTimer = setInterval(monitorTick, monitorSleep);
  pushLog('info', `Монитор запущен: интервал ${monitorSleep} мс (${INTERVAL_ORDER.join(', ')})`);
  monitorTick();
}

function stopMonitor() {
  if (monitorTimer) clearInterval(monitorTimer);
  monitorTimer = null;
  if (monitorClient) {
    monitorClient.end().catch(() => {});
    monitorClient = null;
  }
  pushLog('info', 'Монитор остановлен');
}

app.post('/api/monitor/start', wrap(async (req, res) => {
  startMonitor();
  res.json({ started: true, sleepMs: monitorSleep, threshold: THRESHOLD_BASE });
}));

app.post('/api/monitor/stop', wrap(async (req, res) => {
  stopMonitor();
  res.json({ stopped: true });
}));

app.get('/api/monitor/status', wrap(async (req, res) => {
  res.json(monitorStatus());
}));

// SSE
app.get('/stream', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  sseClients.push(res);
  res.write(`data: ${JSON.stringify({ type: 'hello', ts: Date.now() })}\n\n`);

  if (!sseHeartbeat) {
    // Комментарий-пинг не даёт прокси и браузеру закрыть простаивающий SSE.
    sseHeartbeat = setInterval(() => {
      for (const client of sseClients) {
        try {
          client.write(': ping\n\n');
        } catch (e) {
          /* клиент отвалился */
        }
      }
    }, SSE_HEARTBEAT_MS);
    sseHeartbeat.unref?.();
  }

  const remove = () => {
    sseClients = sseClients.filter((client) => client !== res);
  };
  req.on('close', remove);
  res.on('close', remove);
  res.on('error', remove);
});

// ── Запуск ──
async function main() {
  await withClient(async (client) => {
    await ensureSchema(client);
    const tickers = await listTickers(client);
    if (!tickers.length) {
      pushLog('warn', `Справочник тикеров пуст — добавьте тикер через интерфейс`);
    }
  });

  server.on('error', (e) => {
    console.error(`❌ HTTP-сервер: ${e.message}`);
    pushLog('error', `HTTP-сервер: ${e.message}`);
    // Без выхода процесс останется висеть без слушающего порта.
    if (e.code === 'EADDRINUSE' || e.code === 'EACCES') process.exit(1);
  });

  server.listen(PORT, () => {
    console.log(`🖥️  Dashboard: http://localhost:${PORT}`);
    console.log(`   Ticker: ${TICKER} | Window: ${WINDOW_SIZE}×${FEATURES} | Threshold base: ${THRESHOLD_BASE}`);
    pushLog('info', `Сервер запущен: http://localhost:${PORT} | окно ${WINDOW_SIZE}×${FEATURES} | базовый порог ${THRESHOLD_BASE}`);
    startMonitor();
  });
}

let shuttingDown = false;
function shutdown(signal, exitCode = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  pushLog('warn', `Остановка сервера (${signal})`);
  if (refreshChild) {
    try {
      refreshChild.kill();
    } catch (e) {
      /* уже завершился */
    }
    refreshChild = null;
  }
  if (sseHeartbeat) clearInterval(sseHeartbeat);
  sseHeartbeat = null;
  for (const res of sseClients) {
    try {
      res.end();
    } catch (e) {
      /* уже закрыт */
    }
  }
  sseClients = [];
  stopMonitor();
  pool.end().catch(() => {});
  server.close(() => process.exit(exitCode));
  // Страховка: не даём процессу висеть на keep-alive соединениях.
  setTimeout(() => process.exit(exitCode), 3000).unref();
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

// Перезагрузка сервера: сначала освобождаем порт, затем порождаем новый процесс.
app.post('/api/server/restart', wrap(async (req, res) => {
  res.json({ restarting: true });
  pushLog('warn', 'Перезагрузка сервера инициирована через веб-интерфейс');

  const spawnReplacement = () => {
    const child = spawn(process.execPath, [path.join(__dirname, 'server.js')], {
      cwd: __dirname,
      detached: true,
      stdio: 'ignore',
    });
    child.on('error', (e) => pushLog('error', `Перезапуск сервера: ${e.message}`));
    child.unref();
  };

  setTimeout(() => {
    if (sseHeartbeat) clearInterval(sseHeartbeat);
    sseHeartbeat = null;
    for (const client of sseClients) {
      try {
        client.end();
      } catch (e) {
        /* уже закрыт */
      }
    }
    sseClients = [];
    stopMonitor();
    if (refreshChild) {
      try {
        refreshChild.kill();
      } catch (e) {
        /* уже завершился */
      }
      refreshChild = null;
    }

    let started = false;
    const startOnce = () => {
      if (started) return;
      started = true;
      spawnReplacement();
      process.exit(0);
    };

    server.close(startOnce);
    server.closeIdleConnections?.();
    setTimeout(startOnce, 2000).unref();
  }, 300);
}));

main().catch((e) => {
  console.error('❌ Ошибка запуска сервера:', e);
  process.exit(1);
});
