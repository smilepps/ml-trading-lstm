/**
 * inference_v4.js — инференс моделей scaler v4.
 *
 * Архитектура v4:
 *   • 7 признаков на свечу: ret_z, high_dev, low_dev, close_dev, value_z, dow_z, hour_z
 *     (+ yur_net_z, fiz_net_z для моделей с открытыми позициями futoi)
 *   • отдельные входы ONNX: input_base, input_h1, input_h2
 *   • окно 300 свечей на каждый вход, строка 0 — текущая свеча (current → history)
 *   • старшие интервалы выровнены по закрытию текущей base-свечи
 *
 * Файлы: models/model_{TICKER}_{interval}.onnx, models/scaler_{TICKER}_{interval}.json
 */

const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');
const thresholds = require('./thresholds');

const MODEL_DIR = path.join(__dirname, 'models');
const WINDOW_SIZE = 300;
const BASE_FEATURE_NAMES = ['ret_z', 'high_dev', 'low_dev', 'close_dev', 'value_z', 'dow_z', 'hour_z'];
const OI_FEATURE_NAMES = ['yur_net_z', 'fiz_net_z'];
// Должно совпадать с CONTEXT_MAP в train_gpu.py; порядок задаёт входы ONNX.
const CONTEXT_MAP = Object.freeze({
  '1m': Object.freeze(['10m', '60m']),
  '10m': Object.freeze(['60m', '1D']),
  '60m': Object.freeze(['1D']),
  '1D': Object.freeze([]),
});

function validateScaler(scaler, ticker, interval, label = 'Скейлер') {
  const fail = (detail) => { throw new Error(`${label}: ${detail}`); };
  if (!thresholds.isSupportedInterval(interval)) fail(`интервал ${interval} не поддерживается`);
  if (!scaler || typeof scaler !== 'object' || Array.isArray(scaler)) fail('некорректный JSON');
  if (scaler.version !== 4) fail(`версия ${scaler.version}, ожидается version=4`);
  if (scaler.ticker !== ticker) fail(`тикер ${scaler.ticker}, ожидается ${ticker}`);
  if (scaler.interval !== interval) fail(`интервал ${scaler.interval}, ожидается ${interval}`);
  if (!Number.isInteger(scaler.window) || scaler.window <= 0) fail('нет корректного window');
  if (scaler.order !== 'current_to_history') fail(`порядок ${scaler.order}, ожидается current_to_history`);
  const expected = CONTEXT_MAP[interval];
  if (!Array.isArray(scaler.contexts) || scaler.contexts.some((ctx) => !thresholds.isSupportedInterval(ctx))) {
    fail(`неподдерживаемые или некорректные контексты ${JSON.stringify(scaler.contexts)}, ожидаются ${JSON.stringify(expected)}`);
  }
  if (JSON.stringify(scaler.contexts) !== JSON.stringify(expected)) {
    fail(`контексты ${JSON.stringify(scaler.contexts)}, ожидаются ${JSON.stringify(expected)}`);
  }
  if (scaler.inputs !== 1 + expected.length) fail(`inputs=${scaler.inputs}, ожидается ${1 + expected.length}`);
  const featureNames = scaler.feature_names;
  const base = BASE_FEATURE_NAMES;
  const withOi = [...base, ...OI_FEATURE_NAMES];
  if (!Array.isArray(featureNames) ||
      (JSON.stringify(featureNames) !== JSON.stringify(base) && JSON.stringify(featureNames) !== JSON.stringify(withOi))) {
    fail(`несовместимые feature_names: ${JSON.stringify(featureNames)}`);
  }
  if (scaler.input_size !== featureNames.length) fail(`input_size=${scaler.input_size}, ожидается ${featureNames.length}`);
  const stats = ['ret_mean', 'ret_std', 'value_mean', 'value_std', 'dow_mean', 'dow_std', 'hour_mean', 'hour_std'];
  if (featureNames.length === withOi.length) stats.push('yur_net_mean', 'yur_net_std', 'fiz_net_mean', 'fiz_net_std');
  for (const key of stats) {
    if (typeof scaler[key] !== 'number' || !Number.isFinite(scaler[key])) fail(`нет числового поля ${key}`);
    if (key.endsWith('_std') && scaler[key] <= 0) fail(`${key} должен быть положительным`);
  }
  return scaler;
}

function validateContextScaler(scaler, baseScaler, ticker, interval, label) {
  validateScaler(scaler, ticker, interval, label);
  if (scaler.window !== baseScaler.window) {
    throw new Error(`${label}: окно ${scaler.window}, а у base-модели ${baseScaler.window}`);
  }
  if (JSON.stringify(scaler.feature_names) !== JSON.stringify(baseScaler.feature_names)) {
    throw new Error(`${label}: feature_names не совпадают с base-моделью`);
  }
}

// Для прогноза достаточно windowSize нормализованных строк + 1 свеча для prev_close.
// Берём с запасом: контекстам может понадобиться больше одной свечи на выравнивание.
const HISTORY_LIMIT = 600;
const DEFAULT_DIRECTION_THRESHOLD = 0.55;

const { INTERVAL_MINUTES, intervalMinutes } = thresholds;

function historyLimitFor(windowSize = WINDOW_SIZE) {
  const size = Number.isInteger(windowSize) && windowSize > 0 ? windowSize : WINDOW_SIZE;
  return Math.max(HISTORY_LIMIT, size + 2, size * 2);
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function fileExists(file) {
  return fs.existsSync(file);
}

function modelFingerprint(paths) {
  return paths.map((file) => {
    const st = fs.statSync(file);
    return { file, mtimeMs: st.mtimeMs, ctimeMs: st.ctimeMs, size: st.size, ino: st.ino };
  });
}

/**
 * Загружает ONNX-модель и скейлеры (base + контексты).
 * @param {string} ticker
 * @param {string} interval
 * @returns {Promise<{session, scaler, contexts, contextScalers, window, horizon, modelPath, scalerPath, files}>}
 */
async function loadModel(ticker, interval) {
  const modelPath = path.join(MODEL_DIR, `model_${ticker}_${interval}.onnx`);
  const scalerPath = path.join(MODEL_DIR, `scaler_${ticker}_${interval}.json`);

  if (!fileExists(modelPath)) {
    throw new Error(`Модель не найдена: ${modelPath}`);
  }
  if (!fileExists(scalerPath)) {
    throw new Error(`Скейлер не найден: ${scalerPath}`);
  }

  // Сначала фиксируем base-файлы: скейлер нельзя читать одновременно с заменой.
  const baseFiles = modelFingerprint([modelPath, scalerPath]);
  const scaler = validateScaler(readJson(scalerPath), ticker, interval, `Скейлер ${scalerPath}`);
  const featureNames = scaler.feature_names;
  const inputSize = scaler.input_size;
  const usesOi = featureNames.length === BASE_FEATURE_NAMES.length + OI_FEATURE_NAMES.length;
  const window = scaler.window;
  const horizon = Number(scaler.horizon) || 10;
  const contexts = scaler.contexts;
  const contextScalerPaths = contexts.map((ctx) => path.join(MODEL_DIR, `scaler_${ticker}_${ctx}.json`));
  for (const ctxPath of contextScalerPaths) {
    if (!fileExists(ctxPath)) throw new Error(`Скейлер контекста не найден: ${ctxPath}`);
  }
  const files = [...baseFiles, ...modelFingerprint(contextScalerPaths)];
  const contextScalers = contexts.map((ctx, index) => {
    const ctxPath = contextScalerPaths[index];
    const ctxScaler = readJson(ctxPath);
    validateContextScaler(ctxScaler, scaler, ticker, ctx, `Скейлер контекста ${ctxPath}`);
    return ctxScaler;
  });

  const session = await ort.InferenceSession.create(modelPath);
  const expectedNames = ['input_base', 'input_h1', 'input_h2'].slice(0, 1 + contexts.length);
  if (session.inputNames.length !== expectedNames.length ||
      expectedNames.some((name) => !session.inputNames.includes(name))) {
    throw new Error(`У модели ${modelPath} входы ${session.inputNames.join(', ')}, ожидаются ${expectedNames.join(', ')}`);
  }
  if (!isModelFresh({ files })) throw new Error(`Файлы модели ${modelPath} изменились во время загрузки`);
  return {
    ticker,
    interval,
    session,
    scaler,
    contexts,
    contextScalers,
    window,
    horizon,
    inputSize,
    usesOi,
    featureNames,
    modelPath,
    scalerPath,
    files,
  };
}

/**
 * Проверяет, не изменились ли файлы модели с момента загрузки.
 * Позволяет серверу подхватывать переобученную модель без рестарта.
 */
function isModelFresh(model) {
  if (!model || !Array.isArray(model.files)) return false;
  try {
    for (const item of model.files) {
      const st = fs.statSync(item.file);
      if (st.mtimeMs !== item.mtimeMs || st.ctimeMs !== item.ctimeMs ||
          st.size !== item.size || st.ino !== item.ino) return false;
    }
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * Загружает свечи строго по возрастанию begin.
 * dow и hour считаются в SQL — как при обучении, без сдвига часового пояса.
 * limit позволяет взять только последние N свечей (для инференса).
 */
async function loadCandles(client, ticker, interval, limit = null) {
  const columns = `begin, open, high, low, close, value,
            EXTRACT(ISODOW FROM begin)::int AS dow,
            EXTRACT(HOUR FROM begin)::int AS hour`;
  let sql;
  const params = [ticker, interval];
  if (Number.isInteger(limit) && limit > 0) {
    params.push(limit);
    sql = `SELECT * FROM (
             SELECT ${columns}
             FROM candles
             WHERE ticker = $1 AND interval = $2
             ORDER BY begin DESC
             LIMIT $3
           ) t ORDER BY begin ASC`;
  } else {
    sql = `SELECT ${columns}
           FROM candles
           WHERE ticker = $1 AND interval = $2
           ORDER BY begin ASC`;
  }

  const result = await client.query(sql, params);

  return result.rows.map((r) => ({
    begin: r.begin,
    beginMs: new Date(r.begin).getTime(),
    open: Number(r.open),
    high: Number(r.high),
    low: Number(r.low),
    close: Number(r.close),
    value: r.value == null ? 0 : Number(r.value),
    dow: Number(r.dow),
    hour: Number(r.hour),
  }));
}

/**
 * Снимки открытых позиций futoi для тикера: {ts: [ms], yur: [], fiz: []}.
 * null — если таблицы нет или данных нет (модель без OI или данные не загружены).
 */
async function loadOiSeries(client, ticker) {
  let result;
  try {
    result = await client.query(
      `SELECT ts, clgroup, pos FROM open_interest WHERE ticker = $1 ORDER BY ts ASC`,
      [ticker]
    );
  } catch (e) {
    if (e && e.code === '42P01') return null; // undefined_table
    throw e;
  }
  if (!result.rows.length) return null;

  const byTs = new Map();
  for (const r of result.rows) {
    if (r.clgroup !== 'FIZ' && r.clgroup !== 'YUR') continue;
    const pos = Number(r.pos);
    if (!Number.isFinite(pos)) continue;
    const ms = new Date(r.ts).getTime();
    if (!Number.isFinite(ms)) continue;
    if (!byTs.has(ms)) byTs.set(ms, { yur: null, fiz: null });
    byTs.get(ms)[r.clgroup === 'YUR' ? 'yur' : 'fiz'] = pos;
  }
  const ts = [...byTs.keys()].sort((a, b) => a - b);
  return {
    ts,
    yur: ts.map((t) => byTs.get(t).yur),
    fiz: ts.map((t) => byTs.get(t).fiz),
  };
}

/**
 * As-of join OI к свечам (как attach_oi в train_gpu.py): к закрытию свечи
 * берётся последний снимок с ts <= close; до первого снимка — null (нейтральный
 * 0 после нормализации). Паузы/выходные закрываются forward-fill.
 */
function attachOi(candles, interval, oi) {
  if (!oi) {
    for (const c of candles) {
      c.yurNet = null;
      c.fizNet = null;
    }
    return;
  }
  const stepMs = intervalMinutes(interval) * 60000;
  for (const c of candles) {
    const closeMs = c.beginMs + stepMs;
    // bisect_right по ts
    let lo = 0;
    let hi = oi.ts.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (oi.ts[mid] <= closeMs) lo = mid + 1;
      else hi = mid;
    }
    const i = lo - 1;
    if (i >= 0 && oi.yur[i] != null && oi.fiz[i] != null) {
      c.yurNet = oi.yur[i];
      c.fizNet = oi.fiz[i];
    } else {
      c.yurNet = null;
      c.fizNet = null;
    }
  }
}

function scalerStat(scaler, name) {
  const value = Number(scaler && scaler[name]);
  if (!Number.isFinite(value)) {
    throw new Error(`Скейлер не содержит числовое поле ${name}`);
  }
  return value;
}

/**
 * Нормализует свечи по сохранённому скейлеру.
 * Возвращает массив [n-1][F]: строка j соответствует свече candles[j + 1].
 * F = 7 базовых признаков + 2 OI, если скейлер содержит OI-статистики.
 * Свечи без OI (c.yurNet/c.fizNet == null) получают нейтральный 0.
 */
function normalizeCandles(candles, scaler) {
  const n = candles.length;
  const normalized = [];

  const retMean = scalerStat(scaler, 'ret_mean');
  const retStd = Math.abs(scalerStat(scaler, 'ret_std')) || 1;
  const valueMean = scalerStat(scaler, 'value_mean');
  const valueStd = Math.abs(scalerStat(scaler, 'value_std')) || 1;
  const dowMean = scalerStat(scaler, 'dow_mean');
  const dowStd = Math.abs(scalerStat(scaler, 'dow_std')) || 1;
  const hourMean = scalerStat(scaler, 'hour_mean');
  const hourStd = Math.abs(scalerStat(scaler, 'hour_std')) || 1;
  const withOi = Array.isArray(scaler.feature_names)
    ? OI_FEATURE_NAMES.every((name) => scaler.feature_names.includes(name))
    : Number.isFinite(Number(scaler.yur_net_mean)) && Number.isFinite(Number(scaler.fiz_net_mean));
  let yurMean = 0;
  let yurStd = 1;
  let fizMean = 0;
  let fizStd = 1;
  if (withOi) {
    yurMean = Number(scaler.yur_net_mean);
    yurStd = Math.abs(Number(scaler.yur_net_std)) || 1;
    fizMean = Number(scaler.fiz_net_mean);
    fizStd = Math.abs(Number(scaler.fiz_net_std)) || 1;
  }

  for (let i = 1; i < n; i++) {
    const c = candles[i];
    const prevClose = candles[i - 1].close;

    const prices = [c.open, c.high, c.low, c.close];
    if (!prices.every((value) => Number.isFinite(value) && value > 0)) {
      throw new Error(
        `Некорректная цена в свече ${c.begin}: open=${c.open}, high=${c.high}, low=${c.low}, close=${c.close}`
      );
    }
    if (c.high < c.low) {
      throw new Error(`Свеча ${c.begin}: high=${c.high} меньше low=${c.low}`);
    }
    if (!(prevClose > 0)) {
      throw new Error(`Некорректный prev_close в свече ${c.begin}: ${prevClose}`);
    }

    const value = Number.isFinite(c.value) ? Math.max(0, c.value) : 0;
    const logReturn = Math.log(c.close / prevClose);
    const logValue = Math.log1p(value);

    const row = [
      (logReturn - retMean) / retStd,
      (c.high - prevClose) / prevClose,
      (c.low - prevClose) / prevClose,
      (c.close - prevClose) / prevClose,
      (logValue - valueMean) / valueStd,
      (c.dow - dowMean) / dowStd,
      (c.hour - hourMean) / hourStd,
    ];
    if (withOi) {
      row.push(
        c.yurNet == null ? 0 : (c.yurNet - yurMean) / yurStd,
        c.fizNet == null ? 0 : (c.fizNet - fizMean) / fizStd
      );
    }
    if (!row.every(Number.isFinite)) {
      throw new Error(`Нечисловые признаки в свече ${c.begin} (проверьте скейлер)`);
    }
    normalized.push(row);
  }

  return normalized;
}

/**
 * Окно старшего интервала, выровненное по закрытию base-свечи.
 * Берутся только полностью закрытые свечи (base_end - ctx_duration).
 */
function contextWindow(baseCandles, baseInterval, ctxCandles, ctxNormalized, ctxInterval, windowSize = WINDOW_SIZE) {
  const baseLast = baseCandles[baseCandles.length - 1];
  const baseEnd = baseLast.beginMs + intervalMinutes(baseInterval) * 60000;
  const cutoff = baseEnd - intervalMinutes(ctxInterval) * 60000;

  // bisect_right по beginMs
  let lo = 0;
  let hi = ctxCandles.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ctxCandles[mid].beginMs <= cutoff) lo = mid + 1;
    else hi = mid;
  }
  const count = lo;

  // ctxNormalized[k] соответствует свече k + 1, последняя закрытая свеча — count - 1
  const end = count - 1; // slice end (exclusive)
  const start = end - windowSize;
  if (start < 0) return null;

  return ctxNormalized.slice(start, end).reverse();
}

/**
 * Собирает входы для модели и делает прогноз.
 *
 * @param {import('pg').Client} client
 * @param {object} model результат loadModel()
 * @param {string} ticker
 * @param {string} interval
 * @param {{directionThreshold?: number, fallbackBaseThreshold?: number}} [options]
 * @returns {Promise<object>}
 */
async function predictWithModel(client, model, ticker, interval, options = {}) {
  const { session, scaler, contexts, contextScalers } = model;
  const windowSize = Number.isInteger(model.window) && model.window > 0 ? model.window : WINDOW_SIZE;
  const inputSize = Number.isInteger(model.inputSize) && model.inputSize > 0
    ? model.inputSize
    : BASE_FEATURE_NAMES.length;
  const historyLimit = historyLimitFor(windowSize);
  const directionThreshold = Number.isFinite(Number(options.directionThreshold))
    ? Number(options.directionThreshold)
    : DEFAULT_DIRECTION_THRESHOLD;
  const fallbackBaseThreshold = Number.isFinite(Number(options.fallbackBaseThreshold)) &&
    Number(options.fallbackBaseThreshold) > 0
    ? Number(options.fallbackBaseThreshold)
    : thresholds.DEFAULT_BASE_THRESHOLD;

  const baseCandles = await loadCandles(client, ticker, interval, historyLimit);
  if (baseCandles.length < windowSize + 1) {
    throw new Error(
      `Недостаточно свечей @ ${interval}: ${baseCandles.length}, нужно ${windowSize + 1}`
    );
  }

  // OI нужен, если его ждут base-скейлер или любой контекст.
  const needsOi = model.usesOi || contextScalers.some(
    (s) => Array.isArray(s.feature_names) && OI_FEATURE_NAMES.every((name) => s.feature_names.includes(name))
  );
  const oi = needsOi ? await loadOiSeries(client, ticker) : null;
  if (needsOi) attachOi(baseCandles, interval, oi);

  const baseNormalized = normalizeCandles(baseCandles, scaler);
  const baseWindow = baseNormalized.slice(-windowSize).reverse();

  const windows = [baseWindow];

  for (let i = 0; i < contexts.length; i++) {
    const ctxInterval = contexts[i];
    const ctxCandles = await loadCandles(client, ticker, ctxInterval, historyLimit);
    if (ctxCandles.length < windowSize + 1) {
      throw new Error(
        `Недостаточно свечей контекста @ ${ctxInterval}: ${ctxCandles.length}, нужно ${windowSize + 1}`
      );
    }
    if (needsOi) attachOi(ctxCandles, ctxInterval, oi);
    const ctxNormalized = normalizeCandles(ctxCandles, contextScalers[i]);
    const win = contextWindow(baseCandles, interval, ctxCandles, ctxNormalized, ctxInterval, windowSize);
    if (!win) {
      throw new Error(`Недостаточно истории контекста @ ${ctxInterval} для ${interval}`);
    }
    windows.push(win);
  }

  const inputs = session.inputNames;
  if (inputs.length !== windows.length) {
    throw new Error(
      `У модели ${inputs.length} входов, собрано ${windows.length} окон (${contexts.join(', ') || 'без контекстов'})`
    );
  }

  const preferredNames = ['input_base', 'input_h1', 'input_h2'];
  const feeds = {};
  for (let i = 0; i < windows.length; i++) {
    const preferred = preferredNames[i];
    const inputName = preferred && inputs.includes(preferred) ? preferred : inputs[i];
    if (!inputName) {
      throw new Error(`Не удалось сопоставить вход №${i} модели`);
    }
    const flat = new Float32Array(windowSize * inputSize);
    let k = 0;
    for (let row = 0; row < windows[i].length; row++) {
      for (let f = 0; f < inputSize; f++) flat[k++] = windows[i][row][f];
    }
    feeds[inputName] = new ort.Tensor('float32', flat, [1, windowSize, inputSize]);
  }

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const output = results[outputName];
  if (!output || !output.data || !output.data.length) {
    throw new Error(`Модель не вернула выход ${outputName}`);
  }

  let pUp = Number(output.data[0]);
  if (!Number.isFinite(pUp)) {
    throw new Error('Модель вернула NaN/Infinity вместо вероятности');
  }
  pUp = Math.min(1, Math.max(0, pUp));

  const pDown = 1 - pUp;
  const direction = pUp > directionThreshold ? 'UP' : (pDown > directionThreshold ? 'DOWN' : null);
  const lastCandle = baseCandles[baseCandles.length - 1];
  const meta = thresholds.resolveModelThreshold(scaler, interval, fallbackBaseThreshold);

  return {
    ticker,
    interval,
    pUp,
    pDown,
    direction,
    confidence: Math.max(pUp, pDown),
    lastClose: lastCandle.close,
    lastBegin: lastCandle.begin,
    threshold: meta.effective,
    thresholdBase: meta.base,
    thresholdMultiplier: meta.multiplier,
    thresholdHorizon: meta.horizon,
    thresholdProgressive: meta.progressive,
    thresholdSource: meta.source,
    contexts,
    steps: windows.length,
    window: windowSize,
    horizon: meta.horizon,
    inputShape: [1, windowSize, inputSize],
  };
}

/**
 * Полный цикл: загрузка модели (с кешем) + прогноз.
 */
async function predictInterval(client, ticker, interval, cache = new Map(), options = {}) {
  const key = `${ticker}:${interval}`;
  let model = cache.get(key);
  if (model && !isModelFresh(model)) {
    cache.delete(key);
    model = null;
  }
  if (!model) {
    model = await loadModel(ticker, interval);
    cache.set(key, model);
  }
  try {
    const result = await predictWithModel(client, model, ticker, interval, options);
    if (!isModelFresh(model)) throw new Error('Файлы модели изменились во время расчёта прогноза');
    return result;
  } catch (e) {
    if (!isModelFresh(model) && cache.get(key) === model) cache.delete(key);
    throw e;
  }
}

module.exports = {
  WINDOW_SIZE,
  HISTORY_LIMIT,
  DEFAULT_DIRECTION_THRESHOLD,
  INTERVAL_MINUTES,
  BASE_FEATURE_NAMES,
  OI_FEATURE_NAMES,
  CONTEXT_MAP,
  validateScaler,
  validateContextScaler,
  intervalMinutes,
  historyLimitFor,
  loadModel,
  isModelFresh,
  loadCandles,
  loadOiSeries,
  attachOi,
  normalizeCandles,
  contextWindow,
  predictWithModel,
  predictInterval,
};
