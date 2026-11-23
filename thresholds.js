/**
 * thresholds.js — единая логика прогрессивного порога.
 *
 * Базовый порог (обычно 0.003 = 0.3%) задаёт целевое движение для 1m.
 * На каждом следующем интервале множитель удваивается:
 *   1m ×1, 10m ×2, 60m ×4, 1D ×8.
 * Горизонт (число свечей) не меняет порог.
 *
 * Источник истины для обученной модели — scaler_*.json:
 *   threshold            — эффективный порог, с которым обучалась модель;
 *   threshold_base       — базовый порог;
 *   threshold_multiplier — множитель интервала;
 *   threshold_horizon    — горизонт.
 * Если этих полей нет (старые scaler), эффективный порог всё равно берётся
 * из scaler.threshold: нельзя пересчитывать цель по новому множителю, иначе
 * модель, обученная на 0.3% (SBER 1D), начнёт показывать цель 2.4%.
 */

const fs = require('fs');
const path = require('path');

const CONFIG_PATH = path.join(__dirname, 'thresholds.json');

const DEFAULTS = {
  baseThreshold: 0.003,
  intervalMultipliers: {
    '1m': 1.0,
    '10m': 2.0,
    '60m': 4.0,
    '1D': 8.0,
  },
  fallbackMultipliers: [
    { maxMinutes: 1, multiplier: 1.0 },
    { maxMinutes: 10, multiplier: 2.0 },
    { maxMinutes: 60, multiplier: 4.0 },
    { maxMinutes: 1000000, multiplier: 8.0 },
  ],
};

function positiveNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : null;
}

function positiveInt(value) {
  const n = Number(value);
  return Number.isInteger(n) && n > 0 ? n : null;
}

function loadConfig() {
  let raw;
  try {
    raw = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
  } catch (e) {
    return DEFAULTS;
  }

  const multipliers = {};
  const source = raw && typeof raw.intervalMultipliers === 'object' ? raw.intervalMultipliers : {};
  for (const [interval, value] of Object.entries(source)) {
    const n = positiveNumber(value);
    if (n) multipliers[interval] = n;
  }

  const fallback = Array.isArray(raw && raw.fallbackMultipliers)
    ? raw.fallbackMultipliers
      .map((item) => ({
        maxMinutes: positiveNumber(item && item.maxMinutes),
        multiplier: positiveNumber(item && item.multiplier),
      }))
      .filter((item) => item.maxMinutes && item.multiplier)
      .sort((a, b) => a.maxMinutes - b.maxMinutes)
    : DEFAULTS.fallbackMultipliers;

  return {
    baseThreshold: positiveNumber(raw && raw.baseThreshold) || DEFAULTS.baseThreshold,
    intervalMultipliers: Object.keys(multipliers).length ? multipliers : DEFAULTS.intervalMultipliers,
    fallbackMultipliers: fallback.length ? fallback : DEFAULTS.fallbackMultipliers,
  };
}

const CONFIG = loadConfig();

// Минуты на одну свечу интервала. 1D = 1440.
const INTERVAL_MINUTES = Object.freeze({
  '1m': 1,
  '5m': 5,
  '10m': 10,
  '15m': 15,
  '30m': 30,
  '60m': 60,
  '2h': 120,
  '4h': 240,
  '1D': 1440,
});

// Интервалы, которые поддерживает сервер и загрузчик свечей.
const INTERVAL_ORDER = Object.freeze(['1m', '10m', '60m', '1D']);

function isKnownInterval(interval) {
  return Object.prototype.hasOwnProperty.call(INTERVAL_MINUTES, String(interval || '').trim());
}

function isSupportedInterval(interval) {
  return INTERVAL_ORDER.includes(String(interval || '').trim());
}

function intervalMinutes(interval) {
  const key = String(interval || '').trim();
  const minutes = INTERVAL_MINUTES[key];
  if (!minutes) throw new Error(`Неизвестный интервал: ${interval}`);
  return minutes;
}

function validateInterval(interval) {
  const key = String(interval || '').trim();
  if (!isSupportedInterval(key)) {
    throw new Error(`Неизвестный интервал: ${interval || '(пусто)'}. Допустимы: ${INTERVAL_ORDER.join(', ')}`);
  }
  return key;
}

/**
 * Множитель прогрессивного порога для интервала.
 * Для неизвестных, но валидных минут используется ступенчатый fallback:
 * <=1m ×1, <=10m ×2, <=60m ×4, больше ×8. Неизвестная строка → ×1.
 */
function intervalMultiplier(interval) {
  const key = String(interval || '').trim();
  if (Object.prototype.hasOwnProperty.call(CONFIG.intervalMultipliers, key)) {
    return CONFIG.intervalMultipliers[key];
  }

  const minutes = INTERVAL_MINUTES[key];
  if (!minutes) return 1;
  for (const item of CONFIG.fallbackMultipliers) {
    if (minutes <= item.maxMinutes) return item.multiplier;
  }
  return 1;
}

function baseThreshold(value) {
  return positiveNumber(value) || CONFIG.baseThreshold;
}

/** Эффективный порог по базовому и интервалу (без горизонта). */
function effectiveThreshold(base, interval) {
  return baseThreshold(base) * intervalMultiplier(interval);
}

/** Как effectiveThreshold, но никогда не бросает исключение. */
function safeEffectiveThreshold(base, interval) {
  try {
    return effectiveThreshold(base, interval);
  } catch (e) {
    return baseThreshold(base);
  }
}

function thresholdMeta(base, interval, options = {}) {
  const b = baseThreshold(base);
  const progressive = options.progressive !== false;
  const multiplier = progressive ? intervalMultiplier(interval) : 1;
  return {
    base: b,
    effective: b * multiplier,
    multiplier,
    horizon: positiveInt(options.horizon) || 10,
    progressive,
    source: 'default',
  };
}

/**
 * Разрешает порог обученной модели.
 *
 * Главное правило: если scaler содержит threshold, это и есть цель обучения.
 * Новые scaler дополнительно содержат base/multiplier/horizon/progressive.
 * Старые scaler без base получают эвристику: если threshold совпадает с
 * ожидаемым прогрессивным значением, восстанавливаем base и multiplier;
 * иначе считаем threshold фиксированным базовым порогом (multiplier = 1).
 */
function resolveModelThreshold(scaler, interval, fallbackBase) {
  const sc = scaler && typeof scaler === 'object' ? scaler : {};
  const fallback = baseThreshold(fallbackBase);
  const horizon = positiveInt(sc.horizon) || positiveInt(sc.threshold_horizon) || 10;
  const effectiveFromScaler = positiveNumber(sc.threshold);
  const baseFromScaler = positiveNumber(sc.threshold_base);
  const multiplierFromScaler = positiveNumber(sc.threshold_multiplier);
  const progressiveFlag = sc.threshold_progressive === undefined
    ? null
    : Boolean(sc.threshold_progressive);

  if (effectiveFromScaler) {
    let base;
    let multiplier;

    if (baseFromScaler) {
      base = baseFromScaler;
      multiplier = multiplierFromScaler || effectiveFromScaler / base;
    } else if (multiplierFromScaler) {
      multiplier = multiplierFromScaler;
      base = effectiveFromScaler / multiplier;
    } else {
      // Старый scaler: пытаемся понять, прогрессивный он или фиксированный.
      const expectedMultiplier = intervalMultiplier(interval);
      const expected = fallback * expectedMultiplier;
      const tolerance = Math.max(Math.abs(expected) * 0.25, 1e-6);
      if (Math.abs(effectiveFromScaler - expected) <= tolerance) {
        base = effectiveFromScaler / expectedMultiplier;
        multiplier = expectedMultiplier;
      } else {
        base = effectiveFromScaler;
        multiplier = 1;
      }
    }

    if (!Number.isFinite(base) || base <= 0) base = effectiveFromScaler;
    if (!Number.isFinite(multiplier) || multiplier <= 0) multiplier = 1;

    return {
      base,
      effective: effectiveFromScaler,
      multiplier,
      horizon,
      progressive: progressiveFlag === null ? multiplier !== 1 : progressiveFlag,
      source: 'scaler',
    };
  }

  // Совсем старый scaler без threshold — используем прогрессивные значения.
  const meta = thresholdMeta(baseFromScaler || fallback, interval, {
    horizon,
    progressive: progressiveFlag === null ? true : progressiveFlag,
  });
  meta.source = baseFromScaler ? 'scaler' : 'default';
  return meta;
}

/** Пороги для набора интервалов (для /api/config и логов обучения). */
function thresholdsForIntervals(base, intervals = INTERVAL_ORDER, options = {}) {
  const progressive = options.progressive !== false;
  const b = baseThreshold(base);
  return Object.fromEntries(
    intervals.map((interval) => [interval, progressive ? effectiveThreshold(b, interval) : b])
  );
}

/** Множители для набора интервалов. */
function multipliersForIntervals(intervals = INTERVAL_ORDER, options = {}) {
  const progressive = options.progressive !== false;
  return Object.fromEntries(
    intervals.map((interval) => [interval, progressive ? intervalMultiplier(interval) : 1])
  );
}

module.exports = {
  CONFIG,
  DEFAULT_BASE_THRESHOLD: CONFIG.baseThreshold,
  INTERVAL_MINUTES,
  INTERVAL_ORDER,
  isKnownInterval,
  isSupportedInterval,
  intervalMinutes,
  validateInterval,
  intervalMultiplier,
  baseThreshold,
  effectiveThreshold,
  safeEffectiveThreshold,
  thresholdMeta,
  resolveModelThreshold,
  thresholdsForIntervals,
  multipliersForIntervals,
  positiveNumber,
  positiveInt,
};
