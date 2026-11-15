/**
 * prepare_dataset.js — Подготовка выборки для обучения Brain.js LSTM
 * Загружает свечи из PG, нормализует, создаёт sliding windows 300→10, сохраняет JSON для обучения
 */

require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

const TICKER = process.env.TICKER || 'IMOEX';
const INTERVALS = process.env.INTERVALS ? process.env.INTERVALS.split(',').map(s => s.trim()) : ['60m', '10m', '1D'];
const WINDOW_SIZE = 300;
const HORIZON = 10;
const THRESHOLD = 0.003;
const TRAIN_RATIO = 0.7;
const VAL_RATIO = 0.15;
const TEST_RATIO = 0.15;

const OUT_DIR = path.join(__dirname, 'models');
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

function intervalToMinutes(interval) {
  if (interval === '1D') return 1440;
  if (interval === '4h') return 240;
  if (interval === '1h' || interval === '60m') return 60;
  if (interval === '30m') return 30;
  if (interval === '15m') return 15;
  if (interval === '10m') return 10;
  if (interval === '1m') return 1;
  return parseInt(interval);
}
/**
 * Нормализация: log returns для цены, z-score для объёма
 * Возвращает { data: normalized[][], scaler: {priceMean, priceStd, volMean, volStd} }
 */
function normalizeCandles(candles) {
  const n = candles.length;
  const logReturns = new Array(n - 1);
  const volumes = new Array(n);

  for (let i = 0; i < n; i++) {
    volumes[i] = candles[i].volume || 0;
  }

  for (let i = 1; i < n; i++) {
    const prevClose = candles[i - 1].close;
    const currClose = candles[i].close;
    logReturns[i - 1] = Math.log(currClose / prevClose);
  }

  const priceMean = logReturns.reduce((a, b) => a + b, 0) / logReturns.length;
  const priceStd = Math.sqrt(logReturns.reduce((a, b) => a + (b - priceMean) ** 2, 0) / logReturns.length);
  const volMean = volumes.reduce((a, b) => a + b, 0) / volumes.length;
  const volStd = Math.sqrt(volumes.reduce((a, b) => a + (b - volMean) ** 2, 0) / volumes.length) || 1;

  const normalized = [];
  for (let i = 1; i < n; i++) {
    const c = candles[i];
    const o = candles[i - 1].close;
    normalized.push([
      (logReturns[i - 1] - priceMean) / (priceStd || 1),
      (c.high - o) / o,
      (c.low - o) / o,
      (c.close - o) / o,
      (volumes[i] - volMean) / (volStd || 1)
    ]);
  }

  return {
    data: normalized,
    scaler: { priceMean, priceStd, volMean, volStd }
  };
}
/**
 * Создаёт датасет: X = [window, 5], y = 0/1 (down/up)
 * Лейбл: 1 если max(high) за horizon >= close * (1+threshold)
 *        0 если min(low) за horizon <= close * (1-threshold)
 *        flat (ничего не достигли) — исключаем из обучения
 */
function createDataset(normalized, candles, horizon, threshold) {
  const X = [];
  const y = [];
  const dates = [];

  for (let i = 0; i <= normalized.length - WINDOW_SIZE - horizon; i++) {
    const window = normalized.slice(i, i + WINDOW_SIZE);
    const currentClose = candles[i + WINDOW_SIZE - 1].close;

    let maxHigh = -Infinity;
    let minLow = Infinity;
    for (let j = 1; j <= horizon; j++) {
      const c = candles[i + WINDOW_SIZE - 1 + j];
      maxHigh = Math.max(maxHigh, c.high);
      minLow = Math.min(minLow, c.low);
    }

    const upTarget = currentClose * (1 + threshold);
    const downTarget = currentClose * (1 - threshold);

    let label;
    if (maxHigh >= upTarget && minLow <= downTarget) {
      label = -1; // оба достигли — исключаем
    } else if (maxHigh >= upTarget) {
      label = 1; // up
    } else if (minLow <= downTarget) {
      label = 0; // down
    } else {
      label = -1; // flat — исключаем
    }

    if (label !== -1) {
      X.push(window);
      y.push(label);
      dates.push(candles[i + WINDOW_SIZE - 1].begin);
    }
  }

  return { X, y, dates };
}

async function loadCandles(client, interval, ticker) {
  const result = await client.query(`
    SELECT begin, open, high, low, close, volume
    FROM candles
    WHERE ticker = $1 AND interval = $2
    ORDER BY begin ASC
  `, [ticker, interval]);
  return result.rows.map(r => ({
    begin: r.begin,
    open: parseFloat(r.open),
    high: parseFloat(r.high),
    low: parseFloat(r.low),
    close: parseFloat(r.close),
    volume: parseInt(r.volume) || 0
  }));
}

async function main() {
  const client = new Client({ connectionString: process.env.DB });
  await client.connect();
  console.log('✅ Connected to PostgreSQL');

  for (const interval of INTERVALS) {
    console.log(`\n📊 Processing ${interval}...`);
    const candles = await loadCandles(client, interval, TICKER);
    console.log(`   Loaded ${candles.length} candles`);

    if (candles.length < WINDOW_SIZE + HORIZON + 100) {
      console.log(`   ⚠️ Not enough data, skipping`);
      continue;
    }

    const { data: normalized, scaler } = normalizeCandles(candles);
    console.log(`   Normalized: ${normalized.length} samples`);

    const { X, y, dates } = createDataset(normalized, candles, HORIZON, THRESHOLD);
    console.log(`   Dataset: ${X.length} samples (up: ${y.filter(v => v === 1).length}, down: ${y.filter(v => v === 0).length})`);

    if (X.length < 100) {
      console.log(`   ⚠️ Too few labeled samples, skipping`);
      continue;
    }

    const n = X.length;
    const trainEnd = Math.floor(n * TRAIN_RATIO);
    const valEnd = trainEnd + Math.floor(n * VAL_RATIO);

    const splits = {
      train: { X: X.slice(0, trainEnd), y: y.slice(0, trainEnd), dates: dates.slice(0, trainEnd) },
      val: { X: X.slice(trainEnd, valEnd), y: y.slice(trainEnd, valEnd), dates: dates.slice(trainEnd, valEnd) },
      test: { X: X.slice(valEnd), y: y.slice(valEnd), dates: dates.slice(valEnd) }
    };

    console.log(`   Split: train=${splits.train.X.length}, val=${splits.val.X.length}, test=${splits.test.X.length}`);

    const intervalSafe = interval.replace('m', 'm').replace('D', 'D');
    const scalerPath = path.join(OUT_DIR, `scaler_${intervalSafe}.json`);

    fs.writeFileSync(scalerPath, JSON.stringify(scaler, null, 2));

    console.log(`   💾 Saved scaler: ${scalerPath}`);
  }

  await client.end();
  console.log('\n✅ All intervals processed');
}

main().catch(e => {
  console.error('❌ Error:', e);
  process.exit(1);
});