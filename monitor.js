/**
 * monitor.js — Real-time мониторинг через ONNX Runtime
 * После истечения горизонта (10 свечей) помечает цель:
 *   🟢 зелёный — цель достигнута (прогноз сработал)
 *   🔴 красный — цель не достигнута (прогноз провалился)
 *
 * usage: node monitor.js
 * env:   INTERVALS, TICKER, SLEEP_MS, THRESHOLD
 */

require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');

const WINDOW_SIZE = 300;
const HORIZON = 10;
const THRESHOLD = parseFloat(process.env.THRESHOLD || '0.003');

const OUT_DIR = path.join(__dirname, 'models');
const TICKER = process.env.TICKER || 'IMOEX';
const INTERVALS = process.env.INTERVALS ? process.env.INTERVALS.split(',').map(s => s.trim()) : ['60m', '10m', '4h', '1D', '2h'];
const SLEEP_MS = parseInt(process.env.SLEEP_MS || '60000', 10);

const HIGHER_MAP_DEFAULT = {
  '1m': ['10m', '60m'],
  '10m': ['60m', '4h'],
  '15m': ['60m', '4h'],
  '30m': ['60m', '4h'],
  '60m': ['4h', '1D'],
  '2h': ['4h', '1D'],
  '4h': ['1D', null],
  '1D': [null, null],
};

const sessions = {};
const scalers = {};
const history = {}; // interval -> [{pUp, direction, targetUp, targetDown, begin, checked, result}]

function normalizeWithScaler(candles, scaler) {
  const normalized = [];
  for (let i = 1; i < candles.length; i++) {
    const c = candles[i];
    const o = candles[i - 1].close;
    const lr = Math.log(c.close / o);
    normalized.push([
      (lr - scaler.price_mean) / (scaler.price_std || 1),
      (c.high - o) / o,
      (c.low - o) / o,
      (c.close - o) / o,
      (c.volume - scaler.vol_mean) / (scaler.vol_std || 1)
    ]);
  }
  return normalized;
}

async function loadCandles(client, interval, ticker) {
  const result = await client.query(
    `SELECT begin, open, high, low, close, volume FROM candles WHERE ticker = $1 AND interval = $2 ORDER BY begin ASC`,
    [ticker, interval]
  );
  return result.rows.map(r => ({
    begin: r.begin,
    open: parseFloat(r.open),
    high: parseFloat(r.high),
    low: parseFloat(r.low),
    close: parseFloat(r.close),
    volume: parseInt(r.volume) || 0
  }));
}

function findLatestIdxBefore(candles, targetTime) {
  let idx = -1;
  for (let i = 0; i < candles.length; i++) {
    if (new Date(candles[i].begin) <= new Date(targetTime)) idx = i;
    else break;
  }
  return idx;
}

function loadModel(interval) {
  const modelPath = path.join(OUT_DIR, `model_${interval}.onnx`);
  const scalerPath = path.join(OUT_DIR, `scaler_${interval}.json`);
  if (!fs.existsSync(modelPath) || !fs.existsSync(scalerPath)) return false;
  sessions[interval] = new ort.InferenceSession(modelPath);
  scalers[interval] = JSON.parse(fs.readFileSync(scalerPath, 'utf8'));
  history[interval] = [];
  return true;
}

function nowStr() {
  return new Date().toISOString().replace('T', ' ').substring(0, 19);
}

function colorResult(achieved) {
  return achieved ? '🟢' : '🔴';
}

async function predictInterval(client, interval) {
  const session = sessions[interval];
  const scalerData = scalers[interval];
  if (!session || !scalerData) return;

  const scalerCurr = scalerData.curr;
  const scalerH1 = scalerData.higher1;
  const scalerH2 = scalerData.higher2;
  const higher1Name = scalerData.higher1_name;
  const higher2Name = scalerData.higher2_name;

  try {
    const candlesCurr = await loadCandles(client, interval, ticker);
    if (candlesCurr.length < WINDOW_SIZE + 1) {
      console.log(`${nowStr()} | ${TICKER} ${interval} | ⚠️  ${candlesCurr.length}/${WINDOW_SIZE + 1} свечей`);
      return;
    }

    const targetTime = candlesCurr[candlesCurr.length - 1].begin;
    const normCurr = normalizeWithScaler(candlesCurr, scalerCurr);
    const currWin = normCurr.slice(normCurr.length - WINDOW_SIZE);

    let h1Win = null, h2Win = null;

    if (higher1Name && scalerH1) {
      const candlesH1 = await loadCandles(client, higher1Name, ticker);
      if (candlesH1.length >= WINDOW_SIZE + 1) {
        const normH1 = normalizeWithScaler(candlesH1, scalerH1);
        const idx = findLatestIdxBefore(candlesH1, targetTime);
        if (idx >= WINDOW_SIZE - 1) h1Win = normH1.slice(idx - WINDOW_SIZE + 1, idx + 1);
      }
    }

    if (higher2Name && scalerH2) {
      const candlesH2 = await loadCandles(client, higher2Name, ticker);
      if (candlesH2.length >= WINDOW_SIZE + 1) {
        const normH2 = normalizeWithScaler(candlesH2, scalerH2);
        const idx = findLatestIdxBefore(candlesH2, targetTime);
        if (idx >= WINDOW_SIZE - 1) h2Win = normH2.slice(idx - WINDOW_SIZE + 1, idx + 1);
      }
    }

    const parts = [currWin];
    if (h1Win) parts.push(h1Win);
    if (h2Win) parts.push(h2Win);
    const inputFlat = parts.flat();
    const steps = parts.reduce((a, b) => a + b.length, 0) / WINDOW_SIZE;

    const inputTensor = new ort.Tensor('float32', Float32Array.from(inputFlat), [1, Math.round(steps), 5]);
    const feeds = { input: inputTensor };
    const results = await session.run(feeds);
    const pUp = results.output.data[0] || 0;
    const pDown = 1 - pUp;
    const direction = pUp >= 0.5 ? 'UP' : 'DOWN';

    const lastClose = candlesCurr[candles.length - 1].close;
    const upTarget = (lastClose * (1 + THRESHOLD)).toFixed(2);
    const downTarget = (lastClose * (1 - THRESHOLD)).toFixed(2);

    // Сохраняем в историю для последующей проверки
    history[interval].push({
      pUp, pDown, direction,
      lastClose, lastBegin: targetTime,
      upTarget, downTarget,
      checked: false,
      result: null,
      ts: Date.now()
    });

    // Проверяем старые прогнозы, чей горизонт истёк
    checkHistory(client, interval, candlesCurr);

    const dirEmoji = direction === 'UP' ? '🟢' : '🔴';
    console.log(
      `${nowStr()} | ${TICKER} ${interval} | P(up)=${(pUp * 100).toFixed(2)}% ` +
      `P(down)=${(pDown * 100).toFixed(2)}% | close=${lastClose.toFixed(2)} | ${dirEmoji} ${direction} ` +
      `conf=${(Math.max(pUp, pDown) * 100).toFixed(2)}% | steps=${Math.round(steps)}`
    );
  } catch (e) {
    console.log(`${nowStr()} | ${TICKER} ${interval} | ❌ ${e.message}`);
  }
}

async function checkHistory(client, interval, candlesCurr) {
  const scalerData = scalers[interval];
  if (!scalerData) return;
  const hist = history[interval];
  if (!hist || hist.length === 0) return;

  // Для каждой unverified записи проверяем, прошли ли HORIZON свечей после begin
  for (const rec of hist) {
    if (rec.checked) continue;

    // Находим индекс begin в текущих свечах
    const beginIdx = candlesCurr.findIndex(c => new Date(c.begin).getTime() === new Date(rec.lastBegin).getTime());
    if (beginIdx === -1 || beginIdx + HORIZON >= candlesCurr.length) continue; // ещё не наступили HORIZON свечей

    const futureCandles = candlesCurr.slice(beginIdx + 1, beginIdx + 1 + HORIZON);
    const maxHigh = Math.max(...futureCandles.map(c => c.high));
    const minLow = Math.min(...futureCandles.map(c => c.low));

    const upTarget = rec.lastClose * (1 + THRESHOLD);
    const downTarget = rec.lastClose * (1 - THRESHOLD);

    let achieved = false;
    if (rec.direction === 'UP') achieved = maxHigh >= upTarget;
    if (rec.direction === 'DOWN') achieved = minLow <= downTarget;

    rec.checked = true;
    rec.result = achieved;

    const emoji = colorResult(achieved);
    console.log(
      `${nowStr()} | ${TICKER} ${interval} | ${emoji} ТАРГЕТ ${rec.direction} ` +
      `(от ${rec.lastBegin}, close=${rec.lastClose.toFixed(2)}): ${achieved ? 'ДОСТИГНУТ' : 'ПРОВАЛ'} ` +
      `(high=${maxHigh.toFixed(2)}, low=${minLow.toFixed(2)}, цели up=${upTarget.toFixed(2)}, down=${downTarget.toFixed(2)})`
    );
  }
}

async function main() {
  console.log(`🖥️  Monitor started: ticker=${TICKER} intervals=[${INTERVALS.join(', ')}] sleep=${SLEEP_MS}ms threshold=${THRESHOLD}`);

  for (const interval of INTERVALS) {
    if (loadModel(interval)) {
      console.log(`   ✅ Loaded model ${interval}`);
    } else {
      console.log(`   ⚠️  Model for ${interval} not found (run python train_gpu.py --intervals ${interval} --threshold ${THRESHOLD})`);
    }
  }

  const client = new Client({ connectionString: process.env.DB });
  await client.connect();

  console.log(`\n${nowStr()} — started real-time predictions (Ctrl+C to stop)\n`);

  let first = true;
  while (true) {
    if (!first) {
      await new Promise(resolve => setTimeout(resolve, SLEEP_MS));
    }
    first = false;

    for (const interval of INTERVALS) {
      if (sessions[interval]) await predictInterval(client, interval);
    }
    console.log('');
  }
}

main().catch(e => {
  console.error('❌ Error:', e.message);
  process.exit(1);
});
