/**
 * predict.js — Single-shot инференс через ONNX Runtime
 * Вход: 300 свечей текущего интервала + 300 старшего #1 + 300 старшего #2 = 900 свечей × 5 признаков
 * Порог берется из env THRESHOLD (default 0.003)
 *
 * usage: node predict.js <interval> [ticker]
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

const interval = process.argv[2];
const ticker = process.argv[3] || (process.env.TICKER || 'IMOEX');

if (!interval) {
  console.error('Usage: node predict.js <interval> [ticker]');
  console.error('Example: node predict.js 60m IMOEX');
  process.exit(1);
}

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

async function main() {
  const scalerPath = path.join(OUT_DIR, `scaler_${interval}.json`);
  const modelPath = path.join(OUT_DIR, `model_${interval}.onnx`);

  if (!fs.existsSync(modelPath)) {
    console.error(`❌ Модель не найдена: ${modelPath}`);
    console.error(`   Сначала запустите: python train_gpu.py --intervals ${interval} --threshold ${THRESHOLD}`);
    process.exit(1);
  }
  if (!fs.existsSync(scalerPath)) {
    console.error(`❌ Скейлер не найден: ${scalerPath}`);
    process.exit(1);
  }

  const session = await ort.InferenceSession.create(modelPath);
  const scalerData = JSON.parse(fs.readFileSync(scalerPath, 'utf8'));
  const scalerCurr = scalerData.curr;
  const scalerH1 = scalerData.higher1;
  const scalerH2 = scalerData.higher2;
  const higher1Name = scalerData.higher1_name;
  const higher2Name = scalerData.higher2_name;
  const savedThreshold = scalerData.threshold;

  if (Math.abs(savedThreshold - THRESHOLD) > 1e-9) {
    console.warn(`⚠️  Внимание: порог в скейлере (${savedThreshold}) отличается от текущего (${THRESHOLD}). Переобучите модель.`);
  }

  const client = new Client({ connectionString: process.env.DB });
  await client.connect();

  // Текущий интервал — нужны WINDOW_SIZE+1 свечей для нормализации (первая как база)
  const candlesCurr = await loadCandles(client, interval, ticker);
  await client.end();

  if (candlesCurr.length < WINDOW_SIZE + 1) {
    console.error(`❌ Недостаточно свечей @ ${interval}: ${candlesCurr.length}, нужно ${WINDOW_SIZE + 1}`);
    process.exit(1);
  }

  const targetTime = candlesCurr[candlesCurr.length - 1].begin; // последняя свеча текущего ТФ

  // Нормализуем текущий интервал
  const normCurr = normalizeWithScaler(candlesCurr, scalerCurr);
  const currWin = normCurr.slice(normCurr.length - WINDOW_SIZE);

  let h1Win = null;
  let h2Win = null;

  if (higher1Name && scalerH1) {
    const client2 = new Client({ connectionString: process.env.DB });
    await client2.connect();
    const candlesH1 = await loadCandles(client2, higher1Name, ticker);
    await client2.end();
    if (candlesH1.length >= WINDOW_SIZE + 1) {
      const normH1 = normalizeWithScaler(candlesH1, scalerH1);
      const idx = findLatestIdxBefore(candlesH1, targetTime);
      if (idx >= WINDOW_SIZE - 1) {
        h1Win = normH1.slice(idx - WINDOW_SIZE + 1, idx + 1);
      }
    }
  }

  if (higher2Name && scalerH2) {
    const client3 = new Client({ connectionString: process.env.DB });
    await client3.connect();
    const candlesH2 = await loadCandles(client3, higher2Name, ticker);
    await client3.end();
    if (candlesH2.length >= WINDOW_SIZE + 1) {
      const normH2 = normalizeWithScaler(candlesH2, scalerH2);
      const idx = findLatestIdxBefore(candlesH2, targetTime);
      if (idx >= WINDOW_SIZE - 1) {
        h2Win = normH2.slice(idx - WINDOW_SIZE + 1, idx + 1);
      }
    }
  }

  // Склеиваем вход: [curr, h1, h2] -> 900 x 5 (или меньше если нет старших)
  const parts = [currWin];
  if (h1Win) parts.push(h1Win);
  if (h2Win) parts.push(h2Win);

  const inputFlat = parts.flat(); // длина = steps * 5
  const steps = parts.reduce((a, b) => a + b.length, 0) / WINDOW_SIZE; // число окон

  const lastClose = candlesCurr[candlesCurr.length - 1].close;
  const lastBegin = candlesCurr[candlesCurr.length - 1].begin;

  const inputTensor = new ort.Tensor('float32', Float32Array.from(inputFlat), [1, Math.round(steps), 5]);
  const feeds = { input: inputTensor };
  const results = await session.run(feeds);
  const pUp = results.output.data[0] || 0;
  const pDown = 1 - pUp;

  const upTarget = (lastClose * (1 + THRESHOLD)).toFixed(2);
  const downTarget = (lastClose * (1 - THRESHOLD)).toFixed(2);

  console.log(`\n🔮 PREDICT ${ticker} @ ${interval}`);
  console.log(`   Last candle: ${lastBegin} close=${lastClose.toFixed(2)}`);
  console.log(`   Input steps: ${Math.round(steps)} × 5 (current${h1Win ? ` + ${higher1Name}` : ''}${h2Win ? ` + ${higher2Name}` : ''})`);
  console.log(`   P(up   ≥ +${(THRESHOLD * 100).toFixed(1)}% за ${HORIZON} свеч): ${(pUp * 100).toFixed(2)}%  (цель ≥ ${upTarget})`);
  console.log(`   P(down ≤ -${(THRESHOLD * 100).toFixed(1)}% за ${HORIZON} свеч): ${(pDown * 100).toFixed(2)}%  (цель ≤ ${downTarget})`);
  console.log(`   Direction: ${pUp >= 0.5 ? 'UP' : 'DOWN'}  Confidence: ${(Math.max(pUp, pDown) * 100).toFixed(2)}%`);
  console.log(`   Flat zone: цена не дошла бы ни до ${upTarget}, ни до ${downTarget}`);
}

main().catch(e => {
  console.error('❌ Error:', e.message);
  process.exit(1);
});
