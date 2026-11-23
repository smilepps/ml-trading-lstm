/**
 * train.js — Обучение Brain.js на подготовленных данных
 * brain.js LSTM в этой версии принимает только символьные последовательности,
 * поэтому используется feedforward NeuralNetwork с плоским окном WINDOW_SIZE x 5 = 1500 входов.
 */

require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');
const brain = require('brain.js');

const TICKER = process.env.TICKER || 'IMOEX';
const INTERVALS = process.env.INTERVALS ? process.env.INTERVALS.split(',').map(s => s.trim()) : ['60m', '10m', '1D'];
const WINDOW_SIZE = 300;
const HORIZON = 10;
const THRESHOLD = 0.003;
const TRAIN_RATIO = 0.7;
const VAL_RATIO = 0.15;
const TEST_RATIO = 0.15;

const HIDDEN_LAYERS = [64, 32];
const LEARNING_RATE = 0.001;
const EPOCHS = 100;
const BATCH_SIZE = 32;
const PATIENCE = 10;

const OUT_DIR = path.join(__dirname, 'models');
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

function normalizeCandles(candles) {
  const n = candles.length;
  const logReturns = new Array(n - 1);
  const volumes = new Array(n);

  for (let i = 0; i < n; i++) volumes[i] = candles[i].volume || 0;

  for (let i = 1; i < n; i++) {
    logReturns[i - 1] = Math.log(candles[i].close / candles[i - 1].close);
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

  return { data: normalized, scaler: { priceMean, priceStd, volMean, volStd } };
}

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
    if (maxHigh >= upTarget && minLow <= downTarget) label = -1;
    else if (maxHigh >= upTarget) label = 1;
    else if (minLow <= downTarget) label = 0;
    else label = -1;

    if (label !== -1) {
      X.push(window.flat()); // 1500-мерный вектор
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

function createModel() {
  return new brain.NeuralNetwork({
    hiddenLayers: HIDDEN_LAYERS,
    learningRate: LEARNING_RATE,
    activation: 'sigmoid'
  });
}

function trainModel(net, trainData, valData) {
  console.log(`   Training on ${trainData.length} samples, validating on ${valData.length}...`);

  const trainFormatted = trainData.map(d => ({ input: d.input, output: [d.output] }));
  const valFormatted = valData.map(d => ({ input: d.input, output: [d.output] }));

  let bestValLoss = Infinity;
  let patienceCounter = 0;
  let bestModel = null;

  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    const trainResult = net.train(trainFormatted, {
      iterations: 1,
      errorThresh: 0.005,
      log: false
    });

    let valError = 0;
    let valCorrect = 0;
    for (const { input, output } of valFormatted) {
      const pred = net.run(input);
      const predLabel = pred[0] > 0.5 ? 1 : 0;
      valError += Math.abs(pred[0] - output[0]);
      if (predLabel === output[0]) valCorrect++;
    }
    valError /= valFormatted.length;
    const valAcc = valCorrect / valFormatted.length;

    if (epoch % 5 === 0 || epoch === 1) {
      console.log(`   Epoch ${epoch}: train loss=${trainResult.error.toFixed(4)}, val loss=${valError.toFixed(4)}, val acc=${(valAcc * 100).toFixed(1)}%`);
    }

    if (valError < bestValLoss) {
      bestValLoss = valError;
      patienceCounter = 0;
      bestModel = net.toJSON();
    } else {
      patienceCounter++;
      if (patienceCounter >= PATIENCE) {
        console.log(`   Early stopping at epoch ${epoch} (best val loss: ${bestValLoss.toFixed(4)})`);
        break;
      }
    }
  }

  if (bestModel) net.fromJSON(bestModel);
  return net;
}

function evaluateModel(net, testData) {
  let correct = 0;
  let tp = 0, fp = 0, tn = 0, fn = 0;

  for (const { input, output } of testData) {
    const pred = net.run(input);
    const predLabel = pred[0] > 0.5 ? 1 : 0;
    const trueLabel = output;

    if (predLabel === trueLabel) correct++;
    if (predLabel === 1 && trueLabel === 1) tp++;
    if (predLabel === 1 && trueLabel === 0) fp++;
    if (predLabel === 0 && trueLabel === 0) tn++;
    if (predLabel === 0 && trueLabel === 1) fn++;
  }

  const accuracy = correct / testData.length;
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1 = 2 * precision * recall / (precision + recall) || 0;

  return { accuracy, precision, recall, f1, tp, fp, tn, fn };
}

async function main() {
  const client = new Client({ connectionString: process.env.DB });
  await client.connect();
  console.log('✅ Connected to PostgreSQL');

  for (const interval of INTERVALS) {
    console.log(`\n📊 ===== ${interval} =====`);
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

    const trainData = X.slice(0, trainEnd).map((input, i) => ({ input, output: y[i] }));
    const valData = X.slice(trainEnd, valEnd).map((input, i) => ({ input, output: y[trainEnd + i] }));
    const testData = X.slice(valEnd).map((input, i) => ({ input, output: y[valEnd + i] }));

    console.log(`   Split: train=${trainData.length}, val=${valData.length}, test=${testData.length}`);

    const net = createModel();
    console.log(`   Model: Feedforward ${[X[0].length, ...HIDDEN_LAYERS, 1].join(' → ')}`);
    const trainedNet = trainModel(net, trainData, valData);

    const metrics = evaluateModel(trainedNet, testData);
    console.log(`\n   📈 Test Metrics:`);
    console.log(`      Accuracy:  ${(metrics.accuracy * 100).toFixed(2)}%`);
    console.log(`      Precision: ${(metrics.precision * 100).toFixed(2)}%`);
    console.log(`      Recall:    ${(metrics.recall * 100).toFixed(2)}%`);
    console.log(`      F1:        ${(metrics.f1 * 100).toFixed(2)}%`);
    console.log(`      TP=${metrics.tp}, FP=${metrics.fp}, TN=${metrics.tn}, FN=${metrics.fn}`);

    const modelPath = path.join(OUT_DIR, `model_${interval}.json`);
    const scalerPath = path.join(OUT_DIR, `scaler_${interval}.json`);

    fs.writeFileSync(modelPath, JSON.stringify(trainedNet.toJSON(), null, 2));
    fs.writeFileSync(scalerPath, JSON.stringify(scaler, null, 2));

    console.log(`   💾 Model saved: ${modelPath}`);
    console.log(`   💾 Scaler saved: ${scalerPath}`);
  }

  await client.end();
  console.log('\n✅ All intervals trained');
}

main().catch(e => {
  console.error('❌ Error:', e);
  process.exit(1);
});
