/**
 * server.js — Express веб-сервер с дашбордом для ONNX LSTM прогнозов
 * Обучение: PyTorch CUDA → ONNX. Инференс: onnxruntime-node.
 * Мультитаймфрейм: 300 (текущий) + 300 (старший #1) + 300 (старший #2) = 900 свечей × 5 признаков.
 */

require('dotenv').config();
const express = require('express');
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const ort = require('onnxruntime-node');

const app = express();
const server = http.createServer(app);

const PORT = process.env.PORT || 3000;
const TICKER = process.env.TICKER || 'IMOEX';
const WINDOW_SIZE = 300;
const HORIZON = 10;
const THRESHOLD = parseFloat(process.env.THRESHOLD || '0.003');
const OUT_DIR = path.join(__dirname, 'models');

const DEFAULT_INTERVALS = ['60m', '10m', '4h', '1D', '2h'];
const INTERVALS = process.env.INTERVALS
  ? process.env.INTERVALS.split(',').map(s => s.trim())
  : DEFAULT_INTERVALS;

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

// ── Состояние ──
const sessions = {};
const scalers = {};
const predictions = {};
let monitorInterval = null;
let monitorMs = parseInt(process.env.SLEEP_MS || '60000', 10);
let monitorEnabled = false;
let monitorClient = null;
let sseClients = [];

// ── Вспомогательные ──
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

function nowStr() {
  return new Date().toISOString().replace('T', ' ').substring(0, 19);
}

function findLatestIdxBefore(candles, targetTime) {
  let idx = -1;
  for (let i = 0; i < candles.length; i++) {
    if (new Date(candles[i].begin) <= new Date(targetTime)) idx = i;
    else break;
  }
  return idx;
}

async function loadModel(interval) {
  const modelPath = path.join(OUT_DIR, `model_${interval}.onnx`);
  const scalerPath = path.join(OUT_DIR, `scaler_${interval}.json`);
  if (!fs.existsSync(modelPath) || !fs.existsSync(scalerPath)) return null;
  const session = await ort.InferenceSession.create(modelPath);
  const scaler = JSON.parse(fs.readFileSync(scalerPath, 'utf8'));
  return { session, scaler };
}

async function loadCandles(client, interval) {
  const result = await client.query(
    `SELECT begin, open, high, low, close, volume FROM candles WHERE ticker = $1 AND interval = $2 ORDER BY begin ASC`,
    [TICKER, interval]
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

async function buildInput(client, interval, scalerData) {
  const candlesCurr = await loadCandles(client, interval);
  if (candlesCurr.length < WINDOW_SIZE + 1) return null;

  const targetTime = candlesCurr[candlesCurr.length - 1].begin;
  const normCurr = normalizeWithScaler(candlesCurr, scalerData.curr);
  const currWin = normCurr.slice(normCurr.length - WINDOW_SIZE);

  let h1Win = null, h2Win = null;

  if (scalerData.higher1_name && scalerData.higher1) {
    const candlesH1 = await loadCandles(client, scalerData.higher1_name);
    if (candlesH1.length >= WINDOW_SIZE + 1) {
      const normH1 = normalizeWithScaler(candlesH1, scalerData.higher1);
      const idx = findLatestIdxBefore(candlesH1, targetTime);
      if (idx >= WINDOW_SIZE - 1) h1Win = normH1.slice(idx - WINDOW_SIZE + 1, idx + 1);
    }
  }

  if (scalerData.higher2_name && scalerData.higher2) {
    const candlesH2 = await loadCandles(client, scalerData.higher2_name);
    if (candlesH2.length >= WINDOW_SIZE + 1) {
      const normH2 = normalizeWithScaler(candlesH2, scalerData.higher2);
      const idx = findLatestIdxBefore(candlesH2, targetTime);
      if (idx >= WINDOW_SIZE - 1) h2Win = normH2.slice(idx - WINDOW_SIZE + 1, idx + 1);
    }
  }

  const parts = [currWin];
  if (h1Win) parts.push(h1Win);
  if (h2Win) parts.push(h2Win);
  const inputFlat = parts.flat();
  const steps = parts.reduce((a, b) => a + b.length, 0) / WINDOW_SIZE;

  return {
    inputTensor: new ort.Tensor('float32', Float32Array.from(inputFlat), [1, Math.round(steps), 5]),
    lastClose: candlesCurr[candlesCurr.length - 1].close,
    lastBegin: candlesCurr[candlesCurr.length - 1].begin,
    steps: Math.round(steps),
    hasH1: !!h1Win,
    hasH2: !!h2Win,
  };
}

async function predictInterval(client, interval) {
  const m = sessions[interval];
  if (!m) {
    predictions[interval] = { error: 'Модель не обучена' };
    return;
  }
  try {
    const built = await buildInput(client, interval, m.scaler);
    if (!built) {
      predictions[interval] = { error: 'Недостаточно данных' };
      return;
    }

    const feeds = { input: built.inputTensor };
    const results = await m.session.run(feeds);
    const pUp = results.output.data[0] || 0;
    const pDown = 1 - pUp;
    const direction = pUp >= 0.5 ? 'UP' : 'DOWN';

    const TH = THRESHOLD;
    predictions[interval] = {
      interval,
      pUp, pDown,
      direction,
      confidence: Math.max(pUp, pDown),
      lastClose: built.lastClose,
      lastBegin: built.lastBegin,
      steps: built.steps,
      hasH1: built.hasH1,
      hasH2: built.hasH2,
      updatedAt: new Date().toISOString()
    };
    broadcast({ type: 'prediction', interval, data: predictions[interval] });
  } catch (e) {
    predictions[interval] = { error: e.message };
  }
}

function broadcast(msg) {
  const data = `data: ${JSON.stringify(msg)}\n\n`;
  sseClients.forEach(c => {
    try { c.write(data); } catch (e) { /* ignore */ }
  });
}

async function initModels() {
  for (const interval of INTERVALS) {
    const m = await loadModel(interval);
    if (m) {
      sessions[interval] = m.session;
      scalers[interval] = m.scaler;
      console.log(`[${nowStr()}] ✅ Model loaded: ${interval}`);
    } else {
      console.log(`[${nowStr()}] ⚠️  Model missing: ${interval}`);
    }
  }
}

// ── Express ──
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

app.get('/api/intervals', (req, res) => {
  res.json(INTERVALS.map(interval => ({
    interval,
    hasModel: !!sessions[interval],
    prediction: predictions[interval] || null
  })));
});

app.get('/api/predict', async (req, res) => {
  const client = new Client({ connectionString: process.env.DB });
  await client.connect();
  const intervals = req.query.intervals ? req.query.intervals.split(',') : INTERVALS;
  await Promise.all(intervals.map(i => predictInterval(client, i)));
  await client.end();
  res.json(intervals.map(i => predictions[i] || { error: 'no data' }));
});

app.get('/api/predict/:interval', async (req, res) => {
  const client = new Client({ connectionString: process.env.DB });
  await client.connect();
  await predictInterval(client, req.params.interval);
  await client.end();
  res.json(predictions[req.params.interval] || { error: 'no data' });
});

app.post('/api/train', (req, res) => {
  const { intervals = INTERVALS.join(','), threshold = THRESHOLD } = req.body;
  const env = {
    ...process.env,
    INTERVALS: intervals,
    THRESHOLD: String(threshold)
  };
  const child = spawn('python', ['train_gpu.py', '--intervals', intervals, '--threshold', String(threshold)],
    { env, cwd: __dirname, stdio: ['ignore', 'pipe', 'pipe'] });
  let log = '';
  child.stdout.on('data', d => { log += d.toString(); });
  child.stderr.on('data', d => { log += d.toString(); });
  child.on('close', async code => {
    for (const interval of intervals.split(',')) {
      const m = await loadModel(interval);
      if (m) {
        sessions[interval] = m.session;
        scalers[interval] = m.scaler;
      }
    }
    res.json({ code, log });
  });
  res.json({ started: true, pid: child.pid, intervals, threshold });
});

app.post('/api/monitor/start', (req, res) => {
  if (monitorInterval) return res.json({ already: true });
  monitorEnabled = true;
  monitorClient = new Client({ connectionString: process.env.DB });
  monitorClient.connect().then(() => {
    monitorInterval = setInterval(async () => {
      for (const interval of INTERVALS) {
        if (sessions[interval]) await predictInterval(monitorClient, interval);
      }
    }, monitorMs);
    for (const interval of INTERVALS) {
      if (sessions[interval]) predictInterval(monitorClient, interval);
    }
    res.json({ started: true, sleepMs: monitorMs, threshold: THRESHOLD });
  }).catch(e => {
    res.status(500).json({ error: e.message });
  });
});

app.post('/api/monitor/stop', (req, res) => {
  if (monitorInterval) clearInterval(monitorInterval);
  monitorInterval = null;
  monitorEnabled = false;
  if (monitorClient) { monitorClient.end(); monitorClient = null; }
  res.json({ stopped: true });
});

app.get('/api/monitor/status', (req, res) => {
  res.json({ enabled: monitorEnabled, running: !!monitorInterval, sleepMs: monitorMs, threshold: THRESHOLD });
});

app.get('/stream', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });
  sseClients.push(res);
  res.write(`data: ${JSON.stringify({ type: 'hello', ts: Date.now() })}\n\n`);
  req.on('close', () => {
    sseClients = sseClients.filter(c => c !== res);
  });
});

initModels().then(() => {
  server.listen(PORT, () => {
    console.log(`🖥️  Dashboard server: http://localhost:${PORT}`);
    console.log(`   Ticker: ${TICKER} | Intervals: ${INTERVALS.join(', ')}`);
    console.log(`   Threshold: ${THRESHOLD} | Backend: PyTorch CUDA + ONNX Runtime`);
  });
}).catch(e => {
  console.error('Failed to init models:', e);
  process.exit(1);
});
