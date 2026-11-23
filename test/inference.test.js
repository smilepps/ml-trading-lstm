const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const inf = require('../inference_v4');

function candle(begin, close = 100, high = close + 1, low = close - 1) {
  const d = new Date(begin);
  return {
    begin: d,
    beginMs: d.getTime(),
    open: close,
    high,
    low,
    close,
    value: 1000,
    dow: 3,
    hour: d.getHours(),
  };
}

test('historyLimitFor масштабируется вместе с окном', () => {
  assert.equal(inf.historyLimitFor(300), 600);
  assert.equal(inf.historyLimitFor(500), 1000);
  assert.equal(inf.historyLimitFor(0), 600);
});

test('normalizeCandles считает 7 признаков', () => {
  const scaler = {
    ret_mean: 0,
    ret_std: 1,
    value_mean: 0,
    value_std: 1,
    dow_mean: 4,
    dow_std: 1,
    hour_mean: 12,
    hour_std: 1,
  };
  const rows = inf.normalizeCandles([
    candle('2026-01-05T09:00:00', 100),
    candle('2026-01-05T10:00:00', 101, 102, 99),
  ], scaler);

  assert.equal(rows.length, 1);
  assert.equal(rows[0].length, 7);
  assert.ok(Math.abs(rows[0][0] - Math.log(1.01)) < 1e-12);
  assert.ok(Math.abs(rows[0][1] - 0.02) < 1e-12);
  assert.ok(Math.abs(rows[0][2] - (-0.01)) < 1e-12);
  assert.ok(Math.abs(rows[0][3] - 0.01) < 1e-12);
  assert.ok(Math.abs(rows[0][4] - Math.log1p(1000)) < 1e-12);
  assert.equal(rows[0][5], -1);
  assert.equal(rows[0][6], 10 - 12);
});

test('normalizeCandles не пропускает битые свечи', () => {
  const scaler = {
    ret_mean: 0, ret_std: 1, value_mean: 0, value_std: 1,
    dow_mean: 4, dow_std: 1, hour_mean: 12, hour_std: 1,
  };
  assert.throws(() => inf.normalizeCandles([
    candle('2026-01-05T09:00:00', 100),
    candle('2026-01-05T10:00:00', 101, 99, 102),
  ], scaler), /high=.*меньше low/);
  assert.throws(() => inf.normalizeCandles([
    candle('2026-01-05T09:00:00', 100),
    { ...candle('2026-01-05T10:00:00', 101), close: NaN },
  ], scaler), /Некорректная цена/);
  assert.throws(() => inf.normalizeCandles([
    candle('2026-01-05T09:00:00', 100),
    candle('2026-01-05T10:00:00', 101),
  ], {}), /Скейлер не содержит/);
});

test('contextWindow выравнивает по закрытию base-свечи', () => {
  const base = [candle('2026-01-05T10:00:00', 100)];
  const ctxCandles = [];
  const ctxNormalized = [];
  for (let minute = 9 * 60; minute <= 10 * 60 + 50; minute += 10) {
    const d = new Date(2026, 0, 5, 0, minute);
    ctxCandles.push(candle(d, 100));
    // Строка k соответствует свече k+1, поэтому значение = минуты этой свечи.
    if (ctxCandles.length > 1) ctxNormalized.push([minute]);
  }

  const win = inf.contextWindow(base, '60m', ctxCandles, ctxNormalized, '10m', 3);
  assert.deepEqual(win.map((row) => row[0]), [650, 640, 630]);
});

test('contextWindow возвращает null при нехватке истории', () => {
  const base = [candle('2026-01-05T10:00:00', 100)];
  const ctxCandles = [
    candle('2026-01-05T10:30:00', 100),
    candle('2026-01-05T10:40:00', 100),
    candle('2026-01-05T10:50:00', 100),
  ];
  const ctxNormalized = [[1], [2]];
  assert.equal(inf.contextWindow(base, '60m', ctxCandles, ctxNormalized, '10m', 3), null);
});

test('attachOi: as-of join к закрытию свечи, без заглядывания вперёд', () => {
  const oi = {
    ts: [
      new Date('2026-01-05T10:05:00').getTime(),
      new Date('2026-01-05T10:10:00').getTime(),
    ],
    yur: [-100, -200],
    fiz: [100, 200],
  };
  const candles = [
    candle('2026-01-05T09:50:00'), // close 10:00 — до первого снимка
    candle('2026-01-05T10:00:00'), // close 10:10 — снимок 10:10
    candle('2026-01-05T10:10:00'), // close 10:20 — forward-fill снимка 10:10
  ];
  inf.attachOi(candles, '10m', oi);
  assert.equal(candles[0].yurNet, null);
  assert.equal(candles[0].fizNet, null);
  assert.equal(candles[1].yurNet, -200);
  assert.equal(candles[1].fizNet, 200);
  assert.equal(candles[2].yurNet, -200);
  assert.equal(candles[2].fizNet, 200);
});

test('attachOi: снимок строго после закрытия свечи не используется', () => {
  const oi = {
    ts: [new Date('2026-01-05T10:15:00').getTime()],
    yur: [-100],
    fiz: [100],
  };
  const candles = [candle('2026-01-05T10:00:00')]; // close 10:10 < 10:15
  inf.attachOi(candles, '10m', oi);
  assert.equal(candles[0].yurNet, null);
});

test('attachOi без данных проставляет null', () => {
  const candles = [candle('2026-01-05T10:00:00')];
  inf.attachOi(candles, '10m', null);
  assert.equal(candles[0].yurNet, null);
  assert.equal(candles[0].fizNet, null);
});

test('normalizeCandles с OI-скейлером: 9 признаков, без OI — нейтральный 0', () => {
  const scaler = {
    ret_mean: 0, ret_std: 1, value_mean: 0, value_std: 1,
    dow_mean: 4, dow_std: 1, hour_mean: 12, hour_std: 1,
    yur_net_mean: -150, yur_net_std: 50,
    fiz_net_mean: 150, fiz_net_std: 50,
  };
  const withOi = { ...candle('2026-01-05T10:00:00', 101), yurNet: -200, fizNet: 200 };
  const withoutOi = { ...candle('2026-01-05T11:00:00', 102), yurNet: null, fizNet: null };
  const rows = inf.normalizeCandles([
    candle('2026-01-05T09:00:00', 100),
    withOi,
    withoutOi,
  ], scaler);

  assert.equal(rows.length, 2);
  assert.equal(rows[0].length, 9);
  assert.equal(rows[0][7], (-200 - -150) / 50);
  assert.equal(rows[0][8], (200 - 150) / 50);
  assert.equal(rows[1][7], 0);
  assert.equal(rows[1][8], 0);
});

function scaler(interval, overrides = {}) {
  return {
    ticker: 'SBER', interval, version: 4, window: 300, order: 'current_to_history',
    contexts: [...inf.CONTEXT_MAP[interval]], inputs: 1 + inf.CONTEXT_MAP[interval].length,
    input_size: 7, feature_names: [...inf.BASE_FEATURE_NAMES],
    ret_mean: 0, ret_std: 1, value_mean: 0, value_std: 1,
    dow_mean: 0, dow_std: 1, hour_mean: 0, hour_std: 1,
    ...overrides,
  };
}

test('контракт скейлера: точные контексты, порядок, признаки и статистики', () => {
  for (const interval of Object.keys(inf.CONTEXT_MAP)) {
    assert.doesNotThrow(() => inf.validateScaler(scaler(interval), 'SBER', interval));
  }
  for (const contexts of [['60m', '10m'], ['15m', '60m'], ['10m', '10m'], [], ['10m', '60m', '1D']]) {
    assert.throws(() => inf.validateScaler(scaler('1m', { contexts }), 'SBER', '1m'), /контекст/);
  }
  for (const order of [undefined, 'history_to_current']) {
    assert.throws(() => inf.validateScaler(scaler('1D', { order }), 'SBER', '1D'), /current_to_history/);
  }
  assert.throws(() => inf.validateScaler(scaler('1D', { feature_names: [...inf.BASE_FEATURE_NAMES].reverse() }), 'SBER', '1D'), /feature_names/);
  assert.throws(() => inf.validateScaler(scaler('1D', { ret_mean: null }), 'SBER', '1D'), /ret_mean/);
  assert.throws(() => inf.validateScaler(scaler('1D'), 'SBER', '15m'), /не поддерживается/);
  const oi = scaler('1D', {
    input_size: 9, feature_names: [...inf.BASE_FEATURE_NAMES, ...inf.OI_FEATURE_NAMES],
    yur_net_mean: 0, yur_net_std: 1, fiz_net_mean: 0, fiz_net_std: 1,
  });
  assert.doesNotThrow(() => inf.validateScaler(oi, 'SBER', '1D'));
  assert.throws(() => inf.validateContextScaler(oi, scaler('60m'), 'SBER', '1D', 'Контекст'), /feature_names/);
  assert.throws(() => inf.validateContextScaler(scaler('1D', { window: 301 }), scaler('60m'), 'SBER', '1D', 'Контекст'), /окно/);
});

function isolatedInference(createSession) {
  const source = fs.readFileSync(path.join(__dirname, '..', 'inference_v4.js'), 'utf8');
  const root = path.join(__dirname, 'virtual-inference');
  const files = new Map();
  let revision = 0;
  const fakeFs = {
    existsSync: (name) => files.has(name),
    readFileSync: (name) => { if (!files.has(name)) throw new Error('ENOENT'); return files.get(name).data; },
    statSync: (name) => { if (!files.has(name)) throw new Error('ENOENT'); return files.get(name).stat; },
  };
  const put = (name, data) => {
    revision++;
    files.set(path.join(root, 'models', name), { data, stat: { mtimeMs: revision, ctimeMs: revision, ino: revision, size: data.length } });
  };
  const module = { exports: {} };
  vm.runInNewContext(source, {
    module, __dirname: root, Float32Array,
    require(name) {
      if (name === 'fs') return fakeFs;
      if (name === 'path') return path;
      if (name === 'onnxruntime-node') return { InferenceSession: { create: createSession } };
      if (name === './thresholds') return require('../thresholds');
      throw new Error(name);
    },
  });
  return { inf: module.exports, put };
}

test('predictWithModel подаёт current-to-history в именованные входы зарегистрированных контекстов', async () => {
  let feeds;
  const session = {
    inputNames: ['input_h2', 'input_base', 'input_h1'], outputNames: ['output'],
    async run(value) { feeds = value; return { output: { data: [0.7] } }; },
  };
  const baseTime = Date.parse('2026-01-05T10:00:00Z');
  const client = { async query(_sql, params) {
    const step = { '1m': 60_000, '10m': 600_000, '60m': 3_600_000 }[params[1]];
    const last = params[1] === '1m' ? baseTime : baseTime - step;
    return { rows: Array.from({ length: 4 }, (_, i) => ({
      begin: new Date(last - (3 - i) * step), open: 100 + i, high: 101 + i,
      low: 99 + i, close: 100 + i, value: 1000, dow: 1, hour: 10,
    })) };
  } };
  const model = { session, scaler: scaler('1m', { window: 2 }), contexts: ['10m', '60m'],
    contextScalers: [scaler('10m', { window: 2 }), scaler('60m', { window: 2 })],
    window: 2, inputSize: 7, usesOi: false };
  await inf.predictWithModel(client, model, 'SBER', '1m');
  for (const name of ['input_base', 'input_h1', 'input_h2']) {
    assert.deepEqual(feeds[name].dims, [1, 2, 7]);
    assert.ok(Math.abs(feeds[name].data[0] - Math.log(103 / 102)) < 1e-6);
    assert.ok(Math.abs(feeds[name].data[7] - Math.log(102 / 101)) < 1e-6);
  }
});

test('loadModel проверяет имена входов и замену файлов во время загрузки', async () => {
  let names = ['input_base', 'input_h1', 'input_h2'];
  let duringLoad = () => {};
  const env = isolatedInference(async () => { duringLoad(); return { inputNames: names }; });
  for (const interval of ['1m', '10m', '60m']) env.put(`scaler_SBER_${interval}.json`, JSON.stringify(scaler(interval)));
  env.put('model_SBER_1m.onnx', 'fake model');
  const model = await env.inf.loadModel('SBER', '1m');
  assert.equal(env.inf.isModelFresh(model), true);
  env.put('scaler_SBER_10m.json', JSON.stringify(scaler('10m')));
  assert.equal(env.inf.isModelFresh(model), false);
  names = ['input_base', 'input_h2', 'other'];
  await assert.rejects(env.inf.loadModel('SBER', '1m'), /входы.*input_h1/);
  names = ['input_base', 'input_h1', 'input_h2'];
  duringLoad = () => env.put('scaler_SBER_10m.json', JSON.stringify(scaler('10m')));
  await assert.rejects(env.inf.loadModel('SBER', '1m'), /изменились во время загрузки/);
});
