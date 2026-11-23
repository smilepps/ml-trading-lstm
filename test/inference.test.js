const test = require('node:test');
const assert = require('node:assert/strict');

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
