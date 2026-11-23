const test = require('node:test');
const assert = require('node:assert/strict');

const thresholds = require('../thresholds');

test('прогрессивные пороги по умолчанию', () => {
  assert.equal(thresholds.effectiveThreshold(0.003, '1m'), 0.003);
  assert.equal(thresholds.effectiveThreshold(0.003, '10m'), 0.006);
  assert.equal(thresholds.effectiveThreshold(0.003, '60m'), 0.012);
  assert.equal(thresholds.effectiveThreshold(0.003, '1D'), 0.024);
});

test('fallback для нестандартных интервалов', () => {
  assert.equal(thresholds.intervalMultiplier('5m'), 2);
  assert.equal(thresholds.intervalMultiplier('15m'), 4);
  assert.equal(thresholds.intervalMultiplier('30m'), 4);
  assert.equal(thresholds.intervalMultiplier('2h'), 8);
  assert.equal(thresholds.intervalMultiplier('неизвестно'), 1);
  assert.equal(thresholds.safeEffectiveThreshold(0.003, 'неизвестно'), 0.003);
});

test('новый scaler: порог берётся из threshold', () => {
  const meta = thresholds.resolveModelThreshold({
    threshold: 0.024,
    threshold_base: 0.003,
    threshold_multiplier: 8,
    threshold_progressive: true,
    horizon: 10,
  }, '1D', 0.003);
  assert.equal(meta.effective, 0.024);
  assert.equal(meta.base, 0.003);
  assert.equal(meta.multiplier, 8);
  assert.equal(meta.horizon, 10);
  assert.equal(meta.progressive, true);
  assert.equal(meta.source, 'scaler');
});

test('кастомный множитель NLMK сохраняется', () => {
  const meta = thresholds.resolveModelThreshold({
    threshold: 0.02658459,
    threshold_base: 0.003,
    threshold_multiplier: 8.861531,
    horizon: 10,
  }, '1D', 0.003);
  assert.equal(meta.effective, 0.02658459);
  assert.equal(meta.base, 0.003);
  assert.ok(Math.abs(meta.multiplier - 8.861531) < 1e-9);
});

test('старый фиксированный scaler SBER не пересчитывается', () => {
  const meta = thresholds.resolveModelThreshold({ threshold: 0.003, horizon: 10 }, '1D', 0.003);
  assert.equal(meta.effective, 0.003);
  assert.equal(meta.base, 0.003);
  assert.equal(meta.multiplier, 1);
  assert.equal(meta.progressive, false);
});

test('старый прогрессивный scaler восстанавливает базу', () => {
  const meta = thresholds.resolveModelThreshold({ threshold: 0.024, horizon: 10 }, '1D', 0.003);
  assert.equal(meta.effective, 0.024);
  assert.equal(meta.base, 0.003);
  assert.equal(meta.multiplier, 8);
  assert.equal(meta.progressive, true);
});

test('отключённый прогрессивный режим даёт одинаковый порог', () => {
  const values = thresholds.thresholdsForIntervals(0.003, ['1m', '10m', '60m', '1D'], {
    progressive: false,
  });
  assert.deepEqual(values, { '1m': 0.003, '10m': 0.003, '60m': 0.003, '1D': 0.003 });
  assert.deepEqual(thresholds.multipliersForIntervals(['1m', '1D'], { progressive: false }), {
    '1m': 1,
    '1D': 1,
  });
});

test('validateInterval не пропускает мусор', () => {
  assert.equal(thresholds.validateInterval(' 10m '), '10m');
  assert.throws(() => thresholds.validateInterval('../../etc/passwd'), /Неизвестный интервал/);
  assert.throws(() => thresholds.validateInterval(''), /Неизвестный интервал/);
});
