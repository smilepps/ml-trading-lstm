const test = require('node:test');
const assert = require('node:assert/strict');
const { getSequenceStatus } = require('../sequence_status');

function client(row) {
  return { query: async () => ({ rows: row ? [row] : [] }) };
}

function sequence(overrides = {}) {
  return {
    name: 'public.candles_id_seq',
    start_value: '1',
    increment_by: '1',
    max_value: '9223372036854775807',
    last_value: '2147483647',
    cycle: false,
    accessible: true,
    ...overrides,
  };
}

test('остаток bigint передаётся точно, а не числом JavaScript', async () => {
  const result = await getSequenceStatus(client(sequence()));
  assert.equal(result.maxValue, '9223372036854775807');
  assert.equal(result.lastValue, '2147483647');
  assert.equal(result.remaining, '9223372034707292160');
  assert.equal(result.warning, null);
});

test('неиспользованная sequence учитывает первое значение', async () => {
  const result = await getSequenceStatus(client(sequence({ last_value: null, max_value: '10' })));
  assert.equal(result.remaining, '10');
  assert.equal(result.usedPercent, 0);
});

test('предупреждает об исчерпании и циклической sequence', async () => {
  const nearEnd = await getSequenceStatus(client(sequence({ max_value: '100', last_value: '95' })));
  assert.match(nearEnd.warning, /почти исчерпана/);
  const cycling = await getSequenceStatus(client(sequence({ cycle: true })));
  assert.match(cycling.warning, /зациклена/);
});

test('ошибки доступа и отсутствующая sequence отображаются явно', async () => {
  const denied = await getSequenceStatus(client(sequence({ accessible: false, last_value: null })));
  assert.match(denied.warning, /Нет доступа/);
  const missing = await getSequenceStatus(client(null));
  assert.match(missing.warning, /не найдена/);
  const failed = await getSequenceStatus({ query: async () => { throw new Error('permission denied'); } });
  assert.match(failed.warning, /permission denied/);
});
