const test = require('node:test');
const assert = require('node:assert/strict');

const dbSchema = require('../db_schema');
const tickers = require('../tickers');

function fakeClient() {
  return {
    queries: [],
    async query(sql, params) {
      this.queries.push({ sql: String(sql), params });
      return { rows: [] };
    },
  };
}

test('DDL свечей: BIGSERIAL id и уникальный ключ (ticker, interval, begin)', () => {
  const ddl = dbSchema.CANDLES_DDL;
  assert.match(ddl, /CREATE TABLE IF NOT EXISTS candles/);
  assert.match(ddl, /id\s+BIGSERIAL PRIMARY KEY/);
  assert.match(ddl, /UNIQUE \(ticker, interval, begin\)/);
  // Ключевые колонки должны быть NOT NULL: begin участвует в уникальном ключе.
  for (const column of ['ticker', 'interval', 'begin', 'open', 'close', 'high', 'low']) {
    assert.match(ddl, new RegExp(`${column}\\s+[^,\\n]+NOT NULL`), `нет NOT NULL у ${column}`);
  }
});

test('DDL тикеров идемпотентен', () => {
  assert.match(dbSchema.TICKERS_DDL, /CREATE TABLE IF NOT EXISTS tickers/);
  assert.match(dbSchema.TICKERS_DDL, /ticker\s+varchar\(32\) PRIMARY KEY/);
});

test('ensureCandlesTable и ensureTickersTable отправляют свой DDL', async () => {
  const client = fakeClient();
  await dbSchema.ensureTickersTable(client);
  await dbSchema.ensureCandlesTable(client);
  assert.equal(client.queries.length, 2);
  assert.equal(client.queries[0].sql.trim(), dbSchema.TICKERS_DDL.trim());
  assert.equal(client.queries[1].sql.trim(), dbSchema.CANDLES_DDL.trim());
});

test('ensureSchema создаёт tickers, candles и open_interest до сидирования (чистая база)', async () => {
  const client = fakeClient();
  await tickers.ensureSchema(client);
  const created = client.queries
    .filter((q) => /CREATE TABLE IF NOT EXISTS/.test(q.sql))
    .map((q) => {
      if (q.sql.includes('open_interest')) return 'open_interest';
      return q.sql.includes('candles') ? 'candles' : 'tickers';
    });
  assert.deepEqual(created, ['tickers', 'candles', 'open_interest']);
  // Сидирование идёт после CREATE TABLE, иначе INSERT упал бы на отсутствующую таблицу.
  const firstInsert = client.queries.findIndex((q) => q.sql.includes('INSERT INTO tickers'));
  const candlesCreate = client.queries.findIndex((q) => q.sql.includes('CREATE TABLE IF NOT EXISTS candles'));
  assert.ok(candlesCreate >= 0 && firstInsert > candlesCreate);
});

test('DDL open_interest: уникальный ключ (ticker, ts, clgroup)', () => {
  const ddl = dbSchema.OPEN_INTEREST_DDL;
  assert.match(ddl, /CREATE TABLE IF NOT EXISTS open_interest/);
  assert.match(ddl, /UNIQUE \(ticker, ts, clgroup\)/);
});
