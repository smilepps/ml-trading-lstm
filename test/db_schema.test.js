const test = require('node:test');
const assert = require('node:assert/strict');

const dbSchema = require('../db_schema');
const tickers = require('../tickers');

function fakeClient(options = {}) {
  const info = {
    table_oid: '123', relkind: 'r', relpersistence: 'p', relispartition: false,
    inherited: false, id_type: 'bigint', attnotnull: true, id_primary_key: true,
    attidentity: '', attgenerated: '', ownership: 'a', sequence_oid: '456',
    sequence_name: '"custom schema"."renamed ""sequence"""',
    id_default: "nextval('seq'::regclass)", expected_default: "nextval('seq'::regclass)",
    default_depends_on_sequence: true, sequence_type: 'bigint', seqstart: '1',
    seqmin: '1', seqmax: '9223372036854775807', seqincrement: '1',
    seqcache: '1', seqcycle: false, ...options.info,
  };
  return {
    queries: [], info,
    getTransactionStatus() { return options.transactionStatus || 'I'; },
    async query(sql, params) {
      sql = String(sql);
      this.queries.push({ sql, params });
      if (options.onQuery) await options.onQuery(sql, this);
      if (sql.includes('FROM pg_class c')) return { rows: options.noInfo ? [] : [{ ...info }] };
      if (sql.startsWith('ALTER SEQUENCE')) {
        info.sequence_type = 'bigint';
        info.seqmax = '9223372036854775807';
      }
      if (sql.startsWith('ALTER TABLE candles ALTER')) info.id_type = 'bigint';
      return { rows: [] };
    },
  };
}

function serialClient(options = {}) {
  return fakeClient({
    ...options,
    info: { id_type: 'integer', sequence_type: 'integer', seqmax: '2147483647', ...options.info },
  });
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

test('ensureCandlesTable проверяет BIGSERIAL без миграционной блокировки', async () => {
  const client = fakeClient();
  await dbSchema.ensureTickersTable(client);
  await dbSchema.ensureCandlesTable(client);
  assert.equal(client.queries.length, 3);
  assert.equal(client.queries[0].sql.trim(), dbSchema.TICKERS_DDL.trim());
  assert.equal(client.queries[1].sql.trim(), dbSchema.CANDLES_DDL.trim());
  assert.match(client.queries[2].sql, /FROM pg_class c/);
});

test('SERIAL мигрирует в одной транзакции с ограничением ожидания и реальным именем', async () => {
  const client = serialClient();
  await dbSchema.ensureCandlesTable(client);
  const sql = client.queries.map(q => q.sql.trim());
  const begin = sql.indexOf('BEGIN');
  const timeout = sql.indexOf("SET LOCAL lock_timeout = '5s'");
  const advisory = sql.findIndex(s => s.includes('pg_advisory_xact_lock'));
  const lock = sql.indexOf('LOCK TABLE candles IN ACCESS EXCLUSIVE MODE');
  const sequence = sql.indexOf('ALTER SEQUENCE "custom schema"."renamed ""sequence""" AS bigint');
  const table = sql.indexOf('ALTER TABLE candles ALTER COLUMN id TYPE bigint');
  const commit = sql.indexOf('COMMIT');
  assert.ok(begin > 0 && begin < timeout && timeout < advisory && advisory < lock &&
    lock < sequence && sequence < table && table < commit);
  assert.ok(sql.slice(lock + 1, sequence).some(s => s.includes('FROM pg_class c')));
  assert.ok(sql.slice(sequence + 1, table).some(s => s.includes('FROM pg_class c')));
  assert.ok(!sql.some(s => /\b(?:SELECT\s+nextval|setval)\s*\(/i.test(s)));
  await dbSchema.ensureCandlesTable(client);
  assert.equal(client.queries.filter(q => q.sql === 'BEGIN').length, 1);
});

test('после ожидания advisory lock готовая схема не блокируется повторно', async () => {
  const client = serialClient({ onQuery(sql, c) {
    if (sql.startsWith('SELECT pg_advisory')) {
      c.info.id_type = 'bigint';
      c.info.sequence_type = 'bigint';
      c.info.seqmax = '9223372036854775807';
    }
  } });
  await dbSchema.ensureCandlesTable(client);
  assert.equal(client.queries.at(-1).sql, 'COMMIT');
  assert.ok(!client.queries.some(q => q.sql.startsWith('LOCK TABLE') || q.sql.startsWith('ALTER ')));
});

test('ожидающий миграцию процесс повторно проверяет схему под блокировкой', async () => {
  const client = serialClient({ onQuery(sql, c) {
    if (sql.startsWith('LOCK TABLE')) {
      c.info.id_type = 'bigint';
      c.info.sequence_type = 'bigint';
      c.info.seqmax = '9223372036854775807';
    }
  } });
  await dbSchema.ensureCandlesTable(client);
  assert.equal(client.queries.at(-1).sql, 'COMMIT');
  assert.ok(!client.queries.some(q => q.sql.startsWith('ALTER ')));
});

test('нестандартная старая схема, границы и привязка явно отклоняются', async () => {
  for (const info of [
    { id_type: 'smallint' }, { sequence_type: 'bigint' }, { seqmin: '0' },
    { seqmax: '100' }, { seqincrement: '2' }, { seqstart: '2' },
    { seqcache: '2' }, { seqcycle: true }, { ownership: 'i' },
    { id_default: "nextval('other'::regclass)" }, { default_depends_on_sequence: false },
    { custom_default_function: true }, { id_primary_key: false }, { inherited: true },
  ]) {
    const client = serialClient({ info });
    await assert.rejects(dbSchema.ensureCandlesTable(client), /Миграция candles\.id/);
    assert.ok(!client.queries.some(q => q.sql === 'BEGIN' || q.sql.startsWith('ALTER ')));
  }
  await assert.rejects(dbSchema.ensureCandlesTable(fakeClient({ noInfo: true })), /последовательностью/);
  const limitedBigint = fakeClient({ info: { seqmax: '2147483647' } });
  await assert.rejects(dbSchema.ensureCandlesTable(limitedBigint), /стандартными границами/);
  assert.ok(!limitedBigint.queries.some(q => q.sql === 'BEGIN'));
});

test('внешняя транзакция не завершается миграцией', async () => {
  const client = serialClient({ transactionStatus: 'T' });
  await assert.rejects(dbSchema.ensureCandlesTable(client), /вне внешней транзакции/);
  assert.ok(!client.queries.some(q => /^(BEGIN|COMMIT|ROLLBACK)$/.test(q.sql)));
  const unknown = serialClient();
  delete unknown.getTransactionStatus;
  await assert.rejects(dbSchema.ensureCandlesTable(unknown), /известным состоянием/);
  assert.ok(!unknown.queries.some(q => /^(BEGIN|COMMIT|ROLLBACK)$/.test(q.sql)));
});

test('таймаут, недостаток прав и ошибка ALTER приводят к ROLLBACK', async () => {
  for (const [prefix, message] of [
    ['SELECT pg_advisory', 'lock timeout'],
    ['LOCK TABLE', 'permission denied'],
    ['ALTER SEQUENCE', 'must be owner'],
    ['ALTER TABLE candles ALTER', 'cannot alter'],
  ]) {
    const client = serialClient({ onQuery(sql) {
      if (sql.startsWith(prefix)) throw new Error(message);
    } });
    await assert.rejects(dbSchema.ensureCandlesTable(client), new RegExp(message));
    assert.equal(client.queries.at(-1).sql, 'ROLLBACK');
    assert.ok(!client.queries.some(q => q.sql === 'COMMIT'));
  }
});

test('параллельное изменение последовательности обнаруживается до ALTER TABLE', async () => {
  const client = serialClient({ onQuery(sql, c) {
    if (sql.startsWith('ALTER SEQUENCE')) c.info.seqincrement = '2';
  } });
  await assert.rejects(dbSchema.ensureCandlesTable(client), /изменились во время миграции/);
  assert.equal(client.queries.at(-1).sql, 'ROLLBACK');
  assert.ok(!client.queries.some(q => q.sql.startsWith('ALTER TABLE candles ALTER')));
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
