// Opt-in: CANDLES_RUN_DOCKER_TESTS=1 node --test test/postgres.integration.test.js
// The fixture starts its own disposable PostgreSQL with no network or volumes.
const test = require('node:test');
const assert = require('node:assert/strict');

if (process.env.CANDLES_DOCKER_INNER !== '1') {
  test('PostgreSQL integration in an isolated Docker network namespace', {
    skip: process.env.CANDLES_RUN_DOCKER_TESTS !== '1', timeout: 120_000,
  }, async () => {
    const { execFile } = require('node:child_process');
    const { promisify } = require('node:util');
    const path = require('node:path');
    const exec = promisify(execFile);
    const root = path.resolve(__dirname, '..');
    const name = `candles-test-${process.pid}-${Date.now()}`;
    let started = false;
    try {
      await exec('docker', ['run', '-d', '--pull=never', '--name', name,
        '--network=none', '-e', 'POSTGRES_HOST_AUTH_METHOD=trust', 'postgres:16-alpine']);
      started = true;
      let ready = false;
      for (let i = 0; i < 150; i++) {
        try {
          await exec('docker', ['exec', name, 'pg_isready', '-U', 'postgres']);
          ready = true;
          break;
        } catch (_) {
          await new Promise(resolve => setTimeout(resolve, 200));
        }
      }
      assert.ok(ready, 'isolated PostgreSQL did not start');
      const args = ['run', '--rm', '--pull=never', '--network', `container:${name}`,
        '-e', 'CANDLES_DOCKER_INNER=1', '-w', '/work/test'];
      for (const file of ['db_schema.js', 'tickers.js', 'refresh_data.js', 'thresholds.js',
        'test/postgres.integration.test.js', 'node_modules']) {
        args.push('--mount', `type=bind,source=${path.join(root, file)},target=/work/${file},readonly`);
      }
      args.push('node:26-slim', 'node', '--test', '/work/test/postgres.integration.test.js');
      const result = await exec('docker', args, { timeout: 90_000, maxBuffer: 1024 * 1024 });
      process.stdout.write(result.stdout);
      process.stderr.write(result.stderr);
    } finally {
      if (started) await exec('docker', ['rm', '-f', name]);
    }
  });
} else {
  const fs = require('node:fs');
  const { Client } = require('pg');
  const schema = require('../db_schema');
  const tickers = require('../tickers');
  const refresh = require('../refresh_data');

  // Never accept a database URL from the environment. The child is launched in
  // the private, network-less namespace of the newly created PostgreSQL container.
  assert.equal(process.platform, 'linux');
  assert.ok(!fs.readFileSync('/proc/net/route', 'utf8').split('\n')
    .some(line => /^\S+\s+00000000\s+/.test(line)), 'unexpected external network');
  const config = { host: '127.0.0.1', port: 5432, user: 'postgres', database: 'postgres' };
  let schemaIndex = 0;

  async function fixture(run) {
    const clients = [];
    const name = `fixture_${process.pid}_${++schemaIndex}`;
    async function connect() {
      const client = new Client(config);
      await client.connect();
      clients.push(client);
      await client.query(`SET search_path TO "${name}", pg_catalog`);
      return client;
    }
    const admin = new Client(config);
    await admin.connect();
    try {
      await admin.query(`CREATE SCHEMA "${name}"`);
      await run({ connect, admin, name });
    } finally {
      for (const client of clients) await client.end().catch(() => {});
      await admin.query(`DROP SCHEMA IF EXISTS "${name}" CASCADE`);
      await admin.end();
    }
  }

  const legacyDdl = schema.CANDLES_DDL.replace('BIGSERIAL', 'SERIAL');
  const insert = `INSERT INTO candles (ticker, interval, begin, open, close, high, low, value, volume)
    VALUES ($1, '1m', '2025-01-01 10:00', 1, 2, 3, 1, 10, 5) RETURNING id::text`;
  const seqState = async c => (await c.query('SELECT last_value::text, is_called FROM candles_id_seq')).rows[0];

  async function waitBlocked(admin, client) {
    for (let i = 0; i < 250; i++) {
      const { rows } = await admin.query('SELECT cardinality(pg_blocking_pids($1)) AS n', [client.processID]);
      if (rows[0].n > 0) return;
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    throw new Error('expected concurrent query to block');
  }

  test('fresh BIGSERIAL and legacy SERIAL migration preserve sequence state and bindings', async () => {
    await fixture(async ({ connect }) => {
      const c = await connect();
      await tickers.ensureSchema(c);
      const fresh = await c.query(`SELECT a.atttypid::regtype::text AS id_type, s.seqtypid::regtype::text AS seq_type
        FROM pg_attribute a JOIN pg_sequence s ON s.seqrelid = pg_get_serial_sequence('candles', 'id')::regclass
        WHERE a.attrelid = 'candles'::regclass AND a.attname = 'id'`);
      assert.deepEqual(fresh.rows[0], { id_type: 'bigint', seq_type: 'bigint' });
      await c.query('DROP TABLE candles');
      await c.query(legacyDdl);
      await c.query('ALTER SEQUENCE candles_id_seq RESTART WITH 2147483647');
      assert.equal((await c.query(insert, ['OLD'])).rows[0].id, '2147483647');
      await c.query('ALTER SEQUENCE candles_id_seq RENAME TO "unusual ""sequence"""');
      const before = (await c.query('SELECT last_value::text, is_called FROM "unusual ""sequence"""')).rows[0];
      await tickers.ensureSchema(c);
      const after = (await c.query('SELECT last_value::text, is_called FROM "unusual ""sequence"""')).rows[0];
      assert.deepEqual(after, before);
      await tickers.ensureSchema(c);
      assert.deepEqual((await c.query('SELECT last_value::text, is_called FROM "unusual ""sequence"""')).rows[0], before);
      assert.equal((await c.query(insert, ['NEW'])).rows[0].id, '2147483648');
      assert.equal((await c.query('SELECT COUNT(*)::int AS n FROM candles')).rows[0].n, 2);
    });
  });

  test('an unused sequence stays unused after migration', async () => {
    await fixture(async ({ connect }) => {
      const c = await connect();
      await c.query(legacyDdl);
      const before = await seqState(c);
      assert.equal(before.is_called, false);
      await schema.ensureCandlesTable(c);
      assert.deepEqual(await seqState(c), before);
      assert.equal((await c.query(insert, ['FIRST'])).rows[0].id, '1');
    });
  });

  test('unexpected legacy sequence bounds are rejected without changing state', async () => {
    await fixture(async ({ connect }) => {
      const c = await connect();
      await c.query(legacyDdl);
      await c.query('ALTER SEQUENCE candles_id_seq MAXVALUE 100');
      const before = await seqState(c);
      await assert.rejects(tickers.ensureSchema(c), /границами integer/);
      assert.deepEqual(await seqState(c), before);
      assert.equal((await c.query("SELECT seqtypid::regtype::text AS t FROM pg_sequence WHERE seqrelid = 'candles_id_seq'::regclass")).rows[0].t, 'integer');
    });
  });

  test('wrong sequence default and custom nextval function are rejected', async () => {
    await fixture(async ({ connect }) => {
      const c = await connect();
      await c.query(legacyDdl);
      await c.query('CREATE SEQUENCE another_seq AS integer');
      await c.query("ALTER TABLE candles ALTER COLUMN id SET DEFAULT nextval('another_seq'::regclass)");
      await assert.rejects(schema.ensureCandlesTable(c), /привязка последовательности/);
      await c.query("ALTER TABLE candles ALTER COLUMN id SET DEFAULT nextval('candles_id_seq'::regclass)");
      await c.query('CREATE FUNCTION nextval(regclass) RETURNS bigint LANGUAGE sql AS $$ SELECT 1::bigint $$');
      await c.query("ALTER TABLE candles ALTER COLUMN id SET DEFAULT nextval('candles_id_seq'::regclass)");
      await assert.rejects(schema.ensureCandlesTable(c), /привязка последовательности/);
    });
  });

  test('insufficient ownership rolls back without changing sequence or column', async () => {
    await fixture(async ({ connect, admin, name }) => {
      const c = await connect();
      const role = `limited_${process.pid}`;
      await admin.query(`CREATE ROLE "${role}"`);
      try {
        await c.query(legacyDdl);
        const before = await seqState(c);
        await admin.query(`GRANT USAGE, CREATE ON SCHEMA "${name}" TO "${role}"`);
        await c.query(`GRANT UPDATE ON candles TO "${role}"`);
        await c.query(`SET ROLE "${role}"`);
        await assert.rejects(schema.ensureCandlesTable(c), error => error.code === '42501' && /Миграция candles\.id/.test(error.message));
        assert.equal(c.getTransactionStatus(), 'I');
        await c.query('RESET ROLE');
        assert.deepEqual(await seqState(c), before);
        assert.equal((await c.query("SELECT seqtypid::regtype::text AS t FROM pg_sequence WHERE seqrelid = 'candles_id_seq'::regclass")).rows[0].t, 'integer');
      } finally {
        await c.query('RESET ROLE');
        await admin.query(`DROP OWNED BY "${role}"`);
        await admin.query(`DROP ROLE "${role}"`);
      }
    });
  });

  test('failure after sequence ALTER rolls the whole migration back', async () => {
    await fixture(async ({ connect }) => {
      const c = await connect();
      await c.query(legacyDdl);
      await c.query(insert, ['OLD']);
      const before = await seqState(c);
      await c.query('CREATE VIEW candle_id_view AS SELECT id FROM candles');
      await assert.rejects(tickers.ensureSchema(c), /Миграция candles\.id/);
      assert.equal(c.getTransactionStatus(), 'I');
      assert.deepEqual(await seqState(c), before);
      const result = await c.query(`SELECT a.atttypid::regtype::text AS id_type, s.seqtypid::regtype::text AS seq_type,
        s.seqmax::text AS seqmax FROM pg_attribute a JOIN pg_sequence s ON s.seqrelid = 'candles_id_seq'::regclass
        WHERE a.attrelid = 'candles'::regclass AND a.attname = 'id'`);
      assert.deepEqual(result.rows[0], { id_type: 'integer', seq_type: 'integer', seqmax: '2147483647' });
    });
  });

  test('migration never commits an existing caller transaction', async () => {
    await fixture(async ({ connect }) => {
      const c = await connect();
      await c.query(legacyDdl);
      await c.query('BEGIN');
      await assert.rejects(schema.ensureCandlesTable(c), /вне внешней транзакции/);
      assert.equal(c.getTransactionStatus(), 'T');
      await c.query('ROLLBACK');
      assert.equal((await c.query("SELECT seqtypid::regtype::text AS t FROM pg_sequence WHERE seqrelid = 'candles_id_seq'::regclass")).rows[0].t, 'integer');
    });
  });

  test('advisory lock timeout rolls back and leaves the connection usable', { timeout: 15_000 }, async () => {
    await fixture(async ({ connect }) => {
      const blocker = await connect();
      const c = await connect();
      await c.query(legacyDdl);
      await blocker.query('BEGIN');
      await blocker.query("SELECT pg_advisory_xact_lock(1668177460, 'candles'::regclass::oid::integer)");
      try {
        await assert.rejects(tickers.ensureSchema(c), error => error.code === '55P03' && /Миграция candles\.id/.test(error.message));
        assert.equal(c.getTransactionStatus(), 'I');
        assert.equal((await c.query('SELECT 1 AS n')).rows[0].n, 1);
      } finally {
        await blocker.query('ROLLBACK');
      }
    });
  });

  test('already migrated schema does not acquire a table lock', async () => {
    await fixture(async ({ connect }) => {
      const blocker = await connect();
      const c = await connect();
      await schema.ensureCandlesTable(c);
      await blocker.query('BEGIN');
      await blocker.query('LOCK TABLE candles IN ACCESS EXCLUSIVE MODE');
      try {
        await c.query("SET lock_timeout = '500ms'");
        await schema.ensureCandlesTable(c);
      } finally {
        await blocker.query('ROLLBACK');
      }
    });
  });

  test('table lock timeout rolls migration back', { timeout: 15_000 }, async () => {
    await fixture(async ({ connect }) => {
      const blocker = await connect();
      const c = await connect();
      await c.query(legacyDdl);
      const before = await seqState(c);
      await blocker.query('BEGIN');
      await blocker.query('LOCK TABLE candles IN ACCESS SHARE MODE');
      try {
        await assert.rejects(schema.ensureCandlesTable(c), error => error.code === '55P03' && /Миграция candles\.id/.test(error.message));
        assert.equal(c.getTransactionStatus(), 'I');
        assert.deepEqual(await seqState(c), before);
      } finally {
        await blocker.query('ROLLBACK');
      }
    });
  });

  test('concurrent migration rechecks schema after advisory lock', async () => {
    await fixture(async ({ connect, admin }) => {
      const first = await connect();
      const second = await connect();
      await first.query(legacyDdl);
      await first.query('BEGIN');
      await first.query("SELECT pg_advisory_xact_lock(1668177460, 'candles'::regclass::oid::integer)");
      const pending = schema.ensureCandlesTable(second);
      await waitBlocked(admin, second);
      await first.query('ALTER SEQUENCE candles_id_seq AS bigint');
      await first.query('ALTER TABLE candles ALTER COLUMN id TYPE bigint');
      await first.query('COMMIT');
      await pending;
      assert.equal(second.getTransactionStatus(), 'I');
      assert.equal((await second.query("SELECT atttypid::regtype::text AS t FROM pg_attribute WHERE attrelid = 'candles'::regclass AND attname = 'id'")).rows[0].t, 'bigint');
    });
  });

  test('mixed update/insert is one atomic statement, including on constraint failure', async () => {
    await fixture(async ({ connect }) => {
      const c = await connect();
      await schema.ensureCandlesTable(c);
      await c.query(insert, ['X']);
      const before = await seqState(c);
      const originalFetch = global.fetch;
      global.fetch = async () => ({ ok: true, json: async () => ({ candles: {
        columns: ['begin', 'open', 'close', 'high', 'low', 'value', 'volume'],
        data: [
          ['2025-01-01 10:00:00', 4, 5, 6, 1, 40, 4],
          ['2025-01-01 10:01:00', 7, 8, 9, 1, 70, 7],
        ],
      } }) });
      const save = () => refresh.loadRange(c, 'X', 'https://example.invalid', 'unused', 'unused', 1, '1m',
        { quiet: true, progress: false });
      try {
        await c.query("ALTER TABLE candles ADD CONSTRAINT reject_second CHECK (begin <> '2025-01-01 10:01:00')");
        await assert.rejects(save(), /reject_second/);
        const afterFailure = await c.query('SELECT open FROM candles ORDER BY begin');
        assert.deepEqual(afterFailure.rows, [{ open: 1 }]);
        await c.query('ALTER TABLE candles DROP CONSTRAINT reject_second');
        const beforeSuccess = await seqState(c);
        await save();
        assert.deepEqual((await c.query('SELECT open FROM candles ORDER BY begin')).rows, [{ open: 4 }, { open: 7 }]);
        const afterSuccess = await seqState(c);
        assert.equal(BigInt(afterSuccess.last_value), BigInt(beforeSuccess.last_value) + 1n);
        assert.ok(BigInt(beforeSuccess.last_value) >= BigInt(before.last_value));
      } finally {
        global.fetch = originalFetch;
      }
    });
  });

  test('refresh preserves revisions, sequence values, and concurrent inserts and updates', async () => {
    await fixture(async ({ connect, admin }) => {
      const a = await connect();
      const b = await connect();
      await schema.ensureCandlesTable(a);
      const originalFetch = global.fetch;
      global.fetch = async url => {
        const revision = Number(url.searchParams.get('from'));
        return { ok: true, json: async () => ({ candles: {
          columns: ['begin', 'open', 'close', 'high', 'low', 'value', 'volume'],
          data: [['2025-01-01 10:00:00', revision, revision + 1, revision + 2, 1, revision * 10, revision]],
        } }) };
      };
      const save = (client, ticker, revision) => refresh.loadRange(client, ticker, 'https://example.invalid',
        String(revision), 'unused', 1, '1m', { quiet: true, progress: false });
      const current = async ticker => (await a.query('SELECT open, close, high, low, value, volume::text FROM candles WHERE ticker = $1', [ticker])).rows[0];
      try {
        await save(a, 'X', 2);
        const before = await seqState(a);
        await save(a, 'X', 3);
        await save(a, 'X', 4);
        assert.deepEqual(await seqState(a), before);
        assert.deepEqual(await current('X'), { open: 4, close: 5, high: 6, low: 1, value: 40, volume: '4' });

        await a.query('BEGIN');
        await save(a, 'X', 5);
        const update = save(b, 'X', 6);
        await waitBlocked(admin, b);
        await a.query('COMMIT');
        await update;
        assert.equal((await current('X')).open, 6);
        assert.deepEqual(await seqState(a), before);

        await a.query('BEGIN');
        await save(a, 'Y', 7);
        const conflict = save(b, 'Y', 8);
        await waitBlocked(admin, b);
        await a.query('COMMIT');
        await conflict;
        assert.equal((await current('Y')).open, 8);
        assert.equal((await a.query("SELECT COUNT(*)::int AS n FROM candles WHERE ticker = 'Y'")).rows[0].n, 1);

        await a.query('BEGIN');
        await a.query("DELETE FROM candles WHERE ticker = 'X'");
        const deleted = save(b, 'X', 9);
        await waitBlocked(admin, b);
        await a.query('COMMIT');
        await deleted;
        assert.equal((await current('X')).open, 9);
      } finally {
        global.fetch = originalFetch;
      }
    });
  });
}
