/**
 * db_schema.js — DDL всех таблиц проекта в одном месте.
 *
 * Раньше ensureSchema() в tickers.js создавал только tickers, и на чистой базе
 * первая же загрузка падала с «relation "candles" does not exist»: схема candles
 * существовала лишь в README и в живой базе. Теперь таблица свечей создаётся
 * теми же вызовами ensureSchema(), что уже есть в server.js, refresh_data.js
 * и tickers.js CLI.
 *
 * Новые базы используют BIGSERIAL. Существующий обычный SERIAL переводится
 * вместе со своей последовательностью в bigint при ensureCandlesTable().
 */

const CANDLES_DDL = `
  CREATE TABLE IF NOT EXISTS candles (
    id BIGSERIAL PRIMARY KEY,
    ticker varchar(20) NOT NULL,
    interval varchar(10) NOT NULL,
    begin timestamp NOT NULL,          -- московское время, без timezone
    open double precision NOT NULL,
    close double precision NOT NULL,
    high double precision NOT NULL,
    low double precision NOT NULL,
    value double precision,
    volume bigint,
    created_at timestamp DEFAULT now(),
    UNIQUE (ticker, interval, begin)
  )
`;

// Открытые позиции по фьючерсам (MOEX ISS analyticalproducts/futoi):
// снимки каждые 5 минут в разрезе групп клиентов FIZ (физлица) / YUR (юрлица).
// Для IMOEX это фьючерс на индекс МосБиржи IMOEXF (сумма по всем экспирациям).
// pos_short хранится как отдаёт API — отрицательным.
const OPEN_INTEREST_DDL = `
  CREATE TABLE IF NOT EXISTS open_interest (
    id BIGSERIAL PRIMARY KEY,
    ticker varchar(20) NOT NULL,
    ts timestamp NOT NULL,              -- московское время, без timezone (tradedate + tradetime)
    clgroup varchar(8) NOT NULL,        -- FIZ / YUR
    pos_long bigint,
    pos_short bigint,
    pos bigint,                         -- нетто-позиция группы
    pos_long_num integer,               -- число клиентов в long
    pos_short_num integer,              -- число клиентов в short
    created_at timestamp DEFAULT now(),
    UNIQUE (ticker, ts, clgroup)
  )
`;

const TICKERS_DDL = `
  CREATE TABLE IF NOT EXISTS tickers (
    ticker     varchar(32) PRIMARY KEY,
    engine     varchar(32) NOT NULL,
    market     varchar(32) NOT NULL,
    title      text,
    sort_order integer NOT NULL DEFAULT 100,
    created_at timestamp without time zone NOT NULL DEFAULT now()
  )
`;

// Отдельный индекс (ticker, interval, begin) не нужен: его уже покрывает
// уникальный ключ — EXPLAIN показывает Index Scan / Index Only Scan Backward
// и для «последней свечи», и для диапазона загрузки.

async function ensureTickersTable(client) {
  await client.query(TICKERS_DDL);
}

// Имена последовательностей могли быть изменены; доверяем только каталогу и
// привязке OWNED BY, а не предполагаемому имени candles_id_seq.
const CANDLES_ID_INFO_SQL = `
  SELECT c.oid::text AS table_oid, c.relkind, c.relpersistence,
         c.relispartition, a.atttypid::regtype::text AS id_type,
         a.attnotnull, a.attidentity, a.attgenerated,
         EXISTS (SELECT 1 FROM pg_inherits i
                 WHERE i.inhrelid = c.oid OR i.inhparent = c.oid) AS inherited,
         EXISTS (SELECT 1 FROM pg_constraint p
                 WHERE p.conrelid = c.oid AND p.contype = 'p'
                   AND p.conkey = ARRAY[a.attnum]::smallint[]) AS id_primary_key,
         -- DEFAULT не содержит ссылок на колонки; 0 не блокирует candles при чтении.
         pg_get_expr(d.adbin, 0) AS id_default,
         format('nextval(%L::regclass)', s.oid::regclass::text) AS expected_default,
         EXISTS (SELECT 1 FROM pg_depend dd
                 WHERE dd.classid = 'pg_attrdef'::regclass AND dd.objid = d.oid
                   AND dd.refclassid = 'pg_class'::regclass AND dd.refobjid = s.oid
                   AND dd.refobjsubid = 0 AND dd.deptype = 'n') AS default_depends_on_sequence,
         EXISTS (SELECT 1 FROM pg_depend dd
                 WHERE dd.classid = 'pg_attrdef'::regclass AND dd.objid = d.oid
                   AND dd.refclassid = 'pg_proc'::regclass
                   AND dd.refobjid <> 'pg_catalog.nextval(regclass)'::regprocedure) AS custom_default_function,
         s.oid::text AS sequence_oid,
         format('%I.%I', sn.nspname, s.relname) AS sequence_name,
         owned.deptype AS ownership,
         ps.seqtypid::regtype::text AS sequence_type,
         ps.seqstart::text AS seqstart, ps.seqmin::text AS seqmin,
         ps.seqmax::text AS seqmax, ps.seqincrement::text AS seqincrement,
         ps.seqcache::text AS seqcache, ps.seqcycle
  FROM pg_class c
  JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = 'id'
                     AND a.attnum > 0 AND NOT a.attisdropped
  LEFT JOIN pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum
  LEFT JOIN (
    SELECT dep.refobjid, dep.refobjsubid, dep.objid, dep.deptype
    FROM pg_depend dep
    JOIN pg_class seq ON seq.oid = dep.objid AND seq.relkind = 'S'
    WHERE dep.classid = 'pg_class'::regclass
      AND dep.refclassid = 'pg_class'::regclass AND dep.objsubid = 0
      AND dep.deptype IN ('a', 'i')
  ) owned ON owned.refobjid = c.oid AND owned.refobjsubid = a.attnum
  LEFT JOIN pg_class s ON s.oid = owned.objid
  LEFT JOIN pg_namespace sn ON sn.oid = s.relnamespace
  LEFT JOIN pg_sequence ps ON ps.seqrelid = s.oid
  WHERE c.oid = 'candles'::regclass
`;

async function candlesIdInfo(client) {
  const { rows } = await client.query(CANDLES_ID_INFO_SQL);
  if (rows.length !== 1) throw new Error('ожидается candles.id с одной собственной последовательностью');
  const info = rows[0];
  if (info.relkind !== 'r' || info.relpersistence !== 'p' || info.relispartition || info.inherited ||
      !info.attnotnull || !info.id_primary_key || info.attidentity || info.attgenerated ||
      info.ownership !== 'a' || !info.sequence_oid || !info.sequence_name ||
      !info.default_depends_on_sequence || info.custom_default_function || info.id_default !== info.expected_default) {
    throw new Error('нестандартная таблица candles.id или привязка последовательности');
  }
  return info;
}

function hasStandardSequence(info, type, max) {
  return info.sequence_type === type && info.seqstart === '1' && info.seqmin === '1' &&
    info.seqmax === max && info.seqincrement === '1' && info.seqcache === '1' && !info.seqcycle;
}

async function migrateCandlesId(client) {
  try {
    let info = await candlesIdInfo(client);
    // Уже мигрированная схема: только чтение каталогов, без блокировки таблицы.
    if (info.id_type === 'bigint' && hasStandardSequence(info, 'bigint', '9223372036854775807')) return;
    if (info.id_type !== 'integer' || !hasStandardSequence(info, 'integer', '2147483647')) {
      throw new Error('ожидается обычный SERIAL с границами integer или BIGSERIAL со стандартными границами');
    }

    // BEGIN внутри чужой транзакции в PostgreSQL не вложен и COMMIT закрыл бы её.
    if (typeof client.getTransactionStatus !== 'function' || client.getTransactionStatus() !== 'I') {
      throw new Error('миграция требует клиент с известным состоянием вне внешней транзакции');
    }
    await client.query('BEGIN');
    try {
      await client.query("SET LOCAL lock_timeout = '5s'");
      await client.query("SELECT pg_advisory_xact_lock(1668177460, 'candles'::regclass::oid::integer)");
      // Другой процесс мог уже выполнить миграцию, пока мы ожидали lock.
      info = await candlesIdInfo(client);
      if (info.id_type === 'bigint' && hasStandardSequence(info, 'bigint', '9223372036854775807')) {
        await client.query('COMMIT');
        return;
      }
      if (info.id_type !== 'integer' || !hasStandardSequence(info, 'integer', '2147483647')) {
        throw new Error('ожидается обычный SERIAL с границами integer или BIGSERIAL со стандартными границами');
      }
      await client.query('LOCK TABLE candles IN ACCESS EXCLUSIVE MODE');
      // Повторно проверяем после ожидания блокировки таблицы и стороннего DDL.
      info = await candlesIdInfo(client);
      if (info.id_type === 'bigint' && hasStandardSequence(info, 'bigint', '9223372036854775807')) {
        await client.query('COMMIT');
        return;
      }
      if (info.id_type !== 'integer' || !hasStandardSequence(info, 'integer', '2147483647')) {
        throw new Error('ожидается обычный SERIAL с границами integer или BIGSERIAL со стандартными границами');
      }
      const sequenceOid = info.sequence_oid;
      // ALTER SEQUENCE транзакционен и не вызывает nextval/setval. Повторная
      // проверка после захвата его блокировки обнаруживает параллельный ALTER.
      await client.query(`ALTER SEQUENCE ${info.sequence_name} AS bigint`);
      info = await candlesIdInfo(client);
      if (info.id_type !== 'integer' || info.sequence_oid !== sequenceOid ||
          !hasStandardSequence(info, 'bigint', '9223372036854775807')) {
        throw new Error('параметры последовательности изменились во время миграции');
      }
      await client.query('ALTER TABLE candles ALTER COLUMN id TYPE bigint');
      await client.query('COMMIT');
    } catch (error) {
      try {
        await client.query('ROLLBACK');
      } catch (rollbackError) {
        error.message += `; ROLLBACK: ${rollbackError.message}`;
      }
      throw error;
    }
  } catch (error) {
    error.message = `Миграция candles.id: ${error.message}`;
    throw error;
  }
}

async function ensureCandlesTable(client) {
  await client.query(CANDLES_DDL);
  await migrateCandlesId(client);
}

async function ensureOpenInterestTable(client) {
  await client.query(OPEN_INTEREST_DDL);
}

module.exports = {
  CANDLES_DDL,
  OPEN_INTEREST_DDL,
  TICKERS_DDL,
  ensureTickersTable,
  ensureCandlesTable,
  ensureOpenInterestTable,
};
