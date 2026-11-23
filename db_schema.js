/**
 * db_schema.js — DDL всех таблиц проекта в одном месте.
 *
 * Раньше ensureSchema() в tickers.js создавал только tickers, и на чистой базе
 * первая же загрузка падала с «relation "candles" does not exist»: схема candles
 * существовала лишь в README и в живой базе. Теперь таблица свечей создаётся
 * теми же вызовами ensureSchema(), что уже есть в server.js, refresh_data.js
 * и tickers.js CLI.
 *
 * Отличия от README для новых баз: id = BIGSERIAL, а не serial. Значение
 * nextval резервируется на каждую строку-кандидата апсерта (refresh_data.js),
 * включая отброшенные по уникальному ключу, поэтому счётчик растёт быстрее
 * числа строк:
 * в живой базе 219.3 млн значений счётчика при 1.74 млн строк. Существующую
 * базу этот CREATE TABLE IF NOT EXISTS не меняет: там id остаётся integer, пока
 * не выполнен ALTER SEQUENCE candles_id_seq AS bigint / ALTER TABLE candles
 * ALTER COLUMN id TYPE bigint.
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

async function ensureCandlesTable(client) {
  await client.query(CANDLES_DDL);
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
