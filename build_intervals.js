#!/usr/bin/env node
/**
 * build_intervals.js — строит свечи 5m, 15m и 30m из 1m.
 *
 * 1m-свечи остаются в БД как источник, но в меню и в обучение не попадают.
 * Скрипт идемпотентен: каждый запуск пересобирает 5m/15m/30m с нуля.
 *
 * usage:
 *   node build_intervals.js                  # все тикеры из tickers
 *   node build_intervals.js --ticker IMOEX
 *   node build_intervals.js --ticker SBER,GAZP
 *   node build_intervals.js --force
 */

require('dotenv').config();
const { Client } = require('pg');
const { ensureSchema, listTickers, getTicker } = require('./tickers');

const SOURCE_INTERVAL = '1m';
const TARGETS = [
  { interval: '5m', stride: '5 minutes' },
  { interval: '15m', stride: '15 minutes' },
  { interval: '30m', stride: '30 minutes' }
];

function log(msg) {
  console.log(`[${new Date().toISOString().replace('T', ' ').substring(0, 19)}] ${msg}`);
}

async function buildIntervals(ticker) {
  const client = new Client({ connectionString: process.env.DB });
  await client.connect();
  try {
    const info = await getTicker(client, ticker);
    if (!info) throw new Error(`Тикер ${ticker} не найден в таблице tickers`);

    const source = await client.query(
      `SELECT COUNT(*)::int AS n, MIN(begin) AS first, MAX(begin) AS last
       FROM candles WHERE ticker = $1 AND interval = $2`,
      [ticker, SOURCE_INTERVAL]
    );
    const src = source.rows[0];
    if (!src.n) {
      throw new Error(`Нет исходных свечей ${SOURCE_INTERVAL} для ${ticker}`);
    }
    log(`${ticker}: источник ${SOURCE_INTERVAL} — ${src.n} свечей, ${src.first} → ${src.last}`);

    for (const { interval, stride } of TARGETS) {
      const startedAt = Date.now();
      await client.query('BEGIN');
      try {
        await client.query(
          `DELETE FROM candles WHERE ticker = $1 AND interval = $2`,
          [ticker, interval]
        );
        const result = await client.query(
          `INSERT INTO candles (ticker, interval, begin, open, high, low, close, value, volume)
           SELECT $1 AS ticker,
                  $2 AS interval,
                  bucket AS begin,
                  (array_agg(open ORDER BY begin ASC))[1] AS open,
                  max(high) AS high,
                  min(low) AS low,
                  (array_agg(close ORDER BY begin DESC))[1] AS close,
                  sum(value) AS value,
                  sum(volume) AS volume
           FROM (
             SELECT begin, open, high, low, close, value, volume,
                    date_bin($3::interval, begin, TIMESTAMP '2000-01-01 00:00:00') AS bucket
             FROM candles
             WHERE ticker = $1 AND interval = $4
           ) src
           GROUP BY bucket
           ORDER BY bucket`,
          [ticker, interval, stride, SOURCE_INTERVAL]
        );
        await client.query('COMMIT');
        const check = await client.query(
          `SELECT COUNT(*)::int AS n, MIN(begin) AS first, MAX(begin) AS last,
                  COUNT(*) FILTER (WHERE close IS NULL OR open IS NULL)::int AS bad
           FROM candles WHERE ticker = $1 AND interval = $2`,
          [ticker, interval]
        );
        const row = check.rows[0];
        log(
          `${ticker} ${interval}: вставлено ${result.rowCount}, в БД ${row.n} свечей, ` +
          `${row.first} → ${row.last}, битых=${row.bad}, ${Date.now() - startedAt} мс`
        );
      } catch (e) {
        await client.query('ROLLBACK');
        throw e;
      }
    }

    const all = await client.query(
      `SELECT interval, COUNT(*)::int AS n, MIN(begin) AS first, MAX(begin) AS last
       FROM candles WHERE ticker = $1 GROUP BY interval ORDER BY interval`,
      [ticker]
    );
    log(`Итог ${ticker}:`);
    for (const r of all.rows) {
      log(`  ${r.interval}: ${r.n} свечей, ${r.first} → ${r.last}`);
    }
    return all.rows;
  } finally {
    await client.end().catch(() => {});
  }
}

async function main() {
  const args = process.argv.slice(2);
  const idx = args.indexOf('--ticker');
  const arg = idx >= 0 ? args[idx + 1] : null;

  let tickers;
  if (arg) {
    tickers = arg.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
  } else {
    const client = new Client({ connectionString: process.env.DB });
    await client.connect();
    try {
      await ensureSchema(client);
      tickers = (await listTickers(client)).map(t => t.ticker);
    } finally {
      await client.end().catch(() => {});
    }
  }
  if (!tickers.length) throw new Error('Не указаны тикеры');

  for (const ticker of tickers) {
    await buildIntervals(ticker);
  }
}

if (require.main === module) {
  main().catch(e => {
    console.error('Ошибка сборки интервалов:', e.message);
    process.exit(1);
  });
}

module.exports = { buildIntervals, TARGETS, SOURCE_INTERVAL };
