#!/usr/bin/env node
/**
 * build_intervals.js — заглушка.
 *
 * Все интервалы загружаются напрямую с MOEX ISS (1m, 10m, 60m, 1D).
 * Агрегация промежуточных интервалов (5m, 15m, 30m, 2h, 4h) из 1m отменена.
 *
 * usage:
 *   node build_intervals.js                  # все тикеры из tickers
 *   node build_intervals.js --ticker IMOEX
 */

require('dotenv').config();
const { Client } = require('pg');
const { ensureSchema, listTickers, getTicker } = require('./tickers');

const TARGETS = [];

function log(msg) {
  console.log(`[${new Date().toISOString().replace('T', ' ').substring(0, 19)}] ${msg}`);
}

async function buildIntervals(ticker) {
  const client = new Client({ connectionString: process.env.DB });
  await client.connect();
  try {
    const info = await getTicker(client, ticker);
    if (!info) throw new Error(`Тикер ${ticker} не найден в таблице tickers`);

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

module.exports = { buildIntervals, TARGETS };
