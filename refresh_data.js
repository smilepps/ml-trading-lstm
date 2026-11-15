#!/usr/bin/env node
/**
 * refresh_data.js — догружает свечи с MOEX ISS в PostgreSQL.
 *
 * Тикеры берутся из таблицы tickers (engine/market), поэтому работают не только
 * индексы, но и акции, фьючерсы и т.п.
 *
 * 1m/10m/60m/1D: MOEX отдаёт время уже в московском времени, вставляем как есть.
 * 5m/15m/30m собираются из 1m скриптом build_intervals.js.
 *
 * usage:
 *   node refresh_data.js                         # все тикеры из tickers
 *   node refresh_data.js --ticker IMOEX          # один тикер (полная загрузка)
 *   node refresh_data.js --ticker SBER,GAZP      # несколько тикеров
 *   node refresh_data.js --ticker IMOEX --incremental
 *   node refresh_data.js --from 2024-01-01
 */

require('dotenv').config();
const { Client } = require('pg');
const { ensureSchema, listTickers, ensureTicker, candlesUrl } = require('./tickers');

const PAGE_SIZE = 500;
const PAGE_DELAY_MS = 60;
const REQUEST_TIMEOUT_MS = 30000;
const MAX_RETRIES = 4;

const args = process.argv.slice(2);
function argValue(name, fallback) {
  const i = args.indexOf(name);
  if (i >= 0 && args[i + 1]) return args[i + 1];
  return fallback;
}

function localDate(d = new Date()) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

const TODAY = localDate();
const FROM = argValue('--from', '2024-01-01');
const INCREMENTAL = args.includes('--incremental');
const TICKERS_ARG = argValue('--ticker', null);

const JOBS = [
  { interval: 1, label: '1m' },
  { interval: 10, label: '10m' },
  { interval: 60, label: '60m' },
  { interval: 24, label: '1D' }
];

function log(msg) {
  console.log(`[${new Date().toISOString().replace('T', ' ').substring(0, 19)}] ${msg}`);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchPage(apiUrl, interval, from, till, start) {
  const url = new URL(apiUrl);
  url.searchParams.set('from', from);
  url.searchParams.set('till', till);
  url.searchParams.set('interval', String(interval));
  url.searchParams.set('start', String(start));

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(url, {
        headers: { 'User-Agent': 'ml-candles/3.0' },
        signal: controller.signal
      });
      if (response.status === 429 || response.status >= 500) {
        const retryAfter = Number(response.headers.get('retry-after')) || 0;
        const delay = retryAfter ? retryAfter * 1000 : 500 * 2 ** attempt;
        log(`  HTTP ${response.status}, повтор через ${delay} мс (${attempt + 1}/${MAX_RETRIES})`);
        await sleep(delay);
        continue;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
      return await response.json();
    } catch (e) {
      if (attempt >= MAX_RETRIES) throw e;
      const delay = 500 * 2 ** attempt;
      log(`  ${e.message}, повтор через ${delay} мс (${attempt + 1}/${MAX_RETRIES})`);
      await sleep(delay);
    } finally {
      clearTimeout(timer);
    }
  }
  throw new Error('Не удалось получить страницу API');
}

const UPSERT_SQL = `
  INSERT INTO candles (ticker, interval, begin, open, close, high, low, value, volume)
  SELECT * FROM unnest(
    $1::text[], $2::text[], $3::timestamp[], $4::float8[], $5::float8[],
    $6::float8[], $7::float8[], $8::float8[], $9::bigint[]
  )
  ON CONFLICT (ticker, interval, begin) DO UPDATE SET
    open = EXCLUDED.open,
    close = EXCLUDED.close,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    value = EXCLUDED.value,
    volume = EXCLUDED.volume
`;

function packRows(rows, ticker, label) {
  const tickers = new Array(rows.length).fill(ticker);
  const intervals = new Array(rows.length).fill(label);
  const begins = rows.map(r => r.begin);
  const opens = rows.map(r => r.open);
  const closes = rows.map(r => r.close);
  const highs = rows.map(r => r.high);
  const lows = rows.map(r => r.low);
  const values = rows.map(r => r.value);
  const volumes = rows.map(r => Math.max(0, Math.trunc(r.volume || 0)));
  return [tickers, intervals, begins, opens, closes, highs, lows, values, volumes];
}

async function incrementalFrom(client, ticker, job) {
  const result = await client.query(
    `SELECT MAX(begin) AS last FROM candles WHERE ticker = $1 AND interval = $2`,
    [ticker, job.label]
  );
  const last = result.rows[0]?.last;
  if (!last) return FROM;

  // Небольшой перехлёст в 2 дня: последняя свеча могла быть неполной.
  const d = new Date(last);
  d.setDate(d.getDate() - 2);
  return localDate(d);
}

async function loadJob(client, ticker, apiUrl, job) {
  const startedAt = Date.now();
  const from = INCREMENTAL ? await incrementalFrom(client, ticker, job) : FROM;
  let start = 0;
  let pages = 0;
  let saved = 0;

  log(`${ticker} ${job.label}: ${from} → ${TODAY}${INCREMENTAL ? ' (incremental)' : ''}`);
  while (true) {
    const data = await fetchPage(apiUrl, job.interval, from, TODAY, start);
    const block = data?.candles;
    const rows = Array.isArray(block?.data) ? block.data : [];
    if (!rows.length) break;

    const cols = block.columns;
    const idx = Object.fromEntries(cols.map((name, i) => [name, i]));
    const parsed = rows.map(row => ({
      begin: row[idx.begin],
      open: Number(row[idx.open]),
      close: Number(row[idx.close]),
      high: Number(row[idx.high]),
      low: Number(row[idx.low]),
      value: Number(row[idx.value]) || 0,
      volume: Number(row[idx.volume]) || 0
    }));

    await client.query(UPSERT_SQL, packRows(parsed, ticker, job.label));
    saved += parsed.length;
    pages += 1;
    start += rows.length;

    if (pages % 20 === 0 || rows.length < PAGE_SIZE) {
      log(`  ${ticker} ${job.label}: страниц=${pages}, свечей=${saved}, offset=${start}`);
    }
    if (rows.length < PAGE_SIZE) break;
    if (PAGE_DELAY_MS) await sleep(PAGE_DELAY_MS);
  }

  log(`${ticker} ${job.label}: готово, ${saved} свечей за ${((Date.now() - startedAt) / 1000).toFixed(1)} с`);
  return saved;
}

async function refreshTicker(client, tickerInfo) {
  const { ticker, engine, market } = tickerInfo;
  const apiUrl = process.env.MOEX_API_URL && ticker === 'IMOEX'
    ? process.env.MOEX_API_URL
    : candlesUrl(engine, market, ticker);

  log(`═══ ${ticker} (${engine}/${market}) ═══`);
  log(`API: ${apiUrl}`);

  const before = await client.query(
    `SELECT interval, COUNT(*)::int AS n, MIN(begin) AS first, MAX(begin) AS last
     FROM candles WHERE ticker = $1 GROUP BY interval ORDER BY interval`,
    [ticker]
  );
  for (const r of before.rows) log(`  до: ${r.interval}: ${r.n}, ${r.first} → ${r.last}`);

  let total = 0;
  for (const job of JOBS) {
    total += await loadJob(client, ticker, apiUrl, job);
  }

  const after = await client.query(
    `SELECT interval, COUNT(*)::int AS n, MIN(begin) AS first, MAX(begin) AS last
     FROM candles WHERE ticker = $1 GROUP BY interval ORDER BY interval`,
    [ticker]
  );
  log(`${ticker}: получено ${total} строк`);
  for (const r of after.rows) log(`  после: ${r.interval}: ${r.n}, ${r.first} → ${r.last}`);
  return total;
}

async function main() {
  const client = new Client({ connectionString: process.env.DB });
  await client.connect();
  try {
    await ensureSchema(client);
    let targets;
    if (TICKERS_ARG) {
      const names = TICKERS_ARG.split(',').map(s => s.trim()).filter(Boolean);
      targets = [];
      for (const name of names) targets.push(await ensureTicker(client, name));
    } else {
      targets = await listTickers(client);
    }
    if (!targets.length) throw new Error('В таблице tickers нет ни одного тикера');
    log(`Тикеров к обновлению: ${targets.length} (${targets.map(t => t.ticker).join(', ')})`);

    let grandTotal = 0;
    for (const t of targets) {
      grandTotal += await refreshTicker(client, t);
    }
    log(`Всего получено строк: ${grandTotal}`);
  } finally {
    await client.end().catch(() => {});
  }
}

if (require.main === module) {
  main().catch(e => {
    console.error('Ошибка обновления данных:', e);
    process.exit(1);
  });
}

module.exports = { JOBS, refreshTicker, FROM, INCREMENTAL, TODAY };
