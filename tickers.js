#!/usr/bin/env node
/**
 * tickers.js — справочник тикеров и определение рынка MOEX ISS.
 *
 * Таблица tickers хранит, откуда брать свечи:
 *   engine/market — например stock/index для IMOEX или stock/shares для SBER.
 *
 * Если тикера нет в таблице, рынок определяется по MOEX ISS:
 *   GET https://iss.moex.com/iss/securities/{ticker}.json
 * Берётся основная доска (is_primary = 1): её engine и market.
 */

require('dotenv').config();
const { Client } = require('pg');
const { ensureTickersTable, ensureCandlesTable, ensureOpenInterestTable } = require('./db_schema');

const ISS_TIMEOUT_MS = 15000;
const TICKER_RE = /^[A-Za-z0-9._-]{1,32}$/;

const SEED = [
  { ticker: 'IMOEX', engine: 'stock', market: 'index', title: 'Индекс МосБиржи', sortOrder: 0 }
];

// Соответствие тикера коду базового актива futoi (открытые позиции FIZ/YUR).
// Индексы/акции сами по себе позиций не имеют — берём фьючерс на них.
// История futoi на MOEX ISS начинается с 2023-11-16.
const FUTOI_SECID = {
  IMOEX: 'IMOEXF',
  SBER: 'SBRF',
};

const FUTOI_MIN_DATE = '2023-11-16';

function futoiSecid(ticker) {
  return FUTOI_SECID[normalizeTicker(ticker)] || null;
}

function futoiUrl(secid) {
  return `https://iss.moex.com/iss/analyticalproducts/futoi/securities/${encodeURIComponent(secid)}.json`;
}

function log(msg) {
  console.log(`[${new Date().toISOString().replace('T', ' ').substring(0, 19)}] ${msg}`);
}

function normalizeTicker(value) {
  return String(value || '').trim().toUpperCase();
}

function validateTicker(value) {
  const ticker = normalizeTicker(value);
  if (!TICKER_RE.test(ticker)) {
    throw new Error(`Некорректный тикер «${value}»: допустимы латиница, цифры, точка, дефис и подчёркивание`);
  }
  return ticker;
}

function candlesUrl(engine, market, ticker) {
  return `https://iss.moex.com/iss/engines/${encodeURIComponent(engine)}` +
    `/markets/${encodeURIComponent(market)}/securities/${encodeURIComponent(ticker)}/candles.json`;
}

async function ensureSchema(client) {
  await ensureTickersTable(client);
  // Таблица свечей раньше здесь не создавалась вовсе: на чистой базе
  // сервер поднимался, а первая же загрузка/прогноз падали на candles.
  await ensureCandlesTable(client);
  await ensureOpenInterestTable(client);
  for (const item of SEED) {
    await client.query(
      `INSERT INTO tickers (ticker, engine, market, title, sort_order)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (ticker) DO NOTHING`,
      [item.ticker, item.engine, item.market, item.title, item.sortOrder]
    );
  }
}

async function listTickers(client) {
  const r = await client.query(
    `SELECT ticker, engine, market, title, sort_order, created_at
     FROM tickers ORDER BY sort_order ASC, ticker ASC`
  );
  return r.rows.map(row => ({
    ticker: row.ticker,
    engine: row.engine,
    market: row.market,
    title: row.title,
    sortOrder: row.sort_order,
    createdAt: row.created_at
  }));
}

async function getTicker(client, ticker) {
  const t = normalizeTicker(ticker);
  const r = await client.query(
    `SELECT ticker, engine, market, title, sort_order, created_at
     FROM tickers WHERE ticker = $1`,
    [t]
  );
  if (!r.rows.length) return null;
  const row = r.rows[0];
  return {
    ticker: row.ticker,
    engine: row.engine,
    market: row.market,
    title: row.title,
    sortOrder: row.sort_order,
    createdAt: row.created_at
  };
}

async function fetchJsonWithRetry(url, opts = {}, retries = 3, baseDelayMs = 1000) {
  let lastErr;
  for (let attempt = 0; attempt < retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), ISS_TIMEOUT_MS);
    try {
      const response = await fetch(url, {
        headers: { 'User-Agent': 'ml-dashboard/2.0' },
        signal: controller.signal,
        ...opts,
      });
      if (!response.ok) throw new Error(`MOEX ISS ответил HTTP ${response.status}`);
      return await response.json();
    } catch (e) {
      lastErr = e;
    } finally {
      clearTimeout(timer);
    }
    if (attempt < retries - 1) {
      await new Promise(resolve => setTimeout(resolve, baseDelayMs * (attempt + 1)));
    }
  }
  throw lastErr || new Error(`Не удалось получить данные с MOEX ISS после ${retries} попыток`);
}

async function resolveTicker(ticker) {
  const t = validateTicker(ticker);
  const url = `https://iss.moex.com/iss/securities/${encodeURIComponent(t)}.json?iss.meta=off`;
  let payload;
  try {
    payload = await fetchJsonWithRetry(url);
  } catch (e) {
    throw new Error(`Не удалось опросить MOEX ISS для ${t}: ${e.message}`);
  }

  const boards = payload?.boards;
  if (!boards || !Array.isArray(boards.data) || !boards.data.length) {
    throw new Error(`MOEX ISS не знает тикер ${t}`);
  }
  const idx = Object.fromEntries(boards.columns.map((name, i) => [name, i]));
  const rows = boards.data.map(row => ({
    engine: row[idx.engine],
    market: row[idx.market],
    boardId: row[idx.boardid],
    primary: Number(row[idx.is_primary]) === 1
  }));

  // Приоритет: основная доска, затем акции, индексы, срочный рынок, любая.
  const priority = [
    r => r.primary && r.engine === 'stock' && r.market === 'shares',
    r => r.primary && r.engine === 'stock' && r.market === 'index',
    r => r.primary,
    r => r.engine === 'stock' && r.market === 'shares',
    r => r.engine === 'stock' && r.market === 'index',
    () => true
  ];
  let board = null;
  for (const test of priority) {
    board = rows.find(test);
    if (board) break;
  }
  if (!board) throw new Error(`Не удалось определить рынок для ${t}`);

  let title = t;
  const description = payload?.description;
  if (description && Array.isArray(description.data)) {
    const di = Object.fromEntries(description.columns.map((name, i) => [name, i]));
    const short = description.data.find(row => row[di.name] === 'SHORTNAME');
    if (short && short[di.value]) title = String(short[di.value]);
  }

  return { ticker: t, engine: board.engine, market: board.market, title, boardId: board.boardId };
}

async function addTicker(client, ticker, options = {}) {
  const info = options.engine && options.market
    ? { ticker: validateTicker(ticker), engine: options.engine, market: options.market, title: options.title || normalizeTicker(ticker) }
    : await resolveTicker(ticker);

  const max = await client.query(`SELECT COALESCE(MAX(sort_order), 0) + 10 AS next FROM tickers`);
  const sortOrder = Number(max.rows[0].next) || 10;
  const r = await client.query(
    `INSERT INTO tickers (ticker, engine, market, title, sort_order)
     VALUES ($1, $2, $3, $4, $5)
     ON CONFLICT (ticker) DO UPDATE SET
       engine = EXCLUDED.engine,
       market = EXCLUDED.market,
       title = EXCLUDED.title
     RETURNING ticker, engine, market, title, sort_order`,
    [info.ticker, info.engine, info.market, info.title, sortOrder]
  );
  const row = r.rows[0];
  return {
    ticker: row.ticker,
    engine: row.engine,
    market: row.market,
    title: row.title,
    sortOrder: row.sort_order
  };
}

async function ensureTicker(client, ticker) {
  const existing = await getTicker(client, ticker);
  if (existing) return existing;
  log(`Тикер ${normalizeTicker(ticker)} отсутствует в справочнике — определяю рынок через MOEX ISS`);
  return addTicker(client, ticker);
}

async function removeTicker(client, ticker) {
  const t = validateTicker(ticker);
  const removed = await client.query(`DELETE FROM tickers WHERE ticker = $1 RETURNING ticker`, [t]);
  return removed.rowCount > 0;
}

async function openClient() {
  const client = new Client({ connectionString: process.env.DB });
  await client.connect();
  return client;
}

module.exports = {
  SEED,
  FUTOI_SECID,
  FUTOI_MIN_DATE,
  normalizeTicker,
  validateTicker,
  candlesUrl,
  futoiSecid,
  futoiUrl,
  ensureSchema,
  listTickers,
  getTicker,
  resolveTicker,
  addTicker,
  ensureTicker,
  removeTicker,
  openClient
};

if (require.main === module) {
  (async () => {
    const client = await openClient();
    try {
      await ensureSchema(client);
      const args = process.argv.slice(2);
      if (args[0] === 'add' && args[1]) {
        log(JSON.stringify(await addTicker(client, args[1])));
      } else if (args[0] === 'remove' && args[1]) {
        log(await removeTicker(client, args[1]) ? `удалён ${args[1]}` : `не найден ${args[1]}`);
      } else {
        for (const t of await listTickers(client)) {
          log(`${t.ticker}: ${t.engine}/${t.market} — ${t.title}`);
        }
      }
    } finally {
      await client.end().catch(() => {});
    }
  })().catch(e => {
    console.error('Ошибка tickers.js:', e.message);
    process.exit(1);
  });
}
