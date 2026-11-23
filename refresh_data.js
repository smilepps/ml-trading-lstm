#!/usr/bin/env node
/**
 * refresh_data.js — догружает свечи с MOEX ISS в PostgreSQL.
 *
 * Тикеры берутся из таблицы tickers (engine/market), поэтому работают не только
 * индексы, но и акции, фьючерсы и т.п.
 *
 * 10m/60m/1D: MOEX отдаёт время уже в московском времени, вставляем как есть.
 * Все интервалы (1m/10m/60m/1D) загружаются напрямую с MOEX ISS.
 *
 * usage:
 *   node refresh_data.js                         # все тикеры из tickers
 *   node refresh_data.js --ticker IMOEX          # один тикер (полная загрузка)
 *   node refresh_data.js --ticker SBER,GAZP      # несколько тикеров
 *   node refresh_data.js --ticker IMOEX --incremental
 *   node refresh_data.js --from 2024-01-01       # с указанной даты (в пределах лимита интервала)
 *   node refresh_data.js --no-gaps               # не искать пропуски
 *   node refresh_data.js --gaps-days 14          # окно поиска пропусков
 *   node refresh_data.js --no-oi                 # не грузить открытые позиции (futoi)
 *
 * Для тикеров с фьючерсом на базовый актив (IMOEX → IMOEXF) дополнительно
 * грузятся открытые позиции FIZ/YUR (analyticalproducts/futoi) в open_interest.
 * История futoi: с 2023-11-16, бесплатный доступ — только до сегодня − 14 дней.
 */

require('dotenv').config();
const { Client } = require('pg');
const {
  ensureSchema, listTickers, ensureTicker, candlesUrl,
  futoiSecid, futoiUrl, FUTOI_MIN_DATE,
} = require('./tickers');
const { intervalMinutes } = require('./thresholds');

const PAGE_SIZE = 500;
const PAGE_DELAY_MS = 60;
const REQUEST_TIMEOUT_MS = 30000;
const MAX_RETRIES = 4;
const MAX_PAGES = 20000;
const MAX_GAPS_PER_RUN = 50;
const MAX_GAP_MS = 4 * 60 * 60 * 1000;
const MOSCOW_TIME_ZONE = process.env.MOEX_TIMEZONE || 'Europe/Moscow';

// futoi (открытые позиции FIZ/YUR): API игнорирует start/limit и отдаёт весь
// диапазон from/till, но не больше 1000 строк (самые свежие). Поэтому грузим
// окнами по 2 дня (~700 строк) и при упоре в лимит дробим окно на дни.
// Бесплатный доступ: история только до сегодня − 14 дней.
const OI_PAGE_CAP = 1000;
const OI_WINDOW_DAYS = 2;
const OI_EMBARGO_DAYS = 14;

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

// ── Даты ──
// Даты в БД — timestamp without time zone (московское время). Date из pg
// разбирается в локальном поясе, поэтому localDate/localStamp сохраняют
// настенное время БД. Для «сегодня» используем именно московскую дату.
function localDate(d = new Date()) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function localStamp(d) {
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${localDate(d)} ${hh}:${mm}:${ss}`;
}

function moscowToday() {
  try {
    const parts = new Intl.DateTimeFormat('en-GB', {
      timeZone: MOSCOW_TIME_ZONE,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }).formatToParts(new Date());
    const map = Object.fromEntries(parts.map((part) => [part.type, part.value]));
    if (map.year && map.month && map.day) return `${map.year}-${map.month}-${map.day}`;
  } catch (e) {
    /* неизвестная таймзона — используем локальную дату */
  }
  return localDate();
}

function parseLocalDate(str) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(str || ''));
  if (!m) throw new Error(`Неверный формат даты --from: ${str}, ожидается YYYY-MM-DD`);
  const year = Number(m[1]);
  const month = Number(m[2]);
  const day = Number(m[3]);
  const d = new Date(year, month - 1, day);
  if (d.getFullYear() !== year || d.getMonth() !== month - 1 || d.getDate() !== day) {
    throw new Error(`Несуществующая дата --from: ${str}`);
  }
  return d;
}

function subtractMonths(date, months) {
  const d = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const day = d.getDate();
  d.setDate(1);
  d.setMonth(d.getMonth() - months);
  const lastDay = new Date(d.getFullYear(), d.getMonth() + 1, 0).getDate();
  d.setDate(Math.min(day, lastDay));
  return d;
}

function intervalMaxYears(label) {
  if (label === '1m') return 1;     // 1 год
  if (label === '10m') return 5;    // 5 лет
  if (label === '60m') return 8;    // 8 лет
  if (label === '1D') return 100;   // 1D — максимум (всё доступное)
  return 4;
}

function minDateForInterval(label) {
  const months = Math.round(intervalMaxYears(label) * 12);
  return localDate(subtractMonths(parseLocalDate(moscowToday()), months));
}

function moscowDateMinusDays(days) {
  const d = parseLocalDate(moscowToday());
  d.setDate(d.getDate() - Math.max(0, Number(days) || 0));
  return localDate(d);
}

function formatDb(value) {
  if (!value) return '—';
  const d = value instanceof Date ? value : new Date(value);
  return Number.isNaN(d.getTime()) ? String(value) : localStamp(d);
}

// ── CLI ──
function argValue(args, name, fallback) {
  const i = args.indexOf(name);
  if (i >= 0 && args[i + 1]) return args[i + 1];
  return fallback;
}

function parseCliArgs(argv = process.argv.slice(2)) {
  const args = [...argv];
  const from = argValue(args, '--from', null);
  if (from) parseLocalDate(from);

  const gapDaysRaw = Number(argValue(args, '--gaps-days', '7'));
  const gapDays = Number.isFinite(gapDaysRaw) && gapDaysRaw > 0 ? gapDaysRaw : 7;

  let gaps;
  if (args.includes('--no-gaps')) gaps = false;
  else if (args.includes('--gaps')) gaps = true;

  return {
    from,
    incremental: args.includes('--incremental'),
    tickers: argValue(args, '--ticker', null),
    gaps,
    gapDays,
    noOi: args.includes('--no-oi'),
  };
}

// ── MOEX ISS ──
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
        headers: { 'User-Agent': 'ml-candles/3.1' },
        signal: controller.signal
      });
      if (response.status === 429 || response.status >= 500) {
        if (attempt >= MAX_RETRIES) {
          throw new Error(`MOEX ISS: HTTP ${response.status} после ${MAX_RETRIES + 1} попыток`);
        }
        const retryAfter = Number(response.headers.get('retry-after')) || 0;
        const delay = retryAfter ? retryAfter * 1000 : 500 * 2 ** attempt;
        log(`  HTTP ${response.status}, повтор через ${delay} мс (${attempt + 1}/${MAX_RETRIES})`);
        await sleep(delay);
        continue;
      }
      if (!response.ok) {
        const error = new Error(`HTTP ${response.status} ${response.statusText}`);
        error.retryable = false;
        throw error;
      }
      return await response.json();
    } catch (e) {
      if (e.retryable === false || attempt >= MAX_RETRIES) throw e;
      const delay = 500 * 2 ** attempt;
      log(`  ${e.message}, повтор через ${delay} мс (${attempt + 1}/${MAX_RETRIES})`);
      await sleep(delay);
    } finally {
      clearTimeout(timer);
    }
  }
  throw new Error('Не удалось получить страницу API');
}

// ── Запись ──
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

function parseRow(row, idx) {
  const begin = row[idx.begin];
  if (!begin) return { error: 'нет begin' };

  const open = Number(row[idx.open]);
  const close = Number(row[idx.close]);
  const high = Number(row[idx.high]);
  const low = Number(row[idx.low]);
  if (![open, close, high, low].every((value) => Number.isFinite(value) && value > 0)) {
    return { error: `некорректные OHLC у ${begin}` };
  }
  if (high < low) return { error: `high < low у ${begin}` };

  const value = Number(row[idx.value]);
  const volume = Number(row[idx.volume]);
  return {
    row: {
      begin,
      open,
      close,
      high,
      low,
      value: Number.isFinite(value) ? Math.max(0, value) : 0,
      volume: Number.isFinite(volume) ? Math.max(0, Math.trunc(volume)) : 0,
    },
  };
}

// В одной пачке не должно быть двух строк с одним begin: PostgreSQL не даёт
// ON CONFLICT DO UPDATE дважды затронуть одну строку.
function dedupeRows(rows) {
  const map = new Map();
  for (const row of rows) {
    const key = row.begin instanceof Date ? row.begin.getTime() : String(row.begin);
    map.set(key, row);
  }
  return [...map.values()];
}

async function incrementalFrom(client, ticker, job) {
  const result = await client.query(
    `SELECT MAX(begin) AS last FROM candles WHERE ticker = $1 AND interval = $2`,
    [ticker, job.label]
  );
  const last = result.rows[0]?.last;
  if (!last) return minDateForInterval(job.label);

  // Перехлёст в 2 свечи: последняя свеча могла быть неполной.
  // Для внутридневных интервалов отдаём точное время, чтобы не качать весь день.
  const stepMs = intervalMinutes(job.label) * 60000;
  const fromDate = new Date(new Date(last).getTime() - stepMs * 2);
  const from = job.label === '1D' ? localDate(fromDate) : localStamp(fromDate);
  const minAllowed = minDateForInterval(job.label);
  return from.slice(0, 10) < minAllowed ? minAllowed : from;
}

// ── Поиск пропусков ──
/**
 * Чистая функция поиска реальных дыр внутри торгового дня.
 * @param {number[]} begins timestamps по возрастанию
 * @param {number} stepMs длительность свечи
 * @param {{maxGaps?: number, maxGapMs?: number}} options
 */
function detectGaps(begins, stepMs, options = {}) {
  const maxGaps = options.maxGaps ?? MAX_GAPS_PER_RUN;
  const maxGapMs = options.maxGapMs ?? MAX_GAP_MS;
  const result = { gaps: [], sparse: false, truncated: false, denseShare: 0, total: 0 };

  if (!Array.isArray(begins) || begins.length < 3 || !(stepMs > 0)) return result;

  const spacings = [];
  for (let i = 1; i < begins.length; i++) {
    const diff = begins[i] - begins[i - 1];
    if (diff > 0) spacings.push(diff);
  }
  if (!spacings.length) return result;

  // У малоликвидных акций минуты без сделок выглядят как пропуски.
  // Если больше половины интервалов разрежены, поиск пропусков отключаем.
  const dense = spacings.filter((diff) => diff <= stepMs * 1.5).length;
  result.denseShare = dense / spacings.length;
  if (result.denseShare < 0.5) {
    result.sparse = true;
    return result;
  }

  const gaps = [];
  for (let i = 1; i < begins.length; i++) {
    const diff = begins[i] - begins[i - 1];
    // diff == 2 * step — это ровно одна отсутствующая свеча, её тоже чиним.
    if (diff <= stepMs * 1.5) continue;

    const prev = new Date(begins[i - 1]);
    const next = new Date(begins[i]);
    if (localDate(prev) !== localDate(next)) continue;

    const startMin = prev.getHours() * 60 + prev.getMinutes();
    const endMin = next.getHours() * 60 + next.getMinutes();
    if (startMin < 9 * 60 + 50 || endMin > 23 * 60 + 50) continue;
    // Утренний аукцион (09:50–10:00) и ежедневный техперерыв MOEX — не пропуски.
    // Важно не задевать реальные дыры после открытия: 10:04 → 10:06 чиним.
    if (startMin < 10 * 60 && endMin > 9 * 60 + 45) continue;
    if (startMin < 19 * 60 + 5 && endMin > 18 * 60 + 50) continue;
    if (diff > maxGapMs) continue;

    gaps.push({
      fromMs: begins[i - 1] + stepMs,
      tillMs: begins[i] - stepMs,
      sizeMs: diff - stepMs,
    });
  }

  result.total = gaps.length;
  gaps.sort((a, b) => b.sizeMs - a.sizeMs);
  if (gaps.length > maxGaps) {
    result.truncated = true;
    gaps.length = maxGaps;
  }
  result.gaps = gaps;
  return result;
}

async function findGaps(client, ticker, label, options = {}) {
  if (label === '1D') {
    return { gaps: [], sparse: false, truncated: false, denseShare: 0, total: 0 };
  }

  const stepMs = intervalMinutes(label) * 60000;
  const sinceDays = options.sinceDays ?? 7;
  const since = moscowDateMinusDays(sinceDays);
  const result = await client.query(
    `SELECT begin FROM candles WHERE ticker = $1 AND interval = $2 AND begin >= $3 ORDER BY begin ASC`,
    [ticker, label, since]
  );
  const begins = result.rows
    .map(r => new Date(r.begin).getTime())
    .filter(Number.isFinite)
    .sort((a, b) => a - b);

  const detected = detectGaps(begins, stepMs, options);
  return {
    ...detected,
    gaps: detected.gaps.map(gap => ({
      from: localStamp(new Date(gap.fromMs)),
      till: localStamp(new Date(gap.tillMs)),
      sizeMs: gap.sizeMs,
    })),
  };
}

// ── Загрузка диапазона ──
async function loadRange(client, ticker, apiUrl, from, till, intervalValue, label, options = {}) {
  const logger = options.logger || log;
  const progress = options.progress !== false;
  let start = 0;
  let saved = 0;
  let pages = 0;
  let skipped = 0;
  let skipSample = null;
  let lastPageFirst = null;

  while (true) {
    const data = await fetchPage(apiUrl, intervalValue, from, till, start);
    const block = data?.candles;
    const rows = Array.isArray(block?.data) ? block.data : [];
    if (!rows.length) break;
    if (!Array.isArray(block.columns)) {
      throw new Error('MOEX ISS: в ответе нет описания колонок');
    }

    const idx = Object.fromEntries(block.columns.map((name, i) => [name, i]));
    const missing = ['begin', 'open', 'close', 'high', 'low'].filter((name) => idx[name] === undefined);
    if (missing.length) {
      throw new Error(`MOEX ISS: в ответе нет колонок ${missing.join(', ')}`);
    }
    const parsed = [];
    for (const row of rows) {
      const item = parseRow(row, idx);
      if (item.error) {
        skipped += 1;
        if (!skipSample) skipSample = item.error;
        continue;
      }
      parsed.push(item.row);
    }

    const unique = dedupeRows(parsed);
    if (unique.length) await client.query(UPSERT_SQL, packRows(unique, ticker, label));
    saved += unique.length;
    pages += 1;
    start += rows.length;

    if (progress && (pages % 20 === 0 || rows.length < PAGE_SIZE)) {
      logger(`  ${ticker} ${label}: страниц=${pages}, свечей=${saved}, offset=${start}`);
    }
    if (rows.length < PAGE_SIZE) break;
    if (pages >= MAX_PAGES) {
      logger(`  ${ticker} ${label}: достигнут лимит страниц ${MAX_PAGES}`, 'warn');
      break;
    }

    // Защита от API, который игнорирует start и отдаёт одну и ту же страницу.
    const firstBegin = rows[0]?.[idx.begin];
    if (firstBegin && firstBegin === lastPageFirst) {
      logger(`  ${ticker} ${label}: API повторяет страницу, остановка`, 'warn');
      break;
    }
    lastPageFirst = firstBegin;

    if (PAGE_DELAY_MS) await sleep(PAGE_DELAY_MS);
  }

  if (skipped) {
    logger(`  ${ticker} ${label}: пропущено ${skipped} некорректных строк (${skipSample})`, 'warn');
  }
  if (saved && options.quiet !== true) {
    logger(`  ${ticker} ${label}: дозагружено ${saved} свечей с ${from} по ${till}`);
  }
  return saved;
}

async function loadJob(client, ticker, apiUrl, job, options = {}) {
  const startedAt = Date.now();
  const incremental = !!options.incremental;
  const fromOverride = options.from || null;
  const logger = options.logger || log;

  let from;
  if (incremental) {
    from = await incrementalFrom(client, ticker, job);
    // --from вместе с --incremental задаёт нижнюю границу.
    if (fromOverride && from.slice(0, 10) < fromOverride) from = fromOverride;
  } else if (fromOverride) {
    from = fromOverride;
  } else {
    from = minDateForInterval(job.label);
  }

  // Лимит истории для интервала: --from не может его расширить.
  const minAllowed = minDateForInterval(job.label);
  if (from.slice(0, 10) < minAllowed) {
    logger(`${ticker} ${job.label}: ${fromOverride ? `--from ${fromOverride}` : 'начало'} ограничено ${minAllowed}`);
    from = minAllowed;
  }

  const till = moscowToday();
  if (from.slice(0, 10) > till) {
    logger(`${ticker} ${job.label}: ${from} позже ${till}, пропуск`);
    return 0;
  }

  logger(`${ticker} ${job.label}: ${from} → ${till}${incremental ? ' (incremental)' : ''}`);
  const saved = await loadRange(client, ticker, apiUrl, from, till, job.interval, job.label, { logger });
  logger(`${ticker} ${job.label}: готово, ${saved} свечей за ${((Date.now() - startedAt) / 1000).toFixed(1)} с`);
  return saved;
}

// ── Открытые позиции (futoi) ──
function oiTillDate() {
  return moscowDateMinusDays(OI_EMBARGO_DAYS);
}

async function fetchOiWindow(apiUrl, from, till) {
  const url = new URL(apiUrl);
  url.searchParams.set('from', from);
  url.searchParams.set('till', till);

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(url, {
        headers: { 'User-Agent': 'ml-candles/3.1' },
        signal: controller.signal
      });
      if (response.status === 429 || response.status >= 500) {
        if (attempt >= MAX_RETRIES) {
          throw new Error(`MOEX ISS futoi: HTTP ${response.status} после ${MAX_RETRIES + 1} попыток`);
        }
        const retryAfter = Number(response.headers.get('retry-after')) || 0;
        const delay = retryAfter ? retryAfter * 1000 : 500 * 2 ** attempt;
        log(`  futoi HTTP ${response.status}, повтор через ${delay} мс (${attempt + 1}/${MAX_RETRIES})`);
        await sleep(delay);
        continue;
      }
      if (!response.ok) {
        const error = new Error(`HTTP ${response.status} ${response.statusText}`);
        error.retryable = false;
        throw error;
      }
      const data = await response.json();
      const block = data?.futoi;
      if (!block || !Array.isArray(block.columns) || !Array.isArray(block.data)) {
        throw new Error('MOEX ISS futoi: в ответе нет блока futoi');
      }
      return block;
    } catch (e) {
      if (e.retryable === false || attempt >= MAX_RETRIES) throw e;
      const delay = 500 * 2 ** attempt;
      log(`  futoi ${e.message}, повтор через ${delay} мс (${attempt + 1}/${MAX_RETRIES})`);
      await sleep(delay);
    } finally {
      clearTimeout(timer);
    }
  }
  throw new Error('Не удалось получить окно futoi');
}

const UPSERT_OI_SQL = `
  INSERT INTO open_interest (ticker, ts, clgroup, pos_long, pos_short, pos, pos_long_num, pos_short_num)
  SELECT * FROM unnest(
    $1::text[], $2::timestamp[], $3::text[], $4::bigint[], $5::bigint[],
    $6::bigint[], $7::integer[], $8::integer[]
  )
  ON CONFLICT (ticker, ts, clgroup) DO UPDATE SET
    pos_long = EXCLUDED.pos_long,
    pos_short = EXCLUDED.pos_short,
    pos = EXCLUDED.pos,
    pos_long_num = EXCLUDED.pos_long_num,
    pos_short_num = EXCLUDED.pos_short_num
`;

function oiInt(value) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.trunc(n) : null;
}

/**
 * Разбирает блок futoi. Строки-ошибки API («Invalid date…») и строки без
 * tradedate/clgroup пропускаются. Возвращает {rows, skipped, errorSample}.
 */
function parseOiBlock(block) {
  const idx = Object.fromEntries(block.columns.map((name, i) => [name, i]));
  const required = ['tradedate', 'tradetime', 'clgroup', 'pos'];
  const missing = required.filter((name) => idx[name] === undefined);
  if (missing.length) {
    throw new Error(`MOEX ISS futoi: в ответе нет колонок ${missing.join(', ')}`);
  }
  const rows = [];
  let skipped = 0;
  let errorSample = null;
  for (const row of block.data) {
    if (!Array.isArray(row) || row.length < block.columns.length) {
      skipped += 1;
      if (!errorSample) errorSample = String(row && row[0]).slice(0, 120);
      continue;
    }
    const tradedate = row[idx.tradedate];
    const tradetime = row[idx.tradetime];
    const clgroup = row[idx.clgroup];
    if (!/^\d{4}-\d{2}-\d{2}$/.test(String(tradedate || '')) ||
        !/^\d{2}:\d{2}:\d{2}$/.test(String(tradetime || '')) ||
        (clgroup !== 'FIZ' && clgroup !== 'YUR')) {
      skipped += 1;
      if (!errorSample) errorSample = `${tradedate} ${tradetime} ${clgroup}`;
      continue;
    }
    const pos = oiInt(row[idx.pos]);
    if (pos === null) {
      skipped += 1;
      continue;
    }
    rows.push({
      ts: `${tradedate} ${tradetime}`,
      clgroup,
      pos_long: oiInt(row[idx.pos_long]),
      pos_short: oiInt(row[idx.pos_short]),
      pos,
      pos_long_num: oiInt(row[idx.pos_long_num]),
      pos_short_num: oiInt(row[idx.pos_short_num]),
    });
  }
  return { rows, skipped, errorSample };
}

function dedupeOiRows(rows) {
  const map = new Map();
  for (const row of rows) map.set(`${row.ts}|${row.clgroup}`, row);
  return [...map.values()];
}

async function saveOiRows(client, ticker, rows) {
  const unique = dedupeOiRows(rows);
  if (!unique.length) return 0;
  await client.query(UPSERT_OI_SQL, [
    unique.map(() => ticker),
    unique.map(r => r.ts),
    unique.map(r => r.clgroup),
    unique.map(r => r.pos_long),
    unique.map(r => r.pos_short),
    unique.map(r => r.pos),
    unique.map(r => r.pos_long_num),
    unique.map(r => r.pos_short_num),
  ]);
  return unique.length;
}

/**
 * Грузит futoi окнами по OI_WINDOW_DAYS дней. API отдаёт не больше OI_PAGE_CAP
 * строк на запрос (самые свежие в диапазоне), поэтому при упоре в лимит окно
 * дробится на отдельные дни.
 */
async function loadOiRange(client, ticker, apiUrl, from, till, options = {}) {
  const logger = options.logger || log;
  let saved = 0;
  let skipped = 0;
  let errorSample = null;
  let windows = 0;
  let cursor = parseLocalDate(from);
  const end = parseLocalDate(till);

  while (cursor <= end) {
    const winEndDate = new Date(cursor);
    winEndDate.setDate(winEndDate.getDate() + OI_WINDOW_DAYS - 1);
    const winEnd = winEndDate < end ? winEndDate : end;
    const winFrom = localDate(cursor);
    const winTill = localDate(winEnd);

    let block = await fetchOiWindow(apiUrl, winFrom, winTill);
    let parsed = parseOiBlock(block);
    if (block.data.length >= OI_PAGE_CAP) {
      // Окно упёрлось в лимит 1000 строк — качаем каждый день отдельно.
      logger(`  ${ticker} oi: окно ${winFrom}→${winTill} упёрлось в лимит ${OI_PAGE_CAP}, дробим на дни`, 'warn');
      parsed = { rows: [], skipped: 0, errorSample: null };
      for (let d = new Date(cursor); d <= winEnd; d.setDate(d.getDate() + 1)) {
        const day = localDate(d);
        const dayBlock = await fetchOiWindow(apiUrl, day, day);
        const dayParsed = parseOiBlock(dayBlock);
        parsed.rows.push(...dayParsed.rows);
        parsed.skipped += dayParsed.skipped;
        if (!parsed.errorSample && dayParsed.errorSample) parsed.errorSample = dayParsed.errorSample;
        if (dayBlock.data.length >= OI_PAGE_CAP) {
          logger(`  ${ticker} oi: день ${day} упёрся в лимит ${OI_PAGE_CAP}, возможны потери`, 'warn');
        }
        if (PAGE_DELAY_MS) await sleep(PAGE_DELAY_MS);
      }
    }

    skipped += parsed.skipped;
    if (!errorSample && parsed.errorSample) errorSample = parsed.errorSample;
    saved += await saveOiRows(client, ticker, parsed.rows);
    windows += 1;
    if (windows % 50 === 0) {
      logger(`  ${ticker} oi: окон=${windows}, строк=${saved}, до=${winTill}`);
    }

    cursor = new Date(winEnd);
    cursor.setDate(cursor.getDate() + 1);
    if (PAGE_DELAY_MS) await sleep(PAGE_DELAY_MS);
  }

  if (skipped) {
    logger(`  ${ticker} oi: пропущено ${skipped} некорректных строк (${errorSample})`, 'warn');
  }
  return saved;
}

async function loadOiJob(client, ticker, options = {}) {
  const secid = futoiSecid(ticker);
  if (!secid) return 0;
  const logger = options.logger || log;
  const apiUrl = futoiUrl(secid);
  const till = oiTillDate();

  let from;
  if (options.incremental) {
    const result = await client.query(
      `SELECT MAX(ts) AS last FROM open_interest WHERE ticker = $1`, [ticker]
    );
    const last = result.rows[0]?.last;
    // Перехлёст в день: последний доступный день мог быть неполным.
    from = last ? localDate(new Date(new Date(last).getTime() - 86400000)) : FUTOI_MIN_DATE;
    if (options.from && from < options.from) from = options.from;
  } else {
    from = options.from || FUTOI_MIN_DATE;
  }
  if (from < FUTOI_MIN_DATE) from = FUTOI_MIN_DATE;

  if (from > till) {
    logger(`  ${ticker} oi: новых данных нет (доступно только до ${till}, embargo ${OI_EMBARGO_DAYS} дн.)`);
    return 0;
  }

  logger(`  ${ticker} oi (${secid}): ${from} → ${till}${options.incremental ? ' (incremental)' : ''}`);
  const startedAt = Date.now();
  const saved = await loadOiRange(client, ticker, apiUrl, from, till, { logger });
  const stats = await client.query(
    `SELECT COUNT(*)::int AS n, MIN(ts) AS first, MAX(ts) AS last
     FROM open_interest WHERE ticker = $1`, [ticker]
  );
  const s = stats.rows[0];
  logger(
    `  ${ticker} oi: готово, +${saved} строк за ${((Date.now() - startedAt) / 1000).toFixed(1)} с; ` +
    `всего ${s.n}, ${formatDb(s.first)} → ${formatDb(s.last)}`
  );
  return saved;
}

async function refreshTicker(client, tickerInfo, options = {}) {
  const startedAt = Date.now();
  const { ticker, engine, market } = tickerInfo;
  const incremental = options.incremental === undefined ? false : !!options.incremental;
  const logger = options.logger || log;
  // Полная загрузка сама проходит весь диапазон, поэтому отдельный поиск
  // пропусков нужен только для инкремента (или если явно включён).
  const gapsEnabled = options.gaps === undefined ? incremental : !!options.gaps;
  const gapDays = options.gapDays ?? 7;
  const apiUrl = process.env.MOEX_API_URL && ticker === 'IMOEX'
    ? process.env.MOEX_API_URL
    : candlesUrl(engine, market, ticker);

  logger(`═══ ${ticker} (${engine}/${market}) ═══`);
  logger(`API: ${apiUrl}`);

  const before = await client.query(
    `SELECT interval, COUNT(*)::int AS n, MIN(begin) AS first, MAX(begin) AS last
     FROM candles WHERE ticker = $1 GROUP BY interval ORDER BY interval`,
    [ticker]
  );
  for (const r of before.rows) {
    logger(`  до: ${r.interval}: ${r.n}, ${formatDb(r.first)} → ${formatDb(r.last)}`);
  }

  let total = 0;
  const errors = [];
  for (const job of JOBS) {
    try {
      if (gapsEnabled) {
        const found = await findGaps(client, ticker, job.label, {
          sinceDays: gapDays,
          maxGaps: MAX_GAPS_PER_RUN,
        });
        if (found.sparse) {
          logger(
            `  ${ticker} ${job.label}: пропуски не проверяем — данные разрежены ` +
            `(плотных интервалов ${(found.denseShare * 100).toFixed(1)}%)`,
            'debug'
          );
        } else if (found.total) {
          if (found.truncated) {
            logger(
              `  ${ticker} ${job.label}: найдено ${found.total} пропусков, ` +
              `проверяем ${found.gaps.length} крупнейших`,
              'warn'
            );
          }
          for (const gap of found.gaps) {
            logger(`  ${ticker} ${job.label}: пропуск ${gap.from} → ${gap.till}, дозагрузка`, 'debug');
            total += await loadRange(
              client, ticker, apiUrl, gap.from, gap.till, job.interval, job.label,
              { logger, quiet: true }
            );
          }
        }
      }

      total += await loadJob(client, ticker, apiUrl, job, {
        incremental,
        from: options.from,
        logger,
      });
    } catch (e) {
      errors.push({ interval: job.label, error: e.message });
      logger(`  ${ticker} ${job.label}: ошибка — ${e.message}`, 'error');
    }
  }

  // Открытые позиции FIZ/YUR (futoi) — для тикеров с фьючерсом на базовый актив.
  if (futoiSecid(ticker) && !options.noOi) {
    try {
      total += await loadOiJob(client, ticker, {
        incremental,
        from: options.from,
        logger,
      });
    } catch (e) {
      errors.push({ interval: 'oi', error: e.message });
      logger(`  ${ticker} oi: ошибка — ${e.message}`, 'error');
    }
  }

  const after = await client.query(
    `SELECT interval, COUNT(*)::int AS n, MIN(begin) AS first, MAX(begin) AS last
     FROM candles WHERE ticker = $1 GROUP BY interval ORDER BY interval`,
    [ticker]
  );
  logger(`${ticker}: получено ${total} строк${errors.length ? `, ошибок ${errors.length}` : ''} за ${((Date.now() - startedAt) / 1000).toFixed(1)} с`);
  for (const r of after.rows) {
    logger(`  после: ${r.interval}: ${r.n}, ${formatDb(r.first)} → ${formatDb(r.last)}`);
  }

  return { ticker, total, errors, startedAt, finishedAt: Date.now() };
}

async function main() {
  const cli = parseCliArgs(process.argv.slice(2));
  const client = new Client({ connectionString: process.env.DB });
  await client.connect();
  try {
    await ensureSchema(client);

    let targets;
    if (cli.tickers) {
      const names = cli.tickers.split(',').map(s => s.trim()).filter(Boolean);
      targets = [];
      for (const name of names) targets.push(await ensureTicker(client, name));
    } else {
      targets = await listTickers(client);
    }
    if (!targets.length) throw new Error('В таблице tickers нет ни одного тикера');
    log(`Тикеров к обновлению: ${targets.length} (${targets.map(t => t.ticker).join(', ')})`);

    let grandTotal = 0;
    const allErrors = [];
    for (const t of targets) {
      try {
        const result = await refreshTicker(client, t, cli);
        grandTotal += result.total;
        for (const error of result.errors) allErrors.push({ ticker: t.ticker, ...error });
      } catch (e) {
        allErrors.push({ ticker: t.ticker, error: e.message });
        log(`Ошибка обновления ${t.ticker}: ${e.message}`, 'error');
      }
    }

    log(`Всего получено строк: ${grandTotal}`);
    if (allErrors.length) {
      log(`Ошибок: ${allErrors.length} — ${allErrors.map(e => `${e.ticker} ${e.interval || ''}: ${e.error}`).join('; ')}`, 'error');
      process.exitCode = 1;
    }
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

module.exports = {
  JOBS,
  refreshTicker,
  detectGaps,
  findGaps,
  loadRange,
  loadJob,
  loadOiJob,
  loadOiRange,
  parseOiBlock,
  oiTillDate,
  parseCliArgs,
  intervalMinutes,
  intervalMaxYears,
  minDateForInterval,
  moscowToday,
  moscowDateMinusDays,
  subtractMonths,
  localDate,
  localStamp,
};
