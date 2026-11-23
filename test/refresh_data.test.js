const test = require('node:test');
const assert = require('node:assert/strict');

const refresh = require('../refresh_data');

test('loadRange дедуплицирует ревизии и сохраняет пачку одним атомарным запросом', async () => {
  const originalFetch = global.fetch;
  const queries = [];
  const client = { async query(sql, params) { queries.push({ sql, params }); return { rows: [] }; } };
  global.fetch = async () => ({ ok: true, json: async () => ({
    candles: {
      columns: ['begin', 'open', 'close', 'high', 'low', 'value', 'volume'],
      data: [
        ['2025-01-10 10:00:00', 1, 2, 3, 1, 10, 5],
        ['2025-01-10 10:00:00', 2, 3, 4, 1, 11, 6],
        ['2025-01-10 10:01:00', 3, 4, 5, 2, 12, 7],
      ],
    },
  }) });
  try {
    const saved = await refresh.loadRange(client, 'SBER', 'https://example.invalid/candles',
      '2025-01-10', '2025-01-10', 1, '1m', { progress: false, quiet: true });
    assert.equal(saved, 2);
    assert.equal(queries.length, 1);
    assert.match(queries[0].sql, /WITH incoming AS[\s\S]*UPDATE candles AS c[\s\S]*RETURNING[\s\S]*INSERT INTO candles/);
    assert.match(queries[0].sql, /WHERE NOT EXISTS[\s\S]*FROM updated/);
    assert.match(queries[0].sql, /ON CONFLICT \(ticker, interval, begin\) DO UPDATE/);
    assert.deepEqual(queries[0].params[2], ['2025-01-10 10:00:00', '2025-01-10 10:01:00']);
    assert.deepEqual(queries[0].params[3], [2, 3]);
    assert.deepEqual(queries[0].params[4], [3, 4]);
  } finally {
    global.fetch = originalFetch;
  }
});

function at(hour, minute, day = 5) {
  return new Date(2026, 0, day, hour, minute, 0, 0).getTime();
}

function minutes(startHour, startMinute, count, step = 1) {
  const out = [];
  for (let i = 0; i < count; i++) {
    const total = startHour * 60 + startMinute + i * step;
    out.push(at(Math.floor(total / 60), total % 60));
  }
  return out;
}

test('detectGaps находит одну пропущенную минутную свечу', () => {
  const begins = [
    ...minutes(10, 0, 7),
    // 10:07 и 10:08 отсутствуют
    ...minutes(10, 9, 12),
  ];
  const result = refresh.detectGaps(begins, 60_000);
  assert.equal(result.sparse, false);
  assert.equal(result.total, 1);
  assert.equal(result.gaps[0].sizeMs, 2 * 60_000);
  assert.equal(result.gaps[0].fromMs, at(10, 7));
  assert.equal(result.gaps[0].tillMs, at(10, 8));
});

test('detectGaps чинит дыру после открытия, но не утренний аукцион', () => {
  const afterOpen = [...minutes(10, 0, 5), ...minutes(10, 6, 5)];
  const r1 = refresh.detectGaps(afterOpen, 60_000);
  assert.equal(r1.total, 1);
  assert.equal(r1.gaps[0].fromMs, at(10, 5));
  assert.equal(r1.gaps[0].tillMs, at(10, 5));

  const auction = [...minutes(9, 55, 5), ...minutes(10, 1, 5)];
  const r2 = refresh.detectGaps(auction, 60_000);
  assert.equal(r2.total, 0);
});

test('detectGaps не считает пропуском обеденный перерыв MOEX', () => {
  const begins = [
    ...minutes(10, 0, 20),
    at(18, 45),
    at(19, 10),
    ...minutes(19, 11, 10),
  ];
  const result = refresh.detectGaps(begins, 60_000);
  assert.equal(result.total, 0);
});

test('detectGaps не ищет пропуски через границу дня', () => {
  const begins = [at(23, 40), at(23, 41), at(10, 0, 6)];
  const result = refresh.detectGaps(begins, 60_000);
  assert.equal(result.total, 0);
});

test('detectGaps отключается на разреженных данных', () => {
  const begins = [];
  for (let i = 0; i < 20; i++) begins.push(at(10, 0) + i * 5 * 60_000);
  const result = refresh.detectGaps(begins, 60_000);
  assert.equal(result.sparse, true);
  assert.equal(result.total, 0);
});

test('detectGaps ограничивает число пропусков', () => {
  const begins = [];
  let t = at(10, 0);
  for (let i = 0; i < 20; i++) {
    // Пара плотных минут, затем пропуск в 3 минуты.
    begins.push(t, t + 60_000);
    t += 4 * 60_000;
  }
  const result = refresh.detectGaps(begins, 60_000, { maxGaps: 5 });
  assert.equal(result.sparse, false);
  assert.equal(result.truncated, true);
  assert.equal(result.gaps.length, 5);
  assert.ok(result.total > 5);
});

test('parseCliArgs разбирает флаги и валидирует --from', () => {
  const cli = refresh.parseCliArgs([
    '--ticker', 'SBER,GAZP',
    '--incremental',
    '--no-gaps',
    '--gaps-days', '14',
    '--from', '2025-02-01',
  ]);
  assert.equal(cli.tickers, 'SBER,GAZP');
  assert.equal(cli.incremental, true);
  assert.equal(cli.gaps, false);
  assert.equal(cli.gapDays, 14);
  assert.equal(cli.from, '2025-02-01');

  assert.throws(() => refresh.parseCliArgs(['--from', '2025-02-30']), /Несуществующая дата/);
  assert.throws(() => refresh.parseCliArgs(['--from', '01.02.2025']), /Неверный формат/);
});

test('minDateForInterval не уезжает за конец месяца', () => {
  // 29 февраля в невисокосном году не должно превращаться в 1 марта.
  const date = new Date(2025, 1, 28);
  const shifted = refresh.subtractMonths(date, 1);
  assert.equal(refresh.localDate(shifted), '2025-01-28');
  assert.match(refresh.minDateForInterval('1m'), /^\d{4}-\d{2}-\d{2}$/);
  assert.match(refresh.minDateForInterval('1D'), /^\d{4}-\d{2}-\d{2}$/);
});

test('moscowDateMinusDays возвращает дату в формате БД', () => {
  assert.match(refresh.moscowDateMinusDays(7), /^\d{4}-\d{2}-\d{2}$/);
});

test('parseCliArgs понимает --no-oi', () => {
  assert.equal(refresh.parseCliArgs(['--no-oi']).noOi, true);
  assert.equal(refresh.parseCliArgs([]).noOi, false);
});

test('oiTillDate учитывает 14-дневное embargo бесплатного доступа', () => {
  assert.equal(refresh.oiTillDate(), refresh.moscowDateMinusDays(14));
});

test('parseOiBlock разбирает снимки FIZ/YUR и пропускает строки-ошибки API', () => {
  const columns = [
    'sess_id', 'seqnum', 'tradedate', 'tradetime', 'ticker', 'clgroup',
    'pos', 'pos_long', 'pos_short', 'pos_long_num', 'pos_short_num',
    'systime', 'trade_session_date',
  ];
  const block = {
    columns,
    data: [
      [7222, 57, '2025-01-10', '23:45:00', 'IMOEXF', 'FIZ', 204697, 349067, -144370, 4647, 2376, '2025-06-21 16:22:46', '2025-01-10'],
      [7222, 57, '2025-01-10', '23:45:00', 'IMOEXF', 'YUR', -204697, 39883, -244580, 24, 14, '2025-06-21 16:22:46', '2025-01-10'],
      ["Invalid date. Free users can't receive data for the last 14 days (2026-11-09)."],
      [7222, 58, '2025-01-10', '23:50:00', 'IMOEXF', 'XXX', 1, 1, 0, 1, 0, '2025-06-21 16:22:46', '2025-01-10'],
    ],
  };
  const { rows, skipped } = refresh.parseOiBlock(block);
  assert.equal(rows.length, 2);
  assert.equal(skipped, 2);
  assert.equal(rows[0].ts, '2025-01-10 23:45:00');
  assert.equal(rows[0].clgroup, 'FIZ');
  assert.equal(rows[0].pos, 204697);
  assert.equal(rows[0].pos_short, -144370);
  assert.equal(rows[1].clgroup, 'YUR');
  assert.equal(rows[1].pos, -204697);
});

test('parseOiBlock требует обязательные колонки', () => {
  assert.throws(
    () => refresh.parseOiBlock({ columns: ['tradedate'], data: [] }),
    /нет колонок/
  );
});
