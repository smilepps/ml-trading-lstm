// Тесты состояния модели в public/index.html.
// Фрагменты страницы (словарь причин, statusRow, ветка бейджа в renderRow и
// renderTable) извлекаются из файла и выполняются с заглушками DOM, чтобы
// проверялись именно те строки, которые работают в браузере:
//  - «модель есть, прогноза ещё нет» -> «Нет прогноза», а не «Нет модели»;
//  - «модель сломана» (файл есть, но связка неполна) -> отдельная подпись и причина;
//  - ошибку прогноза не путают с отсутствующей моделью;
//  - интервал со сломанной моделью не исчезает из таблицы.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '..', 'public', 'index.html'), 'utf8').replace(/\r\n/g, '\n');

function between(startMark, endMark, label) {
  const a = html.indexOf(startMark);
  const b = html.indexOf(endMark, a + startMark.length);
  assert.ok(a >= 0, `${label}: начало «${startMark}» не найдено`);
  assert.ok(b >= 0, `${label}: конец не найден`);
  return html.slice(a, b);
}

const statusBlock = (() => {
  const a = html.indexOf('const MODEL_REASON_TEXT = {');
  assert.ok(a >= 0, 'MODEL_REASON_TEXT не найден');
  const fn = html.indexOf('function statusRow(', a);
  assert.ok(fn >= 0, 'statusRow не найден');
  const end = html.indexOf('\n}\n', fn);
  assert.ok(end >= 0, 'конец statusRow не найден');
  return html.slice(a, end + 3);
})();

const renderRowBody = between('const status = modelStatusOf(p.interval, p);', 'tr.innerHTML = `', 'renderRow');
const renderTableSrc = (() => {
  const a = html.indexOf('function renderTable() {');
  assert.ok(a >= 0, 'renderTable не найден');
  const end = html.indexOf('\n}\n', a);
  assert.ok(end >= 0, 'конец renderTable не найден');
  return html.slice(a, end + 3);
})();

function buildUi() {
  const factory = new Function(`
    const intervalsMap = new Map();      // результаты /api/intervals
    const predictionsMap = new Map();    // результаты /api/predict
    const INTERVALS = ['1m', '10m', '60m', '1D'];
    let selectedIntervals = INTERVALS.slice();
    const activeTicker = 'SBER';
    const config = { threshold: 0.012, windowSize: 300, features: 7 };
    const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const timeOf = (v) => String(v);
    const tblBody = {
      innerHTML: '',
      rows: [],
      appendChild(row) { this.rows.push(row); },
    };
    ${statusBlock}
    function renderRow(p) {
      ${renderRowBody}
      return { interval: p.interval, shortStatus, statusCls, statusTitle, problemReason, pending, hasError, modelProblem };
    }
    ${renderTableSrc}
    function selectIntervals(list) { selectedIntervals = list; }
    function clearTable() { tblBody.innerHTML = ''; tblBody.rows = []; }
    return {
      intervalsMap, predictionsMap, tblBody, renderTable, renderRow, statusRow,
      modelStatusOf, modelReasonText, selectIntervals, clearTable,
    };
  `);
  const ui = factory();
  return ui;
}

const INTERVAL_INFO = {
  ok: { interval: '1m', candles: 300000, hasModel: true, modelStatus: { reason: 'ok', error: null }, last: '2026-11-23T12:00:00Z' },
  missingScaler: {
    interval: '10m', candles: 117000, hasModel: false,
    modelStatus: { reason: 'missing_scaler', error: 'Скейлер не найден: scaler_SBER_10m.json' },
    last: '2026-11-23T12:00:00Z',
  },
  missingModel: {
    interval: '1D', candles: 4991, hasModel: false,
    modelStatus: { reason: 'missing_model', error: 'Модель не найдена: model_SBER_1D.onnx' },
    last: '2026-11-23T00:00:00Z',
  },
  missingCtx: {
    interval: '60m', candles: 30522, hasModel: false,
    modelStatus: { reason: 'missing_context_scaler', error: 'Скейлер контекста не найден: scaler_SBER_1D.json' },
    last: '2026-11-23T11:00:00Z',
  },
};

function uiWith(rows) {
  const ui = buildUi();
  for (const row of rows) ui.intervalsMap.set(row.interval, row);
  return ui;
}

test('модель есть, прогноза ещё нет: строка не помечается «Нет модели»', () => {
  const ui = uiWith([INTERVAL_INFO.ok]);
  const row = ui.statusRow('1m');
  assert.equal(row.pending, true);
  assert.equal(row.errorReason, 'pending');
  assert.equal(row.error, 'прогноз ещё не рассчитан');
  const badge = ui.renderRow(row);
  assert.equal(badge.shortStatus, 'Нет прогноза');
  assert.equal(badge.statusCls, 'badge warn');
  assert.equal(badge.hasError, false);
});

test('скейлер пропал: «Модель сломана» с причиной от сервера', () => {
  const ui = uiWith([INTERVAL_INFO.missingScaler]);
  const row = ui.statusRow('10m');
  assert.equal(row.pending, false);
  assert.equal(row.errorReason, 'missing_scaler');
  const badge = ui.renderRow(row);
  assert.equal(badge.shortStatus, 'Модель сломана');
  assert.equal(badge.statusCls, 'badge err');
  assert.equal(badge.problemReason, 'missing_scaler');
  assert.match(badge.statusTitle, /Скейлер не найден: scaler_SBER_10m\.json/);
});

test('модели нет вовсе: «Нет модели»', () => {
  const ui = uiWith([INTERVAL_INFO.missingModel]);
  assert.equal(ui.renderRow(ui.statusRow('1D')).shortStatus, 'Нет модели');
});

test('сломанный скейлер контекста: «Модель сломана»', () => {
  const ui = uiWith([INTERVAL_INFO.missingCtx]);
  assert.equal(ui.renderRow(ui.statusRow('60m')).shortStatus, 'Модель сломана');
});

test('ошибка прогноза не выглядит как отсутствующая модель', () => {
  const ui = uiWith([INTERVAL_INFO.ok]);
  const badge = ui.renderRow({
    interval: '1m', ticker: 'SBER', error: 'ONNX: bad input', errorReason: 'predict_failed',
    modelStatus: { ok: true, reason: 'ok', error: null },
  });
  assert.equal(badge.shortStatus, 'Ошибка');
  assert.equal(badge.statusCls, 'badge err');
  assert.equal(badge.problemReason, 'predict_failed');
  assert.match(badge.statusTitle, /ONNX: bad input/);
});

test('интервал со сломанной моделью остаётся в таблице', () => {
  const ui = uiWith([INTERVAL_INFO.ok, INTERVAL_INFO.missingScaler, INTERVAL_INFO.missingModel, INTERVAL_INFO.missingCtx]);
  ui.predictionsMap.set('1m', { interval: '1m', ticker: 'SBER', pUp: 0.51, lastClose: 288.25 });
  ui.clearTable();
  ui.renderTable();
  const rendered = ui.tblBody.rows.map((r) => [r.interval, r.shortStatus]);
  assert.deepEqual(rendered, [['1m', 'OK'], ['10m', 'Модель сломана'], ['60m', 'Модель сломана'], ['1D', 'Нет модели']]);
});

test('без прогнозов таблица не пустая: показываем все выбранные интервалы', () => {
  const ui = uiWith(['1m', '10m', '60m', '1D'].map((interval) => ({ ...INTERVAL_INFO.ok, interval })));
  ui.clearTable();
  ui.renderTable();
  assert.equal(ui.tblBody.rows.length, 4);
  assert.ok(ui.tblBody.rows.every((r) => r.shortStatus === 'Нет прогноза'));
});

test('текст причины состояния модели человекочитаем: словарь причин', () => {
  const ui = buildUi();
  assert.equal(ui.modelReasonText({ ok: true, reason: 'ok' }), '');
  assert.equal(ui.modelReasonText({ ok: false, reason: 'broken_scaler', error: null }), 'скейлер повреждён');
  assert.equal(
    ui.modelReasonText({ ok: false, reason: 'broken_model', error: 'ONNX load failed' }),
    'ONNX load failed'
  );
  assert.equal(ui.modelReasonText({ ok: false, reason: 'какая-то-причина', error: null }), 'какая-то-причина');
});
