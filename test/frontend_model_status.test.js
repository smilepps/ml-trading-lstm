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
const vm = require('node:vm');

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
    const predictionErrorsMap = new Map();
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

function deferred() {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
}

function browser() {
  class Element {
    constructor() { this.children = []; this._innerHTML = ''; this.value = ''; this.textContent = ''; this.classList = { toggle() {} }; }
    set innerHTML(v) { this._innerHTML = v; this.children = []; }
    get innerHTML() { return this._innerHTML; }
    get options() { return this.children; }
    get childElementCount() { return this.children.length; }
    get selectedOptions() { return this.children.filter(o => o.selected); }
    appendChild(child) { this.children.push(child); if (!this.value && child.value) this.value = child.value; return child; }
    append(...children) { children.forEach(c => this.appendChild(c)); }
    addEventListener() {}
    removeAttribute(name) { delete this[name]; }
  }
  const elements = new Map();
  const document = {
    getElementById(id) { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
    createElement() { return new Element(); },
  };
  const streams = [];
  class EventSource {
    constructor() { streams.push(this); }
    close() { this.closed = true; }
    send(message) { this.onmessage({ data: JSON.stringify(message) }); }
  }
  const timers = [];
  const mocks = { fetchJson: async () => { throw new Error('unexpected request'); } };
  const source = between('<script>', '// Инициализация', 'script').slice('<script>'.length);
  const sandbox = { document, EventSource, mocks, console, URL, Blob, navigator: {},
    setTimeout: (fn) => { timers.push(fn); }, setInterval() {}, clearInterval() {} };
  vm.runInNewContext(source + `
    fetchJson = (url, options) => mocks.fetchJson(url, options);
    this.ui = {
      loadIntervals, fetchPredict, loadConfig, loadSample, selectTicker, openStream, statusRow,
      setTicker(t) { activeTicker = t; tickerVersion++; resetTickerData(); },
      get ticker() { return activeTicker; }, get intervals() { return INTERVALS; },
      get config() { return config; }, predictionsMap, predictionErrorsMap, intervalsMap,
      sampleInfo, sampleInterval
    };`, sandbox);
  return { ui: sandbox.ui, mocks, streams, timers };
}

function row(interval, prediction = null, predictionError = null) {
  return { interval, candles: 100, hasModel: true, modelStatus: { ok: true, reason: 'ok', error: null }, prediction, predictionError };
}
function prediction(ticker, interval, pUp = 0.7) { return { ticker, interval, pUp, lastClose: 100 }; }

test('снимок интервалов удаляет null и исчезнувшие прогнозы, но сохраняет причину ошибки', async () => {
  const { ui, mocks } = browser();
  ui.setTicker('SBER');
  let rows = [row('1m', prediction('SBER', '1m')), row('10m', prediction('SBER', '10m'))];
  mocks.fetchJson = async () => rows;
  await ui.loadIntervals();
  assert.equal(ui.predictionsMap.size, 2);
  rows = [row('1m', null, { ticker: 'SBER', interval: '1m', error: 'Нет свечей', errorReason: 'predict_failed' })];
  await ui.loadIntervals(true);
  assert.equal(ui.predictionsMap.size, 0);
  assert.equal(ui.intervalsMap.has('10m'), false);
  assert.equal(ui.statusRow('1m').error, 'Нет свечей');
  assert.equal(ui.statusRow('1m').errorReason, 'predict_failed');
});

test('ответы старого тикера и старой версии того же тикера игнорируются', async () => {
  const { ui, mocks } = browser();
  const first = deferred();
  ui.setTicker('SBER');
  mocks.fetchJson = () => first.promise;
  const pending = ui.loadIntervals();
  ui.setTicker('GAZP');
  mocks.fetchJson = async () => [row('10m', prediction('GAZP', '10m'))];
  await ui.loadIntervals();
  first.resolve([row('1m', prediction('SBER', '1m'))]);
  await pending;
  assert.equal(ui.predictionsMap.get('10m').ticker, 'GAZP');
  assert.equal(ui.intervalsMap.has('1m'), false);
  const second = deferred();
  ui.setTicker('SBER');
  mocks.fetchJson = () => second.promise;
  const oldConfig = ui.loadConfig();
  ui.setTicker('GAZP');
  ui.setTicker('SBER');
  mocks.fetchJson = async () => ({ threshold: 0.123 });
  await ui.loadConfig();
  second.resolve({ threshold: 0.999 });
  await oldConfig;
  assert.equal(ui.config.threshold, 0.123);
});

test('поздние predict и sample старого тикера не попадают в новый', async () => {
  const { ui, mocks } = browser();
  ui.setTicker('SBER');
  mocks.fetchJson = async () => [row('1m', prediction('SBER', '1m'))];
  await ui.loadIntervals();
  const predictionReply = deferred();
  const sampleReply = deferred();
  mocks.fetchJson = url => url.startsWith('/api/predict') ? predictionReply.promise : sampleReply.promise;
  const pendingPrediction = ui.fetchPredict();
  const pendingSample = ui.loadSample();
  ui.setTicker('GAZP');
  mocks.fetchJson = async () => [row('10m', prediction('GAZP', '10m'))];
  await ui.loadIntervals();
  predictionReply.resolve([prediction('SBER', '1m', 0.99)]);
  sampleReply.reject(new Error('ошибка старого тикера'));
  await Promise.all([pendingPrediction, pendingSample]);
  assert.equal(ui.predictionsMap.size, 1);
  assert.equal(ui.predictionsMap.get('10m').ticker, 'GAZP');
  assert.notEqual(ui.sampleInfo.textContent, 'ошибка старого тикера');
});

test('ошибки fetchPredict не сохраняются как прогноз и не перекрывают новые события SSE', async () => {
  const { ui, mocks, streams } = browser();
  ui.setTicker('SBER');
  mocks.fetchJson = async () => [row('1m', prediction('SBER', '1m'))];
  await ui.loadIntervals();
  ui.openStream();
  const stream = streams[0];
  mocks.fetchJson = async () => [{ ticker: 'SBER', interval: '1m', error: 'ONNX error', errorReason: 'predict_failed' }];
  await ui.fetchPredict();
  assert.equal(ui.predictionsMap.has('1m'), false);
  assert.equal(ui.statusRow('1m').error, 'ONNX error');
  const waiting = deferred();
  mocks.fetchJson = () => waiting.promise;
  const pending = ui.fetchPredict();
  stream.send({ type: 'prediction', ticker: 'SBER', interval: '1m', data: prediction('SBER', '1m', 0.9) });
  waiting.resolve([{ ticker: 'SBER', interval: '1m', error: 'старый сбой', errorReason: 'predict_failed' }]);
  await pending;
  assert.equal(ui.predictionsMap.get('1m').pUp, 0.9);
  assert.equal(ui.predictionErrorsMap.size, 0);
  stream.send({ type: 'prediction_error', ticker: 'SBER', interval: '1m',
    modelStatus: { ok: false, reason: 'missing_scaler', error: 'Нет скейлера' },
    data: { ticker: 'SBER', interval: '1m', error: 'Нет скейлера', errorReason: 'missing_scaler' } });
  assert.equal(ui.predictionsMap.size, 0);
  assert.equal(ui.statusRow('1m').error, 'Нет скейлера');
  stream.send({ type: 'prediction', ticker: 'SBER', interval: '1m', data: prediction('SBER', '1m', 0.8) });
  assert.equal(ui.predictionsMap.get('1m').pUp, 0.8);
  assert.equal(ui.intervalsMap.get('1m').modelStatus.reason, 'ok');
});

test('события SSE новее снимка, переподключение пересинхронизирует и отбрасывает старый поток', async () => {
  const { ui, mocks, streams, timers } = browser();
  ui.setTicker('SBER');
  mocks.fetchJson = async () => [row('1m', prediction('SBER', '1m'))];
  await ui.loadIntervals();
  ui.openStream();
  const first = streams[0];
  const waiting = deferred();
  mocks.fetchJson = () => waiting.promise;
  const pending = ui.loadIntervals(true);
  first.send({ type: 'prediction_invalidated', ticker: 'SBER', interval: '1m',
    modelStatus: { ok: false, reason: 'missing_model', error: 'Модель удалена' } });
  waiting.resolve([row('1m', prediction('SBER', '1m'))]);
  await pending;
  assert.equal(ui.predictionsMap.size, 0);
  assert.equal(ui.statusRow('1m').error, 'Модель удалена');
  first.onerror();
  timers.shift()();
  const second = streams[1];
  first.send({ type: 'prediction', ticker: 'SBER', interval: '1m', data: prediction('SBER', '1m') });
  assert.equal(ui.predictionsMap.size, 0);
  const reconnect = deferred();
  mocks.fetchJson = () => reconnect.promise;
  second.send({ type: 'hello' });
  reconnect.resolve([row('1m', null, { ticker: 'SBER', interval: '1m', error: 'Нет свечей', errorReason: 'predict_failed' })]);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(ui.statusRow('1m').error, 'Нет свечей');
});
