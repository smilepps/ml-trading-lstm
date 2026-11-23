const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const realInf = require('../inference_v4');
const thresholds = require('../thresholds');

function scaler(interval, extra = {}) {
  return {
    ticker: 'SBER', interval, version: 4, window: 300, order: 'current_to_history',
    contexts: [...realInf.CONTEXT_MAP[interval]], inputs: 1 + realInf.CONTEXT_MAP[interval].length,
    input_size: 7, feature_names: [...realInf.BASE_FEATURE_NAMES],
    ret_mean: 0, ret_std: 1, value_mean: 0, value_std: 1,
    dow_mean: 0, dow_std: 1, hour_mean: 0, hour_std: 1, ...extra,
  };
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
}

function buildServer() {
  const root = path.join(__dirname, 'virtual-server');
  const files = new Map();
  let revision = 0;
  const routes = new Map();
  const events = [];
  const settings = { loadError: null, predict: null, rows: [{ interval: '1D', cnt: 400, first: null, last: null }] };
  const file = (name) => path.join(root, 'models', name);
  function put(name, data) {
    revision++;
    files.set(file(name), { data, stat: { size: data.length, mtimeMs: revision, ctimeMs: revision, ino: revision } });
  }
  function remove(name) { files.delete(file(name)); }
  const fakeFs = {
    existsSync: (name) => files.has(name),
    readFileSync(name) { if (!files.has(name)) throw new Error(`ENOENT ${name}`); return files.get(name).data; },
    statSync(name) { if (!files.has(name)) throw new Error(`ENOENT ${name}`); return files.get(name).stat; },
    appendFileSync() {},
  };
  const fakeInf = {
    ...realInf,
    async loadModel(ticker, interval) {
      if (settings.loadError) throw new Error(settings.loadError);
      const base = JSON.parse(fakeFs.readFileSync(file(`scaler_${ticker}_${interval}.json`)));
      const paths = [file(`model_${ticker}_${interval}.onnx`), file(`scaler_${ticker}_${interval}.json`),
        ...base.contexts.map(ctx => file(`scaler_${ticker}_${ctx}.json`))];
      return { files: paths.map(name => ({ name, stat: { ...fakeFs.statSync(name) } })) };
    },
    isModelFresh(model) {
      return model.files.every(({ name, stat }) => {
        try { return JSON.stringify(fakeFs.statSync(name)) === JSON.stringify(stat); } catch { return false; }
      });
    },
    async predictWithModel(client, model, ticker, interval) {
      if (settings.predict) return settings.predict(client, model, ticker, interval);
      return { pUp: 0.7, pDown: 0.3, direction: 'UP', confidence: 0.7, lastClose: 100,
        lastBegin: null, threshold: 0.02, steps: 1, contexts: [], window: 300, inputShape: [1, 300, 7] };
    },
  };
  const client = { async query() { return { rows: settings.rows }; }, release() {}, async end() {}, on() {} };
  const express = () => ({
    use() {},
    get(route, handler) { routes.set(`GET ${route}`, handler); },
    post(route, handler) { routes.set(`POST ${route}`, handler); },
    delete(route, handler) { routes.set(`DELETE ${route}`, handler); },
  });
  express.static = () => () => {};
  express.json = () => () => {};
  class Pool { on() {} async connect() { return client; } }
  const source = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf8').split('// ── Запуск ──')[0];
  const sandbox = {
    __dirname: root, process: { env: {} }, console, setInterval, clearInterval, setTimeout, clearTimeout,
    require(name) {
      if (name === 'dotenv') return { config() {} };
      if (name === 'express') return express;
      if (name === 'http') return { createServer: () => ({}) };
      if (name === 'fs') return fakeFs;
      if (name === 'path') return path;
      if (name === 'child_process') return {};
      if (name === 'pg') return { Pool, Client: class {} };
      if (name === './tickers') return { validateTicker: (v) => v, listTickers: async () => [{ ticker: 'SBER' }] };
      if (name === './refresh_data') return { refreshTicker: async () => {} };
      if (name === './inference_v4') return fakeInf;
      if (name === './thresholds') return thresholds;
      throw new Error(name);
    },
  };
  vm.runInNewContext(source + `\nthis.testApi = {
    modelStatus, listIntervals, predictOne, monitorTick, predictions, predictionErrors,
    setMonitorClient(c) { monitorClient = c; },
    addClient(c) { sseClients.push(c); },
    invalidateIntervalCache
  };`, sandbox);
  const api = sandbox.testApi;
  api.addClient({ write(chunk) { events.push(JSON.parse(chunk.slice(6))); } });
  api.setMonitorClient(client);
  function putScaler(interval, extra = {}) { put(`scaler_SBER_${interval}.json`, JSON.stringify(scaler(interval, extra))); }
  async function route(name, query = {}, params = {}) {
    let response;
    await routes.get(`GET ${name}`)({ query, params }, { json(data) { response = data; }, status() { return this; } });
    return response;
  }
  return { api, client, put, remove, putScaler, events, settings, route };
}

test('modelStatus различает отсутствие модели и скейлера, проверяет контексты', () => {
  const { api, put, putScaler, remove } = buildServer();
  assert.equal(api.modelStatus('SBER', '1D').reason, 'missing_model');
  put('model_SBER_1D.onnx', 'model');
  assert.equal(api.modelStatus('SBER', '1D').reason, 'missing_scaler');
  putScaler('10m', { contexts: ['15m', '1D'] });
  put('model_SBER_10m.onnx', 'model');
  assert.match(api.modelStatus('SBER', '10m').error, /15m/);
  assert.equal(api.modelStatus('SBER', '10m').reason, 'broken_scaler');
  putScaler('10m', { contexts: ['1D', '60m'] });
  assert.match(api.modelStatus('SBER', '10m').error, /ожидаются/);
  putScaler('10m');
  assert.equal(api.modelStatus('SBER', '10m').reason, 'missing_context_scaler');
  putScaler('60m');
  putScaler('1D', { order: 'history_to_current' });
  assert.match(api.modelStatus('SBER', '10m').error, /current_to_history/);
  putScaler('1D');
  assert.equal(api.modelStatus('SBER', '10m').reason, 'ok');
  remove('model_SBER_10m.onnx');
  assert.equal(api.modelStatus('SBER', '10m').reason, 'missing_model');
  assert.equal(api.modelStatus('SBER', '15m').reason, 'unsupported_interval');
});

test('API сохраняет ошибки отдельно, инвалидирует старую модель и восстанавливает прогноз', async () => {
  const env = buildServer();
  const { api, put, putScaler, events, settings, route } = env;
  putScaler('1D');
  put('model_SBER_1D.onnx', 'first');
  let result = await route('/api/predict', { ticker: 'SBER', intervals: '1D' });
  assert.ok(Array.isArray(result));
  assert.equal(result[0].pUp, 0.7);
  assert.equal((await route('/api/intervals', { ticker: 'SBER' }))[0].prediction.pUp, 0.7);
  put('model_SBER_1D.onnx', 'replacement');
  let row = (await route('/api/intervals', { ticker: 'SBER' }))[0];
  assert.equal(row.prediction, null);
  assert.equal(row.hasModel, true);
  assert.ok(events.some(e => e.type === 'prediction_invalidated' && e.interval === '1D'));
  result = await route('/api/predict/:interval', { ticker: 'SBER' }, { interval: '1D' });
  assert.equal(result.pUp, 0.7);
  settings.predict = async () => { throw new Error('Нет свечей'); };
  result = await api.predictOne(env.client, 'SBER', '1D');
  assert.equal(result.errorReason, 'predict_failed');
  row = (await api.listIntervals(env.client, 'SBER'))[0];
  assert.equal(row.prediction, null);
  assert.equal(row.predictionError.error, 'Нет свечей');
  assert.equal(row.modelStatus.reason, 'ok');
  assert.ok(events.some(e => e.type === 'prediction_error' && e.data.error === 'Нет свечей'));
  settings.predict = null;
  await api.predictOne(env.client, 'SBER', '1D');
  assert.equal((await api.listIntervals(env.client, 'SBER'))[0].predictionError, null);
});

test('монитор публикует ошибки недоступной модели и восстанавливается после замены', async () => {
  const env = buildServer();
  const { api, client, put, putScaler, remove, events, settings } = env;
  putScaler('1D'); put('model_SBER_1D.onnx', 'model');
  await api.predictOne(client, 'SBER', '1D');
  remove('scaler_SBER_1D.json');
  await api.monitorTick();
  assert.equal(api.predictions.has('SBER|1D'), false);
  assert.ok(events.some(e => e.type === 'prediction_error' && e.data.errorReason === 'missing_scaler'));
  putScaler('1D');
  settings.loadError = 'ONNX bad';
  await api.monitorTick();
  assert.equal(api.modelStatus('SBER', '1D').reason, 'broken_model');
  settings.loadError = null;
  put('model_SBER_1D.onnx', 'new model');
  await api.monitorTick();
  assert.equal(api.modelStatus('SBER', '1D').reason, 'ok');
  assert.equal(api.predictions.get('SBER|1D').pUp, 0.7);
});

test('завершившаяся раньше старая попытка не перезаписывает новую; замена во время расчёта отвергается', async () => {
  const env = buildServer();
  const { api, client, put, putScaler, settings } = env;
  putScaler('1D'); put('model_SBER_1D.onnx', 'model');
  const waiting = deferred();
  let calls = 0;
  settings.predict = async () => {
    if (++calls === 1) return waiting.promise;
    return { pUp: 0.8, pDown: 0.2, lastClose: 100, threshold: 0.02, inputShape: [1, 300, 7] };
  };
  const old = api.predictOne(client, 'SBER', '1D');
  await new Promise(resolve => setImmediate(resolve));
  await api.predictOne(client, 'SBER', '1D');
  waiting.reject(new Error('старый сбой'));
  assert.equal((await old).errorReason, 'predict_failed');
  assert.equal(api.predictions.get('SBER|1D').pUp, 0.8);
  const waitChange = deferred();
  settings.predict = () => waitChange.promise;
  const running = api.predictOne(client, 'SBER', '1D');
  await new Promise(resolve => setImmediate(resolve));
  put('model_SBER_1D.onnx', 'replacement');
  waitChange.resolve({ pUp: 0.9, pDown: 0.1, lastClose: 100, threshold: 0.02 });
  assert.match((await running).error, /изменились/);
  assert.equal(api.predictions.has('SBER|1D'), false);
});

test('исчезнувший интервал удаляется из серверного снимка и рассылается', async () => {
  const env = buildServer();
  env.putScaler('1D'); env.put('model_SBER_1D.onnx', 'model');
  await env.api.predictOne(env.client, 'SBER', '1D');
  env.settings.rows = [];
  env.api.invalidateIntervalCache('SBER');
  assert.equal((await env.api.listIntervals(env.client, 'SBER')).length, 0);
  assert.equal(env.api.predictions.size, 0);
  assert.ok(env.events.some(e => e.type === 'interval_removed' && e.interval === '1D'));
});
