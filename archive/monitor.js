/**
 * monitor.js — фоновый мониторинг свечей через ONNX-модели scaler v4.
 *
 * Для каждого интервала выводит P(up)/P(down) по последней закрытой свече.
 * После истечения горизонта (10 свечей) помечает цель:
 *   🟢 зелёный — цель достигнута (прогноз сработал)
 *   🔴 красный — цель не достигнута (прогноз провалился)
 *
 * usage: node monitor.js
 * env:   INTERVALS, TICKER, SLEEP_MS, THRESHOLD
 */

require('dotenv').config();
const { Client } = require('pg');
const {
  loadModel,
  predictWithModel,
  WINDOW_SIZE,
} = require('./inference_v4');

const HORIZON = 10;
const THRESHOLD = parseFloat(process.env.THRESHOLD || '0.003');
const TICKER = (process.env.TICKER || 'IMOEX').toUpperCase();
const INTERVALS = process.env.INTERVALS
  ? process.env.INTERVALS.split(',').map((s) => s.trim()).filter(Boolean)
  : ['10m', '60m', '4h', '1D', '2h'];
const SLEEP_MS = parseInt(process.env.SLEEP_MS || '60000', 10);

const models = new Map();
const history = {}; // interval -> [{ direction, lastClose, lastBegin, checked }]

function nowStr() {
  return new Date().toISOString().replace('T', ' ').substring(0, 19);
}

function colorResult(achieved) {
  return achieved ? '🟢' : '🔴';
}

async function predictInterval(client, interval) {
  const model = models.get(interval);
  if (!model) return;

  try {
    const r = await predictWithModel(client, model, TICKER, interval);

    if (!history[interval]) history[interval] = [];
    history[interval].push({
      direction: r.direction,
      lastClose: r.lastClose,
      lastBegin: r.lastBegin,
      pUp: r.pUp,
      checked: false,
    });

    await checkHistory(interval, r.candles);

    const dirEmoji = r.direction === 'UP' ? '🟢' : '🔴';
    console.log(
      `${nowStr()} | ${TICKER} ${interval} | P(up)=${(r.pUp * 100).toFixed(2)}% ` +
      `P(down)=${(r.pDown * 100).toFixed(2)}% | close=${r.lastClose.toFixed(2)} | ` +
      `${dirEmoji} ${r.direction} conf=${(r.confidence * 100).toFixed(2)}% | ` +
      `steps=${r.steps}×${WINDOW_SIZE}×7`
    );
  } catch (e) {
    console.log(`${nowStr()} | ${TICKER} ${interval} | ❌ ${e.message}`);
  }
}

async function checkHistory(interval, candles) {
  const hist = history[interval];
  if (!hist || hist.length === 0) return;

  const indexByTime = new Map(
    candles.map((c, i) => [new Date(c.begin).getTime(), i])
  );

  for (const rec of hist) {
    if (rec.checked) continue;

    // Следующие HORIZON свечей после прогноза
    const idx = indexByTime.get(new Date(rec.lastBegin).getTime());
    if (idx == null || idx + HORIZON >= candles.length) continue;

    const future = candles.slice(idx + 1, idx + 1 + HORIZON);
    const maxHigh = Math.max(...future.map((c) => c.high));
    const minLow = Math.min(...future.map((c) => c.low));

    const upTarget = rec.lastClose * (1 + THRESHOLD);
    const downTarget = rec.lastClose * (1 - THRESHOLD);

    let achieved = false;
    if (rec.direction === 'UP') achieved = maxHigh >= upTarget;
    if (rec.direction === 'DOWN') achieved = minLow <= downTarget;

    rec.checked = true;

    console.log(
      `${nowStr()} | ${TICKER} ${interval} | ${colorResult(achieved)} ТАРГЕТ ${rec.direction} ` +
      `(от ${rec.lastBegin}, close=${rec.lastClose.toFixed(2)}): ` +
      `${achieved ? 'ДОСТИГНУТ' : 'ПРОВАЛ'} ` +
      `(high=${maxHigh.toFixed(2)}, low=${minLow.toFixed(2)}, ` +
      `цели up=${upTarget.toFixed(2)}, down=${downTarget.toFixed(2)})`
    );
  }

  // Не храним бесконечную историю проверенных прогнозов
  history[interval] = hist.filter((rec) => !rec.checked).slice(-HORIZON * 2);
}

async function main() {
  console.log(
    `🖥️  Monitor started: ticker=${TICKER} intervals=[${INTERVALS.join(', ')}] ` +
    `sleep=${SLEEP_MS}ms threshold=${THRESHOLD} (scaler v4)`
  );

  for (const interval of INTERVALS) {
    try {
      const model = await loadModel(TICKER, interval);
      models.set(interval, model);
      history[interval] = [];

      if (Math.abs((model.scaler.threshold || 0) - THRESHOLD) > 1e-9) {
        console.log(
          `   ⚠️  ${interval}: порог в скейлере ${model.scaler.threshold}, ` +
          `а в .env ${THRESHOLD} — переобучите модель`
        );
      }
      console.log(
        `   ✅ Loaded model ${interval} (${1 + model.contexts.length}×${WINDOW_SIZE}×7` +
        `${model.contexts.length ? `, контексты: ${model.contexts.join(', ')}` : ''})`
      );
    } catch (e) {
      console.log(`   ⚠️  Model ${interval} недоступна: ${e.message}`);
    }
  }

  if (models.size === 0) {
    throw new Error(`Нет ни одной модели для ${TICKER} в C:\\ml\\trading\\models`);
  }

  const client = new Client({ connectionString: process.env.DB });
  await client.connect();

  console.log(`\n${nowStr()} — started real-time predictions (Ctrl+C to stop)\n`);

  let first = true;
  while (true) {
    if (!first) {
      await new Promise((resolve) => setTimeout(resolve, SLEEP_MS));
    }
    first = false;

    for (const interval of INTERVALS) {
      if (models.has(interval)) await predictInterval(client, interval);
    }
    console.log('');
  }
}

main().catch((e) => {
  console.error('❌ Error:', e.message);
  process.exit(1);
});
