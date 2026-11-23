/**
 * predict.js — одиночный прогноз через ONNX Runtime (scaler v4).
 *
 * Вход модели: 300 свечей текущего интервала + по 300 свечей старших
 * интервалов, 7 признаков на свечу. Строка 0 — текущая свеча.
 *
 * usage: node predict.js <interval> [ticker]
 */

require('dotenv').config();
const { Client } = require('pg');
const {
  loadModel,
  predictWithModel,
  WINDOW_SIZE,
} = require('./inference_v4');
const thresholds = require('./thresholds');

const interval = String(process.argv[2] || '').trim();
const ticker = (process.argv[3] || process.env.TICKER || 'IMOEX').toUpperCase();
const envBaseThreshold = parseFloat(process.env.THRESHOLD || '0.003');
if (!Number.isFinite(envBaseThreshold) || envBaseThreshold <= 0) {
  throw new Error('THRESHOLD must be a positive number (e.g. 0.003)');
}

if (!interval) {
  console.error('Usage: node predict.js <interval> [ticker]');
  console.error('Example: node predict.js 60m IMOEX');
  process.exit(1);
}

try {
  thresholds.validateInterval(interval);
} catch (e) {
  console.error(`❌ ${e.message}`);
  process.exit(1);
}

async function main() {
  const model = await loadModel(ticker, interval);

  // Порог берётся из скейлера: именно с ним обучалась модель.
  // Пересчёт по текущему .env менял бы цель без переобучения.
  const meta = thresholds.resolveModelThreshold(model.scaler, interval, envBaseThreshold);
  if (Math.abs(meta.base - envBaseThreshold) > 1e-9) {
    console.warn(
      `⚠️  Внимание: базовый порог в скейлере (${meta.base}) отличается от текущего ` +
      `в .env (${envBaseThreshold}). Переобучите модель, если это не задумано.`
    );
  }

  const client = new Client({ connectionString: process.env.DB });
  await client.connect();

  try {
    const r = await predictWithModel(client, model, ticker, interval, {
      fallbackBaseThreshold: envBaseThreshold,
    });

    const threshold = meta.effective;
    const upTarget = (r.lastClose * (1 + threshold)).toFixed(2);
    const downTarget = (r.lastClose * (1 - threshold)).toFixed(2);

    console.log(`\n🔮 PREDICT ${ticker} @ ${interval}`);
    console.log(`   Last candle: ${r.lastBegin} close=${r.lastClose.toFixed(2)}`);
    console.log(
      `   Input: ${r.steps} × ${WINDOW_SIZE} × 7` +
      ` (current${r.contexts.length ? ` + ${r.contexts.join(' + ')}` : ''})`
    );
    console.log(
      `   Threshold: база ${(meta.base * 100).toFixed(2)}% × ${meta.multiplier} ` +
      `= ${(threshold * 100).toFixed(2)}% ` +
      `(${meta.progressive ? 'прогрессивный' : 'фиксированный'}, горизонт ${meta.horizon}, источник ${meta.source})`
    );
    console.log(
      `   P(up   ≥ +${(threshold * 100).toFixed(1)}% за ${meta.horizon} свеч): ` +
      `${(r.pUp * 100).toFixed(2)}%  (цель ≥ ${upTarget})`
    );
    console.log(
      `   P(не UP) за ${meta.horizon} свеч: ` +
      `${(r.pDown * 100).toFixed(2)}%  (down-цель ≤ ${downTarget})`
    );
    console.log(
      `   Direction: ${r.direction}  Confidence: ${(r.confidence * 100).toFixed(2)}%`
    );
  } finally {
    await client.end().catch(() => {});
  }
}

main().catch((e) => {
  console.error('❌ Error:', e.message);
  process.exit(1);
});
