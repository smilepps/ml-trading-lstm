const { Client } = require('pg');

async function main() {
  const c = new Client({ connectionString: 'postgresql://smile:5ty765@127.0.0.1:5432/main' });
  await c.connect();
  try {
    // Последняя 5m свеча
    const r = await c.query(
      "SELECT MAX(begin) AS last_5m FROM candles WHERE ticker='IMOEX' AND interval='5m'"
    );
    console.log('Последняя 5m в БД:', r.rows[0].last_5m);

    // Последняя 10m свеча для сравнения
    const r2 = await c.query(
      "SELECT MAX(begin) AS last_10m FROM candles WHERE ticker='IMOEX' AND interval='10m'"
    );
    console.log('Последняя 10m в БД:', r2.rows[0].last_10m);

    // Текущее время
    const now = new Date();
    console.log('Текущее время (UTC):', now.toISOString());
    console.log('Текущее время (MSK):', now.toLocaleString('ru-RU', { timeZone: 'Europe/Moscow' }));
  } finally {
    await c.end();
  }
}

main().catch(e => { console.error(e.message); process.exit(1); });
