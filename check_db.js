require('dotenv').config();
const { Client } = require('pg');

async function main() {
  const client = new Client({ connectionString: process.env.DB });
  await client.connect();
  console.log('✅ Connected to PostgreSQL');

  const result = await client.query(`
    SELECT interval, COUNT(*) as cnt, MIN(begin) as first, MAX(begin) as last
    FROM candles
    WHERE ticker = 'IMOEX'
    GROUP BY interval
    ORDER BY interval
  `);

  console.table(result.rows);
  await client.end();
}

main().catch(console.error);