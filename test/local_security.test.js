const test = require('node:test');
const assert = require('node:assert/strict');
const express = require('express');
const http = require('node:http');
const localSecurity = require('../local_security');

test('локальная панель отклоняет посторонние Host, Origin и простые межсайтовые запросы', async () => {
  const app = express();
  app.use(localSecurity);
  app.get('/', (_req, res) => res.json({ ok: true }));
  app.post('/change', (_req, res) => res.json({ ok: true }));
  const server = await new Promise((resolve) => {
    const instance = app.listen(0, '127.0.0.1', () => resolve(instance));
  });

  try {
    const base = `http://127.0.0.1:${server.address().port}`;
    assert.equal((await fetch(base)).status, 200);
    const hostileHost = await new Promise((resolve, reject) => {
      http.get(base, { headers: { Host: 'attacker.example' } }, (res) => {
        res.resume();
        res.on('end', () => resolve(res.statusCode));
      }).on('error', reject);
    });
    assert.equal(hostileHost, 403);
    assert.equal((await fetch(base, { headers: { Origin: 'http://attacker.example' } })).status, 403);
    assert.equal((await fetch(`${base}/change`, { method: 'POST' })).status, 403);
    assert.equal((await fetch(`${base}/change`, {
      method: 'POST',
      headers: { Origin: 'http://attacker.example', 'X-Requested-With': 'trading-dashboard' },
    })).status, 403);
    assert.equal((await fetch(`${base}/change`, {
      method: 'POST',
      headers: { Origin: base, 'X-Requested-With': 'trading-dashboard' },
    })).status, 200);
    assert.equal((await fetch(`${base}/change`, {
      method: 'POST',
      headers: { 'X-Requested-With': 'trading-dashboard' },
    })).status, 200);
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
});
