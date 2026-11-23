#!/usr/bin/env node
/**
 * run_python.js — запускает Python-скрипт из .env-пути PYTHON.
 *
 * usage: node run_python.js <script.py> [args...]
 * Пример: node run_python.js train_gpu.py --threshold 0.003
 */

require('dotenv').config();
const { spawn } = require('child_process');
const path = require('path');

const script = process.argv[2];
if (!script) {
  console.error('Usage: node run_python.js <script.py> [args...]');
  process.exit(1);
}

const python = process.env.PYTHON || 'python';
const child = spawn(python, [path.join(__dirname, script), ...process.argv.slice(3)], {
  cwd: __dirname,
  stdio: 'inherit',
});

child.on('error', (e) => {
  console.error(`Не удалось запустить Python (${python}): ${e.message}`);
  process.exit(1);
});
child.on('close', (code) => process.exit(code == null ? 1 : code));
