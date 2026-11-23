const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS']);

function localSecurity(req, res, next) {
  const host = req.get('host');
  const match = typeof host === 'string' && /^(localhost|127\.0\.0\.1)(?::(\d{1,5}))?$/i.exec(host);
  if (!match || (match[2] && (Number(match[2]) < 1 || Number(match[2]) > 65535))) {
    return res.status(403).json({ error: 'Недопустимый Host' });
  }

  const origin = req.get('origin');
  if (origin && origin.toLowerCase() !== `http://${host.toLowerCase()}`) {
    return res.status(403).json({ error: 'Недопустимый Origin' });
  }

  if (!SAFE_METHODS.has(req.method) && req.get('x-requested-with') !== 'trading-dashboard') {
    return res.status(403).json({ error: 'Недопустимый запрос' });
  }

  return next();
}

module.exports = localSecurity;
