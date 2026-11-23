const STATUS_SQL = `
  SELECT pg_get_serial_sequence('candles', 'id') AS name,
         s.start_value::text AS start_value,
         s.increment_by::text AS increment_by,
         s.max_value::text AS max_value,
         s.last_value::text AS last_value,
         s.cycle,
         has_sequence_privilege(link.oid, 'USAGE, SELECT') AS accessible
  FROM (SELECT to_regclass(pg_get_serial_sequence('candles', 'id')) AS oid) AS link
  LEFT JOIN pg_class c ON c.oid = link.oid
  LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
  LEFT JOIN pg_sequences s ON s.schemaname = n.nspname AND s.sequencename = c.relname
`;

async function getSequenceStatus(client) {
  try {
    const { rows } = await client.query(STATUS_SQL);
    const row = rows[0];
    if (!row || !row.name || !row.max_value) {
      return { available: false, warning: 'Для candles.id не найдена sequence' };
    }
    if (!row.accessible) {
      return { available: false, name: row.name, warning: 'Нет доступа к sequence candles.id' };
    }

    const start = BigInt(row.start_value);
    const increment = BigInt(row.increment_by);
    const max = BigInt(row.max_value);
    const last = row.last_value == null ? null : BigInt(row.last_value);
    if (increment <= 0n || start > max || (last != null && (last < start || last > max))) {
      return { available: false, name: row.name, warning: 'Неподдерживаемые параметры sequence candles.id' };
    }

    const total = (max - start) / increment + 1n;
    const remaining = last == null ? total : (max - last) / increment;
    const usedBasisPoints = Number(((total - remaining) * 10000n) / total);
    const warning = row.cycle
      ? 'Sequence candles.id зациклена'
      : usedBasisPoints >= 9000
        ? 'Sequence candles.id почти исчерпана'
        : usedBasisPoints >= 8000
          ? 'Sequence candles.id израсходована более чем на 80%'
          : null;
    return {
      available: true,
      name: row.name,
      lastValue: last == null ? null : last.toString(),
      maxValue: max.toString(),
      remaining: remaining.toString(),
      usedPercent: usedBasisPoints / 100,
      cycle: row.cycle,
      warning,
    };
  } catch (error) {
    return { available: false, warning: `Не удалось проверить sequence candles.id: ${error.message}` };
  }
}

module.exports = { getSequenceStatus };
