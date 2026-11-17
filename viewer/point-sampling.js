export function linspaceIndices(total, limit) {
  if (!Number.isSafeInteger(total) || total < 0 || total > 0xffffffff || !Number.isSafeInteger(limit) || limit < 0) throw new Error('Invalid point sampling budget');
  const count = Math.min(total, limit);
  const indices = new Uint32Array(count);
  for (let index = 0; index < count; index++) indices[index] = count > 1 ? Math.floor(index * (total - 1) / (count - 1)) : 0;
  return indices;
}
