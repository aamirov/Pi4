const TYPES = {'<f4': Float32Array, '|u1': Uint8Array};

export function dataURL(path) {
  return new URL(`data/${assetPath(path)}`, window.location.href).href;
}

export async function readJSON(path) {
  const response = await fetch(dataURL(path));
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

export async function readBundle(description, signal) {
  const abort = new AbortController();
  const cancel = () => abort.abort(signal.reason);
  signal?.addEventListener('abort', cancel, {once: true});
  if (signal?.aborted) cancel();
  const timeout = setTimeout(() => abort.abort(new Error(`${description.file}: download timed out; check the forwarded connection`)), 30000);
  try {
    const response = await fetch(dataURL(description.file), {signal: abort.signal});
    if (!response.ok) throw new Error(`${description.file}: HTTP ${response.status}`);
    return decodeBundle(await response.arrayBuffer(), description);
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener('abort', cancel);
  }
}

export function playbackTimestamps(scene) {
  const source = scene.source.source_timestamps_s;
  if (Array.isArray(source) && source.length === scene.frames.length &&
      source.every((time, index) => Number.isFinite(time) && (!index || time > source[index - 1]))) {
    return source.map(time => time - source[0]);
  }
  return scene.frames.map((unused, index) => index / 10);
}

function requireValue(condition, message) {
  if (!condition) throw new Error(message);
}

export function assetPath(value) {
  requireValue(typeof value === 'string' && value.length > 0 && !/[\\?#:%]/.test(value), 'Pi4 资源路径无效');
  requireValue(!value.startsWith('/') && value.split('/').every(part => part && part !== '.' && part !== '..'), 'Pi4 资源必须使用安全的相对路径');
  return value;
}

export function decodeBundle(buffer, description) {
  requireValue(buffer instanceof ArrayBuffer && buffer.byteLength === description.bytes, 'Pi4 二进制文件长度不匹配');
  const arrays = {};
  const ranges = [];
  for (const [name, field] of Object.entries(description.arrays)) {
    const Type = TYPES[field.dtype];
    requireValue(Type && Array.isArray(field.shape) && field.shape.length > 0, `数组 ${name} 类型或维度无效`);
    requireValue(field.shape.every(size => Number.isSafeInteger(size) && size >= 0), `数组 ${name} 形状无效`);
    const count = field.shape.reduce((total, size) => total * size, 1);
    requireValue(Number.isSafeInteger(count) && Number.isSafeInteger(field.offset) && field.offset >= 0 && field.offset % Math.max(4, Type.BYTES_PER_ELEMENT) === 0, `数组 ${name} 偏移无效`);
    requireValue(field.bytes === count * Type.BYTES_PER_ELEMENT && field.offset + field.bytes <= buffer.byteLength, `数组 ${name} 越界或长度错误`);
    if (field.bytes) ranges.push([field.offset, field.offset + field.bytes]);
    arrays[name] = new Type(buffer, field.offset, count);
  }
  ranges.sort((left, right) => left[0] - right[0]);
  requireValue(ranges.every((range, index) => !index || range[0] >= ranges[index - 1][1]), 'Pi4 二进制数组重叠');
  return arrays;
}


export function frameAtTime(timestamps, time) {
  let low = 0;
  let high = timestamps.length;
  const target = Number.isFinite(time) ? time + 1e-5 : 0;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (timestamps[middle] <= target) low = middle + 1;
    else high = middle;
  }
  return Math.max(0, low - 1);
}
