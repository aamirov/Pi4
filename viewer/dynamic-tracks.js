import * as THREE from 'three';

export function trackSegmentIndices(valid, queryCount, timestamps, segmentIds, frame, tailSeconds, full = false) {
  if (!Number.isInteger(queryCount) || queryCount < 0 || valid.length !== timestamps.length * queryCount || segmentIds.length !== timestamps.length) throw new Error('Track 时间线维度错误');
  if (!Number.isInteger(frame) || frame < 0 || frame >= timestamps.length || !Number.isFinite(tailSeconds) || tailSeconds <= 0) throw new Error('Track 播放参数错误');
  const firstTime = timestamps[frame] - tailSeconds;
  let start = 1;
  const stop = full ? timestamps.length : frame + 1;
  if (!full) while (start < stop && timestamps[start - 1] < firstTime) start++;
  const selected = new Uint32Array(Math.max(0, stop - start) * queryCount);
  let count = 0;
  for (let target = start; target < stop; target++) {
    if (segmentIds[target] !== segmentIds[target - 1]) continue;
    for (let query = 0; query < queryCount; query++) {
      const index = target * queryCount + query;
      if (valid[index] && valid[index - queryCount]) selected[count++] = index;
    }
  }
  return selected.subarray(0, count);
}

export function spatialQueryColors(queryUV) {
  if (queryUV.length % 2) throw new Error('Query UV 维度错误');
  const colors = new Float32Array(queryUV.length / 2 * 3);
  let minimumX = Infinity, minimumY = Infinity, maximumX = -Infinity, maximumY = -Infinity;
  for (let index = 0; index < queryUV.length; index += 2) {
    const horizontal = queryUV[index], vertical = queryUV[index + 1];
    if (!Number.isFinite(horizontal) || !Number.isFinite(vertical)) throw new Error('Query UV 包含非有限值');
    minimumX = Math.min(minimumX, horizontal); maximumX = Math.max(maximumX, horizontal);
    minimumY = Math.min(minimumY, vertical); maximumY = Math.max(maximumY, vertical);
  }
  const color = new THREE.Color();
  for (let query = 0; query < colors.length / 3; query++) {
    const horizontal = maximumX === minimumX ? .5 : (queryUV[query * 2] - minimumX) / (maximumX - minimumX);
    const vertical = maximumY === minimumY ? .5 : (queryUV[query * 2 + 1] - minimumY) / (maximumY - minimumY);
    color.setHSL(.72 * (.7 * horizontal + .3 * vertical), .82, .53 + .08 * vertical).toArray(colors, query * 3);
  }
  return colors;
}

const paletteShader = `
  attribute float queryId, queryRank;
  uniform sampler2D uPalette;
  uniform float uPaletteWidth, uPaletteRows, uQueryLimit;
  varying vec3 vColor;
  varying float vSelected;
  vec3 queryColor() {
    vec2 uv = vec2((mod(queryId, uPaletteWidth) + .5) / uPaletteWidth,
      (floor(queryId / uPaletteWidth) + .5) / uPaletteRows);
    return texture2D(uPalette, uv).rgb;
  }`;

export function randomQueryRanks(queryCount, seed = 20261111) {
  const ranks = Uint32Array.from({length: queryCount}, (unused, index) => index);
  let state = seed >>> 0;
  for (let remaining = queryCount - 1; remaining > 0; remaining--) {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ state >>> 15, state | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    const selected = Math.floor(((value ^ value >>> 14) >>> 0) / 2 ** 32 * (remaining + 1));
    const rank = ranks[remaining];
    ranks[remaining] = ranks[selected];
    ranks[selected] = rank;
  }
  return ranks;
}

function createFastLines(positions, queryCount, indices, timestamps, uniforms, ranks) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const queryIds = new Float32Array(positions.length / 3);
  const sampleTimes = new Float32Array(queryIds.length);
  const queryRanks = new Float32Array(queryIds.length);
  for (let index = 0; index < queryIds.length; index++) {
    queryIds[index] = index % queryCount;
    sampleTimes[index] = timestamps[Math.floor(index / queryCount)];
    queryRanks[index] = ranks[index % queryCount];
  }
  geometry.setAttribute('queryId', new THREE.BufferAttribute(queryIds, 1));
  geometry.setAttribute('sampleTime', new THREE.BufferAttribute(sampleTimes, 1));
  geometry.setAttribute('queryRank', new THREE.BufferAttribute(queryRanks, 1));
  const endpoints = new Uint32Array(indices.length * 2);
  for (let edge = 0; edge < indices.length; edge++) {
    endpoints[edge * 2] = indices[edge] - queryCount;
    endpoints[edge * 2 + 1] = indices[edge];
  }
  geometry.setIndex(new THREE.BufferAttribute(endpoints, 1));
  geometry.setDrawRange(0, 0);
  const material = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, depthTest: true, toneMapped: false, uniforms,
    vertexShader: `
      ${paletteShader}
      attribute float sampleTime;
      uniform float uOpacity, uTime, uTail;
      uniform bool uFull;
      varying float vOpacity;
      void main() {
        vColor = queryColor();
        vSelected = queryRank < uQueryLimit ? 1.0 : 0.0;
        vOpacity = uOpacity * (uFull ? 1.0 : .15 + .85 * clamp(1.0 - (uTime - sampleTime) / uTail, 0.0, 1.0));
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      varying float vOpacity;
      varying vec3 vColor;
      varying float vSelected;
      void main() {
        if (vSelected < .5 || vOpacity < .003) discard;
        gl_FragColor = vec4(vColor, vOpacity);
        #include <colorspace_fragment>
      }`,
  });
  const lines = new THREE.LineSegments(geometry, material);
  lines.frustumCulled = false;
  lines.renderOrder = 4;
  lines.visible = false;
  return lines;
}

export class DynamicTracks {
  constructor(arrays, description, timestamps, normalize) {
    this.arrays = arrays;
    this.queries = description.query_count;
    this.timestamps = timestamps;
    if (!Number.isSafeInteger(this.queries) || this.queries < 0 || arrays.valid.length !== timestamps.length * this.queries || arrays.world_xyz.length !== arrays.valid.length * 3 || arrays.query_uv.length !== this.queries * 2) throw new Error('Invalid track dimensions');
    if (arrays.confidence && arrays.confidence.length !== arrays.valid.length) throw new Error('Invalid track confidence');
    this.positions = normalize(arrays.world_xyz);
    this.valid = arrays.valid.slice();
    for (let index = 0; index < this.valid.length; index++) {
      const offset = index * 3;
      if (!Number.isFinite(this.positions[offset]) || !Number.isFinite(this.positions[offset + 1]) || !Number.isFinite(this.positions[offset + 2])) {
        this.valid[index] = 0;
        this.positions.fill(0, offset, offset + 3);
      }
    }
    if (arrays.confidence) {
      for (let index = 0; index < this.valid.length; index++) if (!Number.isFinite(arrays.confidence[index])) this.valid[index] = 0;
    }
    this.baseValid = this.valid.slice();
    this.group = new THREE.Group();
    this.spatial = spatialQueryColors(arrays.query_uv);
    const paletteWidth = Math.min(1024, Math.max(1, this.queries));
    const paletteRows = Math.ceil(Math.max(1, this.queries) / paletteWidth);
    const paletteValues = new Float32Array(paletteWidth * paletteRows * 4);
    for (let query = 0; query < this.queries; query++) paletteValues.set(this.spatial.subarray(query * 3, query * 3 + 3), query * 4);
    this.palette = new THREE.DataTexture(paletteValues, paletteWidth, paletteRows, THREE.RGBAFormat, THREE.FloatType);
    this.palette.needsUpdate = true;
    this.uniforms = {
      uPalette: {value: this.palette}, uPaletteWidth: {value: paletteWidth}, uPaletteRows: {value: paletteRows},
      uOpacity: {value: .8}, uTime: {value: 0}, uTail: {value: .8},
      uFull: {value: false}, uQueryLimit: {value: this.queries},
    };
    this.ranks = randomQueryRanks(this.queries);
    const segments = new Uint32Array(timestamps.length);
    this.indices = trackSegmentIndices(this.valid, this.queries, timestamps, segments, 0, 1, true);
    this.offsets = new Uint32Array(timestamps.length + 1);
    for (const index of this.indices) this.offsets[Math.floor(index / this.queries) + 1]++;
    for (let frame = 1; frame < this.offsets.length; frame++) this.offsets[frame] += this.offsets[frame - 1];
    this.lines = createFastLines(this.positions, this.queries, this.indices, timestamps, this.uniforms, this.ranks);
    this.lines.visible = true;
    this.group.add(this.lines);
    this.headGeometry = new THREE.BufferGeometry();
    this.headGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.queries * 3), 3).setUsage(THREE.DynamicDrawUsage));
    this.headGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.queries * 3), 3).setUsage(THREE.DynamicDrawUsage));
    this.heads = new THREE.Points(this.headGeometry, new THREE.PointsMaterial({size: 5, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: .9, depthWrite: false}));
    this.heads.frustumCulled = false;
    this.heads.renderOrder = 5;
    this.group.add(this.heads);
  }

  setConfidence(threshold) {
    threshold = Math.fround(threshold);
    if (threshold === this.confidenceThreshold) return;
    this.confidenceThreshold = threshold;
    this.valid = this.baseValid.map((valid, index) => valid && (!this.arrays.confidence || this.arrays.confidence[index] >= threshold));
    const segments = new Uint32Array(this.timestamps.length);
    this.indices = trackSegmentIndices(this.valid, this.queries, this.timestamps, segments, 0, 1, true);
    this.offsets.fill(0);
    const endpoints = this.lines.geometry.index;
    for (let edge = 0; edge < this.indices.length; edge++) {
      const index = this.indices[edge];
      this.offsets[Math.floor(index / this.queries) + 1]++;
      endpoints.array[edge * 2] = index - this.queries;
      endpoints.array[edge * 2 + 1] = index;
    }
    for (let frame = 1; frame < this.offsets.length; frame++) this.offsets[frame] += this.offsets[frame - 1];
    endpoints.needsUpdate = true;
    this.selectedQueryCount = undefined;
  }

  update(frame, {enabled = true, count = this.queries, tail = .8, full = false, confidence = 0} = {}) {
    this.setConfidence(confidence);
    const limit = Math.max(0, Math.min(this.queries, Math.floor(count)));
    if (limit !== this.selectedQueryCount) {
      this.selectedQueryCount = limit;
      this.selectedOffsets = new Uint32Array(this.timestamps.length + 1);
      for (const index of this.indices) {
        if (this.ranks[index % this.queries] < limit) this.selectedOffsets[Math.floor(index / this.queries) + 1]++;
      }
      for (let index = 1; index < this.selectedOffsets.length; index++) this.selectedOffsets[index] += this.selectedOffsets[index - 1];
    }
    this.group.visible = enabled && limit > 0;
    this.uniforms.uQueryLimit.value = limit;
    this.uniforms.uTail.value = Math.max(.001, tail);
    this.uniforms.uFull.value = full;
    this.uniforms.uTime.value = this.timestamps[frame];
    let startFrame = 1;
    const stopFrame = full ? this.timestamps.length : frame + 1;
    if (!full) while (startFrame < stopFrame && this.timestamps[startFrame - 1] < this.timestamps[frame] - tail) startFrame++;
    const start = this.offsets[startFrame], stop = this.offsets[stopFrame];
    this.lines.geometry.setDrawRange(start * 2, (stop - start) * 2);
    this.segmentCount = this.selectedOffsets[stopFrame] - this.selectedOffsets[startFrame];
    let headCount = 0;
    for (let query = 0; query < this.queries; query++) {
      const index = frame * this.queries + query;
      if (this.ranks[query] >= limit || !this.valid[index]) continue;
      this.headGeometry.attributes.position.array.set(this.positions.subarray(index * 3, index * 3 + 3), headCount * 3);
      this.headGeometry.attributes.color.array.set(this.spatial.subarray(query * 3, query * 3 + 3), headCount * 3);
      headCount++;
    }
    for (const attribute of Object.values(this.headGeometry.attributes)) attribute.needsUpdate = true;
    this.headGeometry.setDrawRange(0, headCount);
    this.headCount = headCount;
  }

  dispose() {
    this.group.removeFromParent();
    this.lines.geometry.dispose();
    this.lines.material.dispose();
    this.headGeometry.dispose();
    this.heads.material.dispose();
    this.palette.dispose();
  }
}

