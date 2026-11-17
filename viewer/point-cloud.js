import * as THREE from 'three';
import {linspaceIndices} from './point-sampling.js';

const MOTION_STATIC = 0;
const MOTION_NATIVE = 1;
const MOTION_MARGIN = 2;

function isNativeMotion(motion) { return (motion & MOTION_NATIVE) !== 0; }
function isBackgroundMasked(motion) { return motion !== MOTION_STATIC; }

function makeMaterial(maxSize) {
  return new THREE.ShaderMaterial({
    transparent: true, depthWrite: true, toneMapped: false,
    uniforms: {
      uSize: {value: 2}, uOpacity: {value: 1}, uPixelRatio: {value: 1},
      uMaxSize: {value: maxSize}, uFrame: {value: 0}, uReveal: {value: true},
      uMotionMode: {value: 2}, uHighlight: {value: true},
      uConfidence: {value: 0},
    },
    vertexShader: `
      attribute vec3 color;
      attribute float motion;
      attribute float frameId;
      attribute float confidence;
      uniform float uSize, uPixelRatio, uMaxSize, uFrame;
      uniform float uConfidence;
      uniform int uMotionMode;
      uniform bool uReveal, uHighlight;
      varying vec3 vColor;
      varying float vPresent;
      vec3 linearColor(vec3 value) { return mix(value / 12.92, pow((value + .055) / 1.055, vec3(2.4)), step(vec3(.04045), value)); }
      void main() {
        bool current = abs(frameId - uFrame) < .5;
        bool nativeMotion = mod(motion, ${MOTION_MARGIN.toFixed(1)}) >= ${MOTION_NATIVE.toFixed(1)};
        bool backgroundMasked = motion >= ${MOTION_NATIVE.toFixed(1)};
        bool visible = !uReveal || frameId <= uFrame;
        if (confidence < uConfidence) visible = false;
        if (uMotionMode == 1 && backgroundMasked) visible = false;
        if (uMotionMode == 2 && backgroundMasked && !current) visible = false;
        if (uMotionMode == 3 && (!nativeMotion || !current)) visible = false;
        vPresent = visible ? 1.0 : 0.0;
        vColor = linearColor(color);
        float emphasis = uHighlight && current ? 1.6 : 1.0;
        vColor *= emphasis;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = clamp(uSize * emphasis * uPixelRatio, 1.0, uMaxSize);
      }`,
    fragmentShader: `
      uniform float uOpacity;
      varying vec3 vColor;
      varying float vPresent;
      void main() {
        if (vPresent < .5) discard;
        float radius = length(gl_PointCoord * 2.0 - 1.0);
        if (radius > 1.0) discard;
        float alpha = uOpacity * (1.0 - smoothstep(1.0 - max(fwidth(radius), .015), 1.0, radius));
        if (alpha < .001) discard;
        gl_FragColor = vec4(vColor, alpha);
        #include <colorspace_fragment>
      }`,
  });
}

export class PointCloud extends THREE.Points {
  constructor(capacity, renderer, camera, invalidate) {
    const geometry = new THREE.BufferGeometry();
    for (const [name, array, width, normalized] of [
      ['position', new Float32Array(capacity * 3), 3, false],
      ['color', new Uint8Array(capacity * 3), 3, true],
      ['motion', new Uint8Array(capacity), 1, false],
      ['frameId', new Float32Array(capacity), 1, false],
      ['confidence', new Float32Array(capacity), 1, false],
    ]) geometry.setAttribute(name, new THREE.BufferAttribute(array, width, normalized).setUsage(THREE.DynamicDrawUsage));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(capacity), 1).setUsage(THREE.DynamicDrawUsage));
    geometry.setDrawRange(0, 0);
    super(geometry, makeMaterial(renderer.getContext().getParameter(renderer.getContext().ALIASED_POINT_SIZE_RANGE)[1]));
    this.camera = camera;
    this.invalidate = invalidate;
    this.sort = {worker: null, version: 0, busy: false, timer: null, completed: 0};
    this.frustumCulled = false;
    this.count = 0;
    this.pointBudget = 2_000_000;
    this.sampleStorage = new Uint32Array(capacity);
    this.sampleIndices = new Uint32Array();
    this.frames = new Map();
    this.bounds = new THREE.Box3();
    this.settings = {frame: 0, reveal: true, motionScope: 'current', highlight: true, confidence: 0};
  }

  addFrame(frame, position, color, motion, confidence = new Float32Array(motion.length).fill(1)) {
    if (this.frames.has(frame)) return;
    if (position.length !== motion.length * 3 || color.length !== position.length || confidence.length !== motion.length || this.count + motion.length > this.geometry.attributes.motion.count) throw new Error('Invalid point frame');
    confidence = Float32Array.from(confidence, value => Number.isFinite(value) ? value : -1);
    this.clearSorter();
    const start = this.count;
    const moving = motion.reduce((total, value) => total + Number(isNativeMotion(value)), 0);
    const record = {start, count: motion.length, moving, visibleStatic: 0, visibleMoving: 0, visibleMargin: 0};
    this.frames.set(frame, record);
    for (const [name, values] of Object.entries({position, color, motion, confidence})) {
      const attribute = this.geometry.attributes[name];
      const offset = start * attribute.itemSize;
      attribute.array.set(values, offset);
      if (values.length) attribute.addUpdateRange(offset, values.length);
      attribute.needsUpdate = true;
    }
    this.count += motion.length;
    const frameIds = this.geometry.attributes.frameId;
    frameIds.array.fill(frame, start, this.count);
    if (motion.length) frameIds.addUpdateRange(start, motion.length);
    frameIds.needsUpdate = true;
    this.updatePointSelection(record);
    const point = new THREE.Vector3();
    for (let index = 0; index < motion.length; index++) this.bounds.expandByPoint(point.fromArray(position, index * 3));
    this.queueSort();
    this.invalidate();
  }

  setCurrentFrame(frame) {
    this.settings.frame = frame;
    this.material.uniforms.uFrame.value = frame;
    this.invalidate();
  }

  isPointVisible(index) {
    const frame = this.geometry.attributes.frameId.array[index];
    const motion = this.geometry.attributes.motion.array[index];
    return this.geometry.attributes.confidence.array[index] >= this.settings.confidence && this.isFramePointVisible(frame, motion);
  }

  isFramePointVisible(frame, motion) {
    const settings = this.settings;
    if (settings.reveal && frame > settings.frame) return false;
    if (settings.motionScope === 'hidden') return !isBackgroundMasked(motion);
    if (settings.motionScope === 'current') return !isBackgroundMasked(motion) || frame === settings.frame;
    if (settings.motionScope === 'current-only') return isNativeMotion(motion) && frame === settings.frame;
    return true;
  }

  get visibleCount() {
    let count = 0;
    for (const [frame, record] of this.frames) {
      if (this.isFramePointVisible(frame, MOTION_STATIC)) count += record.visibleStatic;
      if (this.isFramePointVisible(frame, MOTION_NATIVE)) count += record.visibleMoving;
      if (this.isFramePointVisible(frame, MOTION_MARGIN)) count += record.visibleMargin;
    }
    return count;
  }

  visibleBounds() {
    const bounds = new THREE.Box3();
    const point = new THREE.Vector3();
    for (const index of this.sampleIndices) {
      if (this.isPointVisible(index)) bounds.expandByPoint(point.fromBufferAttribute(this.geometry.attributes.position, index));
    }
    return bounds;
  }

  setPointBudget(limit) {
    if (!Number.isSafeInteger(limit) || limit < 1) throw new Error('Invalid point budget');
    if (limit === this.pointBudget) return;
    this.pointBudget = limit;
    this.clearSorter();
    this.updatePointSelection();
    this.queueSort();
    this.invalidate();
  }

  updatePointSelection(appended = null) {
    const incremental = appended && this.count <= this.pointBudget && this.sampleIndices.length === appended.start;
    const start = incremental ? appended.start : 0;
    if (this.count <= this.pointBudget) {
      for (let index = start; index < this.count; index++) this.sampleStorage[index] = index;
      this.sampleIndices = this.sampleStorage.subarray(0, this.count);
    } else {
      this.sampleStorage.set(linspaceIndices(this.count, this.pointBudget));
      this.sampleIndices = this.sampleStorage.subarray(0, this.pointBudget);
    }
    this.geometry.index.array.set(this.sampleIndices.subarray(start), start);
    if (!incremental) this.geometry.index.clearUpdateRanges();
    if (this.sampleIndices.length > start) this.geometry.index.addUpdateRange(start, this.sampleIndices.length - start);
    this.geometry.index.needsUpdate = true;
    this.geometry.setDrawRange(0, this.sampleIndices.length);
    this.updateConfidenceCounts(start);
  }

  updateConfidenceCounts(start = 0) {
    const attributes = this.geometry.attributes;
    if (start === 0) {
      for (const record of this.frames.values()) {
        record.visibleStatic = 0;
        record.visibleMoving = 0;
        record.visibleMargin = 0;
      }
    }
    for (let sample = start; sample < this.sampleIndices.length; sample++) {
      const index = this.sampleIndices[sample];
      if (attributes.confidence.array[index] < this.settings.confidence) continue;
      const record = this.frames.get(attributes.frameId.array[index]);
      const motion = attributes.motion.array[index];
      if (isNativeMotion(motion)) record.visibleMoving++;
      else if (isBackgroundMasked(motion)) record.visibleMargin++;
      else record.visibleStatic++;
    }
  }

  applySettings({size, opacity, pixelRatio, reveal, motionScope, highlight, confidence = this.settings.confidence}) {
    confidence = Math.fround(confidence);
    const confidenceChanged = confidence !== this.settings.confidence;
    Object.assign(this.settings, {reveal, motionScope, highlight, confidence});
    if (confidenceChanged) this.updateConfidenceCounts();
    const uniforms = this.material.uniforms;
    uniforms.uSize.value = size;
    uniforms.uOpacity.value = opacity;
    uniforms.uPixelRatio.value = pixelRatio;
    uniforms.uReveal.value = reveal;
    uniforms.uMotionMode.value = {all: 0, hidden: 1, current: 2, 'current-only': 3}[motionScope] ?? 0;
    uniforms.uHighlight.value = highlight;
    uniforms.uConfidence.value = confidence;
    this.material.depthWrite = opacity >= 1;
    this.renderOrder = opacity >= 1 ? 0 : 1;
    this.queueSort();
    this.invalidate();
  }

  clearSorter() {
    clearTimeout(this.sort.timer);
    this.sort.worker?.terminate();
    this.sort.worker = null;
    this.sort.busy = false;
    this.sort.version++;
  }

  queueSort(immediate = false) {
    clearTimeout(this.sort.timer);
    if (!this.count || this.material.uniforms.uOpacity.value >= 1) {
      this.clearSorter();
      return;
    }
    this.sort.version++;
    if (immediate) this.sendSort();
    else this.sort.timer = setTimeout(() => this.sendSort(), 180);
  }

  sendSort() {
    const sort = this.sort;
    if (sort.busy) return;
    if (!sort.worker) {
      const worker = new Worker(new URL('./sort-worker.js', import.meta.url));
      sort.worker = worker;
      const positions = new Float32Array(this.sampleIndices.length * 3);
      const indices = this.sampleIndices.slice();
      const source = this.geometry.attributes.position.array;
      for (let sample = 0; sample < indices.length; sample++) positions.set(source.subarray(indices[sample] * 3, indices[sample] * 3 + 3), sample * 3);
      worker.postMessage({type: 'positions', positions: positions.buffer, indices: indices.buffer}, [positions.buffer, indices.buffer]);
      worker.onmessage = ({data}) => {
        if (sort.worker !== worker) return;
        sort.busy = false;
        if (data.version === sort.version) {
          this.geometry.index.array.set(new Uint32Array(data.indices));
          this.geometry.index.clearUpdateRanges();
          this.geometry.index.addUpdateRange(0, this.sampleIndices.length);
          this.geometry.index.needsUpdate = true;
          sort.completed++;
          this.invalidate();
        } else this.sendSort();
      };
      worker.onerror = () => { this.clearSorter(); this.invalidate(); };
    }
    this.camera.updateMatrixWorld(true);
    const matrix = this.camera.matrixWorldInverse.elements;
    sort.busy = true;
    sort.worker.postMessage({type: 'sort', direction: [matrix[2], matrix[6], matrix[10]], version: sort.version});
  }
}
