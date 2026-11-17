import * as THREE from 'three';
import {DynamicTracks} from './dynamic-tracks.js';
import {dataURL, frameAtTime, playbackTimestamps, readBundle} from './static-demo-data.js';

export class StaticDemoViewer {
  constructor(scene, cloud, hooks) {
    this.scene = scene;
    this.cloud = cloud;
    this.hooks = hooks;
    this.loadedChunks = new Set();
    this.chunkFrames = scene.viewer.chunks.map(() => []);
    scene.frames.forEach((record, frame) => this.chunkFrames[record.chunk].push(frame));
    this.generation = 0;
    this.frame = -1;
    this.requestedFrame = -1;
    this.playing = false;
    this.loading = false;
    this.pending = new Map();
    this.failedChunks = new Set();
    this.activeDownloads = 0;
    this.commitQueue = Promise.resolve();
    this.prefetchFrames = 32;
    this.maxDownloads = 3;
    this.abort = new AbortController();
    this.timestamps = playbackTimestamps(scene);
    this.settings = {reveal: true, speed: 1, tracks: {enabled: true, count: 256, tail: .8, full: false}};
    this.ready = false;
    this.disposed = false;
    this.tracksLoading = false;
  }

  async open() {
    await this.setFrame(this.scene.tracks?.query_frame ?? 0);
    if (this.disposed) return;
    if (this.frame < 0) throw new Error('Could not load the initial point cloud');
    this.ready = true;
    this.tracksPromise = this.loadTracks();
    this.fillGeometry();
  }

  async loadTracks() {
    if (!this.scene.tracks) return;
    this.tracksLoading = true;
    try {
      const arrays = await readBundle(this.scene.tracks, this.abort.signal);
      await new Promise(resolve => setTimeout(resolve, 0));
      if (this.disposed) return;
      this.tracks = new DynamicTracks(arrays, this.scene.tracks, this.timestamps, values => this.normalize(values));
      this.hooks.add(this.tracks.group);
      this.updateTracks();
      this.hooks.tracks?.(this.tracks);
      this.hooks.invalidate();
    } catch (error) {
      if (!this.disposed && error.name !== 'AbortError') {
        this.hooks.tracks?.(null);
        this.hooks.error(error);
      }
    } finally {
      this.tracksLoading = false;
      this.fillGeometry();
    }
  }

  normalize(coordinates) {
    const normalized = new Float32Array(coordinates.length);
    for (let index = 0; index < coordinates.length; index++) normalized[index] = (coordinates[index] - this.reference.center[index % 3]) / this.reference.scale;
    return normalized;
  }

  async requestPointChunk(index, wanted = () => true) {
    if (this.disposed || !wanted()) return;
    if (this.loadedChunks.has(index)) return;
    this.failedChunks.delete(index);
    const pending = this.enqueueChunk(index, true);
    this.fillGeometry();
    return pending.promise;
  }

  enqueueChunk(index, required = false) {
    if (this.pending.has(index)) {
      const pending = this.pending.get(index);
      pending.required ||= required;
      return pending;
    }
    const pending = {index, started: false, required};
    pending.promise = new Promise((resolve, reject) => Object.assign(pending, {resolve, reject}));
    pending.promise.catch(() => {});
    this.pending.set(index, pending);
    return pending;
  }

  async downloadChunk(pending) {
    pending.started = true;
    this.activeDownloads++;
    try {
      const arrays = await readBundle(this.scene.viewer.chunks[pending.index], this.abort.signal);
      const commit = this.commitQueue.then(() => this.applyPointChunk(pending.index, arrays));
      this.commitQueue = commit.catch(() => {});
      await commit;
      pending.resolve();
    } catch (error) {
      this.failedChunks.add(pending.index);
      pending.reject(error);
    } finally {
      this.activeDownloads--;
      this.pending.delete(pending.index);
      this.fillGeometry();
    }
  }

  async applyPointChunk(index, arrays) {
    if (this.disposed) return;
    if (!arrays.motion || !arrays.position || !arrays.color || arrays.position.length !== arrays.motion.length * 3 || arrays.color.length !== arrays.position.length) throw new Error('Invalid point chunk');
    if ((this.scene.viewer.confidence && !arrays.confidence) || (arrays.confidence && arrays.confidence.length !== arrays.motion.length)) throw new Error('Invalid point confidence');
    if (!this.reference) {
      const bounds = new THREE.Box3();
      const point = new THREE.Vector3();
      for (let offset = 0; offset < arrays.position.length; offset += 3) bounds.expandByPoint(point.fromArray(arrays.position, offset));
      const extent = bounds.getSize(new THREE.Vector3());
      this.reference = {center: bounds.getCenter(new THREE.Vector3()).toArray(), scale: Math.max(extent.x, extent.y, extent.z) || 1};
    }
    let deadline = performance.now() + 4;
    for (const frame of this.chunkFrames[index]) {
      if (this.disposed) return;
      const record = this.scene.frames[frame];
      const start = record.start, stop = start + record.count;
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(record.count) || start < 0 || record.count < 0 || stop > arrays.motion.length) throw new Error('Invalid point frame range');
      this.cloud.addFrame(frame, this.normalize(arrays.position.subarray(start * 3, stop * 3)), arrays.color.subarray(start * 3, stop * 3), arrays.motion.subarray(start, stop), arrays.confidence?.subarray(start, stop));
      if (performance.now() >= deadline) {
        await new Promise(resolve => setTimeout(resolve, 0));
        deadline = performance.now() + 4;
      }
    }
    if (this.disposed) return;
    this.loadedChunks.add(index);
    this.hooks.progress?.();
  }

  setSettings(settings) {
    Object.assign(this.settings, settings);
    this.updateTracks();
    this.fillGeometry();
  }

  updateTracks() {
    if (this.frame >= 0) this.tracks?.update(this.frame, this.settings.tracks);
  }

  fillGeometry() {
    if (this.disposed) return;
    const current = Math.max(0, this.requestedFrame);
    const target = this.settings.reveal ? current + this.prefetchFrames : this.scene.frames.length - 1;
    for (const [index, pending] of this.pending) {
      if (!pending.started && !pending.required && this.chunkFrames[index].every(frame => frame > target)) {
        pending.resolve();
        this.pending.delete(index);
      }
    }
    if (this.ready) {
      this.chunkFrames.forEach((frames, index) => {
        if (!this.loadedChunks.has(index) && !this.failedChunks.has(index) && frames.some(frame => frame <= target)) this.enqueueChunk(index);
      });
    }
    const urgent = this.scene.frames[current].chunk;
    const priority = pending => {
      if (pending.index === urgent) return -Infinity;
      const frames = this.chunkFrames[pending.index];
      if (frames.at(-1) >= current && frames[0] <= target) return frames[0] - current;
      return this.scene.frames.length + Math.abs(frames[0] - current);
    };
    const queued = [...this.pending.values()].filter(pending => !pending.started).sort((left, right) => priority(left) - priority(right));
    const limit = this.maxDownloads - Number(this.tracksLoading);
    for (const pending of queued) {
      if (this.activeDownloads >= limit) break;
      this.downloadChunk(pending);
    }
  }

  async setFrame(frame) {
    if (this.disposed || !Number.isInteger(frame) || frame < 0 || frame >= this.scene.frames.length) return;
    if (frame === this.frame && !this.loading) return;
    const generation = ++this.generation;
    this.requestedFrame = frame;
    this.loading = true;
    const record = this.scene.frames[frame];
    this.hooks.loading?.(this.loadedChunks.has(record.chunk) ? null : this.scene.viewer.chunks[record.chunk]);
    try {
      await this.requestPointChunk(record.chunk, () => generation === this.generation);
      if (this.disposed || generation !== this.generation) return;
      this.cloud.setCurrentFrame(frame);
      this.frame = frame;
      this.updateTracks();
      this.hooks.frame(frame, record, dataURL(record.image));
      this.hooks.invalidate();
    } catch (error) {
      if (error.name !== 'AbortError' && generation === this.generation) {
        this.stopPlayback();
        this.hooks.error(error);
      }
    } finally {
      if (generation === this.generation) {
        this.loading = false;
        this.hooks.loading?.(null);
        this.fillGeometry();
      }
    }
  }

  togglePlayback() {
    if (this.playing) { this.stopPlayback(); return; }
    const frame = this.frame >= this.scene.frames.length - 1 ? 0 : this.frame;
    this.lastTick = performance.now();
    this.playbackTime = this.timestamps[Math.max(0, frame)];
    this.playing = true;
    this.hooks.playback(true);
    this.setFrame(Math.max(0, frame));
  }

  stopPlayback() {
    this.playing = false;
    this.hooks.playback(false);
  }

  tick(now) {
    if (!this.playing) return;
    const elapsed = Math.max(0, (now - this.lastTick) / 1000);
    this.lastTick = now;
    if (this.loading || this.frame < 0) return;
    this.playbackTime += elapsed * this.settings.speed;
    const frame = frameAtTime(this.timestamps, this.playbackTime);
    if (frame !== this.requestedFrame) this.setFrame(frame);
    if (this.frame === this.scene.frames.length - 1) this.stopPlayback();
  }

  dispose() {
    this.stopPlayback();
    this.disposed = true;
    this.generation++;
    this.abort.abort();
    for (const pending of this.pending.values()) pending.resolve();
    this.pending.clear();
    this.loading = false;
    this.loadedChunks.clear();
    this.tracks?.dispose();
    this.tracks = null;
    this.hooks.loading?.(null);
  }
}
