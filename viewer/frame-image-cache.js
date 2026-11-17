import {dataURL} from './static-demo-data.js';

export class FrameImageCache {
  constructor(frames) {
    this.frames = frames;
    this.entries = new Map();
    this.inflight = new Set();
    this.current = 0;
    this.ahead = 24;
    this.behind = 4;
    this.active = 0;
    this.concurrency = 3;
    this.generation = 0;
    this.disposed = false;
  }

  show(frame, element) {
    if (this.disposed) return;
    this.current = frame;
    const generation = ++this.generation;
    const first = Math.max(0, frame - this.behind);
    const last = Math.min(this.frames.length - 1, frame + this.ahead);
    for (const [index, entry] of this.entries) {
      if (index < first || index > last) {
        this.entries.delete(index);
        if (!entry.started) entry.resolve(false);
      }
    }
    for (let index = first; index <= last; index++) {
      if (this.entries.has(index)) continue;
      const entry = {frame: index, started: false, ready: false};
      entry.promise = new Promise(resolve => { entry.resolve = resolve; });
      this.entries.set(index, entry);
    }
    const entry = this.entries.get(frame);
    const display = ready => {
      if (this.disposed || generation !== this.generation) return;
      const image = entry.image;
      if (image === element) return;
      image.id = element.id;
      image.alt = ready ? 'Input RGB frame' : `RGB frame ${frame + 1} unavailable`;
      element.removeAttribute('id');
      element.replaceWith(image);
    };
    if (entry.ready) display(true);
    else entry.promise.then(display);
    this.pump();
  }

  pump() {
    if (this.disposed) return;
    const priority = entry => entry.frame >= this.current ? entry.frame - this.current : this.frames.length + this.current - entry.frame;
    const queued = [...this.entries.values()].filter(entry => !entry.started).sort((left, right) => priority(left) - priority(right));
    for (const entry of queued) {
      if (this.active >= this.concurrency) break;
      this.load(entry);
    }
  }

  async load(entry) {
    entry.started = true;
    this.active++;
    this.inflight.add(entry);
    entry.image = new Image();
    entry.image.decoding = 'async';
    try {
      entry.image.src = dataURL(this.frames[entry.frame].image);
      await entry.image.decode();
      entry.ready = true;
      entry.resolve(true);
    } catch {
      entry.resolve(false);
    } finally {
      this.active--;
      this.inflight.delete(entry);
      this.pump();
    }
  }

  dispose() {
    this.disposed = true;
    this.generation++;
    for (const entry of this.entries.values()) {
      entry.resolve(false);
    }
    for (const entry of this.inflight) entry.image.src = '';
    this.entries.clear();
  }
}
