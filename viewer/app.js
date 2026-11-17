import * as THREE from 'three';
import {ScreenTrackballControls} from './controls.js';
import {PointCloud} from './point-cloud.js';
import {pickPointCloudPoint} from './point-picking.js';
import {dataURL, readJSON} from './static-demo-data.js';
import {StaticDemoViewer} from './static-demo-viewer.js';
import {predictedCameraView, SmoothCameraFollower} from './camera-follow.js';
import {CameraLayer} from './camera-layer.js';
import {linspaceIndices} from './point-sampling.js';
import {FrameImageCache} from './frame-image-cache.js';

const byId = id => document.getElementById(id);
const formatPoints = count => count >= 1_000_000 ? `${(count / 1_000_000).toFixed(2)}M` : count >= 1_000 ? `${(count / 1_000).toFixed(count % 1_000 ? 1 : 0)}K` : String(count);
const formatTime = seconds => {
  const centiseconds = Math.max(0, Math.round((Number.isFinite(seconds) ? seconds : 0) * 100));
  return `${String(Math.floor(centiseconds / 6000)).padStart(2, '0')}:${(centiseconds % 6000 / 100).toFixed(2).padStart(5, '0')}`;
};
function updateRangeStyle(input) {
  const minimum = Number(input.min), maximum = Number(input.max);
  const progress = maximum > minimum ? (Number(input.value) - minimum) / (maximum - minimum) : 0;
  input.style.setProperty('--range-fill', `${Math.max(0, Math.min(1, progress)) * 100}%`);
}
const rangeInputs = [...document.querySelectorAll('input[type="range"]')];
for (const input of rangeInputs) input.addEventListener('input', () => updateRangeStyle(input));
const viewport = byId('viewport');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, .00001, 100);
const renderer = new THREE.WebGLRenderer({antialias: false, alpha: true});
viewport.prepend(renderer.domElement);
const controls = new ScreenTrackballControls(camera, renderer.domElement);
controls.rotateSpeed = 1.5;
controls.zoomSpeed = 1.15;
controls.panSpeed = 1;
controls.staticMoving = true;
controls.keys = [];
controls.mouseButtons = {LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.PAN};
controls.minDistance = .0001;
controls.maxDistance = 1e12;
controls.minZoom = 1e-6;
controls.maxZoom = 1e6;
controls.reverseZoom = true;
const cameraFollower = new SmoothCameraFollower(camera, controls);
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let dirty = true;
let cloud;
let demoViewer;
let cameraLayer;
let toastTimer;
let scrubState;
let scrubEndTimer;
let seekPromise;
let budgetTimer;
let filmstripImages = [];
let frameImages;
let lastRender = 0;
let lastAnimation = performance.now();
let followDistance = 1;
const followNote = byId('follow-note').textContent;

function markDirty() { dirty = true; }

function drawFilmstrip() {
  const canvas = byId('filmstrip');
  canvas.width = Math.max(1, Math.round(canvas.clientWidth));
  canvas.height = 36;
  const context = canvas.getContext('2d');
  if (!context || !filmstripImages.length) return;
  const width = canvas.width / filmstripImages.length;
  filmstripImages.forEach((image, index) => {
    if (image.complete && image.naturalWidth) context.drawImage(image, index * width, 0, width + 1, canvas.height);
  });
}

function loadFilmstrip(description) {
  filmstripImages = Array.from(linspaceIndices(description.frames.length, 12), frame => {
    const image = new Image();
    image.onload = drawFilmstrip;
    image.src = dataURL(description.frames[frame].image);
    return image;
  });
}

function toast(message, error = false) {
  clearTimeout(toastTimer);
  byId('toast').textContent = message;
  byId('toast').classList.toggle('error', error);
  byId('toast').hidden = false;
  toastTimer = setTimeout(() => { byId('toast').hidden = true; }, error ? 8500 : 2500);
}

function updateClipping() {
  const distance = camera.position.distanceTo(controls.target);
  const extent = cloud?.bounds?.getSize(new THREE.Vector3()).length() || 1;
  camera.near = Math.max(1e-6, distance * 1e-5);
  camera.far = Math.max(100, distance + extent * 3);
  camera.updateProjectionMatrix();
}

function stopFollowing() {
  cameraFollower.stop();
  if (!byId('follow-camera').checked) return;
  byId('follow-camera').checked = false;
  byId('follow-note').textContent = followNote;
}

function updateFollowCamera(immediate = false) {
  if (!byId('follow-camera').checked || !demoViewer || demoViewer.frame < 0) return;
  const view = predictedCameraView(demoViewer.scene.frames[demoViewer.frame].camera_to_world, demoViewer.reference);
  if (!view) {
    byId('follow-note').textContent = 'No valid camera pose for this frame; keeping the last view.';
    return;
  }
  byId('follow-note').textContent = followNote;
  cameraFollower.setView(view, followDistance, immediate || reducedMotion.matches);
  updateClipping();
  cloud?.queueSort();
  markDirty();
}

controls.addEventListener('viewchange', () => {
  updateClipping();
  markDirty();
  cloud?.queueSort();
});
controls.addEventListener('end', () => cloud?.queueSort());

function applySettings() {
  const reveal = byId('reveal').checked;
  cloud?.applySettings({
    size: Number(byId('point-size').value), opacity: Number(byId('opacity').value),
    pixelRatio: renderer.getPixelRatio(), reveal,
    motionScope: byId('motion-scope').value, highlight: byId('highlight').checked,
    confidence: Number(byId('confidence').value),
  });
  demoViewer?.setSettings({reveal, speed: Number(byId('speed').value), tracks: {
    enabled: byId('show-tracks').checked, count: Number(byId('track-count').value),
    tail: Number(byId('track-tail').value), full: byId('full-trails').checked,
    confidence: Number(byId('track-confidence').value),
  }});
  byId('size-value').textContent = `${Number(byId('point-size').value).toFixed(1)} px`;
  byId('opacity-value').textContent = `${Math.round(Number(byId('opacity').value) * 100)}%`;
  byId('track-count-value').textContent = `${Number(byId('track-count').value).toLocaleString()} / ${Number(byId('track-count').max).toLocaleString()}`;
  byId('track-tail-value').textContent = `${Number(byId('track-tail').value).toFixed(2)} s`;
  byId('confidence-value').textContent = Number(byId('confidence').value).toFixed(2);
  byId('track-confidence-value').textContent = Number(byId('track-confidence').value).toFixed(2);
  const tracksEnabled = !byId('show-tracks').disabled && byId('show-tracks').checked;
  byId('track-count').disabled = !tracksEnabled;
  byId('full-trails').disabled = !tracksEnabled;
  byId('track-tail').disabled = !tracksEnabled || byId('full-trails').checked;
  byId('track-confidence').disabled = !tracksEnabled || !demoViewer?.tracks?.arrays.confidence;
  for (const input of rangeInputs) updateRangeStyle(input);
  updateStatus();
  markDirty();
}

function updateStatus() {
  if (!cloud || !demoViewer || demoViewer.frame < 0) return;
  const source = demoViewer.scene.source;
  const frame = demoViewer.frame;
  const count = cloud.visibleCount;
  const timestamp = source.source_timestamps_s?.[frame];
  byId('status').textContent = `Source frame ${source.source_frame_indices?.[frame] ?? frame}` +
    (Number.isFinite(timestamp) ? ` · ${timestamp.toFixed(2)}s` : '') + ` · ${count.toLocaleString()} visible points · ${cloud.frames.size} / ${demoViewer.scene.frames.length} frames loaded`;
  Object.assign(byId('status').dataset, {frame, visiblePoints: count, loadedFrames: cloud.frames.size, highlighted: cloud.settings.highlight, pointBudget: cloud.pointBudget, drawnPoints: cloud.geometry.drawRange.count, sampledPoints: cloud.sampleIndices.length, totalPoints: cloud.count});
  byId('visible-points').textContent = count.toLocaleString();
  byId('loaded-frames').textContent = `${cloud.frames.size} / ${demoViewer.scene.frames.length}`;
  byId('load-progress').style.width = `${cloud.frames.size / demoViewer.scene.frames.length * 100}%`;
  const currentTime = demoViewer.timestamps[frame] ?? 0;
  byId('time-value').textContent = `${formatTime(currentTime)} / ${formatTime(demoViewer.timestamps.at(-1))}`;
  byId('source-time').textContent = formatTime(Number.isFinite(timestamp) ? timestamp : currentTime);
  const tracks = demoViewer.tracks;
  byId('track-status').textContent = tracks ? `${tracks.selectedQueryCount.toLocaleString()} selected · ${tracks.headCount.toLocaleString()} valid now · ${tracks.segmentCount.toLocaleString()} trail segments` : 'No tracks available';
  if (tracks) Object.assign(byId('track-status').dataset, {selected: tracks.selectedQueryCount, heads: tracks.headCount, segments: tracks.segmentCount, visible: tracks.group.visible});
  byId('track-empty').hidden = !tracks?.group.visible || tracks.segmentCount > 0;
  byId('track-empty').textContent = frame === 0 && !byId('full-trails').checked ? 'No history at the first frame: play, seek forward, or enable Full trajectories to see lines.' : 'No valid trail segments in this time window. Try a longer duration or Full trajectories.';
}

function fitCloud(reset = false) {
  stopFollowing();
  const bounds = cloud?.visibleBounds();
  if (!bounds || bounds.isEmpty()) return;
  controls.cancelInteraction();
  const center = bounds.getCenter(new THREE.Vector3());
  const direction = camera.position.clone().sub(controls.target).normalize();
  if (reset || direction.lengthSq() === 0) {
    const view = predictedCameraView(demoViewer.scene.frames[demoViewer.frame].camera_to_world, demoViewer.reference);
    direction.copy(view ? view.forward : new THREE.Vector3(0, 0, -1)).negate();
    camera.up.copy(view ? view.up : new THREE.Vector3(0, 1, 0));
  }
  camera.position.copy(center).add(direction);
  camera.lookAt(center);
  camera.updateMatrixWorld(true);
  const inverse = camera.quaternion.clone().invert();
  const tangent = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const positions = cloud.geometry.attributes.position.array;
  const point = new THREE.Vector3();
  let distance = .01;
  for (const index of cloud.sampleIndices) {
    if (!cloud.isPointVisible(index)) continue;
    point.fromArray(positions, index * 3).sub(center).applyQuaternion(inverse);
    distance = Math.max(distance, point.z + Math.abs(point.x) / (tangent * camera.aspect), point.z + Math.abs(point.y) / tangent);
  }
  distance *= 1.13;
  camera.position.copy(center).addScaledVector(direction, distance);
  controls.target.copy(center);
  controls.update();
  updateClipping();
  cloud.queueSort(true);
  markDirty();
}

function resize() {
  const {width, height} = viewport.getBoundingClientRect();
  renderer.setPixelRatio(Math.min(1, Math.sqrt(2_000_000 / Math.max(1, width * height))));
  renderer.setSize(Math.max(1, width), Math.max(1, height));
  camera.aspect = Math.max(1, width) / Math.max(1, height);
  camera.updateProjectionMatrix();
  controls.handleResize();
  drawFilmstrip();
  applySettings();
  markDirty();
}

renderer.domElement.addEventListener('pointerdown', event => {
  stopFollowing();
  controls.mouseButtons.LEFT = event.shiftKey || event.metaKey ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
}, true);
renderer.domElement.addEventListener('wheel', stopFollowing, {capture: true, passive: true});
renderer.domElement.addEventListener('dblclick', event => {
  event.preventDefault();
  stopFollowing();
  if (!cloud) return;
  controls.cancelInteraction();
  const rect = renderer.domElement.getBoundingClientRect();
  const hit = pickPointCloudPoint(cloud, camera, {x: event.clientX - rect.left, y: event.clientY - rect.top, width: rect.width, height: rect.height});
  if (!hit) { toast('Double-click a visible point to set the rotation center.'); return; }
  camera.position.add(hit.point.clone().sub(controls.target));
  controls.target.copy(hit.point);
  controls.update();
  markDirty();
  toast('Rotation center updated; view direction preserved.');
});
for (const id of ['point-size', 'opacity', 'track-count', 'track-tail', 'confidence', 'track-confidence']) byId(id).oninput = applySettings;
for (const id of ['show-tracks', 'show-cameras', 'motion-scope', 'reveal', 'highlight', 'full-trails', 'speed']) byId(id).onchange = applySettings;
function updatePointBudget(immediate = false) {
  clearTimeout(budgetTimer);
  const budget = Number(byId('point-budget').value);
  byId('point-budget-value').textContent = formatPoints(budget);
  const apply = () => { cloud?.setPointBudget(budget); updateStatus(); markDirty(); };
  if (immediate) apply();
  else budgetTimer = setTimeout(apply, 80);
}
byId('point-budget').oninput = () => updatePointBudget();
byId('point-budget').onchange = () => updatePointBudget(true);
byId('fit').onclick = () => fitCloud();
byId('reset').onclick = () => fitCloud(true);
byId('fullscreen').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await byId('viewer-shell').requestFullscreen();
  } catch { toast('Fullscreen is unavailable here. Open the viewer in a new tab.'); }
};
function enterCamera(immediate = false) {
  if (byId('follow-camera').disabled) return;
  if (!byId('follow-camera').checked) followDistance = Math.max(.01, camera.position.distanceTo(controls.target));
  byId('follow-camera').checked = true;
  updateFollowCamera(immediate);
}

function beginScrub() {
  clearTimeout(scrubEndTimer);
  if (!demoViewer || scrubState) return;
  scrubState = {resume: demoViewer.playing};
  demoViewer.stopPlayback();
  enterCamera();
}

function seekFrame() {
  beginScrub();
  seekPromise = demoViewer?.setFrame(Number(byId('frame').value));
}

function endScrub() {
  clearTimeout(scrubEndTimer);
  const state = scrubState;
  if (!state) return;
  scrubEndTimer = setTimeout(async () => {
    const pending = seekPromise;
    await pending;
    if (scrubState !== state || seekPromise !== pending) return;
    scrubState = null;
    if (state.resume && !document.hidden && !demoViewer.disposed && !demoViewer.playing) demoViewer.togglePlayback();
  }, 0);
}

byId('frame').addEventListener('pointerdown', beginScrub);
byId('frame').oninput = seekFrame;
byId('frame').onchange = () => { seekFrame(); endScrub(); };
window.addEventListener('pointerup', endScrub);
window.addEventListener('pointercancel', endScrub);
byId('play').onclick = () => demoViewer?.togglePlayback();
byId('follow-camera').onchange = () => {
  byId('follow-note').textContent = followNote;
  if (!byId('follow-camera').checked) { cameraFollower.stop(); return; }
  followDistance = Math.max(.01, camera.position.distanceTo(controls.target));
  updateFollowCamera();
};
byId('enter-camera').onclick = () => enterCamera();
document.addEventListener('visibilitychange', () => {
  if (document.hidden) demoViewer?.stopPlayback();
});
window.addEventListener('pagehide', event => {
  clearTimeout(scrubEndTimer);
  scrubState = null;
  clearTimeout(budgetTimer);
  if (event.persisted) demoViewer?.stopPlayback();
  else { demoViewer?.dispose(); frameImages?.dispose(); cloud?.clearSorter(); }
});
window.addEventListener('keydown', event => {
  if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(document.activeElement?.tagName) || event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.key.toLowerCase() === 'f') fitCloud();
  if (event.key.toLowerCase() === 'r') fitCloud(true);
  if (event.code === 'Space') { event.preventDefault(); demoViewer?.togglePlayback(); }
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault();
    if (!demoViewer) return;
    demoViewer.stopPlayback();
    enterCamera();
    demoViewer.setFrame(Math.max(0, Math.min(demoViewer.scene.frames.length - 1, demoViewer.frame + (event.key === 'ArrowRight' ? 1 : -1))));
  }
});
new ResizeObserver(resize).observe(viewport);
renderer.domElement.addEventListener('webglcontextlost', event => {
  event.preventDefault();
  demoViewer?.stopPlayback();
  toast('Graphics context lost. Close other 3D previews, then reload this page.', true);
});

async function open() {
  const description = await readJSON('scene.json');
  if (description.format !== 'pi4_scene_v1' || !description.viewer || !description.frames.length) throw new Error('Run example.py with --viewer first.');
  const source = description.source ?? {};
  byId('scene-meta').textContent = [`${description.frames.length} frames`, source.image_size?.join(' × ')].filter(Boolean).join(' · ');
  byId('source-name').textContent = String(source.source ?? '').split(/[\\/]/).filter(Boolean).at(-1) || 'RGB sequence';
  byId('source-name').title = byId('source-name').textContent;
  const capacity = description.frames.reduce((total, frame) => total + frame.count, 0);
  if (!Number.isSafeInteger(capacity) || capacity < 0) throw new Error('Invalid point-cloud capacity');
  cloud = new PointCloud(capacity, renderer, camera, markDirty);
  scene.add(cloud);
  const budgetSlider = byId('point-budget');
  budgetSlider.max = Math.max(1, capacity);
  budgetSlider.min = Math.min(50_000, Number(budgetSlider.max));
  budgetSlider.value = Math.min(2_000_000, Number(budgetSlider.max));
  budgetSlider.disabled = capacity < 2;
  byId('point-budget-min').textContent = formatPoints(Number(budgetSlider.min));
  byId('point-budget-max').textContent = formatPoints(Number(budgetSlider.max));
  updatePointBudget(true);
  frameImages = new FrameImageCache(description.frames);
  byId('confidence').disabled = !description.viewer.confidence;
  byId('confidence').value = description.viewer.confidence?.default_threshold ?? 0;
  if (byId('confidence').disabled) byId('confidence-note').textContent = 'This older export has no geometry confidence. Re-export with example.py --viewer to enable filtering.';
  byId('show-tracks').disabled = !description.tracks?.query_count;
  byId('show-tracks').checked = !byId('show-tracks').disabled;
  byId('show-tracks').disabled = true;
  byId('track-count').max = description.tracks?.query_count ?? 0;
  byId('track-count').value = Math.min(256, Number(byId('track-count').max));
  byId('motion-scope').disabled = !description.viewer.has_motion_mask;
  if (byId('motion-scope').disabled) {
    byId('motion-scope').value = 'all';
    byId('motion-note').textContent = 'No motion mask in this export. Progressive construction and highlighting remain available.';
  } else if (description.viewer.motion_mask_dilation_pixels) {
    byId('motion-note').textContent = `Dynamic-region hiding adds a ${description.viewer.motion_mask_dilation_pixels}px margin to each frame's native mask (≥ 0.5). Current motion only uses the native mask. Track filtering stays independent.`;
  } else {
    byId('motion-note').textContent = "Uses each frame's native motion mask (≥ 0.5). Track filtering stays independent.";
  }
  byId('frame').max = description.frames.length - 1;
  const pointLimit = description.viewer.max_points_per_frame;
  const budget = description.viewer.max_bytes;
  const previewLabel = budget ? `${(budget / 1e6).toFixed(0)} MB point-cloud budget` : 'Light preview';
  byId('preview-note').hidden = pointLimit === undefined;
  byId('preview-note').textContent = pointLimit === undefined ? '' : `${previewLabel} · up to ${pointLimit.toLocaleString()} sampled points/frame. Display controls do not change model results.`;
  demoViewer = new StaticDemoViewer(description, cloud, {
    add: object => scene.add(object), invalidate: markDirty,
    error: error => toast(error.message, true),
    playback: playing => {
      byId('play').setAttribute('aria-pressed', String(playing));
      byId('play').setAttribute('aria-label', playing ? 'Pause playback' : 'Start playback');
      byId('play').title = playing ? 'Pause playback' : 'Start playback';
    },
    loading: chunk => {
      byId('loading').hidden = !chunk;
      byId('loading').textContent = chunk ? `Buffering ${(chunk.bytes / 1048576).toFixed(2)} MiB` : '';
    },
    progress: () => { updateStatus(); updateClipping(); markDirty(); },
    tracks: tracks => {
      byId('show-tracks').disabled = !tracks || !description.tracks?.query_count;
      if (tracks && !tracks.arrays.confidence) byId('track-confidence-note').textContent = 'This export has no tracking confidence; filtering is unavailable.';
      applySettings();
    },
    frame: (frame, record, image) => {
      if (!scrubState) byId('frame').value = frame;
      byId('frame').setAttribute('aria-valuetext', `Frame ${frame + 1} of ${description.frames.length}`);
      byId('frame-value').textContent = `${frame + 1} / ${description.frames.length}`;
      updateRangeStyle(byId('frame'));
      frameImages.show(frame, byId('preview'));
      updateFollowCamera();
      updateStatus();
      updateClipping();
    },
  });
  const duration = demoViewer.timestamps.at(-1) ?? 0;
  byId('track-tail').max = Math.max(1, Math.ceil(duration));
  applySettings();
  await demoViewer.open();
  loadFilmstrip(description);
  cameraLayer = new CameraLayer(description, demoViewer.reference);
  scene.add(cameraLayer.group);
  byId('show-cameras').disabled = !cameraLayer.cameras.length;
  if (demoViewer.tracks && !demoViewer.tracks.arrays.confidence) byId('track-confidence-note').textContent = 'This export has no tracking confidence; filtering is unavailable.';
  byId('show-tracks').disabled = !demoViewer.tracks || !description.tracks?.query_count;
  byId('play').disabled = description.frames.length < 2;
  byId('follow-camera').disabled = !description.frames.some(record => predictedCameraView(record.camera_to_world, demoViewer.reference));
  byId('enter-camera').disabled = byId('follow-camera').disabled;
  if (byId('follow-camera').disabled) byId('follow-note').textContent = 'This export has no valid camera poses. Free-view navigation remains available.';
  byId('frame').disabled = false;
  applySettings();
  fitCloud(true);
  if (!byId('follow-camera').disabled) enterCamera(true);
  if (description.frames.length > 1 && !document.hidden) demoViewer.togglePlayback();
}

function animate(now = performance.now()) {
  requestAnimationFrame(animate);
  const elapsed = Math.max(0, Math.min((now - lastAnimation) / 1000, .2));
  lastAnimation = now;
  demoViewer?.tick(now);
  if (cameraFollower.update(elapsed)) {
    updateClipping();
    cloud?.queueSort();
    markDirty();
  }
  controls.update();
  if (!dirty || now - lastRender < 1000 / 30) return;
  lastRender = now;
  dirty = false;
  cameraLayer?.update(camera, demoViewer?.frame, byId('follow-camera').checked, byId('show-cameras').checked);
  const started = performance.now();
  renderer.render(scene, camera);
  byId('render-stats').textContent = `${cloud?.visibleCount.toLocaleString() ?? 0} points · ${(performance.now() - started).toFixed(1)} ms`;
}

resize();
animate();
open().catch(error => toast(error.message, true));
