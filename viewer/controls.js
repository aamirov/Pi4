import * as THREE from 'three';
import {TrackballControls} from './vendor/TrackballControls.js';

export class ScreenTrackballControls extends TrackballControls {
  constructor(camera, element) {
    super(camera, element);
    this.reverseZoom = true;
    const mouseMove = this._onMouseMove;
    const touchMove = this._onTouchMove;
    this._onMouseMove = event => { mouseMove(event); this.update(); };
    this._onTouchMove = event => { touchMove(event); this.update(); };
    element.removeEventListener('wheel', this._onMouseWheel);
    this._onMouseWheel = event => {
      if (!this.enabled || this.noZoom) return;
      event.preventDefault();
      const units = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.screen.height : 1;
      const direction = this.reverseZoom && !event.ctrlKey ? -1 : 1;
      const factor = Math.exp(THREE.MathUtils.clamp(event.deltaY * units * direction * this.zoomSpeed * .001, -2, 2));
      this.dispatchEvent({type: 'start'});
      if (this.object.isPerspectiveCamera) {
        const offset = this.object.position.clone().sub(this.target);
        offset.setLength(THREE.MathUtils.clamp(offset.length() * factor, this.minDistance, this.maxDistance));
        this.object.position.copy(this.target).add(offset);
      } else {
        this.object.zoom = THREE.MathUtils.clamp(this.object.zoom / factor, this.minZoom, this.maxZoom);
        this.object.updateProjectionMatrix();
      }
      this.update();
      this.dispatchEvent({type: 'end'});
    };
    element.addEventListener('wheel', this._onMouseWheel, {passive: false});
    this._cancelOnPointer = event => {
      if (this._pointers.some(pointer => pointer.pointerId === event.pointerId)) this.cancelInteraction();
    };
    this._cancelOnBlur = () => this.cancelInteraction();
    this._cancelOnVisibility = () => { if (document.hidden) this.cancelInteraction(); };
    element.addEventListener('pointercancel', this._cancelOnPointer, true);
    element.addEventListener('lostpointercapture', this._cancelOnPointer);
    window.addEventListener('blur', this._cancelOnBlur);
    document.addEventListener('visibilitychange', this._cancelOnVisibility);
  }

  update() {
    super.update();
    const previous = this._viewPose;
    if (!previous) {
      this._viewPose = {position: this.object.position.clone(), quaternion: this.object.quaternion.clone(), target: this.target.clone(), zoom: this.object.zoom};
      return;
    }
    if (previous.position.equals(this.object.position) && previous.quaternion.equals(this.object.quaternion) && previous.target.equals(this.target) && previous.zoom === this.object.zoom) return;
    previous.position.copy(this.object.position);
    previous.quaternion.copy(this.object.quaternion);
    previous.target.copy(this.target);
    previous.zoom = this.object.zoom;
    this.dispatchEvent({type: 'viewchange'});
  }

  _trackballPoint(pointer) {
    const ratio = this.screen.width / Math.max(1, Math.min(this.screen.width, this.screen.height));
    const horizontal = pointer.x * ratio;
    const vertical = pointer.y * ratio;
    const distanceSquared = horizontal * horizontal + vertical * vertical;
    const depth = distanceSquared <= .5 ? Math.sqrt(1 - distanceSquared) : .5 / Math.sqrt(distanceSquared);
    return new THREE.Vector3(horizontal, vertical, depth).normalize();
  }

  _rotateCamera() {
    if (this._movePrev.equals(this._moveCurr)) return;
    const previous = this._trackballPoint(this._movePrev);
    const current = this._trackballPoint(this._moveCurr);
    this._movePrev.copy(this._moveCurr);
    const rotation = new THREE.Quaternion().setFromUnitVectors(current, previous);
    const sine = Math.hypot(rotation.x, rotation.y, rotation.z);
    if (!sine) return;
    const angle = 2 * Math.atan2(sine, rotation.w) * this.rotateSpeed;
    rotation.setFromAxisAngle(new THREE.Vector3(rotation.x, rotation.y, rotation.z).divideScalar(sine), angle);
    const backward = this._eye.clone().normalize();
    const vertical = this.object.up.clone().addScaledVector(backward, -this.object.up.dot(backward));
    if (vertical.lengthSq() < 1e-12) {
      vertical.set(Math.abs(backward.y) < .9 ? 0 : 1, Math.abs(backward.y) < .9 ? 1 : 0, 0);
      vertical.addScaledVector(backward, -vertical.dot(backward));
    }
    vertical.normalize();
    const right = new THREE.Vector3().crossVectors(vertical, backward).normalize();
    const basis = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, vertical, backward));
    rotation.premultiply(basis).multiply(basis.clone().invert()).normalize();
    this._eye.applyQuaternion(rotation);
    this.object.up.copy(vertical).applyQuaternion(rotation).normalize();
  }

  cancelInteraction() {
    const pointers = this._pointers.slice();
    this._pointers.length = 0;
    this._pointerPositions = {};
    this.state = this.keyState = -1;
    this._movePrev.copy(this._moveCurr);
    this._panStart.copy(this._panEnd);
    this._zoomStart.copy(this._zoomEnd);
    this._lastAngle = 0;
    this.domElement.removeEventListener('pointermove', this._onPointerMove);
    this.domElement.removeEventListener('pointerup', this._onPointerUp);
    for (const pointer of pointers) if (this.domElement.hasPointerCapture(pointer.pointerId)) this.domElement.releasePointerCapture(pointer.pointerId);
    if (pointers.length) this.dispatchEvent({type: 'end'});
  }

  dispose() {
    this.cancelInteraction();
    this.domElement.removeEventListener('pointercancel', this._cancelOnPointer, true);
    this.domElement.removeEventListener('lostpointercapture', this._cancelOnPointer);
    window.removeEventListener('blur', this._cancelOnBlur);
    document.removeEventListener('visibilitychange', this._cancelOnVisibility);
    super.dispose();
  }

  _panCamera() {
    const delta = this._panEnd.clone().sub(this._panStart);
    if (!delta.lengthSq()) return;
    const height = this.object.isPerspectiveCamera ? 2 * this._eye.length() * Math.tan(THREE.MathUtils.degToRad(this.object.fov / 2)) : (this.object.top - this.object.bottom) / this.object.zoom;
    const width = this.object.isPerspectiveCamera ? height * this.object.aspect : (this.object.right - this.object.left) / this.object.zoom;
    const shift = this._eye.clone().cross(this.object.up).normalize().multiplyScalar(delta.x * width * this.panSpeed).add(this.object.up.clone().normalize().multiplyScalar(delta.y * height * this.panSpeed));
    this.object.position.add(shift);
    this.target.add(shift);
    this._panStart.copy(this._panEnd);
  }
}

