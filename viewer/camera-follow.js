import * as THREE from 'three';

export function predictedCameraView(pose, reference) {
  if (!Array.isArray(pose) || pose.length !== 4 || !pose.every(row => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite))) return null;
  if (!reference || !Number.isFinite(reference.scale) || reference.scale <= 0 || !Array.isArray(reference.center) || reference.center.length !== 3 || !reference.center.every(Number.isFinite)) return null;
  const position = new THREE.Vector3(pose[0][3], pose[1][3], pose[2][3]);
  position.sub(new THREE.Vector3().fromArray(reference.center)).divideScalar(reference.scale);
  const forward = new THREE.Vector3(pose[0][2], pose[1][2], pose[2][2]);
  const up = new THREE.Vector3(-pose[0][1], -pose[1][1], -pose[2][1]);
  if (forward.lengthSq() < 1e-12 || up.lengthSq() < 1e-12 || new THREE.Vector3().crossVectors(forward, up).lengthSq() < 1e-12) return null;
  return {position, forward: forward.normalize(), up: up.normalize()};
}

export class SmoothCameraFollower {
  constructor(camera, controls) {
    this.camera = camera;
    this.controls = controls;
    this.goal = null;
    this.distance = 1;
    this.forward = new THREE.Vector3();
  }

  setView(view, distance, immediate = false) {
    this.controls.cancelInteraction();
    this.distance = Math.max(.0001, this.camera.position.distanceTo(this.controls.target));
    const target = view.position.clone().add(view.forward);
    const quaternion = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(view.position, target, view.up));
    this.goal = {position: view.position.clone(), quaternion, distance};
    this.update(immediate ? Infinity : 0);
  }

  update(elapsed) {
    if (!this.goal) return false;
    const goal = this.goal;
    const positionAlpha = 1 - Math.exp(-8 * Math.max(0, elapsed));
    const rotationAlpha = 1 - Math.exp(-11 * Math.max(0, elapsed));
    this.camera.position.lerp(goal.position, positionAlpha);
    this.camera.quaternion.slerp(goal.quaternion, rotationAlpha);
    this.distance = THREE.MathUtils.lerp(this.distance, goal.distance, positionAlpha);
    if (this.camera.position.distanceToSquared(goal.position) < 1e-12 && this.camera.quaternion.angleTo(goal.quaternion) < 1e-6 && Math.abs(this.distance - goal.distance) < 1e-6) {
      this.camera.position.copy(goal.position);
      this.camera.quaternion.copy(goal.quaternion);
      this.distance = goal.distance;
      this.goal = null;
    }
    this.camera.up.set(0, 1, 0).applyQuaternion(this.camera.quaternion);
    this.forward.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
    this.controls.target.copy(this.camera.position).addScaledVector(this.forward, this.distance);
    this.camera.updateMatrixWorld(true);
    return true;
  }

  stop() {
    this.goal = null;
  }
}
