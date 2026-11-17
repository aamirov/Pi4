import * as THREE from 'three';
import {predictedCameraView} from './camera-follow.js';
import {linspaceIndices} from './point-sampling.js';

export class CameraLayer {
  constructor(description, reference) {
    this.group = new THREE.Group();
    this.cameras = [];
    const samples = linspaceIndices(description.frames.length, 72);
    const aspect = description.image_size[0] / description.image_size[1];
    const corners = [[-aspect, -1, 2.5], [aspect, -1, 2.5], [aspect, 1, 2.5], [-aspect, 1, 2.5]];
    const vertices = [];
    corners.forEach((corner, index) => vertices.push(0, 0, 0, ...corner, ...corner, ...corners[(index + 1) % corners.length]));
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    for (const frame of samples) {
      const pose = predictedCameraView(description.frames[frame].camera_to_world, reference);
      if (!pose) continue;
      const color = new THREE.Color().setHSL(.76 - .3 * frame / Math.max(1, description.frames.length - 1), .65, .58);
      const material = new THREE.LineBasicMaterial({color, transparent: true, opacity: .65, depthWrite: false});
      const frustum = new THREE.LineSegments(this.geometry, material);
      frustum.position.copy(pose.position);
      frustum.quaternion.setFromRotationMatrix(new THREE.Matrix4().lookAt(pose.position, pose.position.clone().sub(pose.forward), pose.up));
      frustum.frustumCulled = false;
      this.cameras.push({frame, frustum});
      this.group.add(frustum);
    }
  }

  update(camera, frame, following, enabled) {
    this.group.visible = enabled;
    if (!enabled) return;
    const direction = camera.getWorldDirection(new THREE.Vector3());
    const offset = new THREE.Vector3();
    for (const record of this.cameras) {
      const depth = offset.copy(record.frustum.position).sub(camera.position).dot(direction);
      record.frustum.visible = depth > camera.near * 10 && (!following || record.frame !== frame);
      record.frustum.scale.setScalar(Math.min(.025, Math.max(0, depth) * .008));
      record.frustum.material.opacity = record.frame === frame ? .85 : .4;
    }
  }
}
