export function pickPointCloudPoint(cloud, camera, {x, y, width, height, radius = 6}) {
  if (!cloud.count || cloud.material.uniforms.uOpacity.value <= .001) return null;
  camera.updateMatrixWorld(true);
  cloud.updateWorldMatrix(true, false);
  const transform = camera.projectionMatrix.clone().multiply(camera.matrixWorldInverse).multiply(cloud.matrixWorld).elements;
  const positions = cloud.geometry.attributes.position;
  const uniforms = cloud.material.uniforms;
  let best = null;
  for (const index of cloud.sampleIndices) {
    if (!cloud.isPointVisible(index)) continue;
    const horizontal = positions.getX(index), vertical = positions.getY(index), depth = positions.getZ(index);
    const clipW = transform[3] * horizontal + transform[7] * vertical + transform[11] * depth + transform[15];
    const clipZ = transform[2] * horizontal + transform[6] * vertical + transform[10] * depth + transform[14];
    if (!(clipW > 0) || clipZ < -clipW || clipZ > clipW) continue;
    const screenX = (transform[0] * horizontal + transform[4] * vertical + transform[8] * depth + transform[12]) / clipW;
    const screenY = (transform[1] * horizontal + transform[5] * vertical + transform[9] * depth + transform[13]) / clipW;
    const deltaX = (screenX - (x / width * 2 - 1)) * width * .5;
    const deltaY = (screenY - (1 - y / height * 2)) * height * .5;
    const distance = deltaX * deltaX + deltaY * deltaY;
    if (!Number.isFinite(distance) || distance > radius * radius + 1e-8) continue;
    const emphasis = cloud.settings.highlight && cloud.geometry.attributes.frameId.getX(index) === cloud.settings.frame ? 1.6 : 1;
    const size = Math.max(1, Math.min(uniforms.uSize.value * emphasis * uniforms.uPixelRatio.value, uniforms.uMaxSize.value)) / uniforms.uPixelRatio.value;
    const priority = distance <= size * size * .25 ? 1 : 0;
    const projectedDepth = clipZ / clipW;
    const score = priority ? projectedDepth : distance;
    const secondary = priority ? distance : projectedDepth;
    if (best && (priority < best.priority || (priority === best.priority && (score > best.score + 1e-8 || (Math.abs(score - best.score) < 1e-8 && secondary >= best.secondary))))) continue;
    best = {index, priority, score, secondary};
  }
  if (best) best.point = camera.position.clone().fromBufferAttribute(positions, best.index).applyMatrix4(cloud.matrixWorld);
  return best;
}
