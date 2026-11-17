let positions;
let pointIndices;
self.onmessage = ({data}) => {
  if (data.type === 'positions') {
    positions = new Float32Array(data.positions);
    pointIndices = new Uint32Array(data.indices);
    return;
  }
  if (!positions) return;
  const count = positions.length / 3;
  const depths = new Float32Array(count);
  const indices = new Uint32Array(count);
  for (let index = 0; index < count; index++) {
    depths[index] = positions[index * 3] * data.direction[0] + positions[index * 3 + 1] * data.direction[1] + positions[index * 3 + 2] * data.direction[2];
    indices[index] = index;
  }
  indices.sort((left, right) => depths[left] - depths[right]);
  const selected = Uint32Array.from(indices, index => pointIndices[index]);
  postMessage({indices: selected.buffer, version: data.version}, [selected.buffer]);
};
