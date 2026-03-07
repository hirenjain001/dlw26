import type { LightColor, LightDelta } from "../api/protocol";


const W = 40;
const H = 40;

type Grid = LightColor[][];

function makeGrid(fill: LightColor): Grid {
  const g: Grid = [];
  for (let x = 0; x < W; x++) {
    const col: LightColor[] = [];
    for (let y = 0; y < H; y++) col.push(fill);
    g.push(col);
  }
  return g;
}

let grid: Grid = makeGrid("OFF");
const subs = new Set<() => void>();

export function getLightGrid(): Grid {
  return grid;
}

export function resetLightGrid(): void {
  grid = makeGrid("OFF");
  subs.forEach((fn) => fn());
}

export function applyLightsDelta(delta: LightDelta[]): void {
  if (!delta || delta.length === 0) return;

  // shallow copy grid + mutated columns so updates are “atomic”
  const next = grid.map((col) => col.slice());

  for (const [x, y, c] of delta) {
    if (x >= 0 && x < W && y >= 0 && y < H) next[x][y] = c;
  }

  grid = next;
  subs.forEach((fn) => fn());
}

export function subscribeLightGrid(fn: () => void): () => void {
  subs.add(fn);
  return () => subs.delete(fn);
}