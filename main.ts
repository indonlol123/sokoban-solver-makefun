// Sokoban solver for make.fun sokoban-solver-01-run-20261121.
//
// Push-optimal A* per room. The generator's levels are a hallway with rooms off
// it, and a box never leaves its room (a push through a door into the hallway is
// lost), so each room is an independent push-optimal subproblem and the level's
// optimum P* is the sum of its rooms' optima. We decompose the level into rooms
// (cutting the hallway's dead squares out of the floor), solve each room
// push-optimally, and stitch the rooms together with the player's walks.
//
// Search: A* over pushes. A state is the boxes' squares plus the player's
// reachable region (named by its lowest square): within a region the player can
// reach any square, so the exact square does not matter for future pushes.
// Heuristic: the Hungarian (minimum-assignment) sum of per-box BFS distances to
// goals, ignoring other boxes. Each push moves one box one square, so the
// assignment value lower-bounds the pushes still needed: admissible, hence the
// first goal state popped is push-optimal. Pruning: dead squares (a box that can
// never be pulled onto any goal) and frozen 2x2 blocks of walls and boxes.
// Fallback: if A* exhausts its expansion budget, the same search reruns with a
// zero heuristic (uniform-cost / BFS over pushes), which the generator's own
// reference uses and which solves every kept room within its state budget.
//
// Determinism: every budget is counted in expansions, never the clock, so two
// runs on one set write byte-identical outputs. No randomness, no clock reads,
// no environment reads beyond the input/output directories.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Per-room A* expansion budget; per-room BFS (zero-heuristic) budget. */
const ASTAR_CAP = 600_000;
const BFS_CAP = 1_200_000;
/** Whole-set expansion budget: bounds the worst-case wall time well under 60 s. */
const TOTAL_CAP = 4_000_000;

const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/** The squares from which a box alone can be pulled onto some goal. */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** The squares the player reaches from `from` with `occupied` in the way; returns the lowest. */
function reachLow(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    // Neighbours, in a fixed order: up, down, left, right.
    if (wall[p - w] === 0 && occupied[p - w] === 0 && seen[p - w] !== stamp) { seen[p - w] = stamp; stack[n++] = p - w; }
    if (wall[p + w] === 0 && occupied[p + w] === 0 && seen[p + w] !== stamp) { seen[p + w] = stamp; stack[n++] = p + w; }
    if (wall[p - 1] === 0 && occupied[p - 1] === 0 && seen[p - 1] !== stamp) { seen[p - 1] = stamp; stack[n++] = p - 1; }
    if (wall[p + 1] === 0 && occupied[p + 1] === 0 && seen[p + 1] !== stamp) { seen[p + 1] = stamp; stack[n++] = p + 1; }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(w: number, wall: Uint8Array, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  door: number;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a
 * box there reaches no goal) out of the floor; each remaining part that holds a
 * box is a room. Null when the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], door: -1 });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  // Each room's door: the room square next to a hallway (player-row) square.
  for (const r of rooms) {
    for (const s of r.squares) {
      if ((s + w >= row * w && s + w < (row + 1) * w && !wall[s + w]) || (s - w >= row * w && s - w < (row + 1) * w && !wall[s - w])) {
        r.door = s;
        break;
      }
    }
  }
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(w: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how: string[] = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: unreachable');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** Shared per-search scratch, counted against the global work budget. */
const budget = { used: 0 };

interface SearchResult {
  pushes: number;
  moves: string;
}

/**
 * Push-optimal search of one sub-level (a room, or a whole level as fallback).
 * `requireExit`: when true, a solved state also needs the player able to reach
 * `door` from its final square, so the player can leave the room afterwards.
 * Returns null when the search gives up (budget) or the level has no solution.
 * `weighted`: 1 for A* (Hungarian heuristic), 0 for uniform-cost/BFS.
 */
function solvePush(
  level: Level,
  cap: number,
  weighted: 0 | 1,
  requireExit: boolean,
  door: number,
): SearchResult | null {
  const { w, wall, goals, boxes: startBoxes, player: startPlayer } = level;
  const size = wall.length;
  const n = startBoxes.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  if (startBoxes.some((b) => !live[b])) return null;

  // BFS distances from each goal through walls only (boxes ignored).
  const dist: Int32Array[] = goals.map((g) => {
    const d = new Int32Array(size).fill(-1);
    const q = [g];
    d[g] = 0;
    for (let qi = 0; qi < q.length; qi++) {
      const p = q[qi]!;
      for (const [dx, dy] of DIRS) {
        const t = p + dy * w + dx;
        if (!wall[t] && d[t] < 0) {
          d[t] = d[p]! + 1;
          q.push(t);
        }
      }
    }
    return d;
  });

  // Heuristic: Hungarian minimum assignment of boxes to goals (n <= 6), else the
  // sum of each box's nearest-goal distance. Infinity when some box reaches no goal.
  const heuristic = (bs: number[]): number => {
    if (weighted === 0) return 0;
    if (n <= 6) {
      // Enumerate every assignment (permutation of goal indices).
      const perm = goals.map((_, i) => i);
      let best = Infinity;
      const rec = (k: number, acc: number): void => {
        if (acc >= best) return;
        if (k === n) {
          best = acc;
          return;
        }
        for (let i = k; i < n; i++) {
          const t = perm[k]!;
          perm[k] = perm[i]!;
          perm[i] = t;
          const c = dist[perm[k]!]![bs[k]!]!;
          if (c >= 0) rec(k + 1, acc + c);
          const t2 = perm[k]!;
          perm[k] = perm[i]!;
          perm[i] = t2;
        }
      };
      rec(0, 0);
      return best;
    }
    let sum = 0;
    for (const b of bs) {
      let m = Infinity;
      for (const d of dist) {
        const v = d[b]!;
        if (v >= 0 && v < m) m = v;
      }
      if (m === Infinity) return Infinity;
      sum += m;
    }
    return sum;
  };

  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);

  // State storage: boxes, player square, parent, the push that made it.
  const sb: number[][] = [startBoxes];
  const sp: number[] = [startPlayer];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushDir: number[] = [-1]; // index into DIRS
  const sg: number[] = [0];
  const sh: number[] = [heuristic(startBoxes)];
  const sf: number[] = [sh[0]!];
  const skey: (number | string)[] = [];
  const numericKey = (n + 1) * Math.ceil(Math.log2(size)) < 52;
  const keyOf = (bs: number[], low: number): number | string => {
    if (numericKey) {
      let k = low;
      for (const b of bs) k = k * size + b;
      return k;
    }
    return low + ':' + bs.join(',');
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const best = new Map<number | string, number>();
  for (const b of startBoxes) occupied[b] = 1;
  const low0 = reachLow(w, wall, occupied, startPlayer, seen, ++stamp, stack);
  for (const b of startBoxes) occupied[b] = 0;
  const k0 = keyOf(startBoxes, low0);
  skey.push(k0);
  best.set(k0, 0);

  // Binary heap of state indices; A*: (f, h, -g, index); BFS: (f, index).
  const heap: number[] = [0];
  const less = (a: number, b: number): boolean => {
    if (sf[a]! !== sf[b]!) return sf[a]! < sf[b]!;
    if (weighted === 0) return a < b;
    if (sh[a]! !== sh[b]!) return sh[a]! < sh[b]!;
    if (sg[a]! !== sg[b]!) return sg[a]! > sg[b]!;
    return a < b;
  };
  const heapPush = (s: number): void => {
    let i = heap.length;
    heap.push(s);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  // Door-reachability check for a solved state: flood fill from the player's
  // square around walls and boxes; the door must be in the region.
  const canExit = (bs: number[], from: number): boolean => {
    if (door < 0) return true;
    for (const b of bs) occupied[b] = 1;
    const low = reachLow(w, wall, occupied, from, seen, ++stamp, stack);
    void low;
    // reachLow marks the region with `stamp` in `seen`; test the door directly.
    const ok = seen[door] === stamp;
    for (const b of bs) occupied[b] = 0;
    return ok;
  };

  let found = solved(startBoxes) && (!requireExit || canExit(startBoxes, startPlayer)) ? 0 : -1;
  let expansions = 0;
  while (heap.length > 0 && found < 0 && expansions < cap && budget.used < TOTAL_CAP) {
    const s = heapPop();
    if (best.get(skey[s]!) !== sg[s]) continue; // stale heap entry
    const bs = sb[s]!;
    if (solved(bs)) {
      if (!requireExit || canExit(bs, sp[s]!)) {
        found = s;
        break;
      }
      // Solved but the player cannot leave: keep searching for another way.
      continue;
    }
    expansions++;
    budget.used++;
    for (const b of bs) occupied[b] = 1;
    reachLow(w, wall, occupied, sp[s]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi]!;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di]!;
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, t);
        let key: number | string = -1;
        let next: number[] | null = null;
        let h = Infinity;
        if (!frozen) {
          next = bs.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          h = heuristic(next);
          if (h !== Infinity) key = keyOf(next, reachLow(w, wall, occupied, b, mark, ++stamp, stack));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || h === Infinity) continue;
        const seenG = best.get(key);
        if (seenG !== undefined && seenG <= sg[s]! + 1) continue;
        best.set(key, sg[s]! + 1);
        const ns = sb.length;
        sb.push(next!);
        sp.push(b);
        parent.push(s);
        pushFrom.push(b);
        pushDir.push(di);
        sg.push(sg[s]! + 1);
        sh.push(h);
        sf.push(sg[s]! + 1 + h);
        skey.push(key);
        heapPush(ns);
      }
    }
    for (const b of bs) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Reconstruct: the pushes, then the walks between them.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes;
  for (const s of path) {
    const nextBoxes = sb[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const di = pushDir[s]!;
    const d = DIRS[di]![1] * w + DIRS[di]![0];
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += DIRS[di]![3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/** Solve one level's text into its moves string (room by room, else whole). */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (rooms === null) {
    const res = solvePush(level, ASTAR_CAP, 1, false, -1) ?? solvePush(level, BFS_CAP, 0, false, -1);
    return res === null ? '' : res.moves;
  }
  const done: boolean[] = rooms.map(() => false);
  let player = level.player;
  let moves = '';
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri]!;
    const wall = level.wall.slice();
    for (let rj = 0; rj < rooms.length; rj++) {
      if (rj === ri) continue;
      for (const b of done[rj]! ? rooms[rj]!.goals : rooms[rj]!.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const lastRoom = ri === rooms.length - 1;
    let res = solvePush(sub, ASTAR_CAP, 1, !lastRoom, room.door);
    if (res === null) res = solvePush(sub, BFS_CAP, 0, !lastRoom, room.door);
    if (res === null) return moves; // unsolved room: keep the valid moves so far
    moves += res.moves;
    player = endOf(level.w, player, res.moves);
    done[ri] = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
