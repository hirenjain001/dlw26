import {Particle, type Rect} from './frontend/src/Particle';
import { GRID_SIZE } from "../simulation/frontend/src/config/grid";

export interface InitPayload {
    type: "init";
    session_id: string;
    grid: { w: number; h: number };
    layout: {
        walls: [number, number][];
        exits: [number, number][];
    };
    opts: { max_exits: number };
}

export interface TickPayload {
    type: "tick";
    session_id: string;
    t: number;
    ts_ms: number;
    crowd_delta: [number, number, number][]; // [x, y, count]
    fire_on: [number, number][];             // [x, y]
    fire_off: [number, number][];            // [x, y]
}

export class Link {
    private sessionId: string = "demo";
    private gridSize: number = GRID_SIZE;
    private canvasW: number;
    private canvasH: number;
    private tickCounter: number = 0;

    // Store the PREVIOUS state here so we can calculate what changed (the deltas)
    private prevCrowd: number[][];
    private prevFire: boolean[][];

    constructor(canvasWidth: number, canvasHeight: number) {
        this.canvasW = canvasWidth;
        this.canvasH = canvasHeight;
        
        // Initialize empty 40x40 grids filled with 0s and false
        this.prevCrowd = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
        this.prevFire = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(false));
    }

    private rectToCenterCell(rect: Rect): [number, number] {
        const cellW = this.canvasW / this.gridSize;
        const cellH = this.canvasH / this.gridSize;

        const centerX = rect.x + rect.w / 2;
        const centerY = rect.y + rect.h / 2;

        const gx = Math.max(0, Math.min(GRID_SIZE - 1, Math.floor(centerX / cellW)));
        const gy = Math.max(0, Math.min(GRID_SIZE - 1, Math.floor(centerY / cellH)));

        return [gx, gy];
    }

    private rectToCells(rect: Rect): [number, number][] {
        const cellW = this.canvasW / this.gridSize;
        const cellH = this.canvasH / this.gridSize;

        // Find the start and end grid columns/rows, clamping them between 0 and 39
        const startX = Math.max(0, Math.floor(rect.x / cellW));
        const endX = Math.min(GRID_SIZE-1, Math.floor((rect.x + rect.w) / cellW));
        const startY = Math.max(0, Math.floor(rect.y / cellH));
        const endY = Math.min(GRID_SIZE-1, Math.floor((rect.y + rect.h) / cellH));

        const cells: [number, number][] = [];
        for (let x = startX; x <= endX; x++) {
            for (let y = startY; y <= endY; y++) {
                cells.push([x, y]);
            }
        }
        return cells;
    }

    public generateInit(walls: Rect[], exits: Rect[]): InitPayload {
        const wallSet = new Set<string>();
        walls.forEach(w => this.rectToCells(w).forEach(c => wallSet.add(`${c[0]},${c[1]}`)));

        const exitSet = new Set<string>();
        exits.forEach(e => {
            const [x, y] = this.rectToCenterCell(e);
            exitSet.add(`${x},${y}`);
        });

        return {
            type: "init",
            session_id: this.sessionId,
            grid: { w: this.gridSize, h: this.gridSize },
            layout: {
                walls: Array.from(wallSet).map(s => s.split(',').map(Number) as [number, number]),
                exits: Array.from(exitSet).map(s => s.split(',').map(Number) as [number, number])
            },
            opts: { max_exits: 3 }
        };
    }

    public generateTick(particles: Particle[], fires: Rect[]): TickPayload {
        const cellW = this.canvasW / this.gridSize;
        const cellH = this.canvasH / this.gridSize;

        // 1. Calculate Current Crowd Grid
        const currentCrowd = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
        particles.forEach(p => {
            if (p.escaped) return; // Ignore people who left the building
            const x = Math.max(0, Math.min(GRID_SIZE-1, Math.floor(p.x / cellW)));
            const y = Math.max(0, Math.min(GRID_SIZE-1, Math.floor(p.y / cellH)));
            currentCrowd[x][y]++;
        });

        // 2. Compare Current Crowd to Previous Crowd (Find the Deltas)
        const crowd_delta: [number, number, number][] = [];
        for (let x = 0; x < GRID_SIZE; x++) {
            for (let y = 0; y < GRID_SIZE; y++) {
                if (currentCrowd[x][y] !== this.prevCrowd[x][y]) {
                    crowd_delta.push([x, y, currentCrowd[x][y]]);
                }
            }
        }

        // 3. Calculate Current Fire Grid
        const currentFire = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(false));
        fires.forEach(f => {
            this.rectToCells(f).forEach(([x, y]) => {
                currentFire[x][y] = true;
            });
        });

        // 4. Compare Current Fire to Previous Fire (Find fire_on and fire_off)
        const fire_on: [number, number][] = [];
        const fire_off: [number, number][] = [];
        for (let x = 0; x < GRID_SIZE; x++) {
            for (let y = 0; y < GRID_SIZE; y++) {
                const isNowOn = currentFire[x][y];
                const wasOn = this.prevFire[x][y];
                
                if (isNowOn && !wasOn) fire_on.push([x, y]);
                if (!isNowOn && wasOn) fire_off.push([x, y]);
            }
        }

        // 5. Save current state as previous state for the next tick
        this.prevCrowd = currentCrowd;
        this.prevFire = currentFire;
        this.tickCounter++;

        return {
            type: "tick",
            session_id: this.sessionId,
            t: this.tickCounter,
            ts_ms: Date.now(),
            crowd_delta,
            fire_on,
            fire_off
        };
    }
}