export interface Rect { x: number; y: number; w: number; h: number; dirX?: number; dirY?: number; }

export class Particle {
    x: number; y: number;
    vx: number; vy: number;
    ax: number; ay: number;
    maxSpeed: number; maxForce: number; radius: number;
    color: string; baseColor: string; crushColor: string;
    escaped: boolean;

    constructor(x: number, y: number) {
        this.x = x; this.y = y;
        this.vx = 0; this.vy = 0; this.ax = 0; this.ay = 0;
        this.maxSpeed = 2 + Math.random() * 1.5; 
        this.maxForce = 0.05; 
        this.radius = 3;
        this.baseColor = '#00ffff'; 
        this.crushColor = '#ff0000'; 
        this.color = this.baseColor;
        this.escaped = false; 
    }

    applyForce(fx: number, fy: number) {
        this.ax += fx; this.ay += fy;
    }

    checkEscaped(exits: Rect[]) {
        exits.forEach(exit => {
            const centerX = exit.x + exit.w / 2;
            const centerY = exit.y + exit.h / 2;
            const d = Math.hypot(centerX - this.x, centerY - this.y);
            if (d < Math.max(exit.w, exit.h) / 2) {
                this.escaped = true; 
            }
        });
    }

    applyLightGrid(lightGrid: string[][], cellW: number, cellH: number, exits: Rect[]) {
        let steerX = 0; let steerY = 0;
        let whiteCount = 0; let redCount = 0;

        const myCol = Math.max(0, Math.min(19, Math.floor(this.x / cellW)));
        const myRow = Math.max(0, Math.min(19, Math.floor(this.y / cellH)));

        const onWhite = lightGrid[myCol] && lightGrid[myCol][myRow] === "WHITE";

        const searchRadius = 2; 
        const startX = Math.max(0, myCol - searchRadius);
        const endX = Math.min(19, myCol + searchRadius);
        const startY = Math.max(0, myRow - searchRadius);
        const endY = Math.min(19, myRow + searchRadius);

        for (let x = startX; x <= endX; x++) {
            for (let y = startY; y <= endY; y++) {
                const cellColor = lightGrid[x][y];
                if (cellColor === "OFF") continue;

                const cellCenterX = (x * cellW) + (cellW / 2);
                const cellCenterY = (y * cellH) + (cellH / 2);

                const dx = cellCenterX - this.x;
                const dy = cellCenterY - this.y;
                const dist = Math.hypot(dx, dy);

                if (dist > 0) {
                    if (cellColor === "WHITE") {
                        const weight = 1 / dist;
                        steerX += (dx / dist) * weight;
                        steerY += (dy / dist) * weight;
                        whiteCount++;
                    } else if (cellColor === "RED") {
                        const weight = 1 / dist;
                        steerX -= (dx / dist) * weight * 3.5; 
                        steerY -= (dy / dist) * weight * 3.5;
                        redCount++;
                    }
                }
            }
        }

        let compassX = 0; let compassY = 0;
        if (exits.length > 0) {
            let closest = exits[0];
            let recordDist = Infinity;
            exits.forEach(exit => {
                const cx = exit.x + exit.w / 2;
                const cy = exit.y + exit.h / 2;
                const d = Math.hypot(cx - this.x, cy - this.y);
                if (d < recordDist) { recordDist = d; closest = exit; }
            });
            const dx = (closest.x + closest.w / 2) - this.x;
            const dy = (closest.y + closest.h / 2) - this.y;
            const dist = Math.hypot(dx, dy);
            if (dist > 0) {
                compassX = dx / dist;
                compassY = dy / dist;
            }
        }

        if (whiteCount > 0 || redCount > 0) {
            // FIX 1: Nerfed the compass heavily (from 1.2 down to 0.15)
            // Now they will trust the AI's pathing around walls rather than blindly rushing the exit
            steerX += compassX * 0.15;
            steerY += compassY * 0.15;

            steerX += (Math.random() - 0.5) * 0.8;
            steerY += (Math.random() - 0.5) * 0.8;

            const currentMaxSpeed = onWhite ? this.maxSpeed * 1.5 : this.maxSpeed;

            const mag = Math.hypot(steerX, steerY);
            if (mag > 0) {
                steerX = (steerX / mag) * currentMaxSpeed - this.vx;
                steerY = (steerY / mag) * currentMaxSpeed - this.vy;

                const steerMag = Math.hypot(steerX, steerY);
                if (steerMag > this.maxForce) {
                    steerX = (steerX / steerMag) * this.maxForce;
                    steerY = (steerY / steerMag) * this.maxForce;
                }
                this.applyForce(steerX, steerY);
            }
            this.color = redCount > whiteCount ? '#ff8800' : '#ffffff';
        } else {
            this.applyForce((Math.random() - 0.5) * 0.05, (Math.random() - 0.5) * 0.05);
            this.color = this.baseColor;
        }
    }

    resolveWalls(walls: Rect[]) {
        walls.forEach(w => {
            let testX = this.x; let testY = this.y;
            if (this.x < w.x) testX = w.x; else if (this.x > w.x + w.w) testX = w.x + w.w;
            if (this.y < w.y) testY = w.y; else if (this.y > w.y + w.h) testY = w.y + w.h;

            let distX = this.x - testX; let distY = this.y - testY;
            let distance = Math.sqrt((distX * distX) + (distY * distY));

            if (distance <= this.radius && distance > 0) {
                // FIX 2: Vector Sliding instead of Pong Bouncing
                // Push the particle perfectly out of the wall based on radius overlap
                const overlap = this.radius - distance;
                this.x += (distX / distance) * overlap;
                this.y += (distY / distance) * overlap;
                
                // Dampen the velocity heavily so they "slide" frictionally along the wall 
                // to find the edge, rather than ricocheting backwards.
                this.vx *= 0.5; 
                this.vy *= 0.5;
            }
        });
    }

    separate(particles: Particle[]) {
        let steerX = 0; let steerY = 0; let crushCount = 0;
        const desiredSeparation = this.radius * 3; 

        for (let i = 0; i < particles.length; i++) {
            const other = particles[i];
            if (this === other) continue;
            const dx = this.x - other.x; const dy = this.y - other.y;
            const distSq = dx * dx + dy * dy;

            if (distSq > 0 && distSq < desiredSeparation * desiredSeparation) {
                const dist = Math.sqrt(distSq);
                steerX += (dx / dist) / dist; steerY += (dy / dist) / dist;
                crushCount++;
            }
        }

        if (this.color !== '#ffffff' && this.color !== '#ff8800') {
            this.color = crushCount > 3 ? this.crushColor : this.baseColor;
        }

        if (crushCount > 0) {
            steerX /= crushCount; steerY /= crushCount;
            const mag = Math.sqrt(steerX * steerX + steerY * steerY);
            if (mag > 0) {
                steerX = (steerX / mag) * this.maxSpeed - this.vx;
                steerY = (steerY / mag) * this.maxSpeed - this.vy;
                const repulseForce = this.maxForce * 1.5; 
                const steerMag = Math.sqrt(steerX * steerX + steerY * steerY);
                if (steerMag > repulseForce) {
                    steerX = (steerX / steerMag) * repulseForce;
                    steerY = (steerY / steerMag) * repulseForce;
                }
            }
        }
        this.applyForce(steerX, steerY);
    }

    update() {
        this.vx += this.ax; this.vy += this.ay;
        this.x += this.vx; this.y += this.vy;
        this.ax = 0; this.ay = 0;
    }

    draw(ctx: CanvasRenderingContext2D) {
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();
    }
}