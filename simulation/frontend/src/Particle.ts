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

    // 1. DESPAWN ONLY: No primitive steering, just delete them if they cross the threshold
    checkEscaped(exits: Rect[]) {
        exits.forEach(exit => {
            const centerX = exit.x + exit.w / 2;
            const centerY = exit.y + exit.h / 2;
            const d = Math.hypot(centerX - this.x, centerY - this.y);
            // If they are inside the exit, mark for garbage collection
            if (d < Math.max(exit.w, exit.h) / 2) {
                this.escaped = true; 
            }
        });
    }

    // 2. Blindly Follow the AI Light Grid
    applyLightGrid(lightGrid: string[][], cellW: number, cellH: number) {
        let steerX = 0; let steerY = 0;
        let whiteCount = 0; let redCount = 0;

        const myCol = Math.max(0, Math.min(19, Math.floor(this.x / cellW)));
        const myRow = Math.max(0, Math.min(19, Math.floor(this.y / cellH)));

        // Scan a 5x5 grid area around the particle for AI lights
        const searchRadius = 2; 
        const startX = Math.max(0, myCol - searchRadius);
        const endX = Math.min(19, myCol + searchRadius);
        const startY = Math.max(0, myRow - searchRadius);
        const endY = Math.min(19, myRow + searchRadius);

        for (let x = startX; x <= endX; x++) {
            for (let y = startY; y <= endY; y++) {
                const cellColor = lightGrid[x][y];
                if (cellColor === "OFF") continue;

                // Find the physical center of the glowing cell
                const cellCenterX = (x * cellW) + (cellW / 2);
                const cellCenterY = (y * cellH) + (cellH / 2);

                const dx = cellCenterX - this.x;
                const dy = cellCenterY - this.y;
                const dist = Math.hypot(dx, dy);

                if (dist > 0) {
                    if (cellColor === "WHITE") {
                        // GRAVITY: Pull towards white (stronger when closer)
                        const weight = 1 / dist;
                        steerX += (dx / dist) * weight;
                        steerY += (dy / dist) * weight;
                        whiteCount++;
                    } else if (cellColor === "RED") {
                        // ANTI-GRAVITY: Push away from red violently
                        const weight = 1 / dist;
                        steerX -= (dx / dist) * weight * 2.5; 
                        steerY -= (dy / dist) * weight * 2.5;
                        redCount++;
                    }
                }
            }
        }

        // Apply the AI's collective vector force
        if (whiteCount > 0 || redCount > 0) {
            const mag = Math.hypot(steerX, steerY);
            if (mag > 0) {
                steerX = (steerX / mag) * this.maxSpeed - this.vx;
                steerY = (steerY / mag) * this.maxSpeed - this.vy;

                const steerMag = Math.hypot(steerX, steerY);
                if (steerMag > this.maxForce) {
                    steerX = (steerX / steerMag) * this.maxForce;
                    steerY = (steerY / steerMag) * this.maxForce;
                }
                this.applyForce(steerX, steerY);
            }
            // Turn white if guided, orange if fleeing
            this.color = redCount > whiteCount ? '#ff8800' : '#ffffff';
        } else {
            // If they are in the dark, they slow down slightly and return to base color
            this.color = this.baseColor;
        }
    }

    // 3. Keep Hard Physical Walls
    resolveWalls(walls: Rect[]) {
        walls.forEach(w => {
            let testX = this.x; let testY = this.y;
            if (this.x < w.x) testX = w.x; else if (this.x > w.x + w.w) testX = w.x + w.w;
            if (this.y < w.y) testY = w.y; else if (this.y > w.y + w.h) testY = w.y + w.h;

            let distX = this.x - testX; let distY = this.y - testY;
            let distance = Math.sqrt((distX * distX) + (distY * distY));

            if (distance <= this.radius) {
                this.vx *= -1; this.vy *= -1;
                this.x += this.vx * 2; this.y += this.vy * 2;
            }
        });
    }

    // 4. Keep Crowd Crush physics
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