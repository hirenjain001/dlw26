export interface Rect { x: number; y: number; w: number; h: number; dirX?: number; dirY?: number; }

export class Particle {
    x: number; y: number;
    vx: number; vy: number;
    ax: number; ay: number;
    maxSpeed: number; maxForce: number; radius: number;
    color: string; baseColor: string; crushColor: string;
    escaped: boolean;
    readonly compassPower: number = 0.12;

    constructor(x: number, y: number) {
        this.x = x; this.y = y;
        this.vx = 0; this.vy = 0; this.ax = 0; this.ay = 0;
        this.maxSpeed = 3 + Math.random() * 2; 
        this.maxForce = 0.08; 
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

    applyLightGrid(lightGrid: string[][], distMap: number[][], cellW: number, cellH: number, exits: Rect[]) {
        let steerX = 0; let steerY = 0;
        let redCount = 0;

        const myCol = Math.max(0, Math.min(distMap.length - 1, Math.floor(this.x / cellW)));
        const myRow = Math.max(0, Math.min(distMap.length - 1, Math.floor(this.y / cellH)));

        const onWhite = lightGrid[myCol] && lightGrid[myCol][myRow] === "WHITE";

        // 1. FLOW FIELD STEERING
        // We look for the neighboring WHITE cell with the lowest distance score
        let bestDist = distMap[myCol] && distMap[myCol][myRow] !== undefined ? distMap[myCol][myRow] : Infinity;
        let targetX = 0; let targetY = 0;
        let foundPath = false;

        const searchRadius = 2; 
        const startX = Math.max(0, myCol - searchRadius);
        const endX = Math.min(distMap.length - 1, myCol + searchRadius);
        const startY = Math.max(0, myRow - searchRadius);
        const endY = Math.min(distMap.length - 1, myRow + searchRadius);

        for (let x = startX; x <= endX; x++) {
            for (let y = startY; y <= endY; y++) {
                const cellColor = lightGrid[x][y];
                if (cellColor === "OFF") continue;

                const cellCenterX = (x * cellW) + (cellW / 2);
                const cellCenterY = (y * cellH) + (cellH / 2);

                const dx = cellCenterX - this.x;
                const dy = cellCenterY - this.y;
                const distToCell = Math.hypot(dx, dy);

                if (distToCell > 0) {
                    if (cellColor === "RED") {
                        const weight = 1 / distToCell;
                        steerX -= (dx / distToCell) * weight * 3.5; 
                        steerY -= (dy / distToCell) * weight * 3.5;
                        redCount++;
                    } else if (cellColor === "WHITE") {
                        // THE FIX: Is this cell further "downhill" toward the exit?
                        if (distMap[x][y] < bestDist) {
                            bestDist = distMap[x][y];
                            targetX = cellCenterX;
                            targetY = cellCenterY;
                            foundPath = true;
                        }
                    }
                }
            }
        }

        // Apply strong steering down the correct path
        if (foundPath) {
            const dx = targetX - this.x;
            const dy = targetY - this.y;
            const dist = Math.hypot(dx, dy);
            if (dist > 0) {
                steerX += (dx / dist) * 1.5; 
                steerY += (dy / dist) * 1.5;
            }
        }

        // 2. THE COMPASS (Fallback)
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

        if (foundPath || redCount > 0) {
            // Compass is heavily nerfed (0.10) so they trust the Flow Field to route around obstacles
            steerX += compassX * 0.10;
            steerY += compassY * 0.10;

            // Brownian noise to prevent perfectly straight lines and simulate panic
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
            this.color = redCount > 0 ? '#ff8800' : '#ffffff';
        } else {
            this.applyForce((Math.random() - 0.5) * 0.05, (Math.random() - 0.5) * 0.05);
            this.color = this.baseColor;
        }
    }

    resolveWalls(walls: Rect[]) {
        let repulseX = 0; let repulseY = 0;

        walls.forEach(w => {
            let testX = this.x; let testY = this.y;
            if (this.x < w.x) testX = w.x; else if (this.x > w.x + w.w) testX = w.x + w.w;
            if (this.y < w.y) testY = w.y; else if (this.y > w.y + w.h) testY = w.y + w.h;

            let distX = this.x - testX; let distY = this.y - testY;
            let distance = Math.hypot(distX, distY);

            if (distance > 0) {
                // 1. The Ice Wall: Hard collision without losing momentum
                if (distance <= this.radius) {
                    const overlap = this.radius - distance;
                    this.x += (distX / distance) * overlap;
                    this.y += (distY / distance) * overlap;
                    // REMOVED: vx *= 0.5 friction. They will now slide instantly.
                }

                // 2. The Forcefield: Push them away before they get stuck
                const forcefieldRadius = 20; // Projects 20 pixels off the wall
                if (distance < forcefieldRadius) {
                    const pushWeight = (forcefieldRadius - distance) / forcefieldRadius;
                    repulseX += (distX / distance) * pushWeight;
                    repulseY += (distY / distance) * pushWeight;
                }
            }
        });

        // Apply the wall's repulsive steering force
        if (repulseX !== 0 || repulseY !== 0) {
            const mag = Math.hypot(repulseX, repulseY);
            repulseX = (repulseX / mag) * this.maxSpeed - this.vx;
            repulseY = (repulseY / mag) * this.maxSpeed - this.vy;
            
            // Make the wall push significantly stronger than the compass pull
            const repulseForce = this.maxForce * 0.8; 
            const forceMag = Math.hypot(repulseX, repulseY);
            if (forceMag > repulseForce) {
                repulseX = (repulseX / forceMag) * repulseForce;
                repulseY = (repulseY / forceMag) * repulseForce;
            }
            this.applyForce(repulseX, repulseY);
        }
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
                const repulseForce = this.maxForce * 1.2; // changed from 1.5 to 1.2
                const steerMag = Math.sqrt(steerX * steerX + steerY * steerY);
                if (steerMag > repulseForce) {
                    steerX = (steerX / steerMag) * repulseForce;
                    steerY = (steerY / steerMag) * repulseForce;
                }
            }
        }
        this.applyForce(steerX, steerY);
    }

    keepInBounds(width: number, height: number) {
        if (this.x - this.radius < 0) {
            this.x = this.radius;
            this.vx *= 0.0; // Bounce off the left wall
        } else if (this.x + this.radius > width) {
            this.x = width - this.radius;
            this.vx *= 0.0; // Bounce off the right wall
        }

        if (this.y - this.radius < 0) {
            this.y = this.radius;
            this.vy *= 0.0; // Bounce off the ceiling --> changed from -0.5 to 0 for all 
        } else if (this.y + this.radius > height) {
            this.y = height - this.radius;
            this.vy *= 0.0; // Bounce off the floor
        }
    }
    // change made here 
    update() {
        this.vx += this.ax;
        this.vy += this.ay;

        const speed = Math.hypot(this.vx, this.vy);
        const maxStepSpeed = 2.2;

        if (speed > maxStepSpeed) {
            this.vx = (this.vx / speed) * maxStepSpeed;
            this.vy = (this.vy / speed) * maxStepSpeed;
        }

        this.x += this.vx;
        this.y += this.vy;

        this.ax = 0;
        this.ay = 0;
    }

    draw(ctx: CanvasRenderingContext2D) {
        // Change 3: Draw pressure halo if crowding -> Crowd Pressure Visualisation
        if (this.color === this.crushColor) {
            ctx.fillStyle = "rgba(255,80,0,0.15)";
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * 5, 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();
    }
}