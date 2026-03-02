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

    seekClosestExit(exits: Rect[]) {
        if (exits.length === 0) return; 
        
        let closest = exits[0];
        let recordDist = Infinity;

        exits.forEach(exit => {
            const centerX = exit.x + exit.w / 2;
            const centerY = exit.y + exit.h / 2;
            const d = Math.hypot(centerX - this.x, centerY - this.y);
            if (d < recordDist) {
                recordDist = d;
                closest = exit;
            }
        });

        // Despawn Mechanic
        if (recordDist < Math.max(closest.w, closest.h) / 2) {
            this.escaped = true; 
            return;
        }

        const targetX = closest.x + closest.w / 2;
        const targetY = closest.y + closest.h / 2;
        let desiredX = targetX - this.x;
        let desiredY = targetY - this.y;
        const distance = Math.sqrt(desiredX * desiredX + desiredY * desiredY);
        
        desiredX = (desiredX / distance) * this.maxSpeed;
        desiredY = (desiredY / distance) * this.maxSpeed;
        
        let steerX = desiredX - this.vx;
        let steerY = desiredY - this.vy;
        
        const steerMag = Math.sqrt(steerX * steerX + steerY * steerY);
        if (steerMag > this.maxForce) {
            steerX = (steerX / steerMag) * this.maxForce;
            steerY = (steerY / steerMag) * this.maxForce;
        }
        this.applyForce(steerX, steerY);
    }

    applyPhototaxis(fires: Rect[]) {
        let steerX = 0; let steerY = 0; let isFleeing = false;

        fires.forEach(fire => {
            const centerX = fire.x + fire.w / 2;
            const centerY = fire.y + fire.h / 2;
            const dist = Math.hypot(this.x - centerX, this.y - centerY);
            const fearRadius = Math.max(fire.w, fire.h) * 1.2;

            if (dist > 0 && dist < fearRadius) {
                isFleeing = true;
                const fleeWeight = fearRadius / dist; 
                steerX += ((this.x - centerX) / dist) * fleeWeight;
                steerY += ((this.y - centerY) / dist) * fleeWeight;
            }
        });

        if (isFleeing) {
            const mag = Math.sqrt(steerX * steerX + steerY * steerY);
            steerX = (steerX / mag) * this.maxSpeed;
            steerY = (steerY / mag) * this.maxSpeed;
            steerX -= this.vx; steerY -= this.vy;

            const fleeForce = this.maxForce * 3; 
            const steerMag = Math.sqrt(steerX * steerX + steerY * steerY);
            if (steerMag > fleeForce) {
                steerX = (steerX / steerMag) * fleeForce;
                steerY = (steerY / steerMag) * fleeForce;
            }
            this.applyForce(steerX, steerY);
            this.color = '#ff8800'; 
        }
    }

    applyRunway(runways: Rect[]) {
        let steerX = 0; let steerY = 0; let isAttracted = false;
        let insideRunway = false;
        let activeRunway: Rect | null = null;

        // FIX: Using a standard for-loop instead of .forEach so TypeScript 
        // can track the variable assignment perfectly without getting confused by closures.
        for (let i = 0; i < runways.length; i++) {
            const r = runways[i];
            if (this.x > r.x && this.x < r.x + r.w && this.y > r.y && this.y < r.y + r.h) {
                insideRunway = true;
                activeRunway = r;
            } else {
                const centerX = r.x + r.w / 2;
                const centerY = r.y + r.h / 2;
                const dist = Math.hypot(this.x - centerX, this.y - centerY);
                const attractRadius = Math.max(r.w, r.h) * 1.5;

                if (dist < attractRadius) {
                    isAttracted = true;
                    steerX += (centerX - this.x) / dist;
                    steerY += (centerY - this.y) / dist;
                }
            }
        }

        if (insideRunway && activeRunway) {
            this.maxSpeed = 5.0; 
            this.color = '#ffffff';

            // Directed Vector Alignment
            if (activeRunway.dirX !== undefined && activeRunway.dirY !== undefined) {
                let desiredX = activeRunway.dirX * this.maxSpeed;
                let desiredY = activeRunway.dirY * this.maxSpeed;
                
                let steerX = desiredX - this.vx;
                let steerY = desiredY - this.vy;
                
                this.applyForce(steerX * 0.8, steerY * 0.8);
            }
        } else {
            this.maxSpeed = 2 + Math.random() * 1.5; 
        }

        if (isAttracted && !insideRunway) {
            const mag = Math.hypot(steerX, steerY);
            steerX = (steerX / mag) * this.maxSpeed;
            steerY = (steerY / mag) * this.maxSpeed;
            steerX -= this.vx; steerY -= this.vy;

            const attractForce = this.maxForce * 1.2; 
            const steerMag = Math.hypot(steerX, steerY);
            if (steerMag > attractForce) {
                steerX = (steerX / steerMag) * attractForce;
                steerY = (steerY / steerMag) * attractForce;
            }
            this.applyForce(steerX, steerY);
            
            if (this.color === this.baseColor) {
                this.color = '#aaddff'; 
            }
        }
    }

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