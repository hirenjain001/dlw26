import React, { useRef, useEffect, useState } from 'react';
import { Particle, type Rect } from '../Particle';
import { Link } from '../../../Link';
import { socket } from "../api/evacSocket";
import { getLightGrid, resetLightGrid } from "../state/lightGrid";

// 1. Replaced 'runway' with 'spawn'
type DrawMode = 'wall' | 'exit' | 'fire' | 'spawn' | null;

export const Simulation: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const requestRef = useRef<number>(0);
    const swarmRef = useRef<Particle[]>([]);

    const [drawMode, setDrawMode] = useState<DrawMode>(null);
    const [walls, setWalls] = useState<Rect[]>([]);
    const [exits, setExits] = useState<Rect[]>([]);
    const [fires, setFires] = useState<Rect[]>([]);
    const [crowdSize, setCrowdSize] = useState(0);

    const isDrawing = useRef(false);
    const startPos = useRef({ x: 0, y: 0 });
    const currentRect = useRef<Rect | null>(null);
    
    const aiLinkRef = useRef<Link | null>(null);
    const tickIntervalRef = useRef<number | null>(null);
    const [isAIDeployed, setIsAIDeployed] = useState(false);

    // NEW: The live ref needed so the AI ticking loop can see new fires
    const firesRef = useRef<Rect[]>([]);
    useEffect(() => {
        firesRef.current = fires;
    }, [fires]);

    useEffect(() => {
        return () => {
            if (tickIntervalRef.current) clearInterval(tickIntervalRef.current);
        };
    }, []);

    useEffect(() => {
        const resize = () => {
            if (canvasRef.current) {
                canvasRef.current.width = window.innerWidth;
                canvasRef.current.height = window.innerHeight;
            }
        };
        window.addEventListener('resize', resize);
        resize();
        return () => window.removeEventListener('resize', resize);
    }, []);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const render = () => {
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // === AI LIGHT OVERLAY (20x20) ===
            const lg = getLightGrid();
            const cellW = canvas.width / 20;
            const cellH = canvas.height / 20;

            for (let x = 0; x < 20; x++) {
                for (let y = 0; y < 20; y++) {
                    const c = lg[x][y];
                    if (c === "OFF") continue;

                    const px = x * cellW;
                    const py = y * cellH;

                    if (c === "WHITE") {
                        ctx.shadowBlur = 18;
                        ctx.shadowColor = "rgba(255,255,255,0.9)";
                        ctx.fillStyle = "rgba(255,255,255,0.28)";
                    } else {
                        ctx.shadowBlur = 12;
                        ctx.shadowColor = "rgba(255,0,0,0.9)";
                        ctx.fillStyle = "rgba(255,0,0,0.30)";
                    }

                    ctx.fillRect(px, py, cellW, cellH);
                    ctx.shadowBlur = 0;
                }
            }

            exits.forEach(e => {
                ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
                ctx.fillRect(e.x, e.y, e.w, e.h);
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2;
                ctx.strokeRect(e.x, e.y, e.w, e.h);
            });

            walls.forEach(w => {
                ctx.fillStyle = '#333333';
                ctx.fillRect(w.x, w.y, w.w, w.h);
            });

            fires.forEach(f => {
                ctx.fillStyle = 'rgba(255, 0, 0, 0.4)';
                ctx.fillRect(f.x, f.y, f.w, f.h);
                ctx.strokeStyle = '#ff0000';
                ctx.lineWidth = 4;
                ctx.strokeRect(f.x, f.y, f.w, f.h);
            });

            if (isDrawing.current && currentRect.current) {
                const r = currentRect.current;
                // Cyan dashed line for the spawn zone
                ctx.strokeStyle = drawMode === 'wall' ? '#aaaaaa' : 
                                  drawMode === 'exit' ? '#00ff00' : 
                                  drawMode === 'fire' ? '#ff0000' : '#00ffff'; 
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(r.x, r.y, r.w, r.h);
                ctx.setLineDash([]); 
            }

            // Execute Physics Loop
            swarmRef.current.forEach(particle => {
                particle.applyLightGrid(lg, cellW, cellH, exits);
                particle.resolveWalls(walls);          
                particle.separate(swarmRef.current);   
                particle.checkEscaped(exits);
                particle.keepInBounds(canvas.width, canvas.height); // Keeps them on screen
                
                particle.update();
                particle.draw(ctx);
            });

            swarmRef.current = swarmRef.current.filter(p => !p.escaped);
            
            const popCounter = document.getElementById('population-counter');
            if (popCounter) popCounter.innerText = swarmRef.current.length.toString();

            requestRef.current = requestAnimationFrame(render);
        };

        requestRef.current = requestAnimationFrame(render);
        return () => cancelAnimationFrame(requestRef.current);
    }, [walls, exits, fires, drawMode]);

    const getMousePos = (e: React.MouseEvent) => {
        const rect = canvasRef.current?.getBoundingClientRect();
        return { x: e.clientX - (rect?.left || 0), y: e.clientY - (rect?.top || 0) };
    };

    const handleMouseDown = (e: React.MouseEvent) => {
        if (!drawMode) return;
        isDrawing.current = true;
        startPos.current = getMousePos(e);
        currentRect.current = { x: startPos.current.x, y: startPos.current.y, w: 0, h: 0 };
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!isDrawing.current || !currentRect.current) return;
        const pos = getMousePos(e);
        currentRect.current = {
            x: Math.min(startPos.current.x, pos.x),
            y: Math.min(startPos.current.y, pos.y),
            w: Math.abs(pos.x - startPos.current.x),
            h: Math.abs(pos.y - startPos.current.y),
        };
    };

    const handleMouseUp = (e: React.MouseEvent) => {
        if (!isDrawing.current || !currentRect.current || !drawMode) return;
        
        if (currentRect.current.w > 5 && currentRect.current.h > 5) {
            
            // 2. THE CUSTOM SPAWN ZONE LOGIC
            if (drawMode === 'spawn') {
                // Calculate area to determine how many people to spawn (bigger box = more people)
                const area = currentRect.current.w * currentRect.current.h;
                const spawnCount = Math.min(250, Math.max(10, Math.floor(area / 800)));
                
                for (let i = 0; i < spawnCount; i++) {
                    const spawnX = currentRect.current.x + Math.random() * currentRect.current.w;
                    const spawnY = currentRect.current.y + Math.random() * currentRect.current.h;
                    swarmRef.current.push(new Particle(spawnX, spawnY));
                }
                setCrowdSize(swarmRef.current.length);
            } 
            else if (drawMode === 'wall') setWalls([...walls, currentRect.current]);
            else if (drawMode === 'exit') setExits([...exits, currentRect.current]);
            else if (drawMode === 'fire') setFires([...fires, currentRect.current]);
        }
        
        isDrawing.current = false;
        currentRect.current = null;
    };

    const clearSim = () => {
        swarmRef.current = [];
        setWalls([]); setExits([]); setFires([]); setCrowdSize(0);
    };

    const toggleAI = () => {
        if (isAIDeployed) {
            if (tickIntervalRef.current) clearInterval(tickIntervalRef.current);
                tickIntervalRef.current = null;

            socket.disconnect();
            resetLightGrid();

            setIsAIDeployed(false);
            console.log("AI PIPELINE OFFLINE");
            return;
        }

        if (!canvasRef.current) return;

        socket.connect();

        const link = new Link(canvasRef.current.width, canvasRef.current.height);
        aiLinkRef.current = link;

        const initPayload = link.generateInit(walls, exits);
        console.log("SENDING INIT", initPayload);
        socket.sendInit(initPayload);

        tickIntervalRef.current = window.setInterval(() => {
            // FED firesRef.current SO THE LOOP SEES NEW FIRES
            const tickPayload = aiLinkRef.current?.generateTick(swarmRef.current, firesRef.current);
            if (!tickPayload) return;

            if (
            tickPayload.crowd_delta.length > 0 ||
            tickPayload.fire_off.length > 0 ||
            tickPayload.fire_on.length > 0
            ) {
                socket.sendTick(tickPayload);
            }
        }, 200);
        
        setIsAIDeployed(true);
    };

    return (
        <div className="relative w-screen h-screen overflow-hidden bg-black">
            <canvas
                ref={canvasRef}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                className={`absolute top-0 left-0 ${drawMode ? 'cursor-crosshair' : 'cursor-default'}`}
            />

            <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 flex items-center space-x-4 bg-gray-900 border border-gray-700 p-4 rounded-xl shadow-2xl font-mono text-sm z-10">
                <div className="text-gray-400 mr-4">
                    POPULATION: <span id="population-counter" className="text-cyan-400 font-bold">{crowdSize}</span>
                </div>
                
                <button 
                    onClick={() => setDrawMode(drawMode === 'wall' ? null : 'wall')}
                    className={`px-4 py-2 rounded transition-colors ${drawMode === 'wall' ? 'bg-gray-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}>
                    [+] WALL
                </button>
                <button 
                    onClick={() => setDrawMode(drawMode === 'exit' ? null : 'exit')}
                    className={`px-4 py-2 rounded transition-colors ${drawMode === 'exit' ? 'bg-green-700 text-white' : 'bg-gray-800 text-green-400 hover:bg-gray-700'}`}>
                    [+] EXIT
                </button>
                <button 
                    onClick={() => setDrawMode(drawMode === 'fire' ? null : 'fire')}
                    className={`px-4 py-2 rounded transition-colors ${drawMode === 'fire' ? 'bg-red-700 text-white' : 'bg-gray-800 text-red-400 hover:bg-gray-700'}`}>
                    [+] FIRE
                </button>
                
                {/* 3. REPLACED RUNWAY AND SPAWN 100 WITH THIS TARGETED SPAWN TOOL */}
                <button 
                    onClick={() => setDrawMode(drawMode === 'spawn' ? null : 'spawn')}
                    className={`px-4 py-2 rounded transition-colors ${drawMode === 'spawn' ? 'bg-cyan-700 text-white' : 'bg-gray-800 text-cyan-400 hover:bg-gray-700'}`}>
                    [+] SPAWN ZONE
                </button>

                <div className="h-8 w-px bg-gray-700 mx-2"></div>

                <button 
                    onClick={clearSim}
                    className="px-4 py-2 bg-red-900/50 text-red-400 rounded hover:bg-red-900 transition-colors border border-red-900">
                    NUKE
                </button>
                <div className="h-8 w-px bg-gray-700 mx-2"></div>
                <button 
                    onClick={toggleAI}
                    className={`px-4 py-2 font-bold rounded transition-colors ${isAIDeployed ? 'bg-purple-600 text-white animate-pulse' : 'bg-gray-800 text-purple-400 border border-purple-900 hover:bg-gray-700'}`}>
                    {isAIDeployed ? 'AI ACTIVE' : 'DEPLOY AI'}
                </button>
            </div>
            
            {drawMode && (
                <div className="absolute top-6 left-1/2 transform -translate-x-1/2 bg-black/80 border border-gray-700 px-6 py-2 rounded text-white font-mono text-sm animate-pulse">
                    Click and drag on the canvas to draw a {drawMode.toUpperCase()} zone
                </div>
            )}
        </div>
    );
};