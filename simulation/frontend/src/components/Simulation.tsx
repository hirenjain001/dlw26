import React, { useRef, useEffect, useState } from 'react';
import { Particle, type Rect } from '../Particle';
import {Link} from '../../../Link';

type DrawMode = 'wall' | 'exit' | 'fire' | 'runway' | null;

export const Simulation: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const requestRef = useRef<number>(0);
    const swarmRef = useRef<Particle[]>([]);

    const [drawMode, setDrawMode] = useState<DrawMode>(null);
    const [walls, setWalls] = useState<Rect[]>([]);
    const [exits, setExits] = useState<Rect[]>([]);
    const [fires, setFires] = useState<Rect[]>([]);
    const [runways, setRunways] = useState<Rect[]>([]);
    const [crowdSize, setCrowdSize] = useState(0);

    const isDrawing = useRef(false);
    const startPos = useRef({ x: 0, y: 0 });
    const currentRect = useRef<Rect | null>(null);
    
    const aiLinkRef = useRef<Link | null>(null);
    const tickIntervalRef = useRef<number | null>(null);
    const [isAIDeployed, setIsAIDeployed] = useState(false);

    useEffect(() => {
        return () => {
            if (tickIntervalRef.current) clearInterval(tickIntervalRef.current);
        };
    }, []);

    // 1. Fullscreen Canvas Sizing
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

    // 2. The Main Render Engine
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const render = () => {
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

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

            runways.forEach(r => {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
                ctx.fillRect(r.x, r.y, r.w, r.h);
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
                ctx.lineWidth = 2;
                ctx.strokeRect(r.x, r.y, r.w, r.h);
                
                // Draw the Vector Arrow
                if (r.dirX !== undefined && r.dirY !== undefined) {
                    const cx = r.x + r.w / 2;
                    const cy = r.y + r.h / 2;
                    const length = Math.min(r.w, r.h) / 3;
                    
                    ctx.beginPath();
                    ctx.moveTo(cx - r.dirX * length, cy - r.dirY * length);
                    ctx.lineTo(cx + r.dirX * length, cy + r.dirY * length);
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
                    ctx.lineWidth = 4;
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.arc(cx + r.dirX * length, cy + r.dirY * length, 6, 0, Math.PI * 2);
                    ctx.fillStyle = '#ffffff';
                    ctx.fill();
                }
            });

            if (isDrawing.current && currentRect.current) {
                const r = currentRect.current;
                ctx.strokeStyle = drawMode === 'wall' ? '#aaaaaa' : 
                                  drawMode === 'exit' ? '#00ff00' : 
                                  drawMode === 'fire' ? '#ff0000' : '#ffffff';
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(r.x, r.y, r.w, r.h);
                ctx.setLineDash([]); 
            }

            // Execute Physics Loop
            swarmRef.current.forEach(particle => {
                particle.applyPhototaxis(fires);       
                particle.applyRunway(runways);         
                particle.resolveWalls(walls);          
                particle.separate(swarmRef.current);   
                particle.seekClosestExit(exits);       
                particle.update();
                particle.draw(ctx);
            });

            // The Garbage Collector
            swarmRef.current = swarmRef.current.filter(p => !p.escaped);
            
            // Live DOM update (Bypasses React State for 60fps performance)
            const popCounter = document.getElementById('population-counter');
            if (popCounter) popCounter.innerText = swarmRef.current.length.toString();

            requestRef.current = requestAnimationFrame(render);
        };

        requestRef.current = requestAnimationFrame(render);
        return () => cancelAnimationFrame(requestRef.current);
    }, [walls, exits, fires, runways, drawMode]);

    // 3. Mouse Handlers
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
            const pos = getMousePos(e);
            const dx = pos.x - startPos.current.x;
            const dy = pos.y - startPos.current.y;
            const mag = Math.hypot(dx, dy);
            
            const dirX = mag > 0 ? dx / mag : 0;
            const dirY = mag > 0 ? dy / mag : 0;
            
            const directedRect: Rect = { ...currentRect.current, dirX, dirY };

            if (drawMode === 'wall') setWalls([...walls, currentRect.current]);
            if (drawMode === 'exit') setExits([...exits, currentRect.current]);
            if (drawMode === 'fire') setFires([...fires, currentRect.current]);
            if (drawMode === 'runway') setRunways([...runways, directedRect]);
        }
        isDrawing.current = false;
        currentRect.current = null;
    };

    // 4. UI Controls
    const spawnPeople = (amount: number) => {
        for (let i = 0; i < amount; i++) {
            const spawnX = window.innerWidth * 0.1 + Math.random() * 200;
            const spawnY = window.innerHeight * 0.5 - 150 + Math.random() * 300;
            swarmRef.current.push(new Particle(spawnX, spawnY));
        }
        setCrowdSize(swarmRef.current.length);
    };

    const clearSim = () => {
        swarmRef.current = [];
        setWalls([]); setExits([]); setFires([]); setRunways([]); setCrowdSize(0);
    };

    const toggleAI = () => {
        if (isAIDeployed) {
            if (tickIntervalRef.current) clearInterval(tickIntervalRef.current);
            setIsAIDeployed(false);
            console.log("AI PIPELINE OFFLINE");
            return;
        }

        if(!canvasRef.current) return;

        // Instantiate custom math class
        const link = new Link(canvasRef.current.width, canvasRef.current.height);
        aiLinkRef.current = link;

        // Generate and output the INIT payload
        const initPayload = link.generateInit(walls, exits);
        console.log("SENDING INIT", initPayload);

        // FIXME: @txryn - plug WebSocket SEND here -> ws.send(JSON.stringify(initPayload));

        // Start the 200ms TICK loop
        tickIntervalRef.current = window.setInterval(() => {
            const tickPayload = aiLinkRef.current?.generateTick(swarmRef.current, fires);

            if (tickPayload) {
                // Only log changes
                if (tickPayload.crowd_delta.length > 0 ||
                    tickPayload.fire_off.length > 0 ||
                    tickPayload.fire_on.length > 0) {
                        console.log("SENDING TICK DATA");

                        // FIXME: @txryn - plug WebSocket SEND here -> ws.send(JSON.stringify(initPayload));
                    }
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
                <button 
                    onClick={() => setDrawMode(drawMode === 'runway' ? null : 'runway')}
                    className={`px-4 py-2 rounded transition-colors ${drawMode === 'runway' ? 'bg-gray-200 text-black font-bold' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'}`}>
                    [+] RUNWAY
                </button>

                <div className="h-8 w-px bg-gray-700 mx-2"></div>

                <button 
                    onClick={() => spawnPeople(100)}
                    className="px-4 py-2 bg-cyan-900 text-cyan-300 rounded hover:bg-cyan-800 transition-colors">
                    SPAWN 100
                </button>
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
                    Click and drag on the canvas to draw a {drawMode.toUpperCase()}
                </div>
            )}
        </div>
    );
};