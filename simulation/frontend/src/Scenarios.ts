import { type Rect } from './Particle';

export interface Scenario {
    name: string;
    walls: Rect[];
    exits: Rect[];
    fires: Rect[];
}

export const getScenarios = (w: number, h: number): Scenario[] => {
    return [
        {
            name: "1. Bottleneck",
            walls: [
                // Top wall pushing down
                { x: w * 0.4, y: 0, w: 40, h: h * 0.35 },
                // Bottom wall pushing up, leaving a tight gap in the middle
                { x: w * 0.4, y: h * 0.65, w: 40, h: h * 0.35 }
            ],
            exits: [{ x: w * 0.9, y: h * 0.4, w: 80, h: h * 0.2 }],
            fires: []
        },
        {
            name: "2. Blocked Primary Exit",
            walls: [
                // A central dividing wall
                { x: w * 0.5, y: h * 0.2, w: 40, h: h * 0.6 }
            ],
            exits: [
                { x: 20, y: h * 0.4, w: 40, h: h * 0.2 },       // Left Exit
                { x: w - 60, y: h * 0.4, w: 40, h: h * 0.2 }    // Right Exit
            ],
            fires: [
                // A massive fire sitting directly in front of the Right Exit
                { x: w - 250, y: h * 0.3, w: 150, h: h * 0.4 } 
            ]
        },
        {
            name: "3. Office Maze",
            walls: [
                { x: w * 0.2, y: h * 0.2, w: w * 0.4, h: 40 },
                { x: w * 0.4, y: h * 0.5, w: w * 0.4, h: 40 },
                { x: w * 0.2, y: h * 0.8, w: w * 0.4, h: 40 }
            ],
            exits: [{ x: w * 0.9, y: h * 0.7, w: 60, h: 150 }],
            fires: []
        }
    ];
};