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
                { x: w * 0.1, y: h * 0.24, w: 90, h: 12 },
                { x: w * 0.2, y: h * 0.83, w: 80, h: 13 },
                { x: w * 0.3, y: h * 0.72, w: 82, h: 42 },
                { x: w * 0.4, y: h * 0.65, w: 87, h: 12 },
                { x: w * 0.5, y: h * 0.52, w: 45, h: 34 },
                { x: w * 0.6, y: h * 0.12, w: 98, h: 54 },
                { x: w * 0.7, y: h * 0.37, w: 29, h: 45 },
                { x: w * 0.8, y: h * 0.88, w: 54, h: 56 },
                { x: w * 0.9, y: h * 0.36, w: 77, h: 23 },
                { x: w * 0.83, y: h * 0.45, w: 65, h: 44 }
            ],
            exits: [{ x: w * 0.98, y: h * 0.1, w: 80, h: 80 }],
            fires: [{ x: w * 0.34, y: h * 0.12, w: 23, h: 12 }]
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