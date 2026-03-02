export type XY = [number, number];
export type XYCount = [number, number, number];
export type LightColor = "WHITE" | "RED" | "OFF";
export type LightDelta = [number, number, LightColor];

export interface InitPayload {
  type: "init";
  session_id: string;
  grid: { w: number; h: number };
  layout: {
    walls: XY[];
    exits: XY[];
  };
  opts: { max_exits: number };
}

export interface TickPayload {
  type: "tick";
  session_id: string;
  t: number;
  ts_ms: number;
  crowd_delta: XYCount[];
  fire_on: XY[];
  fire_off: XY[];
}

export interface AckMsg {
  type: "ack";
  message: string;
  session_id: string;
}

export interface ErrorMsg {
  type: "error";
  message: string;
}

export interface CmdMsg {
  type: "cmd";
  t: number;
  ts_ms: number;
  ttl_ms: number;
  policy: { action: number; mode: "AUTO_NEAREST" | "GUIDE_EXIT" };
  lights_delta: LightDelta[];
  counts: { n_white: number; n_red: number };
}

export type ServerMsg = AckMsg | ErrorMsg | CmdMsg;