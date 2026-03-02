import type { InitPayload, TickPayload, ServerMsg } from "./protocol";
import { applyLightsDelta } from "../state/lightGrid";

type Status = "idle" | "connecting" | "open" | "closed" | "error";
type ClientMsg = InitPayload | TickPayload;

class EvacSocket {
  private ws: WebSocket | null = null;
  private status: Status = "idle";
  private outbox: ClientMsg[] = [];

  private url(): string {
    const envUrl = (import.meta as any).env?.VITE_WS_URL as string | undefined;
    return envUrl ?? "ws://127.0.0.1:8000/ws";
  }

  public connect(): void {
    if (this.ws && (this.status === "open" || this.status === "connecting")) return;

    console.log("[ws] connecting", this.url());
    this.status = "connecting";
    this.ws = new WebSocket(this.url());

    this.ws.onopen = () => {
      this.status = "open";
      console.log("[ws] open", this.url());

      // flush queued init/ticks
      for (const msg of this.outbox) this.ws!.send(JSON.stringify(msg));
      this.outbox = [];
    };

    this.ws.onclose = () => {
      this.status = "closed";
      console.log("[ws] closed");
    };

    this.ws.onerror = () => {
      this.status = "error";
      console.log("[ws] error");
    };

    this.ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data) as ServerMsg;

        if (msg.type === "ack") {
          console.log("[ws] ack", msg);
          return;
        }
        if (msg.type === "error") {
          console.error("[ws] backend error:", msg.message);
          return;
        }
        if (msg.type === "cmd") {
          console.log("[ws] cmd", msg.counts, msg.policy, "delta:", msg.lights_delta.length);
          applyLightsDelta(msg.lights_delta);
          return;
        }

        console.warn("[ws] unknown msg", msg);
      } catch (e) {
        console.error("[ws] bad JSON", e);
      }
    };
  }

  public disconnect(): void {
    this.outbox = [];
    if (!this.ws) return;
    this.ws.close();
    this.ws = null;
    this.status = "closed";
  }

  private send(obj: ClientMsg): void {
    if (this.ws && this.status === "open") {
      this.ws.send(JSON.stringify(obj));
      return;
    }
    // queue until open
    this.outbox.push(obj);
  }

  public sendInit(payload: InitPayload): void {
    this.send(payload);
  }

  public sendTick(payload: TickPayload): void {
    this.send(payload);
  }

  public isOpen(): boolean {
    return this.status === "open";
  }
}

export const socket = new EvacSocket();