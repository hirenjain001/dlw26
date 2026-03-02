import type { InitPayload, TickPayload, ServerMsg } from "./protocol";
import { applyLightsDelta } from "../state/lightGrid";

type Status = "idle" | "connecting" | "open" | "closed" | "error";

class EvacSocket {
  private ws: WebSocket | null = null;
  private status: Status = "idle";

  private url(): string {
    // Use Vite env if present, else default to local backend
    const envUrl = (import.meta as any).env?.VITE_WS_URL as string | undefined;
    return envUrl ?? "ws://127.0.0.1:8000/ws";
  }

  public connect(): void {
    if (this.ws && (this.status === "open" || this.status === "connecting")) return;

    this.status = "connecting";
    this.ws = new WebSocket(this.url());

    this.ws.onopen = () => {
      this.status = "open";
      console.log("[ws] open", this.url());
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
          // core behaviour: apply lights delta to grid store
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
    if (!this.ws) return;
    this.ws.close();
    this.ws = null;
    this.status = "closed";
  }

  private send(obj: ServerMsg | InitPayload | TickPayload): void {
    if (!this.ws || this.status !== "open") return;
    this.ws.send(JSON.stringify(obj));
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