import asyncio
import json
import websockets

WS_URL = "ws://127.0.0.1:8000/ws"

def build_border_walls():
    walls = []
    for i in range(20):
        walls.append([i, 0]); walls.append([i, 19])
    for j in range(20):
        walls.append([0, j]); walls.append([19, j])
    return walls

async def contract_test():
    print("Connecting to backend:", WS_URL)

    async with websockets.connect(WS_URL) as ws:
        init_msg = {
            "type": "init",
            "session_id": "contract_test",
            "grid": {"w": 20, "h": 20},
            "layout": {
                "walls": build_border_walls(),
                "exits": [[1, 1], [18, 18]],
            },
            "opts": {"max_exits": 3},
        }

        await ws.send(json.dumps(init_msg))
        resp = json.loads(await ws.recv())

        # === INIT CONTRACT ===
        assert resp.get("type") == "ack", f"Expected type=ack, got {resp}"
        assert resp.get("message") == "init_ok", f"Expected message=init_ok, got {resp}"
        assert resp.get("session_id") == "contract_test", f"Session id mismatch: {resp}"
        print("INIT OK")

        # === TICK CONTRACT ===
        for t in range(5):
            tick_msg = {
                "type": "tick",
                "session_id": "contract_test",
                "t": t,
                "ts_ms": 0,
                "crowd_delta": [[10, 9, 5], [10, 8, 3]],
                "fire_on": [[10, 10]] if t == 0 else [],
                "fire_off": [],
            }

            await ws.send(json.dumps(tick_msg))
            resp = json.loads(await ws.recv())

            assert resp.get("type") == "cmd", f"Expected type=cmd, got {resp}"
            assert resp.get("t") == t, f"Tick id mismatch: expected {t}, got {resp.get('t')}"
            assert "lights_delta" in resp, f"Missing lights_delta: {resp}"
            assert isinstance(resp["lights_delta"], list), f"lights_delta must be list: {type(resp['lights_delta'])}"

            # validate delta entries shape
            for item in resp["lights_delta"][:10]:
                assert isinstance(item, list) and len(item) == 3, f"Bad delta item: {item}"
                x, y, s = item
                assert 0 <= int(x) < 20 and 0 <= int(y) < 20, f"Bad coords: {item}"
                assert s in ("WHITE", "RED", "OFF"), f"Bad state: {item}"

            print(f"TICK {t} OK — delta_len={len(resp['lights_delta'])}")

        print("\nCONTRACT TEST PASSED ✅")

if __name__ == "__main__":
    asyncio.run(contract_test())