import asyncio
import json
import websockets

WS_URL = "ws://127.0.0.1:8000/ws"


def build_border_walls():
    walls = []
    for i in range(20):
        walls.append([i, 0])
        walls.append([i, 19])
    for j in range(20):
        walls.append([0, j])
        walls.append([19, j])
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
            assert resp.get("ttl_ms") == 500, f"Expected ttl_ms=500, got {resp}"
            assert "ts_ms" in resp, f"Missing ts_ms: {resp}"

            assert "policy" in resp, f"Missing policy: {resp}"
            assert isinstance(resp["policy"], dict), f"policy must be dict: {resp}"
            assert "action" in resp["policy"], f"Missing policy.action: {resp}"
            assert isinstance(resp["policy"]["action"], int), f"policy.action must be int: {resp}"
            assert resp["policy"].get("mode") in ("AUTO_NEAREST", "GUIDE_EXIT"), f"Bad policy.mode: {resp}"

            assert "lights_delta" in resp, f"Missing lights_delta: {resp}"
            assert isinstance(resp["lights_delta"], list), f"lights_delta must be list: {type(resp['lights_delta'])}"

            for item in resp["lights_delta"][:10]:
                assert isinstance(item, list) and len(item) == 3, f"Bad delta item: {item}"
                x, y, s = item
                assert 0 <= int(x) < 20 and 0 <= int(y) < 20, f"Bad coords: {item}"
                assert s in ("WHITE", "RED", "OFF"), f"Bad state: {item}"

            assert "counts" in resp, f"Missing counts: {resp}"
            assert isinstance(resp["counts"], dict), f"counts must be dict: {resp}"
            assert "n_white" in resp["counts"], f"Missing n_white: {resp}"
            assert "n_red" in resp["counts"], f"Missing n_red: {resp}"
            assert "n_congestion_red" in resp["counts"], f"Missing n_congestion_red: {resp}"

            print(
                f"TICK {t} OK — delta_len={len(resp['lights_delta'])}, "
                f"mode={resp['policy']['mode']}, counts={resp['counts']}"
            )

        # === STALE TICK CONTRACT ===
        stale_tick_msg = {
            "type": "tick",
            "session_id": "contract_test",
            "t": 4,  
            "crowd_delta": [[10, 9, 5]],
            "fire_on": [],
            "fire_off": [],
        }

        await ws.send(json.dumps(stale_tick_msg))
        resp = json.loads(await ws.recv())

        assert resp.get("type") == "error", f"Expected stale tick to fail, got {resp}"
        assert "stale tick" in resp.get("message", ""), f"Expected stale tick error, got {resp}"
        print("STALE TICK OK")

        print("\nCONTRACT TEST PASSED ✅")


if __name__ == "__main__":
    asyncio.run(contract_test())