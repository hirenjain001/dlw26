import argparse
import asyncio
import json
import websockets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=20, help="Grid width for init/tick test")
    parser.add_argument("--height", type=int, default=20, help="Grid height for init/tick test")
    parser.add_argument("--ws_url", type=str, default="ws://127.0.0.1:8000/ws", help="WebSocket URL")
    return parser.parse_args()


def build_border_walls(width: int, height: int):
    walls = []
    for x in range(width):
        walls.append([x, 0])
        walls.append([x, height - 1])
    for y in range(height):
        walls.append([0, y])
        walls.append([width - 1, y])
    return walls


async def contract_test(ws_url: str, width: int, height: int):
    print("Connecting to backend:", ws_url)
    print(f"Testing grid size: {width}x{height}")

    center_x = width // 2
    center_y = height // 2

    exit_a = [1, 1]
    exit_b = [width - 2, height - 2]

    crowd_cells = [
        [center_x, max(1, center_y - 1), 5],
        [center_x, max(1, center_y - 2), 3],
    ]
    fire_cell = [center_x, center_y]

    async with websockets.connect(ws_url) as ws:
        init_msg = {
            "type": "init",
            "session_id": "contract_test",
            "grid": {"w": width, "h": height},
            "layout": {
                "walls": build_border_walls(width, height),
                "exits": [exit_a, exit_b],
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
                "crowd_delta": crowd_cells,
                "fire_on": [fire_cell] if t == 0 else [],
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
                assert 0 <= int(x) < width and 0 <= int(y) < height, f"Bad coords: {item}"
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
            "crowd_delta": [[center_x, max(1, center_y - 1), 5]],
            "fire_on": [],
            "fire_off": [],
        }

        await ws.send(json.dumps(stale_tick_msg))
        resp = json.loads(await ws.recv())

        assert resp.get("type") == "error", f"Expected stale tick to fail, got {resp}"
        assert "stale tick" in resp.get("message", ""), f"Expected stale tick error, got {resp}"
        print("STALE TICK OK")

        # === OUT-OF-BOUNDS INPUT CONTRACT ===
        oob_tick_msg = {
            "type": "tick",
            "session_id": "contract_test",
            "t": 5,
            "crowd_delta": [
                [-1, center_y, 4],
                [width + 3, 2, 7],
                [2, height + 5, 1],
                [center_x, center_y, 6],
            ],
            "fire_on": [
                [-2, -2],
                [width, height],
                [center_x, min(height - 1, center_y + 1)],
            ],
            "fire_off": [
                [999, 999]
            ],
        }

        await ws.send(json.dumps(oob_tick_msg))
        resp = json.loads(await ws.recv())

        assert resp.get("type") == "cmd", f"Expected cmd after out-of-bounds tick, got {resp}"
        assert resp.get("t") == 5, f"Expected t=5, got {resp}"
        assert "lights_delta" in resp and isinstance(resp["lights_delta"], list), f"Bad lights_delta: {resp}"

        for item in resp["lights_delta"][:10]:
            assert isinstance(item, list) and len(item) == 3, f"Bad delta item after OOB tick: {item}"
            x, y, s = item
            assert 0 <= int(x) < width and 0 <= int(y) < height, f"Backend returned out-of-bounds coords: {item}"
            assert s in ("WHITE", "RED", "OFF"), f"Bad state after OOB tick: {item}"

        print("OUT-OF-BOUNDS INPUT OK")

        print("\nCONTRACT TEST PASSED ✅")


if __name__ == "__main__":
    args = parse_args()

    if args.width < 3 or args.height < 3:
        raise ValueError("width and height must both be at least 3")

    asyncio.run(contract_test(args.ws_url, args.width, args.height))