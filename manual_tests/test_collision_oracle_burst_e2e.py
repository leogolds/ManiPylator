#!/usr/bin/env python3
"""
Burst-load e2e: many GET /collision requests with q1..q6 query parameters (short window).

Cycles canonical clear, +90deg hit, and -90deg hit on q3 (empiric world) and asserts each
JSON ``collided`` matches the expected boolean.

Requires a running collision oracle (same as test_collision_oracle_e2e.py).

  python manual_tests/test_collision_oracle_burst_e2e.py

  E2E_BURST_COUNT=200 E2E_ORACLE_BASE=http://127.0.0.1:8765 python manual_tests/test_collision_oracle_burst_e2e.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from urllib.error import URLError
from urllib.parse import urlencode

BASE = os.environ.get("E2E_ORACLE_BASE", "http://127.0.0.1:8765").rstrip("/")
WAIT_GENESIS_SEC = float(os.environ.get("E2E_WAIT_GENESIS_SEC", "400"))
BURST_COUNT = int(os.environ.get("E2E_BURST_COUNT", "200"))

# Same canonical poses as test_collision_oracle_e2e.py (empiric world).
_Q_CLEAR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_Q_HIT_POS = [0.0, 0.0, 1.5707963267948966, 0.0, 0.0, 0.0]  # +90 deg
_Q_HIT_NEG = [0.0, 0.0, -1.5707963267948966, 0.0, 0.0, 0.0]  # -90 deg


def _get(path: str) -> tuple[int, bytes]:
    req = urllib.request.Request(f"{BASE}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except URLError:
        return 0, b""


def _get_collision_q(qs: list[float]) -> tuple[int, bytes]:
    params = {f"q{i + 1}": qs[i] for i in range(6)}
    return _get(f"/collision?{urlencode(params)}")


def _wait_for_genesis_ready() -> bool:
    deadline = time.monotonic() + WAIT_GENESIS_SEC
    while time.monotonic() < deadline:
        code, body = _get("/health")
        if code == 200 and body:
            try:
                j = json.loads(body.decode())
                if j.get("genesis_ready"):
                    return True
            except json.JSONDecodeError:
                pass
        time.sleep(2.0)
    return False


def _pose_for_burst_index(i: int) -> tuple[list[float], bool]:
    """Return (q, expected_collided). Cycle clear, +90 hit, -90 hit; tiny q0 nudge keeps URLs distinct."""
    nudge = float(i) * 1e-9
    r = i % 3
    if r == 0:
        q = list(_Q_CLEAR)
        q[0] += nudge
        return q, False
    if r == 1:
        q = list(_Q_HIT_POS)
        q[0] += nudge
        return q, True
    q = list(_Q_HIT_NEG)
    q[0] += nudge
    return q, True


def main() -> int:
    print(f"E2E oracle base: {BASE}")
    print(f"GET /collision burst (q1..q6 params): count={BURST_COUNT}")

    if not _wait_for_genesis_ready():
        print(
            f"TIMEOUT: no genesis_ready on {BASE}/health within {WAIT_GENESIS_SEC}s",
            file=sys.stderr,
        )
        return 1

    failures: list[tuple[int, int, bytes]] = []
    wrong_collided: list[tuple[int, bool, object, str]] = []
    t0 = time.perf_counter()
    for i in range(BURST_COUNT):
        q, expect_collided = _pose_for_burst_index(i)
        code, body = _get_collision_q(q)
        if code != 200:
            failures.append((i, code, body[:200]))
            continue
        try:
            j = json.loads(body.decode())
        except json.JSONDecodeError:
            failures.append((i, -1, body[:200]))
            continue
        got = j.get("collided")
        if got is not True and got is not False:
            wrong_collided.append(
                (i, expect_collided, got, "collided missing or not bool")
            )
            continue
        if got != expect_collided:
            wrong_collided.append((i, expect_collided, got, json.dumps(j)[:200]))

    elapsed = time.perf_counter() - t0
    qps = BURST_COUNT / elapsed if elapsed > 0 else 0.0

    print(
        f"Completed {BURST_COUNT} GET /collision requests in {elapsed:.3f}s "
        f"({qps:.1f} req/s client-side)"
    )

    if failures:
        for idx, code, snippet in failures[:10]:
            print(
                f"  FAIL i={idx} HTTP {code} body={snippet!r}",
                file=sys.stderr,
            )
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more", file=sys.stderr)
        print(
            "Burst failed (non-200). If you see 503, raise COLLISION_ORACLE_MAX_QPS "
            "or COLLISION_ORACLE_MAX_QUEUE on the server.",
            file=sys.stderr,
        )
        return 1

    if wrong_collided:
        for idx, exp, got, detail in wrong_collided[:15]:
            print(
                f"  WRONG collided i={idx} expected={exp} got={got} {detail}",
                file=sys.stderr,
            )
        if len(wrong_collided) > 15:
            print(f"  ... and {len(wrong_collided) - 15} more", file=sys.stderr)
        return 1

    print("Burst OK (collided flags match clear vs +90 vs -90 hit poses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
