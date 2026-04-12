#!/usr/bin/env python3
"""
End-to-end test against a running collision oracle HTTP server (stdlib urllib only).

Default base URL: ``http://127.0.0.1:8765`` (override with ``E2E_ORACLE_BASE``).

Examples:

  # Oracle already running (e.g. docker compose collision-oracle)
  python manual_tests/test_collision_oracle_e2e.py

  E2E_ORACLE_BASE=http://localhost:9000 python manual_tests/test_collision_oracle_e2e.py

Timing (similar poses, averaged wall time per request):

  E2E_TIMING_SAMPLES=30 E2E_TIMING_WARMUP=5 python manual_tests/test_collision_oracle_e2e.py

  E2E_TIMING_SAMPLES=0   # skip timing, correctness checks only
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.error import URLError
from urllib.parse import urlencode

BASE = os.environ.get("E2E_ORACLE_BASE", "http://127.0.0.1:8765").rstrip("/")
WAIT_GENESIS_SEC = float(os.environ.get("E2E_WAIT_GENESIS_SEC", "400"))
TIMING_SAMPLES = int(os.environ.get("E2E_TIMING_SAMPLES", "20"))
TIMING_WARMUP = int(os.environ.get("E2E_TIMING_WARMUP", "3"))

# Poses for timing: (label, q, expected_collided or None to skip assert)
_Q_CLEAR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_Q_HIT = [0.0, 0.0, 1.5707963267948966, 0.0, 0.0, 0.0]
_Q_NEAR_CLEAR = [0.0, 0.0, 0.01, 0.0, 0.0, 0.0]  # similar to home, still free


def _get(path: str) -> tuple[int, bytes]:
    req = urllib.request.Request(f"{BASE}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except URLError:
        return 0, b""


def _post_json(path: str, body: dict) -> tuple[int, bytes]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except URLError:
        return 0, b""


def _post_collision(qs: list[float]) -> tuple[int, bytes]:
    return _post_json("/collision", {"q": qs})


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


def _bench_case(
    label: str,
    call: Callable[[], tuple[int, bytes]],
    samples: int,
    warmup: int,
    expect_collided: bool | None,
) -> None:
    for _ in range(warmup):
        code, body = call()
        if code != 200:
            print(
                f"  [warmup] {label}: HTTP {code} {body[:200]!r}",
                file=sys.stderr,
            )
            return

    times_ms: list[float] = []
    for i in range(samples):
        t0 = time.perf_counter()
        code, body = call()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if code != 200:
            print(
                f"  [{label}] sample {i}: HTTP {code} {body[:200]!r}",
                file=sys.stderr,
            )
            return
        if expect_collided is not None:
            j = json.loads(body.decode())
            if j.get("collided") != expect_collided:
                print(
                    f"  [{label}] sample {i}: unexpected collided={j.get('collided')!r}",
                    file=sys.stderr,
                )
        times_ms.append(dt_ms)

    if not times_ms:
        return
    mean = statistics.mean(times_ms)
    print(
        f"  {label:32s}  n={len(times_ms):3d}  "
        f"mean={mean:7.2f} ms  min={min(times_ms):7.2f}  max={max(times_ms):7.2f}  "
        f"stdev={statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0:6.2f} ms"
    )


def run_timing_benchmark() -> None:
    if TIMING_SAMPLES <= 0:
        return
    print()
    print(
        f"Timing (wall clock, client side)  samples={TIMING_SAMPLES}  warmup={TIMING_WARMUP}"
    )
    _bench_case(
        "GET /collision q=0 (clear)",
        lambda: _get_collision_q(_Q_CLEAR),
        TIMING_SAMPLES,
        TIMING_WARMUP,
        False,
    )
    _bench_case(
        "GET /collision q3=90deg (hit)",
        lambda: _get_collision_q(_Q_HIT),
        TIMING_SAMPLES,
        TIMING_WARMUP,
        True,
    )
    _bench_case(
        "GET /collision q3=0.01 (near)",
        lambda: _get_collision_q(_Q_NEAR_CLEAR),
        TIMING_SAMPLES,
        TIMING_WARMUP,
        None,
    )
    _bench_case(
        "POST /collision q=0 (clear)",
        lambda: _post_collision(_Q_CLEAR),
        TIMING_SAMPLES,
        TIMING_WARMUP,
        False,
    )


def main() -> int:
    print(f"E2E oracle base: {BASE}")

    if not _wait_for_genesis_ready():
        print(
            f"TIMEOUT: no genesis_ready on {BASE}/health within {WAIT_GENESIS_SEC}s",
            file=sys.stderr,
        )
        return 1

    code, body = _get("/health")
    assert code == 200, (code, body)
    print("GET /health:", body.decode()[:200])

    code, body = _get("/v1/meta")
    assert code == 200, (code, body)
    meta = json.loads(body.decode())
    assert meta.get("n_dofs") == 6, meta
    print("GET /v1/meta: n_dofs=", meta["n_dofs"], "labels=", meta.get("dof_labels"))

    code, body = _get("/collision")
    assert code == 422, (code, body)
    err = json.loads(body.decode())
    detail = err.get("detail", {})
    assert "how_to_use" in detail, err
    print("GET /collision (no q1..q6): 422 with how_to_use OK")

    code, body = _post_json("/collision", {})
    assert code == 422, (code, body)
    err = json.loads(body.decode())
    detail = err.get("detail", {})
    assert "how_to_use" in detail, err
    print("POST /collision (empty body): 422 with how_to_use OK")

    q_clear = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    code, body = _get_collision_q(q_clear)
    assert code == 200, (code, body)
    r0 = json.loads(body.decode())
    assert r0.get("q") == q_clear, r0
    print("GET /collision clear pose collided:", r0.get("collided"))

    code, body = _post_collision(q_clear)
    assert code == 200, (code, body)
    r0b = json.loads(body.decode())
    assert r0b.get("collided") == r0.get("collided")
    print("POST /collision (same pose) matches GET")

    q_hit = [0.0, 0.0, 1.5707963267948966, 0.0, 0.0, 0.0]
    code, body = _get_collision_q(q_hit)
    assert code == 200, (code, body)
    r1 = json.loads(body.decode())
    assert len(r1.get("q", [])) == 6
    assert r1.get("collided") is True, r1
    print("GET /collision +90deg pose collided:", r1.get("collided"))

    q_hit_neg = [0.0, 0.0, -1.5707963267948966, 0.0, 0.0, 0.0]
    code, body = _get_collision_q(q_hit_neg)
    assert code == 200, (code, body)
    r1n = json.loads(body.decode())
    assert r1n.get("q") == q_hit_neg, r1n
    assert r1n.get("collided") is True, r1n
    print("GET /collision -90deg pose collided:", r1n.get("collided"))

    code, body = _get("/view")
    assert code == 200, (code, body)
    assert body[:8] == b"\x89PNG\r\n\x1a\n", "expected PNG signature"
    print("GET /view: PNG bytes", len(body))

    run_timing_benchmark()

    print("E2E OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
