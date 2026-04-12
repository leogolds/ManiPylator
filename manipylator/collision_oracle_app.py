"""
HTTP collision oracle: one long-lived Genesis ``KinematicSimulator`` and ``GET`` / ``POST /collision``.

Configuration via environment variables:

- ``COLLISION_ORACLE_WORLD_FILE`` – path to a ``.py`` file exporting ``world`` (``World``) or
  ``build_world()`` (returns ``World`` or ``WorldMorphs``). Default: ``/workspace/worlds/empiric_collision_world.py``
  when that path exists, else repository ``worlds/empiric_collision_world.py`` relative to cwd.
- ``COLLISION_ORACLE_ROBOT_TEMPLATE`` – passed to ``render_robot_from_template`` (default ``robots/empiric``).
- ``COLLISION_ORACLE_INCLUDE_GROUND`` – ``1``/``true``/``yes`` for ground plane (default true).
- ``COLLISION_ORACLE_MAX_QPS`` – max collision queries per second (0 = unlimited).
- ``COLLISION_ORACLE_MAX_QUEUE`` – max pending requests; when full, respond with 503.
- ``COLLISION_ORACLE_HOST`` – bind host (default ``0.0.0.0``).
- ``COLLISION_ORACLE_PORT`` – port (default ``8765``).

Run: ``uvicorn manipylator.collision_oracle_app:app`` or ``python -m manipylator.collision_oracle_app``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import queue
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from manipylator.base import KinematicSimulator, World
from manipylator.schemas import (
    CollisionCheckBodyV1,
    CollisionCheckResponseV1,
    validate_joint_q_6dof,
)

COLLISION_ENDPOINT_USAGE = (
    "Six joint angles in radians, Genesis solver order (GET /v1/meta → dof_labels). "
    "Same validation as MQTT ``manipylator/robot/state/v1`` field ``q`` (RobotStateV1). "
    "Ergonomic: GET /collision?q1=0&q2=0&q3=0&q4=0&q5=0&q6=0. "
    'MQTT-shaped: POST /collision with JSON body {"q": [0, 0, 0, 0, 0, 0]}.'
)


def _parse_q1_through_q6_params(
    q1: Optional[float],
    q2: Optional[float],
    q3: Optional[float],
    q4: Optional[float],
    q5: Optional[float],
    q6: Optional[float],
) -> list[float]:
    names = ("q1", "q2", "q3", "q4", "q5", "q6")
    vals = [q1, q2, q3, q4, q5, q6]
    missing = [n for n, v in zip(names, vals) if v is None]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Missing query parameter(s): "
                    + ", ".join(missing)
                    + ". All of q1, q2, q3, q4, q5, q6 are required."
                ),
                "how_to_use": COLLISION_ENDPOINT_USAGE,
            },
        )
    raw = [float(v) for v in vals]
    try:
        validate_joint_q_6dof(raw)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(e),
                "how_to_use": COLLISION_ENDPOINT_USAGE,
            },
        )
    return raw


async def _run_collision_query(q_vec: list[float]) -> CollisionCheckResponseV1:
    """Enqueue one ``is_collided`` check and return the MQTT-shaped response."""
    if state.simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not ready")
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[bool] = loop.create_future()
    try:
        state._task_queue.put_nowait((q_vec, loop, fut))
    except queue.Full:
        raise HTTPException(status_code=503, detail="Collision check queue is full")
    try:
        collided = await asyncio.wait_for(fut, timeout=300.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Collision check timed out")
    return CollisionCheckResponseV1(collided=collided, q=q_vec)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _default_world_file() -> Path:
    p = Path("/workspace/worlds/empiric_collision_world.py")
    if p.is_file():
        return p
    return Path(__file__).resolve().parents[1] / "worlds" / "empiric_collision_world.py"


def load_world_from_file(path: Path) -> World:
    """Load ``world`` or ``build_world()`` from a Python file (trusted lab use only)."""
    spec = importlib.util.spec_from_file_location("collision_oracle_world", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load world module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    if hasattr(mod, "world") and isinstance(getattr(mod, "world"), World):
        return mod.world
    if hasattr(mod, "build_world"):
        bw = mod.build_world()
        if isinstance(bw, World):
            return bw
        return World.from_morphs(bw)  # type: ignore[arg-type]
    raise ValueError(
        f"{path} must define `world: World` or `build_world() -> World | WorldMorphs`"
    )


def _dof_labels_in_solver_order(entity) -> tuple[str, ...]:
    movable = [j for j in entity.joints if j.n_dofs > 0]
    movable.sort(key=lambda j: j.dofs_idx_local[0])
    labels: list[str] = []
    for j in movable:
        if j.n_dofs == 1:
            labels.append(j.name)
        else:
            labels.extend(f"{j.name}[{k}]" for k in range(j.n_dofs))
    return tuple(labels)


class MetaResponse(BaseModel):
    dof_labels: list[str]
    n_dofs: int
    robot_template: str
    world_file: str


class OracleState:
    """Holds simulator and queue worker state."""

    def __init__(self) -> None:
        self.simulator: Optional[KinematicSimulator] = None
        self.last_q: list[float] = [0.0] * 6
        self._sim_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._task_queue: queue.Queue[
            tuple[list[float], asyncio.AbstractEventLoop, asyncio.Future]
        ]
        self._stop = threading.Event()
        self._max_qps: float = 0.0
        self._last_proc_mono: float = 0.0
        self._lock_interval: float = 0.0

    def start_worker(
        self,
        sim: KinematicSimulator,
        max_qps: float,
        max_queue: int,
    ) -> None:
        self.simulator = sim
        self._max_qps = max(0.0, max_qps)
        self._lock_interval = (1.0 / self._max_qps) if self._max_qps > 0 else 0.0
        self._task_queue = queue.Queue(maxsize=max_queue)

        def worker() -> None:
            while not self._stop.is_set():
                try:
                    q_vec, loop, fut = self._task_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    if self._lock_interval > 0:
                        now = time.monotonic()
                        wait = self._lock_interval - (now - self._last_proc_mono)
                        if wait > 0:
                            time.sleep(wait)
                    assert self.simulator is not None
                    collided = self.simulator.is_collided(q_vec)
                    self.last_q = list(q_vec)
                    self._last_proc_mono = time.monotonic()

                    def _set_result() -> None:
                        if not fut.done():
                            fut.set_result(collided)

                    loop.call_soon_threadsafe(_set_result)
                except Exception as e:

                    def _set_exc(exc: BaseException = e) -> None:
                        if not fut.done():
                            fut.set_exception(exc)

                    loop.call_soon_threadsafe(_set_exc)
                finally:
                    self._task_queue.task_done()

        self._worker = threading.Thread(
            target=worker, name="collision-oracle-worker", daemon=True
        )
        self._worker.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)


state = OracleState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    world_path = Path(
        os.environ.get("COLLISION_ORACLE_WORLD_FILE", str(_default_world_file()))
    )
    robot_template = os.environ.get("COLLISION_ORACLE_ROBOT_TEMPLATE", "robots/empiric")
    include_ground = _env_bool("COLLISION_ORACLE_INCLUDE_GROUND", True)
    max_qps = float(os.environ.get("COLLISION_ORACLE_MAX_QPS", "0"))
    max_queue = int(os.environ.get("COLLISION_ORACLE_MAX_QUEUE", "256"))

    world = load_world_from_file(world_path)

    from manipylator.utils import render_robot_from_template

    with render_robot_from_template(robot_template) as urdf_path:
        sim = KinematicSimulator(
            urdf_path,
            headless=True,
            world=world,
            include_ground_plane=include_ground,
        )
        state.start_worker(sim, max_qps=max_qps, max_queue=max_queue)

        app.state.world_path = str(world_path.resolve())
        app.state.robot_template = robot_template
        app.state.include_ground = include_ground

        yield

    state.shutdown()


app = FastAPI(title="ManiPylator collision oracle", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "message": "Request body does not match the expected schema.",
                "errors": exc.errors(),
                "how_to_use": COLLISION_ENDPOINT_USAGE,
            }
        },
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "genesis_ready": state.simulator is not None,
    }


@app.get("/v1/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    if state.simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not ready")
    labels = list(_dof_labels_in_solver_order(state.simulator.robot))
    return MetaResponse(
        dof_labels=labels,
        n_dofs=len(labels),
        robot_template=getattr(app.state, "robot_template", ""),
        world_file=getattr(app.state, "world_path", ""),
    )


@app.get("/collision", response_model=CollisionCheckResponseV1)
async def collision_check_get(
    q1: Optional[float] = Query(None, description="Joint 1 (rad); dof_labels[0] from GET /v1/meta"),
    q2: Optional[float] = Query(None, description="Joint 2 (rad)"),
    q3: Optional[float] = Query(None, description="Joint 3 (rad)"),
    q4: Optional[float] = Query(None, description="Joint 4 (rad)"),
    q5: Optional[float] = Query(None, description="Joint 5 (rad)"),
    q6: Optional[float] = Query(None, description="Joint 6 (rad)"),
) -> CollisionCheckResponseV1:
    """Collision probe via URL query (same ``q`` semantics as POST / MQTT)."""
    q_vec = _parse_q1_through_q6_params(q1, q2, q3, q4, q5, q6)
    return await _run_collision_query(q_vec)


@app.post("/collision", response_model=CollisionCheckResponseV1)
async def collision_check_post(body: CollisionCheckBodyV1) -> CollisionCheckResponseV1:
    """Collision probe using the same ``q`` vector as MQTT ``RobotStateV1``."""
    return await _run_collision_query(list(body.q))


@app.get("/view")
async def view() -> Response:
    """Return a PNG snapshot of the scene at the last ``q`` used in a collision check."""
    if state.simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not ready")
    try:
        import cv2
    except ImportError as e:
        raise HTTPException(status_code=501, detail="OpenCV required for /view") from e

    def _render_rgb() -> np.ndarray:
        with state._sim_lock:
            return state.simulator.render_rgb_uint8(state.last_q)

    # Run on the event-loop thread (not asyncio.to_thread): Genesis/Taichi GPU
    # context is tied to the process main thread; worker thread handles collision only.
    rgb = _render_rgb()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode PNG")
    return Response(content=buf.tobytes(), media_type="image/png")


def main() -> None:
    import uvicorn

    host = os.environ.get("COLLISION_ORACLE_HOST", "0.0.0.0")
    port = int(os.environ.get("COLLISION_ORACLE_PORT", "8765"))
    uvicorn.run(
        "manipylator.collision_oracle_app:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
