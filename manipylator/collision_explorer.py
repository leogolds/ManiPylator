"""
Configuration-space exploration: Sobol cover, mixed-edge graph, bisection, forbidden boxes.

Requires ``scikit-learn`` (``pip install 'manipylator[exploration]'`` or ``scikit-learn``).

Oracle: ``Callable[[Sequence[float]], bool]`` returning True if configuration is in collision.
Use :func:`http_oracle` when the oracle is the collision REST service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import roboticstoolbox as rtb
from scipy.stats import qmc

try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
except ImportError as e:
    raise ImportError(
        "collision_explorer requires scikit-learn. "
        "Install with: pip install 'manipylator[exploration]' or pip install scikit-learn"
    ) from e


@dataclass
class JointLimits:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=np.float64).reshape(-1)
        self.upper = np.asarray(self.upper, dtype=np.float64).reshape(-1)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape")


def get_joint_limits(robot: rtb.Robot) -> JointLimits:
    """Return per-joint lower/upper bounds from RTB ``qlim`` (shape ``(n,)`` each)."""
    q = np.asarray(robot.qlim, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"unexpected qlim shape: {robot.qlim}")
    if q.shape[0] == 2:
        lower, upper = q[0], q[1]
    elif q.shape[1] == 2:
        lower, upper = q[:, 0], q[:, 1]
    else:
        raise ValueError(f"unexpected qlim shape: {q.shape}")
    return JointLimits(lower=lower, upper=upper)


def normalize_q(q: np.ndarray, limits: JointLimits) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    span = limits.upper - limits.lower
    span = np.where(span < 1e-12, 1.0, span)
    return (q - limits.lower) / span


def denormalize_q(u: np.ndarray, limits: JointLimits) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    span = limits.upper - limits.lower
    return limits.lower + u * span


def sobol_sample(n: int, limits: JointLimits, seed: int) -> np.ndarray:
    """``(n, n_dof)`` joint samples in physical radians."""
    d = limits.lower.size
    engine = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = engine.random(n=n)
    out = np.empty_like(u)
    for i in range(d):
        out[:, i] = limits.lower[i] + u[:, i] * (limits.upper[i] - limits.lower[i])
    return out


@dataclass
class Sample:
    q: np.ndarray
    collided: bool
    u: Optional[np.ndarray] = None


@dataclass
class MixedEdge:
    i: int
    j: int


@dataclass
class BoundaryPoint:
    q: np.ndarray
    u: np.ndarray


def find_mixed_edges(
    samples: Sequence[Sample],
    limits: JointLimits,
    k: int = 8,
) -> List[MixedEdge]:
    """k-NN graph edges where endpoints disagree on collision class."""
    if len(samples) < 2:
        return []
    u = np.stack([normalize_q(s.q, limits) for s in samples])
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(samples)), algorithm="auto").fit(u)
    dist, idx = nbrs.kneighbors(u)
    edges: List[MixedEdge] = []
    seen = set()
    for a in range(len(samples)):
        for nb in idx[a, 1:]:
            b = int(nb)
            if samples[a].collided == samples[b].collided:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            edges.append(MixedEdge(i=key[0], j=key[1]))
    return edges


def bisect_collision_boundary(
    oracle: Callable[[Sequence[float]], bool],
    q_free: np.ndarray,
    q_collide: np.ndarray,
    limits: JointLimits,
    tol: float = 1e-3,
    max_steps: int = 48,
) -> BoundaryPoint:
    """Bisect in normalized space between a free and a collided configuration."""
    u_lo = normalize_q(q_free, limits)
    u_hi = normalize_q(q_collide, limits)
    f_lo = oracle(q_free)
    f_hi = oracle(q_collide)
    if f_lo == f_hi:
        mid_u = 0.5 * (u_lo + u_hi)
        return BoundaryPoint(q=denormalize_q(mid_u, limits), u=mid_u)
    for _ in range(max_steps):
        u_mid = 0.5 * (u_lo + u_hi)
        q_mid = denormalize_q(u_mid, limits)
        if np.max(np.abs(u_hi - u_lo)) < tol:
            return BoundaryPoint(q=q_mid, u=u_mid)
        mid_hit = oracle(q_mid)
        if mid_hit == f_lo:
            u_lo = u_mid
        else:
            u_hi = u_mid
    u_mid = 0.5 * (u_lo + u_hi)
    q_mid = denormalize_q(u_mid, limits)
    return BoundaryPoint(q=q_mid, u=u_mid)


def fit_forbidden_boxes(
    collided_samples: np.ndarray,
    limits: JointLimits,
    eps: float = 0.08,
    min_samples: int = 2,
) -> List[dict]:
    """DBSCAN in normalized space; axis-aligned boxes from cluster bounds."""
    if collided_samples.size == 0:
        return []
    u = normalize_q(collided_samples, limits)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(u)
    boxes: List[dict] = []
    for lab in sorted(set(clustering.labels_)):
        if lab < 0:
            continue
        mask = clustering.labels_ == lab
        cluster_u = u[mask]
        u_min = cluster_u.min(axis=0)
        u_max = cluster_u.max(axis=0)
        q_min = denormalize_q(u_min, limits)
        q_max = denormalize_q(u_max, limits)
        boxes.append(
            {
                "min": q_min.tolist(),
                "max": q_max.tolist(),
                "cluster": int(lab),
            }
        )
    return boxes


def export_samples(path: Path, samples: Sequence[Sample]) -> None:
    payload = [
        {
            "q": s.q.tolist(),
            "collided": s.collided,
            "u": s.u.tolist() if s.u is not None else None,
        }
        for s in samples
    ]
    path.write_text(json.dumps(payload, indent=2))


def export_boundary(path: Path, points: Sequence[BoundaryPoint]) -> None:
    payload = [{"q": p.q.tolist(), "u": p.u.tolist()} for p in points]
    path.write_text(json.dumps(payload, indent=2))


def export_forbidden_boxes(path: Path, boxes: Sequence[dict]) -> None:
    path.write_text(json.dumps(list(boxes), indent=2))


def export_all(
    out_dir: Path,
    samples: Sequence[Sample],
    boundary_points: Sequence[BoundaryPoint],
    forbidden_boxes: Sequence[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    export_samples(out_dir / "collision_samples.json", samples)
    export_boundary(out_dir / "collision_boundary.json", boundary_points)
    export_forbidden_boxes(out_dir / "forbidden_boxes.json", forbidden_boxes)


def http_oracle(base_url: str, timeout: float = 60.0) -> Callable[[Sequence[float]], bool]:
    """Oracle that ``GET``s ``{base_url}/collision`` with ``q1``..``q6`` query params (same angles as MQTT ``RobotStateV1.q``)."""
    try:
        import httpx
    except ImportError as e:
        raise ImportError("http_oracle requires httpx: pip install httpx") from e

    base = base_url.rstrip("/")

    def _oracle(q: Sequence[float]) -> bool:
        q_list = [float(x) for x in q]
        if len(q_list) != 6:
            raise ValueError("http_oracle expects exactly 6 joint values")
        params = {f"q{i + 1}": q_list[i] for i in range(6)}
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{base}/collision", params=params)
            r.raise_for_status()
            data = r.json()
            return bool(data["collided"])

    return _oracle


@dataclass
class CollisionExplorer:
    limits: JointLimits
    oracle: Callable[[Sequence[float]], bool]
    samples: List[Sample] = field(default_factory=list)
    boundary_points: List[BoundaryPoint] = field(default_factory=list)
    forbidden_boxes: List[dict] = field(default_factory=list)

    def initial_cover(self, n: int, seed: int) -> None:
        qs = sobol_sample(n, self.limits, seed)
        for row in qs:
            q = np.asarray(row, dtype=np.float64)
            hit = self.oracle(q)
            u = normalize_q(q, self.limits)
            self.samples.append(Sample(q=q, collided=hit, u=u))

    def mixed_edges(self, k: int = 8) -> List[MixedEdge]:
        return find_mixed_edges(self.samples, self.limits, k=k)

    def refine_top_edges(
        self,
        n_edges: int,
        tol: float = 1e-3,
        k: int = 8,
    ) -> None:
        edges = self.mixed_edges(k=k)
        scored: List[Tuple[float, MixedEdge]] = []
        u_all = np.stack([normalize_q(s.q, self.limits) for s in self.samples])
        for e in edges:
            d = np.linalg.norm(u_all[e.i] - u_all[e.j])
            scored.append((d, e))
        scored.sort(reverse=True, key=lambda t: t[0])
        for _, e in scored[:n_edges]:
            qa, qb = self.samples[e.i].q, self.samples[e.j].q
            ca, cb = self.samples[e.i].collided, self.samples[e.j].collided
            if ca == cb:
                continue
            if not ca and cb:
                bp = bisect_collision_boundary(self.oracle, qa, qb, self.limits, tol=tol)
            else:
                bp = bisect_collision_boundary(self.oracle, qb, qa, self.limits, tol=tol)
            self.boundary_points.append(bp)

    def fit_regions(self, eps: float = 0.08, min_samples: int = 2) -> None:
        collided = np.stack([s.q for s in self.samples if s.collided])
        if collided.size == 0:
            self.forbidden_boxes = []
            return
        self.forbidden_boxes = fit_forbidden_boxes(
            collided, self.limits, eps=eps, min_samples=min_samples
        )

    def export(self, out_dir: Path) -> None:
        export_all(out_dir, self.samples, self.boundary_points, self.forbidden_boxes)
