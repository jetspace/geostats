r"""Part V. Brownian motion on $S^2$ by projection [5].

Ambient Euler steps plus radial projection converge to Riemannian Brownian
motion (Stratonovich rolling). The generator is $\tfrac12\Delta_{S^2}$.
On spherical harmonics of degree $1$, $\Delta z=-2z$, so
$\mathrm{E}[z_t]=z_0 e^{-t}$.
"""

from __future__ import annotations

import numpy as np


def project(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-15)


def sphere_bm(
    x0: np.ndarray,
    rng: np.random.Generator,
    t_final: float = 1.0,
    n_steps: int = 400,
) -> np.ndarray:
    r"""Projected Brownian increment: $x\leftarrow \Pi(x+\sqrt{dt}\,P_x W)$."""
    x = project(np.asarray(x0, dtype=float))
    dt = t_final / n_steps
    sdt = np.sqrt(dt)
    path = np.empty((n_steps + 1, 3), dtype=float)
    path[0] = x
    for i in range(n_steps):
        w = rng.normal(size=3)
        w = w - x * np.dot(x, w)
        x = project(x + sdt * w)
        path[i + 1] = x
    return path


def sphere_bm_many(
    rng: np.random.Generator,
    n_paths: int = 40,
    t_final: float = 1.5,
    n_steps: int = 250,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    if x0 is None:
        x0 = np.array([0.0, 0.0, 1.0])
    out = np.empty((n_paths, n_steps + 1, 3), dtype=float)
    for k in range(n_paths):
        out[k] = sphere_bm(x0, rng, t_final=t_final, n_steps=n_steps)
    return out


def constraint_residual(path: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.norm(path, axis=-1) - 1.0)))


def mean_z_decay(paths: np.ndarray) -> np.ndarray:
    return paths[:, :, 2].mean(axis=0)


def parallel_frame(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a tangent frame along a path (first-order parallel transport)."""
    x0 = path[0]
    e1 = np.array([1.0, 0.0, 0.0], dtype=float)
    e1 = e1 - np.dot(e1, x0) * x0
    nrm = np.linalg.norm(e1)
    if nrm < 1e-12:
        e1 = np.array([0.0, 1.0, 0.0], dtype=float)
        e1 = e1 - np.dot(e1, x0) * x0
        nrm = np.linalg.norm(e1)
    e1 = e1 / nrm
    e1s = np.empty_like(path)
    e2s = np.empty_like(path)
    for i, x in enumerate(path):
        e1 = e1 - np.dot(e1, x) * x
        e1 = e1 / max(np.linalg.norm(e1), 1e-15)
        e1s[i] = e1
        e2s[i] = np.cross(x, e1)
    return e1s, e2s


def sphere_bm_chart(
    rng: np.random.Generator,
    t_final: float = 1.0,
    n_steps: int = 400,
    polar0: float = 0.15,
) -> np.ndarray:
    """Naive Euler-Maruyama in spherical coordinates, no It\\^o correction.

    Polar angle $\\vartheta$ from the north pole. This scheme is *not*
    Riemannian Brownian motion: it is the chart It\\^o equation without the
    generator drift. Used only as a foil against the projected scheme.
    """
    dt = t_final / n_steps
    sdt = np.sqrt(dt)
    th = float(polar0)
    ph = 0.0
    path = np.empty((n_steps + 1, 3), dtype=float)
    path[0] = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    for i in range(n_steps):
        gth = 1.0
        gph = 1.0 / max(np.sin(th), 1e-6)
        dW = rng.normal(size=2)
        th = float(np.clip(th + sdt * gth * dW[0], 1e-4, np.pi - 1e-4))
        ph = ph + sdt * gph * dW[1]
        path[i + 1] = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return path


def wire_sphere(n_lat: int = 18, n_lon: int = 32) -> list[np.ndarray]:
    """Latitude and longitude circles for a Plotly wireframe."""
    circles = []
    u = np.linspace(0.0, 2.0 * np.pi, n_lon)
    for lat in np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_lat):
        c, s = np.cos(lat), np.sin(lat)
        circles.append(np.stack([c * np.cos(u), c * np.sin(u), np.full_like(u, s)], axis=1))
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_lat)
    for lon in np.linspace(0.0, np.pi, n_lon // 2, endpoint=False):
        circles.append(
            np.stack([np.cos(v) * np.cos(lon), np.cos(v) * np.sin(lon), np.sin(v)], axis=1)
        )
    return circles
