r"""Part II. Crofton sampling from the invariant line measure [2].

Lines in the plane: $dG=dp\,d\phi$, $\phi\in[0,\pi)$. For a closed convex
curve, $N=2$ on a hitting line and $\int N\,dG=2L$, so the hitting measure
equals the perimeter.
"""

from __future__ import annotations

import numpy as np

from geostats.convex import area_from_support, cauchy_perimeter_2d, support_points


def sample_lines(
    rng: np.random.Generator,
    n: int,
    p_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    phi = rng.uniform(0.0, np.pi, size=n)
    p = rng.uniform(-p_max, p_max, size=n)
    return p, phi


def support_interp(h: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Linear interpolation of a $2\\pi$-periodic sampled support function."""
    n = h.size
    x = np.asarray(phi, dtype=float) / (2.0 * np.pi) * n
    i0 = np.floor(x).astype(int) % n
    i1 = (i0 + 1) % n
    a = x - np.floor(x)
    return (1.0 - a) * h[i0] + a * h[i1]


def hits_convex(p: np.ndarray, phi: np.ndarray, h: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """A line $x\\cos\\phi+y\\sin\\phi=p$ hits iff $-H(\\phi+\\pi)\\le p\\le H(\\phi)$."""
    del theta
    h_phi = support_interp(h, phi)
    h_opp = support_interp(h, phi + np.pi)
    return (p <= h_phi + 1e-12) & (p >= -h_opp - 1e-12)


def chord_lengths(p: np.ndarray, phi: np.ndarray, h: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Secant length $\\sigma$ of each line, or $0$ if the line misses $K$."""
    u = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    up = np.stack([-np.sin(phi), np.cos(phi)], axis=1)
    ut = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    cos_d = u @ ut.T
    sin_d = up @ ut.T
    rhs = h[None, :] - p[:, None] * cos_d
    tmin = np.full(p.shape, -np.inf)
    tmax = np.full(p.shape, np.inf)
    pos = sin_d > 1e-14
    neg = sin_d < -1e-14
    bounds = np.divide(rhs, sin_d, out=np.zeros_like(rhs), where=pos | neg)
    if np.any(pos):
        filled = np.where(pos, bounds, np.inf)
        tmax = np.min(filled, axis=1)
    if np.any(neg):
        filled = np.where(neg, bounds, -np.inf)
        tmin = np.max(filled, axis=1)
    sigma = np.maximum(tmax - tmin, 0.0)
    sigma[~np.isfinite(sigma)] = 0.0
    return sigma


def crofton_running(
    rng: np.random.Generator,
    h: np.ndarray,
    theta: np.ndarray,
    n: int = 4000,
    p_max: float | None = None,
) -> dict:
    if p_max is None:
        p_max = float(h.max()) * 1.15
    p, phi = sample_lines(rng, n, p_max)
    hit = hits_convex(p, phi, h, theta)
    window = 2.0 * p_max * np.pi
    # E[N] * window = 2L with N=2 on hit, 0 else, so L_hat = window * mean(hit)
    cmean = np.cumsum(hit.astype(float)) / np.arange(1, n + 1)
    l_hat = window * cmean
    l_true = cauchy_perimeter_2d(h, theta)
    segs = line_segments(p, phi, p_max * 1.4)
    sigma = chord_lengths(p, phi, h, theta)
    area = area_from_support(h, theta)
    hostinsky_hat = window * float(np.mean(sigma**3))
    hostinsky_true = 3.0 * area * area
    return {
        "l_hat": l_hat,
        "l_true": l_true,
        "hit": hit,
        "p": p,
        "phi": phi,
        "sigma": sigma,
        "segments": segs,
        "rel_err": abs(float(l_hat[-1]) - l_true) / l_true,
        "hostinsky_hat": hostinsky_hat,
        "hostinsky_true": hostinsky_true,
        "hostinsky_rel_err": abs(hostinsky_hat - hostinsky_true) / max(hostinsky_true, 1e-15),
    }


def line_segments(p: np.ndarray, phi: np.ndarray, span: float) -> np.ndarray:
    """Endpoints of displayed chords, shape $(n,2,2)$."""
    c, s = np.cos(phi), np.sin(phi)
    # point on the line nearest origin: p (cos φ, sin φ)
    q = np.stack([p * c, p * s], axis=1)
    t = np.stack([-s, c], axis=1)
    a = q - span * t
    b = q + span * t
    return np.stack([a, b], axis=1)


def body_outline(h: np.ndarray, theta: np.ndarray) -> np.ndarray:
    from geostats.convex import close_loop

    return close_loop(support_points(h, theta))
