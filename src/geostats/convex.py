r"""Part I. Convex bodies via support functions [1].

A planar convex body is stored as a sampled support function $H(\theta)$
on $\theta\in[0,2\pi)$. Minkowski addition is pointwise addition of $H$.
The support point of a $C^2$ body is $x=H u + H_\theta u^\perp$.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def angle_grid(n: int = 360) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)


def ellipse_support(a: float, b: float, theta: np.ndarray) -> np.ndarray:
    """Support of the ellipse $x^2/a^2+y^2/b^2=1$."""
    return np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)


def stadium_support(radius: float, half_len: float, theta: np.ndarray) -> np.ndarray:
    r"""Capsule: disk of radius $r$ Minkowski-summed with a segment of length $2\ell$."""
    return radius + half_len * np.abs(np.cos(theta))


def square_support(half_side: float, theta: np.ndarray) -> np.ndarray:
    r"""Axis-aligned square of half-side $s$: $H=s\max(|\cos\theta|,|\sin\theta|)$."""
    return half_side * np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))


def minkowski_sum_h(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    return h1 + h2


def support_points(h: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Boundary from $x=H u+H_\\theta u^\\perp$."""
    dth = float(theta[1] - theta[0])
    hp = np.gradient(np.r_[h, h[0]], dth)[:-1]
    c, s = np.cos(theta), np.sin(theta)
    x = h * c - hp * s
    y = h * s + hp * c
    return np.stack([x, y], axis=1)


def close_loop(pts: np.ndarray) -> np.ndarray:
    return np.vstack([pts, pts[0]])


def polygonal_area(pts: np.ndarray) -> float:
    hull = ConvexHull(pts)
    return float(hull.volume)


def area_from_support(h: np.ndarray, theta: np.ndarray) -> float:
    """$A=\\tfrac12\\int(H^2-H_\\theta^2)\\,d\\theta$ for a $C^2$ convex body."""
    dth = float(theta[1] - theta[0])
    hp = np.gradient(np.r_[h, h[0]], dth)[:-1]
    return 0.5 * float(np.sum(h * h - hp * hp) * dth)


def brunn_minkowski_curve(
    h0: np.ndarray,
    h1: np.ndarray,
    theta: np.ndarray,
    n_t: int = 41,
) -> dict:
    """$K_t=(1-t)K_0\\oplus t K_1$. Compare $\\sqrt{A(K_t)}$ to the linear bound."""
    ts = np.linspace(0.0, 1.0, n_t)
    root = []
    for t in ts:
        h = (1.0 - t) * h0 + t * h1
        root.append(np.sqrt(max(area_from_support(h, theta), 0.0)))
    root = np.asarray(root)
    linear = (1.0 - ts) * root[0] + ts * root[-1]
    return {"t": ts, "root_area": root, "linear": linear, "gap": root - linear}


def cauchy_perimeter_2d(h: np.ndarray, theta: np.ndarray) -> float:
    """Planar Cauchy: $L=\\int_0^{\\pi} w(\\phi)\\,d\\phi$ with width $w=H(u)+H(-u)$."""
    n = h.size
    half = n // 2
    width = h + np.roll(h, half)
    dth = float(theta[1] - theta[0])
    return float(np.sum(width[:half]) * dth)


def parallel_support(h: np.ndarray, radius: float) -> np.ndarray:
    """Steiner: $H(K\\oplus r B)=H(K)+r$."""
    return h + radius


def steiner_area_plane(h: np.ndarray, theta: np.ndarray, radius: float) -> float:
    """Exact Steiner polynomial in the plane: $A+Lr+\\pi r^2$."""
    area = area_from_support(h, theta)
    length = cauchy_perimeter_2d(h, theta)
    return float(area + length * radius + np.pi * radius * radius)
