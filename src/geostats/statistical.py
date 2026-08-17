r"""Part III. Fisher geometry of the univariate Gaussian [3, 4].

In coordinates $(\mu,\sigma)$, $\sigma>0$,

$$
g
=
\mathrm{diag}(\sigma^{-2},\,2\sigma^{-2}).
$$

This is a Poincaré half-plane (up to scaling). The $\alpha$-Christoffel
symbols are closed form. Natural parameters of $\mathcal{N}(\mu,\sigma^2)$
are $\theta=( \mu/\sigma^2,\,-1/(2\sigma^2) )$, expectation parameters
$\eta=(\mu,\,\mu^2+\sigma^2)$. Straight lines in $\theta$ (resp. $\eta$)
are $\alpha=+1$ (resp. $\alpha=-1$) geodesics.
"""

from __future__ import annotations

import numpy as np


def fisher(mu: float, sigma: float) -> np.ndarray:
    s2 = sigma * sigma
    return np.array([[1.0 / s2, 0.0], [0.0, 2.0 / s2]], dtype=float)


def kl_gaussian(mu0: float, s0: float, mu: float, s: float) -> float:
    return float(np.log(s / s0) + (s0 * s0 + (mu0 - mu) ** 2) / (2.0 * s * s) - 0.5)


def fisher_from_kl_hessian(mu: float, sigma: float, h: float = 1e-5) -> np.ndarray:
    theta0 = np.array([mu, sigma], dtype=float)
    e = np.eye(2)
    hess = np.zeros((2, 2))
    f0 = kl_gaussian(mu, sigma, mu, sigma)
    for i in range(2):
        hess[i, i] = (
            kl_gaussian(mu, sigma, *(theta0 + h * e[i]))
            + kl_gaussian(mu, sigma, *(theta0 - h * e[i]))
            - 2.0 * f0
        ) / (h * h)
        for j in range(i + 1, 2):
            fpp = kl_gaussian(mu, sigma, *(theta0 + h * e[i] + h * e[j]))
            fpm = kl_gaussian(mu, sigma, *(theta0 + h * e[i] - h * e[j]))
            fmp = kl_gaussian(mu, sigma, *(theta0 - h * e[i] + h * e[j]))
            fmm = kl_gaussian(mu, sigma, *(theta0 - h * e[i] - h * e[j]))
            hess[i, j] = hess[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * h * h)
    return hess


def alpha_christoffel_lower(sigma: float, alpha: float) -> np.ndarray:
    r"""$\Gamma_{ijk}^{(\alpha)}$ at any $\mu$, indices $(\mu,\sigma)$."""
    s3 = sigma**3
    g = np.zeros((2, 2, 2), dtype=float)
    g[0, 0, 1] = (1.0 - alpha) / s3
    g[0, 1, 0] = g[1, 0, 0] = (-1.0 - alpha) / s3
    g[1, 1, 1] = (-2.0 - 4.0 * alpha) / s3
    return g


def alpha_duality_residual(sigma: float) -> float:
    gp = alpha_christoffel_lower(sigma, 1.0)
    gm = alpha_christoffel_lower(sigma, -1.0)
    g0 = alpha_christoffel_lower(sigma, 0.0)
    return float(np.max(np.abs(gp + gm - 2.0 * g0)))


def geodesic(
    mu0: float,
    sig0: float,
    v_mu: float,
    v_sig: float,
    alpha: float = 0.0,
    t_final: float = 1.0,
    n_steps: int = 80,
) -> np.ndarray:
    """RK4 on $\\ddot\\theta^k+\\Gamma^k_{ij}\\dot\\theta^i\\dot\\theta^j=0$."""
    dt = t_final / n_steps
    y = np.array([mu0, sig0, v_mu, v_sig], dtype=float)
    path = np.empty((n_steps + 1, 2), dtype=float)
    path[0] = y[:2]

    def acc(mu: float, sig: float, vel: np.ndarray) -> np.ndarray:
        sig = max(sig, 1e-6)
        low = alpha_christoffel_lower(sig, alpha)
        ginv = np.diag([sig * sig, 0.5 * sig * sig])
        gamma = np.einsum("kl,ijl->kij", ginv, low)
        return -np.einsum("kij,i,j->k", gamma, vel, vel)

    def f(state: np.ndarray) -> np.ndarray:
        mu, sig, vu, vs = state
        a = acc(mu, sig, np.array([vu, vs]))
        return np.array([vu, vs, a[0], a[1]])

    for i in range(n_steps):
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y[1] = max(y[1], 1e-4)
        path[i + 1] = y[:2]
    return path


def geodesic_connect(
    start: tuple[float, float],
    end: tuple[float, float],
    alpha: float = 0.0,
    n_steps: int = 80,
    n_iter: int = 8,
) -> np.ndarray:
    """Shoot an $\\alpha$-geodesic from $start$ to $end$ in unit time."""
    v = np.array([end[0] - start[0], end[1] - start[1]], dtype=float)
    target = np.array(end, dtype=float)
    eps = 1e-4
    for _ in range(n_iter):
        path = geodesic(start[0], start[1], v[0], v[1], alpha=alpha, t_final=1.0, n_steps=n_steps)
        err = path[-1] - target
        if float(np.linalg.norm(err)) < 1e-5:
            return path
        jac = np.zeros((2, 2))
        for k in range(2):
            dv = v.copy()
            dv[k] += eps
            landed = geodesic(start[0], start[1], dv[0], dv[1], alpha=alpha, t_final=1.0, n_steps=n_steps)[-1]
            jac[:, k] = (landed - path[-1]) / eps
        try:
            v = v - np.linalg.solve(jac + 1e-10 * np.eye(2), err)
        except np.linalg.LinAlgError:
            break
    return geodesic(start[0], start[1], v[0], v[1], alpha=alpha, t_final=1.0, n_steps=n_steps)


def to_natural(mu: float, sigma: float) -> np.ndarray:
    s2 = sigma * sigma
    return np.array([mu / s2, -0.5 / s2], dtype=float)


def from_natural(th: np.ndarray) -> tuple[float, float]:
    th1, th2 = float(th[0]), float(th[1])
    sigma = np.sqrt(-0.5 / th2)
    mu = th1 * sigma * sigma
    return float(mu), float(sigma)


def to_expectation(mu: float, sigma: float) -> np.ndarray:
    return np.array([mu, mu * mu + sigma * sigma], dtype=float)


def from_expectation(eta: np.ndarray) -> tuple[float, float]:
    mu = float(eta[0])
    second = float(eta[1])
    var = max(second - mu * mu, 1e-12)
    return mu, float(np.sqrt(var))


def dual_straight(
    start: tuple[float, float],
    end: tuple[float, float],
    kind: str,
    n: int = 80,
) -> np.ndarray:
    """Straight line in $\\theta$ ($+1$) or $\\eta$ ($-1$), mapped back to $(\\mu,\\sigma)$."""
    a = np.linspace(0.0, 1.0, n)
    if kind == "exp":
        t0, t1 = to_natural(*start), to_natural(*end)
        pts = np.stack([from_natural((1 - s) * t0 + s * t1) for s in a])
    else:
        e0, e1 = to_expectation(*start), to_expectation(*end)
        pts = np.stack([from_expectation((1 - s) * e0 + s * e1) for s in a])
    return pts
