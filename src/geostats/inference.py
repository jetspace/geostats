r"""Part IV. Efron curvature of a curved exponential family [3, 4].

The parabolic location family $X\sim\mathcal{N}(\eta(u),I_2)$,
$\eta(u)=(u,a u^2)$, sits in a flat Gaussian location family. Statistical
curvature is

$$
\gamma^2(u)=\frac{4a^2}{(1+4a^2 u^2)^3}.
$$

Information loss of the MLE is this intrinsic term (mixture ancillary
curvature vanishes).
"""

from __future__ import annotations

import numpy as np


def parabola_fisher(u: np.ndarray | float, a: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return 1.0 + 4.0 * a * a * u * u


def parabola_gamma2(u: np.ndarray | float, a: float) -> np.ndarray:
    g = parabola_fisher(u, a)
    return (4.0 * a * a) / np.power(g, 3.0)


def parabola_eta(u: np.ndarray | float, a: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return np.stack([u, a * u * u], axis=-1)


def mle_loss_ratio(a: float, u0: float, n: int, rng: np.random.Generator, n_rep: int = 400) -> dict:
    """Monte Carlo $\\mathrm{Var}(\\hat u)$ versus CRLB $1/(n g)$."""
    g = float(parabola_fisher(u0, a))
    crlb = 1.0 / (n * g)
    hats = np.empty(n_rep)
    for r in range(n_rep):
        z = rng.normal(size=(n, 2))
        # observations around eta(u0)
        eta = parabola_eta(u0, a)
        x = eta + z
        # MLE: project onto the parabola in Euclidean (mixture) sense
        us = np.linspace(u0 - 2.0, u0 + 2.0, 81)
        etas = parabola_eta(us, a)
        mean = x.mean(axis=0)
        d = np.sum((etas - mean) ** 2, axis=1)
        hats[r] = us[int(np.argmin(d))]
    var = float(np.var(hats, ddof=1))
    return {
        "var": var,
        "crlb": crlb,
        "ratio": var / crlb,
        "gamma2": float(parabola_gamma2(u0, a)),
        "g": g,
    }
