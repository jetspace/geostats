# geostats

[Jetspace Research](https://github.com/orgs/jetspace/repositories)

Convex geometry, geometric probability, information geometry, and stochastic
analysis on manifolds sit in one line of argument. Size is written as a
support function. Those invariants become expectations once they are
integrated against Haar measure on lines and rigid motions. The same demand
for invariance, imposed on a family of probability laws, is the Fisher metric
and the $\alpha$-connections. When the model is a curved submanifold of an
exponential family, embedding curvature is information that no estimator
recovers. When the state of a process is a point of a manifold, Brownian
motion is a horizontal lift, not an Itô equation in a single chart.

This repository is the start of a research program at that intersection:
definitions and identities first, then numerics that are required to preserve
them. The working board is
[`notebooks/program.ipynb`](notebooks/program.ipynb). What follows is meant
to be extended: mixed volumes in higher dimension, kinematic formulae,
observed geometry, and affine-invariant diffusions on $\mathrm{SPD}$ are
natural next identities, not a closed list.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[notebooks]"
jupyter lab notebooks/program.ipynb
```

Use the kernel from that virtualenv. NumPy, SciPy, and Plotly. Formulae in
`$...$` and `$$...$$`.

## Layout

```
LICENSE                 Jetspace Research License 1.0
NOTICE
REFERENCES.md           primary texts [1]-[5]
src/geostats/           kernels
notebooks/program.ipynb working board
notebooks/_emit.py      regenerates the notebook
tests/                  identity checks
```

## Numerics

A planar convex body is a support function $H$ on an angular grid. Minkowski
addition is pointwise addition of $H$. Area is
$A=\tfrac12\int(H^2-H_\vartheta^2)\,d\vartheta$; perimeter is Cauchy width.
Random lines are drawn from $dp\,d\phi$.

The univariate Gaussian is closed form:
$g=\mathrm{diag}(\sigma^{-2},2\sigma^{-2})$, $\alpha$-Christoffel symbols from
the third-moment tensor of the score, geodesics by RK4. Dual coordinates are
$\theta=(\mu/\sigma^2,-1/(2\sigma^2))$ and $\eta=(\mu,\mu^2+\sigma^2)$.
Brownian motion on $S^2$ is a tangent increment followed by radial
projection. Figures are Plotly `go.Frame` animations with Play / Pause.

```bash
pip install -e ".[dev,notebooks]"
python notebooks/_emit.py
pytest -q
```

## Sources

Bonnesen and Fenchel [1], Santaló [2], Amari et al. [3], Amari [4], Ikeda
and Watanabe [5]. See [`REFERENCES.md`](REFERENCES.md). Those books are not
redistributed here.

## License

[Jetspace Research License 1.0](LICENSE). Study, teaching, and
non-commercial research are free with attribution. Commercial or production
use needs written permission from Jetspace Research. Cite [1]-[5]; this
repository does not own those texts.

Cite as Jetspace Research. See [`CITATION.cff`](CITATION.cff).

Support:
[buymeacoffee.com/jetbundle](https://buymeacoffee.com/jetbundle).
