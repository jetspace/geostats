"""Emit notebooks/program.ipynb. Run from the geostats root."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)

SETUP = r"""
import sys
from pathlib import Path
from IPython.display import HTML, display

_src = None
for _root in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
    _cand = _root / "src"
    if (_cand / "geostats").is_dir():
        _src = _cand
        break
if _src is None:
    raise RuntimeError("could not find src/geostats from the notebook working directory")
_src_s = str(_src)
if _src_s in sys.path:
    sys.path.remove(_src_s)
sys.path.insert(0, _src_s)
for _key in list(sys.modules):
    if _key == "geostats" or _key.startswith("geostats."):
        del sys.modules[_key]

display(HTML('''
<style>
:root {
  --jb-bg: #000000;
  --jb-panel: #0A0A0A;
  --jb-fg: #C4C4C4;
  --jb-dim: #6A6A6A;
  --jb-wire: #222222;
  --jb-link: #AE93EC;
  --jb-hover: #E7B597;
  --jb-white: #FFFFFF;
  --jb-warn: #C24A3A;
}
.jp-RenderedHTMLCommon, .jp-RenderedMarkdown, .text_cell_render {
  color: var(--jb-fg) !important;
  font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, Consolas, monospace;
  line-height: 1.55;
  max-width: 54em;
}
.jp-RenderedHTMLCommon h1, .jp-RenderedHTMLCommon h2, .text_cell_render h1, .text_cell_render h2 {
  color: var(--jb-link) !important;
  letter-spacing: 0.05em;
  font-weight: 500;
  border-bottom: 1px solid var(--jb-wire);
  padding-bottom: 0.35em;
}
.jp-RenderedHTMLCommon h3 { color: var(--jb-hover) !important; }
.jp-RenderedHTMLCommon a { color: var(--jb-link) !important; }
.jp-RenderedHTMLCommon em { color: var(--jb-hover); }
</style>
'''))

from geostats.figures import (
    crofton_figure,
    curvature_figure,
    dual_figure,
    dual_potential_figure,
    fisher_figure,
    fisher_surface_figure,
    gaussian_flow_figure,
    lift_figure,
    minkowski_figure,
    minkowski_stack_figure,
    sphere_figure,
)
"""


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip() + "\n")


def viz(call: str):
    return nbf.v4.new_code_cell(call.strip() + "\n")


def main() -> None:
    cells = [
        md(
            r"""
# Geometric Inference, Statistics, and Stochastics

A statistical model is not a table of numbers. It is a space of laws. Distances
in that space say how distinguishable two laws are. Curves say how an estimator,
a test, or a diffusion may move. Geometry is not an ornament on the analysis.
It is the language that remains legal when coordinates change, when a sample is
replaced by a sufficient statistic, and when the state of a process is a point
of a manifold rather than a vector in Euclidean space.

The lecture builds five objects, in this order, and only this order. Each
symbol is defined in the section that first uses it.

1. A convex body $K$, measured by a support function $H(K,\cdot)$.
2. An invariant (Haar) measure on the space of lines, turning size into an
   expectation.
3. A regular parametric family $\mathcal{S}$ of laws, equipped with the Fisher
   metric $g$ and a one-parameter family of connections $\nabla^{(\alpha)}$.
4. An estimator $\hat\theta$ of a curved submodel, whose information loss is
   embedding curvature.
5. A diffusion $X_t=\pi(r_t)$ whose state lies on a manifold $M$, obtained by
   projecting a horizontal motion $r_t$ on the orthonormal frame bundle of $M$.

Play each figure. Pause it. The argument is the motion.
"""
        ),
        viz(SETUP),
        md(
            r"""
## 1. Convex bodies

**Definition.** A set $K\subset\mathbb{R}^n$ is a *convex body* if it is
compact, convex, and has nonempty interior. Convexity means that if
$x,y\in K$ and $t\in[0,1]$, then $(1-t)x+ty\in K$.

**Definition.** The *support function* of $K$ is the function
$H(K,\cdot):\mathbb{R}^n\to\mathbb{R}$ given by

$$
H(K,u)
=
\max\{\,u\cdot x:x\in K\,\}.
$$

Restricted to the unit sphere $S^{n-1}=\{u:\|u\|=1\}$, the number $H(K,u)$ is
the signed distance from the origin to the supporting hyperplane of $K$ with
outward unit normal $u$. Two convex bodies with the same support function
coincide.

**Example (ellipse).** Let

$$
E_{a,b}
=
\bigl\{(x,y)\in\mathbb{R}^2:x^2/a^2+y^2/b^2\le 1\bigr\},
\qquad a,b>0.
$$

If $u=(\cos\vartheta,\sin\vartheta)$, then

$$
H(E_{a,b},u)
=
\sqrt{a^2\cos^2\vartheta+b^2\sin^2\vartheta}.
$$

The disk of radius $r$ is the case $a=b=r$, and $H\equiv r$.

**Definition.** The *Minkowski sum* of convex bodies $K_1,K_2$ is

$$
K_1\oplus K_2
=
\{x+y:x\in K_1,\,y\in K_2\}.
$$

**Identity.** Support functions turn that sum into addition of numbers:

$$
H(K_1\oplus K_2,u)=H(K_1,u)+H(K_2,u).
$$

If $H$ is $C^2$ on $S^1$, the support point with outward normal
$u=(\cos\vartheta,\sin\vartheta)$ is
$x=H u + H_\vartheta u^\perp$, and the area of $K$ is

$$
A(K)
=
\frac12\int_0^{2\pi}\bigl(H^2-H_\vartheta^2\bigr)\,d\vartheta.
$$

**Mixed volumes.** For $\lambda_i\ge 0$, the volume of
$\lambda_1 K_1+\cdots+\lambda_r K_r$ is a homogeneous polynomial of degree $n$
in the $\lambda_i$. The coefficients $V(K_{i_1},\ldots,K_{i_n})$ are the mixed
volumes: symmetric, nonnegative, and monotone in each argument [1].

**Steiner formula (plane).** The parallel set $K_r=K\oplus r B$, $B$ the unit
disk, has $H(K_r,u)=H(K,u)+r$. Expanding area gives

$$
A(K_r)=A(K)+L(K)\,r+\pi r^2,
$$

where $L(K)$ is the perimeter. Cauchy's formula identifies $L$ with mean
width: $L=\int_0^\pi w(\phi)\,d\phi$ and $w(\phi)=H(u)+H(-u)$.

**Brunn-Minkowski.** Write $K_t=(1-t)K_0\oplus t K_1$ for $t\in[0,1]$, and let
$V$ denote $n$-dimensional volume. Then $t\mapsto V(K_t)^{1/n}$ is concave:

$$
V(K_0\oplus K_1)^{1/n}
\ge
V(K_0)^{1/n}+V(K_1)^{1/n}.
$$

Aleksandrov-Fenchel is the same concavity in mixed-volume form:
$V(K_1,K_2,K_3,\ldots)^2\ge V(K_1,K_1,K_3,\ldots)\,V(K_2,K_2,K_3,\ldots)$.
In the plane, $n=2$ and $V$ is area $A$. The figure draws two ellipses $A$ and
$B$, the body $A\oplus t B$ in violet, $H(A\oplus t B)$ in sand, and
$\sqrt{A(A\oplus t B)}$ against the chord that would hold if root-area were
linear.
"""
        ),
        viz("minkowski_figure()"),
        md(
            r"""
The same family $\{A\oplus t B:t\in[0,1]\}$, extruded with height equal to $t$.
A plane drawing hides that Minkowski interpolation is a path in the space of
shapes. The solid makes that path visible.
"""
        ),
        viz("minkowski_stack_figure()"),
        md(
            r"""
## 2. Invariant measure on lines

Part 1 produced numbers attached to a *fixed* set. Geometric probability asks
for the measure of a *set of positions* of a moving figure. The measure must
not depend on an arbitrary origin.

**Definition.** A line $G$ in the plane may be written in *normal coordinates*
$(p,\phi)$, where $\phi\in[0,\pi)$ is the direction of a unit normal and
$p\in\mathbb{R}$ is the signed distance from the origin. Explicitly, $G$ is
the set of $x\in\mathbb{R}^2$ with $x\cdot(\cos\phi,\sin\phi)=p$.

**Definition.** The *invariant line density* (Haar measure on the space of
lines, up to scale) is

$$
dG=dp\,d\phi.
$$

**Crofton's formula.** Let $C$ be a rectifiable curve of length $L$, and let
$N(p,\phi)$ be the number of intersections of $C$ with the line of coordinates
$(p,\phi)$. Then

$$
\int N(p,\phi)\,dp\,d\phi=2L.
$$

**Example (unit circle).** Take $C=S^1$, so $L=2\pi$. A line hits $C$ if and
only if $|p|\le 1$, and then $N=2$. Hence

$$
\int N\,dG
=
\int_0^\pi\int_{-1}^{1} 2\,dp\,d\phi
=
4\pi
=
2L.
$$

For any closed convex curve the same counting holds: $N=2$ on a hitting line
and $N=0$ otherwise, so the measure of hitting lines equals the perimeter.
The figure samples lines from $dG$. Sand strokes are the newest hits; violet
is recent history. The right panel is the running estimator

$$
\hat L_N
=
\bigl(2\,p_{\max}\,\pi\bigr)\cdot\frac1N\sum_{i=1}^N \mathbf{1}_{\{\text{line }i\text{ hits }K\}},
$$

which converges to the perimeter of $K$. A line hits $K$ (origin in the
interior) if and only if $-H(\phi+\pi)\le p\le H(\phi)$.

**Hostinský.** If $\sigma$ is chord length, then
$\int_{G\cap K\neq\emptyset}\sigma^3\,dG=3A(K)^2$. For the unit disk both
Crofton and Hostinský are elementary quadratures; both identities are used as
numerical checks for the kernels.
"""
        ),
        viz("crofton_figure()"),
        md(
            r"""
## 3. A manifold of laws

The same invariance, imposed on families of probability laws rather than on
subsets of the plane, produces a Riemannian manifold.

**Definition.** A *regular parametric family* is a set of densities

$$
\mathcal{S}
=
\{p(\,\cdot\,;\theta):\theta\in\Theta\subset\mathbb{R}^n\}
$$

on a common sample space, with $\Theta$ open, such that the map
$\theta\mapsto p(\,\cdot\,;\theta)$ is injective and the scores below are
square-integrable and linearly independent. A point of $\mathcal{S}$ is a
probability law, not a sample. Write

$$
\ell(x,\theta)=\log p(x;\theta),
\qquad
\partial_i=\frac{\partial}{\partial\theta^i}.
$$

**Definition (Fisher-Rao metric).** Under the usual regularity that permits
differentiation under the integral,

$$
g_{ij}(\theta)
=
\mathrm{E}_\theta[\partial_i\ell\cdot\partial_j\ell]
=
-\mathrm{E}_\theta[\partial_i\partial_j\ell].
$$

Chentsov's theorem states that $g$ is, up to a scalar, the unique Riemannian
metric on $\mathcal{S}$ invariant under sufficient statistics. The unit ball
of $g$ at $\theta$ is the set of infinitesimal displacements $d\theta$ that are
equally hard to detect with one observation.

**Example (univariate Gaussian).** Let $\theta=(\mu,\sigma)$ with $\sigma>0$ and

$$
p(x;\mu,\sigma)
=
\frac{1}{\sqrt{2\pi}\,\sigma}
\exp\Bigl(-\frac{(x-\mu)^2}{2\sigma^2}\Bigr),
\qquad x\in\mathbb{R}.
$$

Then

$$
\ell(x;\mu,\sigma)
=
-\log\sigma-\frac{(x-\mu)^2}{2\sigma^2}-\frac12\log(2\pi),
$$

$$
\partial_\mu\ell=\frac{x-\mu}{\sigma^2},
\qquad
\partial_\sigma\ell=\frac{(x-\mu)^2}{\sigma^3}-\frac1\sigma.
$$

Write $z=(x-\mu)/\sigma\sim\mathcal{N}(0,1)$. Then
$\partial_\mu\ell=z/\sigma$ and $\partial_\sigma\ell=(z^2-1)/\sigma$. The
second-moment computation is

$$
g_{\mu\mu}=\mathrm{E}[z^2]/\sigma^2=\sigma^{-2},
\qquad
g_{\sigma\sigma}=\mathrm{E}[(z^2-1)^2]/\sigma^2=2\sigma^{-2},
\qquad
g_{\mu\sigma}=0,
$$

so

$$
g
=
\begin{pmatrix}
\sigma^{-2} & 0 \\
0 & 2\sigma^{-2}
\end{pmatrix}.
$$

The same matrix is the Hessian of $D_{\mathrm{KL}}(p_{\theta_0}\|p_\theta)$ at
$\theta=\theta_0$. This is a Poincaré half-plane metric, up to the constant
$2$ in the $\sigma$-slot. Ellipses in the next figure are unit balls of $g$.

**Definition ($\alpha$-connections).** For each real number $\alpha$, the
Chentsov-Amari connection $\nabla^{(\alpha)}$ has coefficients

$$
\Gamma_{ijk}^{(\alpha)}
=
\mathrm{E}_\theta
\Bigl[
\bigl(\partial_i\partial_j\ell+\tfrac{1-\alpha}{2}\partial_i\ell\cdot\partial_j\ell\bigr)
\partial_k\ell
\Bigr].
$$

Three values organise the subject. Here $\nabla^{(0)}$ is the Levi-Civita
connection of $g$ (metric geodesics). The value $\alpha=+1$ is the
*exponential* connection. The value $\alpha=-1$ is the *mixture* connection.
They are conjugate with respect to $g$:

$$
X\,g(Y,Z)
=
g\bigl(\nabla_X^{(\alpha)}Y,Z\bigr)
+
g\bigl(Y,\nabla_X^{(-\alpha)}Z\bigr).
$$

Equivalently $\Gamma^{(\alpha)}_{ijk}=\Gamma^{(0)}_{ijk}-(\alpha/2)T_{ijk}$ with
$T_{ijk}=\mathrm{E}[\partial_i\ell\,\partial_j\ell\,\partial_k\ell]$. On the
Gaussian chart the surviving components are $T_{\mu\mu\sigma}=2\sigma^{-3}$ and
$T_{\sigma\sigma\sigma}=8\sigma^{-3}$, so

$$
\Gamma_{\mu\mu\sigma}^{(\alpha)}=\frac{1-\alpha}{\sigma^3},
\qquad
\Gamma_{\mu\sigma\mu}^{(\alpha)}=\frac{-1-\alpha}{\sigma^3},
\qquad
\Gamma_{\sigma\sigma\sigma}^{(\alpha)}=\frac{-2-4\alpha}{\sigma^3}.
$$

In particular $\Gamma^{(+1)}+\Gamma^{(-1)}=2\Gamma^{(0)}$. A geodesic
$\theta(t)$ of $\nabla^{(\alpha)}$ solves
$\ddot\theta^k+\Gamma^k_{ij}(\theta)\dot\theta^i\dot\theta^j=0$.

The three curves below join the same pair of Gaussian laws
$(\mu,\sigma)=(0,1.05)$ and $(1.35,0.58)$. Violet is $\alpha=0$, sand is
$\alpha=+1$, oxide is $\alpha=-1$.
"""
        ),
        viz("fisher_figure()"),
        md(
            r"""
Those ellipses are slices of a three-dimensional geometry. Because
$g=\sigma^{-2}\,\mathrm{diag}(1,2)$, the conformal factor of the metric is
proportional to $\sigma^{-1}$. The next figure graphs the surface
$(\mu,\sigma)\mapsto 1/\sigma$ and lifts the three geodesics onto it. Distances
along the surface are Fisher distances. This is the precise content of the
sentence "the Gaussian family is hyperbolic."
"""
        ),
        viz("fisher_surface_figure()"),
        md(
            r"""
A geodesic $\theta(t)$ on $\mathcal{S}$ is a path of laws. For the Gaussian
family that path is a pair $(\mu(t),\sigma(t))$, and the law sampled at time
$t$ is $p(\,\cdot\,;\mu(t),\sigma(t))$. The next figure extrudes that density
in the $t$-direction. The sand ridge is the present density. Mean and scale
move together because they are coordinates of a single curve on
$(\mathcal{S},g)$.
"""
        ),
        viz("gaussian_flow_figure()"),
        md(
            r"""
## 4. Dual flatness

**Definition.** A *full exponential family* on a sample space with sufficient
statistic $T$ has densities of the form

$$
p(x;\theta)
=
\exp\bigl(\theta\cdot T(x)-\psi(\theta)\bigr)\,h(x),
$$

where $\theta$ ranges over an open convex set on which the *log-partition*
$\psi(\theta)=\log\int e^{\theta\cdot T}h$ is finite. In the coordinates
$\theta$ (called *natural* or *canonical*), one has $\Gamma^{(+1)}\equiv 0$:
the manifold is exponentially flat. The *expectation coordinates*

$$
\eta
=
\nabla\psi(\theta)
=
\mathrm{E}_\theta[T]
$$

flatten the mixture connection: $\Gamma^{(-1)}\equiv 0$ in $\eta$. The two
charts are dual. If $\phi$ denotes the convex conjugate of $\psi$, the
Legendre identity

$$
\psi(\theta)+\phi(\eta)-\theta\cdot\eta=0
$$

holds on the dual pairing $\eta=\nabla\psi(\theta)$, and
$g_{ij}=\partial_i\partial_j\psi$. Kullback-Leibler divergence is the Bregman
divergence of $\psi$,

$$
D_{\mathrm{KL}}(p_\theta\|p_{\theta'})
=
\psi(\theta)-\psi(\theta')-(\theta-\theta')\cdot\eta',
$$

which is the $\alpha=-1$ divergence. If an $\alpha$-geodesic from $P$ to $Q$
meets a $(-\alpha)$-geodesic from $Q$ to $R$ orthogonally at $Q$, then
$D(P,Q)+D(Q,R)=D(P,R)$. Maximum likelihood, for a regular curved exponential
family, is that orthogonal projection along a dual geodesic.

**Example (Gaussian, continued).** The univariate Gaussian is a full
exponential family with $T(x)=(x,x^2)$. The natural and expectation
coordinates are

$$
\theta
=
\Bigl(\frac{\mu}{\sigma^2},\,-\frac{1}{2\sigma^2}\Bigr),
\qquad
\eta
=
(\mu,\,\mu^2+\sigma^2),
$$

and the log-partition, writing $\theta=(\theta_1,\theta_2)$ with $\theta_2<0$, is

$$
\psi(\theta_1,\theta_2)
=
-\frac12\log(-2\theta_2)-\frac{\theta_1^2}{4\theta_2}.
$$

Consequently: a curve that is a Euclidean straight line in $\theta$ is an
$\alpha=+1$ geodesic; a Euclidean straight line in $\eta$ is an $\alpha=-1$
geodesic. In the coordinates $(\mu,\sigma)$ both are curved. The next board
draws the *same* two laws in all three charts at once.
"""
        ),
        viz("dual_figure()"),
        md(
            r"""
Graph the convex potential $\psi$ over the $\theta$-plane. An exponential
geodesic is then a Euclidean straight line in the base, riding on the graph of
$\psi$. A mixture geodesic, written in the same chart, remains curved. Dual
flatness is exactly this: two affine structures, conjugate with respect to
$g$, each of which straightens one family of geodesics.
"""
        ),
        viz("dual_potential_figure()"),
        md(
            r"""
## 5. Curved exponential families

**Definition.** Let $S$ be a full exponential family, hence $1$-flat and
$(-1)$-flat. A *curved exponential family* is a smooth submanifold
$M\subset S$ that is not itself a full exponential family. Write
$H^{(e)}$ for the Euler-Schouten embedding curvature of $M$ in $S$ with
respect to $\nabla^{(+1)}$, and $H_A^{(m)}$ for the embedding curvature of
the ancillary foliation $A$ of an estimator, computed in $\nabla^{(-1)}$.

**Information loss.** If $u$ is an efficient estimator of the parameter of
$M$ based on $N$ i.i.d. observations, the Fisher information lost by retaining
$u$ instead of the full sample admits the expansion

$$
\Delta g
=
(H^{(e)})^2
+
\tfrac12 (H_A^{(m)})^2
+
O(N^{-1}).
$$

The first term is intrinsic to $M$: it cannot be removed by a clever
estimator. The maximum likelihood estimator is characterised by an ancillary
family orthogonal in the mixture connection, which forces $H_A^{(m)}=0$. What
remains is Efron's *statistical curvature* $\gamma$, built from $H^{(e)}$.

**Example (parabola in a location family).** Let the ambient family $S$ be
the two-dimensional Gaussian location family
$X\sim\mathcal{N}(\eta,I_2)$ on $\mathbb{R}^2$, which is flat. Let $M$ be the
curve

$$
\eta(u)=(u,\,a u^2),
\qquad u\in\mathbb{R},
$$

for a fixed $a>0$. The tangent is $\eta'(u)=(1,2au)$, so the induced Fisher
information (ambient metric Euclidean) is
$g(u)=\|\eta'(u)\|^2=1+4a^2 u^2$. The second derivative is
$\eta''(u)=(0,2a)$. Efron curvature is the squared geodesic curvature of this
embedding,

$$
\gamma^2(u)
=
\frac{\|\eta'\wedge\eta''\|^2}{\|\eta'\|^6}
=
\frac{4a^2}{(1+4a^2 u^2)^3}.
$$

In the figure the horizontal plane is $S$, the violet curve is $M$, and height
is $\gamma^2(u)$. The sand cloud is a sample from $\mathcal{N}(\eta(u),I_2)$.
Where the ridge is high, $M$ bends hardest in the exponential connection, and
the information that the maximum likelihood estimator cannot recover is
largest.
"""
        ),
        viz("curvature_figure()"),
        md(
            r"""
## 6. Diffusions whose state lies on a manifold

Until now the manifold was a family of laws, and the sample was Euclidean. If
the *state* $X_t$ of a process takes values in a smooth manifold $M$, a
diffusion cannot be defined by writing Itô's formula in one chart and hoping
the chain rule survives a change of coordinates.

**Itô and Stratonovich.** Let $B_t$ be Brownian motion in $\mathbb{R}^d$. The
Itô integral $\int\Phi\,dB$ evaluates the integrand at the left endpoint of
each increment and is a martingale. The Stratonovich integral
$\int\Phi\circ dB$ evaluates at the midpoint and obeys the Newton chain rule.
On a manifold, legality means invariance under diffeomorphisms of charts, so
the equation is written in Stratonovich form.

**Definition.** Let $A_0,A_1,\ldots,A_r$ be smooth vector fields on $M$. A
continuous $M$-valued semimartingale $X$ solves

$$
dX_t
=
\sum_{k=1}^r A_k(X_t)\circ dB^k_t
+
A_0(X_t)\,dt
$$

if, for every $f\in C^\infty(M)$,

$$
f(X_t)-f(X_0)
=
\sum_{k=1}^r\int_0^t (A_k f)(X_s)\circ dB^k_s
+\int_0^t (A_0 f)(X_s)\,ds.
$$

**Horizontal Brownian motion.** Let $(M,g)$ be Riemannian of dimension $d$,
and let $O(M)$ be its orthonormal frame bundle: a point $r=(x,e)\in O(M)$ is a
point $x\in M$ together with an orthonormal basis $e$ of $T_x M$. The
Levi-Civita connection splits $T_r O(M)$ into vertical vectors (which rotate
the frame) and horizontal vectors (which move $x$ by parallel transport of
$e$). Canonical horizontal fields $L_1,\ldots,L_d$ on $O(M)$ implement rolling
without slipping. If

$$
dr_t=\sum_{i=1}^d L_i(r_t)\circ dB^i_t,
$$

the projection $X_t=\pi(r_t)$ onto $M$ is *Riemannian Brownian motion*. Its
generator is $\tfrac12\Delta_M$, where $\Delta_M$ is the Laplace-Beltrami
operator of $g$.

The conversion between the two integrals, for an $\mathbb{R}^n$-valued
semimartingale, is
$Y\circ dX = Y\,dX + \tfrac12 d\langle Y,X\rangle$. The extra quadratic term
is why Itô's formula has a Laplacian and Stratonovich does not.

**Example (the sphere).** Let $M=S^2=\{x\in\mathbb{R}^3:\|x\|=1\}$, with the
round metric. One step of the projected scheme is
$x\leftarrow \Pi\bigl(x+\sqrt{dt}\,P_x W\bigr)$, where $P_x=I-xx^\top$ and
$\Pi$ is radial projection. For small $dt$ the radial correction contributes
drift $-x\,dt$. The generator on ambient functions, restricted to $S^2$, is
therefore $\tfrac12\Delta_{S^2}$. On the coordinate $z(x)=x_3$ one has
$\Delta_{S^2}z=-2z$, so $\tfrac12\Delta_{S^2}z=-z$ and

$$
\mathrm{E}[z(X_t)]=z(X_0)\,e^{-t}.
$$

The right panel of the next figure is that identity, read from an ensemble of
paths. The camera orbits because a sphere drawn from one angle is a disk.
"""
        ),
        viz("sphere_figure()"),
        md(
            r"""
A point $x\in S^2$ does not determine a roll. The horizontal lift carries an
orthonormal frame $(e_1,e_2)$ of $T_x S^2$, transported as the path moves.
Sand and oxide in the last figure are that frame; the dim radius is $x$
itself. Brownian motion on a manifold is not a random walk in coordinates. It
is a roll without slip, then a projection $\pi:O(M)\to M$.
"""
        ),
        viz("lift_figure()"),
        md(
            r"""
## What has been defined

A convex body is measured by $H$. Minkowski addition is addition of $H$.
Brunn-Minkowski is concavity of $V^{1/n}$. Crofton converts length into an
integral against $dp\,d\phi$. A regular family $\mathcal{S}$ carries the
Fisher metric $g$ and the connections $\nabla^{(\alpha)}$. On a full
exponential family, $\theta$ flattens $\alpha=+1$ and $\eta$ flattens
$\alpha=-1$, related by a Legendre transform of $\psi$. Embedding curvature of
a curved submodel is information the maximum likelihood estimator cannot
recover. Riemannian Brownian motion is $X=\pi(r)$ for horizontal $r$ on
$O(M)$, with generator $\tfrac12\Delta_M$.

The primary texts are Bonnesen and Fenchel [1], Santaló [2], Amari with
Barndorff-Nielsen, Kass, Lauritzen, and Rao [3], Amari [4], and Ikeda and
Watanabe [5]. This lecture is an exposition of an architecture those books
already contain. It is not a substitute for them.
"""
        ),
    ]
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = cells
    path = OUT / "program.ipynb"
    nbf.write(nb, path)
    print("wrote", path)


if __name__ == "__main__":
    main()
