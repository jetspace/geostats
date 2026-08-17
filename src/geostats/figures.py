"""Plotly boards: play/pause frames, jetbundle-noir.

Every animation has a constant trace count, a Play/Pause pair, and a slider.
Motion is slow enough to read. New geometry appears a few strokes at a time.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LINK = "#AE93EC"
HOVER = "#E7B597"
WARN = "#C24A3A"
FG = "#C4C4C4"
BG = "#000000"
WIRE = "#222222"
DIM = "#6A6A6A"
FILL = "rgba(174,147,236,0.16)"
FILL_SAND = "rgba(231,181,151,0.14)"

__all__ = [
    "crofton_figure",
    "curvature_figure",
    "dual_figure",
    "dual_potential_figure",
    "fisher_figure",
    "fisher_surface_figure",
    "gaussian_flow_figure",
    "lift_figure",
    "minkowski_figure",
    "minkowski_stack_figure",
    "sphere_figure",
]

LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=FG, family="DejaVu Sans Mono, monospace", size=13),
    margin=dict(l=40, r=24, t=96, b=56),
    height=720,
)


def play_pause(frames: list[go.Frame], duration: int = 110, prefix: str = "") -> dict:
    n = len(frames)
    every = max(1, n // 7)
    steps = []
    for i, fr in enumerate(frames):
        lab = fr.name if (i % every == 0 or i == n - 1) else ""
        steps.append(
            {
                "args": [[fr.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": lab,
                "method": "animate",
            }
        )
    return dict(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "pad": {"r": 10, "t": 72},
                "x": 0.04,
                "y": 1.14,
                "bgcolor": "#0A0A0A",
                "bordercolor": WIRE,
                "font": {"color": FG},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": duration, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 40},
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": prefix,
                    "visible": True,
                    "font": {"color": HOVER, "size": 12},
                },
                "pad": {"t": 18, "b": 8},
                "x": 0.04,
                "len": 0.92,
                "bgcolor": "#0A0A0A",
                "bordercolor": WIRE,
                "tickcolor": DIM,
                "font": {"color": DIM, "size": 10},
                "steps": steps,
            }
        ],
    )


def _xy_style(fig: go.Figure) -> None:
    fig.update_xaxes(
        gridcolor=WIRE, zerolinecolor=WIRE, linecolor=WIRE, tickfont=dict(color=DIM), color=DIM
    )
    fig.update_yaxes(
        gridcolor=WIRE, zerolinecolor=WIRE, linecolor=WIRE, tickfont=dict(color=DIM), color=DIM
    )


def _scene(eye=None) -> dict:
    if eye is None:
        eye = dict(x=1.55, y=1.45, z=1.05)
    hide = dict(
        backgroundcolor=BG,
        gridcolor=WIRE,
        showbackground=False,
        showgrid=True,
        zeroline=False,
        color=DIM,
        tickfont=dict(color=DIM, size=10),
    )
    return dict(
        bgcolor=BG,
        xaxis=hide,
        yaxis=hide,
        zaxis=hide,
        aspectmode="cube",
        camera=dict(eye=eye),
    )


def _orbit_eye(k: int, n: int, radius: float = 1.85, z: float = 0.95) -> dict:
    a = 0.45 + 1.25 * k / max(n - 1, 1)
    return dict(x=radius * np.cos(a), y=radius * np.sin(a), z=z)


def _finish(fig: go.Figure, frames: list[go.Frame], title: str, duration: int, prefix: str = "", **kw):
    fig.frames = frames
    fig.update_layout(
        **LAYOUT,
        **play_pause(frames, duration=duration, prefix=prefix),
        title=dict(text=title, font=dict(color="#FFFFFF", size=15)),
        **kw,
    )
    return fig


def _gauss_pdf(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2.0 * np.pi))


def _psi(th1: np.ndarray, th2: np.ndarray) -> np.ndarray:
    return -0.5 * np.log(np.maximum(-2.0 * th2, 1e-12)) - (th1 * th1) / (4.0 * th2)


def minkowski_figure(n_frames: int = 64) -> go.Figure:
    from geostats.convex import (
        angle_grid,
        brunn_minkowski_curve,
        close_loop,
        ellipse_support,
        support_points,
    )

    theta = angle_grid(280)
    h_a = ellipse_support(1.4, 0.7, theta)
    h_b = ellipse_support(0.55, 1.15, theta)
    h_sum = h_a + h_b
    pts_a = close_loop(support_points(h_a, theta))
    pts_b = close_loop(support_points(h_b, theta))
    bm = brunn_minkowski_curve(h_a, h_b, theta, n_t=n_frames)

    def body_at(t: float):
        h = (1.0 - t) * h_a + t * h_sum
        return close_loop(support_points(h, theta)), h

    frames = []
    for i, t in enumerate(bm["t"]):
        body, h = body_at(t)
        frames.append(
            go.Frame(
                name=f"{t:.2f}",
                traces=[2, 3, 4],
                data=[
                    go.Scatter(
                        x=body[:, 0],
                        y=body[:, 1],
                        mode="lines",
                        fill="toself",
                        fillcolor=FILL,
                        line=dict(color=LINK, width=3),
                    ),
                    go.Scatter(x=theta, y=h, mode="lines", line=dict(color=HOVER, width=2.4)),
                    go.Scatter(
                        x=bm["t"][: i + 1],
                        y=bm["root_area"][: i + 1],
                        mode="lines",
                        line=dict(color=LINK, width=2.4),
                    ),
                ],
            )
        )
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        subplot_titles=("Minkowski sum in the plane", "support function H", "root area stays above the chord"),
        column_widths=[0.56, 0.44],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    body0, h0 = body_at(0.0)
    fig.add_trace(
        go.Scatter(x=pts_a[:, 0], y=pts_a[:, 1], mode="lines", line=dict(color=DIM, width=1.4), name="A"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=pts_b[:, 0], y=pts_b[:, 1], mode="lines", line=dict(color=DIM, width=1.4, dash="dot"), name="B"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=body0[:, 0],
            y=body0[:, 1],
            mode="lines",
            fill="toself",
            fillcolor=FILL,
            line=dict(color=LINK, width=3),
            name="A ⊕ t B",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=theta, y=h0, mode="lines", line=dict(color=HOVER, width=2.4), name="H"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=[0], y=[bm["root_area"][0]], mode="lines", line=dict(color=LINK, width=2.4), name="root A"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=bm["t"], y=bm["linear"], mode="lines", line=dict(color=WARN, width=1.2, dash="dot"), name="linear"),
        row=2,
        col=2,
    )
    _finish(
        fig,
        frames,
        "Adding bodies is adding support functions. Root area is concave along the sum.",
        duration=95,
        prefix="t = ",
        showlegend=False,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    _xy_style(fig)
    fig.update_xaxes(title_text="direction", row=1, col=2)
    fig.update_xaxes(title_text="t", row=2, col=2)
    fig.update_yaxes(title_text="H", row=1, col=2)
    fig.update_yaxes(title_text="sqrt area", row=2, col=2)
    return fig


def minkowski_stack_figure(n_frames: int = 40) -> go.Figure:
    """The morph A ⊕ t B extruded as a solid in time."""
    from geostats.convex import angle_grid, close_loop, ellipse_support, support_points

    theta = angle_grid(180)
    h_a = ellipse_support(1.35, 0.68, theta)
    h_b = ellipse_support(0.5, 1.05, theta)
    ts = np.linspace(0.0, 1.0, n_frames)
    slices = []
    for t in ts:
        h = h_a + t * h_b
        pts = close_loop(support_points(h, theta))
        slices.append(pts)

    def stack_xyz(upto: int):
        xs, ys, zs = [], [], []
        for i in range(upto + 1):
            xs.extend(list(slices[i][:, 0]) + [None])
            ys.extend(list(slices[i][:, 1]) + [None])
            zs.extend(list(np.full(len(slices[i]), ts[i])) + [None])
        return xs, ys, zs

    frames = []
    for k in range(n_frames):
        xs, ys, zs = stack_xyz(k)
        rim = slices[k]
        frames.append(
            go.Frame(
                name=f"{ts[k]:.2f}",
                traces=[0, 1],
                data=[
                    go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=LINK, width=2)),
                    go.Scatter3d(
                        x=rim[:, 0],
                        y=rim[:, 1],
                        z=np.full(len(rim), ts[k]),
                        mode="lines",
                        line=dict(color=HOVER, width=6),
                    ),
                ],
                layout=go.Layout(scene_camera=dict(eye=_orbit_eye(k, n_frames, radius=1.7, z=1.15))),
            )
        )
    xs, ys, zs = stack_xyz(0)
    rim = slices[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=LINK, width=2), name="history"))
    fig.add_trace(
        go.Scatter3d(
            x=rim[:, 0],
            y=rim[:, 1],
            z=np.full(len(rim), 0.0),
            mode="lines",
            line=dict(color=HOVER, width=6),
            name="present",
        )
    )
    _finish(
        fig,
        frames,
        "The same sum, read as a solid in time. Height is the Minkowski parameter t.",
        duration=120,
        prefix="t = ",
        scene=_scene(),
        showlegend=False,
    )
    return fig


def _lines_xy(segs: np.ndarray, idx: np.ndarray) -> tuple[list, list]:
    xs, ys = [], []
    for j in np.asarray(idx, dtype=int).ravel():
        xs.extend([segs[j, 0, 0], segs[j, 1, 0], None])
        ys.extend([segs[j, 0, 1], segs[j, 1, 1], None])
    if not xs:
        return [None], [None]
    return xs, ys


def crofton_figure(n_frames: int = 72, n_lines: int = 216, seed: int = 0) -> go.Figure:
    from geostats.convex import angle_grid, ellipse_support
    from geostats.integral import body_outline, crofton_running

    theta = angle_grid(240)
    h = ellipse_support(1.55, 0.88, theta)
    outline = body_outline(h, theta)
    rng = np.random.default_rng(seed)
    run = crofton_running(rng, h, theta, n=n_lines)
    segs = run["segments"]
    hit = run["hit"]

    counts = np.unique(np.linspace(1, n_lines, n_frames, dtype=int))
    frames = []
    for nshow in counts:
        shown = np.arange(nshow)
        h_show = shown[hit[:nshow]]
        m_show = shown[~hit[:nshow]]
        # keep the picture readable: faint history, a few fresh strokes
        h_old = h_show[:-2][-48:] if h_show.size > 2 else h_show
        h_new = h_show[-2:]
        m_old = m_show[-24:]
        hx, hy = _lines_xy(segs, h_old)
        mx, my = _lines_xy(segs, m_old)
        nx, ny = _lines_xy(segs, h_new)
        frames.append(
            go.Frame(
                name=str(int(nshow)),
                traces=[1, 2, 3, 4],
                data=[
                    go.Scatter(x=hx, y=hy, mode="lines", line=dict(color=LINK, width=1.2)),
                    go.Scatter(x=mx, y=my, mode="lines", line=dict(color="#2A2A2A", width=1)),
                    go.Scatter(x=nx, y=ny, mode="lines", line=dict(color=HOVER, width=3.2)),
                    go.Scatter(
                        x=np.arange(1, nshow + 1),
                        y=run["l_hat"][:nshow],
                        mode="lines",
                        line=dict(color=LINK, width=2.4, shape="spline"),
                    ),
                ],
            )
        )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("invariant lines meeting a convex body", "running Crofton estimate of perimeter"),
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=outline[:, 0],
            y=outline[:, 1],
            mode="lines",
            fill="toself",
            fillcolor=FILL_SAND,
            line=dict(color=HOVER, width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=LINK, width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#2A2A2A", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=HOVER, width=3.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1], y=[run["l_hat"][0]], mode="lines", line=dict(color=LINK, width=2.4)), row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=[1, n_lines],
            y=[run["l_true"], run["l_true"]],
            mode="lines",
            line=dict(color=WARN, width=1.3, dash="dot"),
        ),
        row=1,
        col=2,
    )
    _finish(
        fig,
        frames,
        "Each new line is one sample from Haar measure on lines. The average of hits is the perimeter.",
        duration=160,
        prefix="lines  ",
        showlegend=False,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    _xy_style(fig)
    fig.update_xaxes(range=[-3.2, 3.2], row=1, col=1)
    fig.update_yaxes(range=[-2.4, 2.4], row=1, col=1)
    fig.update_xaxes(title_text="samples", row=1, col=2)
    fig.update_yaxes(title_text="length", row=1, col=2)
    return fig


def fisher_figure(n_frames: int = 56) -> go.Figure:
    from geostats.statistical import dual_straight, fisher, geodesic_connect

    mu_g = np.linspace(-2.0, 2.0, 9)
    sig_g = np.linspace(0.42, 2.15, 7)
    ex, ey = [], []
    t = np.linspace(0.0, 2.0 * np.pi, 48)
    circ = np.stack([np.cos(t), np.sin(t)])
    for m in mu_g:
        for s in sig_g:
            g = fisher(m, s)
            w, v = np.linalg.eigh(g)
            shape = (v * (0.15 / np.sqrt(w))) @ circ
            ex.extend(list(m + shape[0]) + [None])
            ey.extend(list(s + shape[1]) + [None])
    start, end = (0.0, 1.05), (1.35, 0.58)
    geo0 = geodesic_connect(start, end, alpha=0.0, n_steps=n_frames)
    exp = dual_straight(start, end, "exp", n=n_frames)
    mix = dual_straight(start, end, "mix", n=n_frames)
    frames = []
    for i in range(n_frames):
        frames.append(
            go.Frame(
                name=str(i),
                traces=[1, 2, 3, 4],
                data=[
                    go.Scatter(x=geo0[: i + 1, 0], y=geo0[: i + 1, 1], mode="lines", line=dict(color=LINK, width=3.2)),
                    go.Scatter(x=exp[: i + 1, 0], y=exp[: i + 1, 1], mode="lines", line=dict(color=HOVER, width=2.4)),
                    go.Scatter(x=mix[: i + 1, 0], y=mix[: i + 1, 1], mode="lines", line=dict(color=WARN, width=2.4)),
                    go.Scatter(
                        x=[geo0[i, 0], exp[i, 0], mix[i, 0]],
                        y=[geo0[i, 1], exp[i, 1], mix[i, 1]],
                        mode="markers",
                        marker=dict(size=8, color=[LINK, HOVER, WARN]),
                    ),
                ],
            )
        )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#333333", width=1), name="unit balls of Fisher"))
    fig.add_trace(go.Scatter(x=geo0[:1, 0], y=geo0[:1, 1], mode="lines", line=dict(color=LINK, width=3.2), name="Levi-Civita"))
    fig.add_trace(go.Scatter(x=exp[:1, 0], y=exp[:1, 1], mode="lines", line=dict(color=HOVER, width=2.4), name="exponential"))
    fig.add_trace(go.Scatter(x=mix[:1, 0], y=mix[:1, 1], mode="lines", line=dict(color=WARN, width=2.4), name="mixture"))
    fig.add_trace(
        go.Scatter(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            mode="markers",
            marker=dict(color=FG, size=10),
            name="laws",
        )
    )
    _finish(
        fig,
        frames,
        "Each ellipse is the set of nearby Gaussians that are equally hard to tell apart. The three curves are three notions of straight.",
        duration=90,
        prefix="step  ",
        xaxis_title="mean μ",
        yaxis_title="scale σ",
        legend=dict(bgcolor="#0A0A0A", bordercolor=WIRE, font=dict(color=FG), x=0.72, y=0.98),
        yaxis=dict(range=[0.28, 2.35]),
        xaxis=dict(range=[-2.2, 2.2]),
    )
    _xy_style(fig)
    return fig


def fisher_surface_figure(n_frames: int = 48) -> go.Figure:
    """Conformal factor 1/σ as a surface. Geodesics live on that geometry."""
    from geostats.statistical import dual_straight, geodesic_connect

    mu = np.linspace(-2.0, 2.0, 42)
    sig = np.linspace(0.38, 2.2, 36)
    MU, SIG = np.meshgrid(mu, sig)
    Z = 1.0 / SIG
    start, end = (0.0, 1.05), (1.35, 0.58)
    geo0 = geodesic_connect(start, end, alpha=0.0, n_steps=n_frames)
    exp = dual_straight(start, end, "exp", n=n_frames)
    mix = dual_straight(start, end, "mix", n=n_frames)

    def lift(path, i):
        p = path[: i + 1]
        return p[:, 0], p[:, 1], 1.0 / p[:, 1]

    frames = []
    for i in range(n_frames):
        x0, y0, z0 = lift(geo0, i)
        x1, y1, z1 = lift(exp, i)
        x2, y2, z2 = lift(mix, i)
        frames.append(
            go.Frame(
                name=str(i),
                traces=[1, 2, 3],
                data=[
                    go.Scatter3d(x=x0, y=y0, z=z0, mode="lines", line=dict(color=LINK, width=8)),
                    go.Scatter3d(x=x1, y=y1, z=z1, mode="lines", line=dict(color=HOVER, width=6)),
                    go.Scatter3d(x=x2, y=y2, z=z2, mode="lines", line=dict(color=WARN, width=6)),
                ],
                layout=go.Layout(scene_camera=dict(eye=_orbit_eye(i, n_frames, radius=1.8, z=1.2))),
            )
        )
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=MU,
            y=SIG,
            z=Z,
            colorscale=[[0, "#0A0A0A"], [0.55, "#3D2E6B"], [1, "#AE93EC"]],
            showscale=False,
            opacity=0.88,
            name="1/σ",
        )
    )
    x0, y0, z0 = lift(geo0, 0)
    x1, y1, z1 = lift(exp, 0)
    x2, y2, z2 = lift(mix, 0)
    fig.add_trace(go.Scatter3d(x=x0, y=y0, z=z0, mode="lines", line=dict(color=LINK, width=8), name="Levi-Civita"))
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode="lines", line=dict(color=HOVER, width=6), name="exponential"))
    fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode="lines", line=dict(color=WARN, width=6), name="mixture"))
    _finish(
        fig,
        frames,
        "Height is 1/σ, the conformal factor of Fisher-Rao. Distances shrink as scale grows because wide Gaussians look alike.",
        duration=100,
        prefix="step  ",
        scene=dict(
            **_scene(),
            xaxis_title="μ",
            yaxis_title="σ",
            zaxis_title="1/σ",
        ),
        legend=dict(bgcolor="#0A0A0A", font=dict(color=FG)),
    )
    return fig


def gaussian_flow_figure(n_frames: int = 48) -> go.Figure:
    """What a geodesic does to the law you actually sample."""
    from geostats.statistical import geodesic_connect

    start, end = (0.0, 1.05), (1.35, 0.58)
    geo0 = geodesic_connect(start, end, alpha=0.0, n_steps=n_frames)
    xs = np.linspace(-4.2, 4.6, 80)
    ts = np.linspace(0.0, 1.0, n_frames)
    dens = np.stack([_gauss_pdf(xs, float(m), float(s)) for m, s in geo0])
    frames = []
    for i in range(n_frames):
        yi = ts[: i + 1]
        zi = dens[: i + 1]
        if i == 0:
            yi = np.array([0.0, 1e-3])
            zi = np.vstack([dens[0], dens[0]])
        frames.append(
            go.Frame(
                name=str(i),
                traces=[0, 1],
                data=[
                    go.Surface(
                        x=xs,
                        y=yi,
                        z=zi,
                        colorscale=[[0, "#000000"], [0.4, "#3D2E6B"], [1, "#E7B597"]],
                        showscale=False,
                        opacity=0.95,
                    ),
                    go.Scatter3d(
                        x=xs,
                        y=np.full_like(xs, ts[i]),
                        z=dens[i],
                        mode="lines",
                        line=dict(color=HOVER, width=6),
                    ),
                ],
            )
        )
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=xs,
            y=np.array([0.0, 1e-3]),
            z=np.vstack([dens[0], dens[0]]),
            colorscale=[[0, "#000000"], [0.4, "#3D2E6B"], [1, "#E7B597"]],
            showscale=False,
            opacity=0.95,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=np.full_like(xs, 0.0),
            z=dens[0],
            mode="lines",
            line=dict(color=HOVER, width=6),
            name="present law",
        )
    )
    _finish(
        fig,
        frames,
        "A geodesic is not an abstract curve. It is a continuous morph of the density you sample from.",
        duration=110,
        prefix="t = ",
        scene=dict(
            **_scene(eye=dict(x=1.7, y=-1.55, z=0.9)),
            xaxis_title="sample x",
            yaxis_title="time along geodesic",
            zaxis_title="density",
        ),
        showlegend=False,
    )
    return fig


def dual_figure(n_frames: int = 52) -> go.Figure:
    from geostats.statistical import dual_straight, geodesic_connect, to_expectation, to_natural

    start, end = (0.0, 1.05), (1.2, 0.62)
    geo0 = geodesic_connect(start, end, alpha=0.0, n_steps=n_frames)
    exp = dual_straight(start, end, "exp", n=n_frames)
    mix = dual_straight(start, end, "mix", n=n_frames)
    paths = {"0": geo0, "+1": exp, "-1": mix}
    colors = {"0": LINK, "+1": HOVER, "-1": WARN}
    th = {k: np.stack([to_natural(float(m), float(s)) for m, s in p]) for k, p in paths.items()}
    et = {k: np.stack([to_expectation(float(m), float(s)) for m, s in p]) for k, p in paths.items()}

    frames = []
    for i in range(n_frames):
        data = []
        for kind in ("0", "+1", "-1"):
            data.append(go.Scatter(x=paths[kind][: i + 1, 0], y=paths[kind][: i + 1, 1]))
        for kind in ("0", "+1", "-1"):
            data.append(go.Scatter(x=th[kind][: i + 1, 0], y=th[kind][: i + 1, 1]))
        for kind in ("0", "+1", "-1"):
            data.append(go.Scatter(x=et[kind][: i + 1, 0], y=et[kind][: i + 1, 1]))
        frames.append(go.Frame(name=str(i), traces=list(range(9)), data=data))

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("the laws themselves, (μ, σ)", "exponential chart θ  (flat for α = +1)", "mixture chart η  (flat for α = −1)"),
        horizontal_spacing=0.07,
    )
    for kind in ("0", "+1", "-1"):
        fig.add_trace(
            go.Scatter(
                x=paths[kind][:1, 0],
                y=paths[kind][:1, 1],
                mode="lines",
                line=dict(color=colors[kind], width=2.6),
                name={"0": "Levi-Civita", "+1": "exponential", "-1": "mixture"}[kind],
            ),
            row=1,
            col=1,
        )
    for kind in ("0", "+1", "-1"):
        fig.add_trace(
            go.Scatter(x=th[kind][:1, 0], y=th[kind][:1, 1], mode="lines", line=dict(color=colors[kind], width=2.6), showlegend=False),
            row=1,
            col=2,
        )
    for kind in ("0", "+1", "-1"):
        fig.add_trace(
            go.Scatter(x=et[kind][:1, 0], y=et[kind][:1, 1], mode="lines", line=dict(color=colors[kind], width=2.6), showlegend=False),
            row=1,
            col=3,
        )
    _finish(
        fig,
        frames,
        "Dual flatness: the sand curve is a straight line in θ, the oxide curve is a straight line in η. Same two laws, three geometries.",
        duration=95,
        prefix="step  ",
        legend=dict(bgcolor="#0A0A0A", font=dict(color=FG), orientation="h", y=1.12, x=0.15),
    )
    _xy_style(fig)
    fig.update_xaxes(title_text="μ", row=1, col=1)
    fig.update_yaxes(title_text="σ", row=1, col=1)
    fig.update_xaxes(title_text="θ¹", row=1, col=2)
    fig.update_yaxes(title_text="θ²", row=1, col=2)
    fig.update_xaxes(title_text="η¹", row=1, col=3)
    fig.update_yaxes(title_text="η²", row=1, col=3)
    return fig


def dual_potential_figure(n_frames: int = 44) -> go.Figure:
    """Legendre potential ψ(θ). Exponential geodesics are Euclidean straight lines in the base."""
    from geostats.statistical import dual_straight, to_natural

    start, end = (0.0, 1.05), (1.2, 0.62)
    exp = dual_straight(start, end, "exp", n=n_frames)
    mix = dual_straight(start, end, "mix", n=n_frames)
    th_e = np.stack([to_natural(float(m), float(s)) for m, s in exp])
    th_m = np.stack([to_natural(float(m), float(s)) for m, s in mix])
    t1 = np.linspace(-1.8, 3.2, 36)
    t2 = np.linspace(-2.4, -0.18, 36)
    T1, T2 = np.meshgrid(t1, t2)
    PSI = _psi(T1, T2)
    PSI = np.clip(PSI, -1.5, 4.0)

    def lift(th, i):
        p = th[: i + 1]
        return p[:, 0], p[:, 1], _psi(p[:, 0], p[:, 1])

    frames = []
    for i in range(n_frames):
        x0, y0, z0 = lift(th_e, i)
        x1, y1, z1 = lift(th_m, i)
        frames.append(
            go.Frame(
                name=str(i),
                traces=[1, 2],
                data=[
                    go.Scatter3d(x=x0, y=y0, z=z0, mode="lines", line=dict(color=HOVER, width=8)),
                    go.Scatter3d(x=x1, y=y1, z=z1, mode="lines", line=dict(color=WARN, width=6)),
                ],
                layout=go.Layout(scene_camera=dict(eye=_orbit_eye(i, n_frames, radius=1.75, z=1.15))),
            )
        )
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=T1,
            y=T2,
            z=PSI,
            colorscale=[[0, "#000000"], [0.5, "#3D2E6B"], [1, "#AE93EC"]],
            showscale=False,
            opacity=0.86,
        )
    )
    x0, y0, z0 = lift(th_e, 0)
    x1, y1, z1 = lift(th_m, 0)
    fig.add_trace(go.Scatter3d(x=x0, y=y0, z=z0, mode="lines", line=dict(color=HOVER, width=8), name="exponential path"))
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode="lines", line=dict(color=WARN, width=6), name="mixture path, in θ"))
    _finish(
        fig,
        frames,
        "The convex potential ψ(θ) of the Gaussian. An exponential geodesic is a Euclidean straight line in θ, riding on ψ.",
        duration=110,
        prefix="step  ",
        scene=dict(**_scene(), xaxis_title="θ¹", yaxis_title="θ²", zaxis_title="ψ"),
        legend=dict(bgcolor="#0A0A0A", font=dict(color=FG)),
    )
    return fig


def curvature_figure(n_frames: int = 48) -> go.Figure:
    """Parabola in the flat ambient family, with Efron curvature as height."""
    from geostats.inference import parabola_eta, parabola_gamma2

    u = np.linspace(-1.45, 1.45, 160)
    a = 0.85
    eta = parabola_eta(u, a)
    g2 = parabola_gamma2(u, a)
    us = np.linspace(-1.15, 1.15, n_frames)
    eta_s = parabola_eta(us, a)
    g_s = parabola_gamma2(us, a)
    rng = np.random.default_rng(7)

    plane_x = np.array([-1.6, 1.6, 1.6, -1.6, -1.6])
    plane_y = np.array([-0.2, -0.2, 2.0, 2.0, -0.2])
    plane_z = np.zeros(5)

    frames = []
    for i in range(n_frames):
        cloud = eta_s[i] + 0.16 * rng.normal(size=(28, 2))
        vx = [eta_s[i, 0], eta_s[i, 0], None]
        vy = [eta_s[i, 1], eta_s[i, 1], None]
        vz = [0.0, float(g_s[i]), None]
        frames.append(
            go.Frame(
                name=str(i),
                traces=[3, 4, 5],
                data=[
                    go.Scatter3d(
                        x=cloud[:, 0],
                        y=cloud[:, 1],
                        z=np.zeros(len(cloud)),
                        mode="markers",
                        marker=dict(size=3, color=DIM, opacity=0.7),
                    ),
                    go.Scatter3d(
                        x=[eta_s[i, 0]],
                        y=[eta_s[i, 1]],
                        z=[0.0],
                        mode="markers",
                        marker=dict(size=8, color=HOVER),
                    ),
                    go.Scatter3d(x=vx, y=vy, z=vz, mode="lines", line=dict(color=HOVER, width=6)),
                ],
                layout=go.Layout(scene_camera=dict(eye=_orbit_eye(i, n_frames, radius=1.9, z=1.05))),
            )
        )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(x=plane_x, y=plane_y, z=plane_z, mode="lines", line=dict(color=WIRE, width=2), name="ambient family")
    )
    fig.add_trace(
        go.Scatter3d(
            x=eta[:, 0],
            y=eta[:, 1],
            z=np.zeros_like(u),
            mode="lines",
            line=dict(color=LINK, width=7),
            name="model M",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=eta[:, 0],
            y=eta[:, 1],
            z=g2,
            mode="lines",
            line=dict(color=WARN, width=5),
            name="Efron γ²",
        )
    )
    fig.add_trace(
        go.Scatter3d(x=[eta_s[0, 0]], y=[eta_s[0, 1]], z=[0], mode="markers", marker=dict(size=3, color=DIM), name="sample")
    )
    fig.add_trace(
        go.Scatter3d(x=[eta_s[0, 0]], y=[eta_s[0, 1]], z=[0], mode="markers", marker=dict(size=8, color=HOVER), name="law on M")
    )
    fig.add_trace(
        go.Scatter3d(
            x=[eta_s[0, 0], eta_s[0, 0]],
            y=[eta_s[0, 1], eta_s[0, 1]],
            z=[0.0, float(g_s[0])],
            mode="lines",
            line=dict(color=HOVER, width=6),
            name="curvature",
        )
    )
    _finish(
        fig,
        frames,
        "The model is a parabola in a flat Gaussian family. Height is the information the MLE cannot recover.",
        duration=120,
        prefix="u = ",
        scene=_scene(),
        legend=dict(bgcolor="#0A0A0A", font=dict(color=FG)),
    )
    return fig


def sphere_figure(n_frames: int = 48, n_paths: int = 18, seed: int = 1) -> go.Figure:
    from geostats.diffusion import mean_z_decay, sphere_bm_many, wire_sphere

    rng = np.random.default_rng(seed)
    n_steps = 200
    t_final = 1.8
    paths = sphere_bm_many(rng, n_paths=n_paths, t_final=t_final, n_steps=n_steps)
    mz = mean_z_decay(paths)
    times = np.linspace(0.0, t_final, n_steps + 1)
    theory = np.exp(-times)
    wx, wy, wz = [], [], []
    for circ in wire_sphere(12, 24):
        wx.extend(list(circ[:, 0]) + [None])
        wy.extend(list(circ[:, 1]) + [None])
        wz.extend(list(circ[:, 2]) + [None])

    def path_xyz(upto: int):
        xs, ys, zs = [], [], []
        for p in range(n_paths):
            xs.extend(list(paths[p, : upto + 1, 0]) + [None])
            ys.extend(list(paths[p, : upto + 1, 1]) + [None])
            zs.extend(list(paths[p, : upto + 1, 2]) + [None])
        return xs, ys, zs

    idx = np.unique(np.linspace(0, n_steps, n_frames, dtype=int))
    frames = []
    for k, ti in enumerate(idx):
        ti = int(ti)
        xs, ys, zs = path_xyz(ti)
        frames.append(
            go.Frame(
                name=f"{times[ti]:.2f}",
                traces=[1, 2, 3],
                data=[
                    go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=LINK, width=3)),
                    go.Scatter3d(
                        x=paths[:, ti, 0],
                        y=paths[:, ti, 1],
                        z=paths[:, ti, 2],
                        mode="markers",
                        marker=dict(size=4, color=HOVER),
                    ),
                    go.Scatter(x=times[: ti + 1], y=mz[: ti + 1], mode="lines", line=dict(color=LINK, width=2.6)),
                ],
                layout=go.Layout(scene_camera=dict(eye=_orbit_eye(k, len(idx), radius=1.85, z=0.88))),
            )
        )
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("Brownian motion on the sphere", "mean height against the generator"),
        column_widths=[0.58, 0.42],
    )
    xs, ys, zs = path_xyz(0)
    fig.add_trace(go.Scatter3d(x=wx, y=wy, z=wz, mode="lines", line=dict(color="#2A2A2A", width=1.4), name="S²"), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=LINK, width=3), name="paths"), row=1, col=1)
    fig.add_trace(
        go.Scatter3d(
            x=paths[:, 0, 0],
            y=paths[:, 0, 1],
            z=paths[:, 0, 2],
            mode="markers",
            marker=dict(size=4, color=HOVER),
            name="tips",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=times[:1], y=mz[:1], mode="lines", line=dict(color=LINK, width=2.6), name="mean z"), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=times, y=theory, mode="lines", line=dict(color=WARN, width=1.3, dash="dot"), name="e^{-t}"),
        row=1,
        col=2,
    )
    _finish(
        fig,
        frames,
        "The generator is half the Laplace-Beltrami operator. Height z decays as e^{-t} because Δz = −2z.",
        duration=130,
        prefix="t = ",
        showlegend=False,
        scene=_scene(),
    )
    fig.update_xaxes(title_text="t", gridcolor=WIRE, linecolor=WIRE, row=1, col=2)
    fig.update_yaxes(title_text="mean z", gridcolor=WIRE, linecolor=WIRE, row=1, col=2)
    return fig


def lift_figure(n_frames: int = 56, seed: int = 4) -> go.Figure:
    from geostats.diffusion import parallel_frame, sphere_bm, wire_sphere

    rng = np.random.default_rng(seed)
    n_steps = 200
    path = sphere_bm(np.array([0.0, 0.0, 1.0]), rng, t_final=1.7, n_steps=n_steps)
    e1s, e2s = parallel_frame(path)
    wx, wy, wz = [], [], []
    for circ in wire_sphere(12, 24):
        wx.extend(list(circ[:, 0]) + [None])
        wy.extend(list(circ[:, 1]) + [None])
        wz.extend(list(circ[:, 2]) + [None])
    scale = 0.32

    def triad(i: int):
        x = path[i]
        e1, e2 = e1s[i], e2s[i]
        b = x + scale * e1
        c = x + scale * e2
        n = 1.12 * x
        return (
            [x[0], b[0]],
            [x[1], b[1]],
            [x[2], b[2]],
            [x[0], c[0]],
            [x[1], c[1]],
            [x[2], c[2]],
            [0.0, n[0]],
            [0.0, n[1]],
            [0.0, n[2]],
        )

    idx = np.unique(np.linspace(0, n_steps, n_frames, dtype=int))
    frames = []
    for k, ti in enumerate(idx):
        ti = int(ti)
        x1, y1, z1, x2, y2, z2, xn, yn, zn = triad(ti)
        frames.append(
            go.Frame(
                name=str(k),
                traces=[1, 2, 3, 4, 5],
                data=[
                    go.Scatter3d(
                        x=path[: ti + 1, 0],
                        y=path[: ti + 1, 1],
                        z=path[: ti + 1, 2],
                        mode="lines",
                        line=dict(color=LINK, width=5),
                    ),
                    go.Scatter3d(x=x1, y=y1, z=z1, mode="lines", line=dict(color=HOVER, width=10)),
                    go.Scatter3d(x=x2, y=y2, z=z2, mode="lines", line=dict(color=WARN, width=10)),
                    go.Scatter3d(x=xn, y=yn, z=zn, mode="lines", line=dict(color=DIM, width=3)),
                    go.Scatter3d(
                        x=[path[ti, 0]],
                        y=[path[ti, 1]],
                        z=[path[ti, 2]],
                        mode="markers",
                        marker=dict(size=6, color=FG),
                    ),
                ],
                layout=go.Layout(scene_camera=dict(eye=_orbit_eye(k, len(idx), radius=1.8, z=0.92))),
            )
        )
    x1, y1, z1, x2, y2, z2, xn, yn, zn = triad(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=wx, y=wy, z=wz, mode="lines", line=dict(color="#2A2A2A", width=1.4), name="S²"))
    fig.add_trace(
        go.Scatter3d(x=path[:1, 0], y=path[:1, 1], z=path[:1, 2], mode="lines", line=dict(color=LINK, width=5), name="path")
    )
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode="lines", line=dict(color=HOVER, width=10), name="e₁"))
    fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode="lines", line=dict(color=WARN, width=10), name="e₂"))
    fig.add_trace(go.Scatter3d(x=xn, y=yn, z=zn, mode="lines", line=dict(color=DIM, width=3), name="normal"))
    fig.add_trace(
        go.Scatter3d(x=[path[0, 0]], y=[path[0, 1]], z=[path[0, 2]], mode="markers", marker=dict(size=6, color=FG), name="x")
    )
    _finish(
        fig,
        frames,
        "A point on the sphere is not enough. Brownian motion is a rolling frame: two tangent vectors, carried without slip.",
        duration=125,
        prefix="step  ",
        scene=_scene(),
        legend=dict(bgcolor="#0A0A0A", font=dict(color=FG)),
    )
    return fig
