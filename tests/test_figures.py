"""Plotly boards construct without opening a browser."""

from __future__ import annotations

import pytest

plotly = pytest.importorskip("plotly")

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


@pytest.mark.parametrize(
    "builder",
    [
        minkowski_figure,
        minkowski_stack_figure,
        crofton_figure,
        fisher_figure,
        fisher_surface_figure,
        gaussian_flow_figure,
        dual_figure,
        dual_potential_figure,
        curvature_figure,
        sphere_figure,
        lift_figure,
    ],
)
def test_figure_builds(builder):
    fig = builder()
    assert len(fig.data) >= 1
    assert fig.frames is not None and len(fig.frames) > 0
