"""Identities the program must not lose."""

from __future__ import annotations

import numpy as np

from geostats.convex import (
    angle_grid,
    area_from_support,
    brunn_minkowski_curve,
    ellipse_support,
    minkowski_sum_h,
    parallel_support,
    stadium_support,
    steiner_area_plane,
)
from geostats.diffusion import constraint_residual, mean_z_decay, parallel_frame, sphere_bm, sphere_bm_many
from geostats.inference import parabola_fisher, parabola_gamma2
from geostats.integral import chord_lengths, crofton_running
from geostats.statistical import (
    alpha_duality_residual,
    dual_straight,
    fisher,
    fisher_from_kl_hessian,
    geodesic_connect,
    to_expectation,
    to_natural,
)


def test_support_additivity():
    th = angle_grid(180)
    h1 = ellipse_support(1.2, 0.6, th)
    h2 = stadium_support(0.4, 0.5, th)
    assert np.max(np.abs(minkowski_sum_h(h1, h2) - (h1 + h2))) == 0.0


def test_brunn_minkowski_gap_nonnegative():
    th = angle_grid(240)
    h0 = ellipse_support(1.0, 1.0, th)
    h1 = ellipse_support(1.8, 0.5, th)
    bm = brunn_minkowski_curve(h0, h1, th, n_t=21)
    assert np.min(bm["gap"]) > -5e-3


def test_area_disk():
    th = angle_grid(360)
    h = ellipse_support(1.0, 1.0, th)
    a = area_from_support(h, th)
    assert abs(a - np.pi) / np.pi < 0.01


def test_steiner_disk():
    th = angle_grid(360)
    h = ellipse_support(1.0, 1.0, th)
    r = 0.3
    discrete = area_from_support(parallel_support(h, r), th)
    poly = steiner_area_plane(h, th, r)
    exact = np.pi * (1.0 + r) ** 2
    assert abs(discrete - exact) / exact < 0.015
    assert abs(poly - exact) / exact < 0.015


def test_crofton_disk():
    rng = np.random.default_rng(0)
    th = angle_grid(240)
    h = ellipse_support(1.0, 1.0, th)
    run = crofton_running(rng, h, th, n=6000)
    assert run["rel_err"] < 0.08


def test_hostinsky_disk():
    rng = np.random.default_rng(1)
    th = angle_grid(240)
    h = ellipse_support(1.0, 1.0, th)
    run = crofton_running(rng, h, th, n=8000)
    assert run["hostinsky_rel_err"] < 0.12


def test_chord_disk():
    th = angle_grid(360)
    h = ellipse_support(1.0, 1.0, th)
    p = np.array([0.0, 0.5, 0.9])
    phi = np.array([0.0, 0.4, 1.1])
    sig = chord_lengths(p, phi, h, th)
    exact = 2.0 * np.sqrt(np.maximum(1.0 - p * p, 0.0))
    assert np.max(np.abs(sig - exact)) < 0.03


def test_fisher_is_kl_hessian():
    g = fisher(0.2, 1.1)
    h = fisher_from_kl_hessian(0.2, 1.1)
    rel = np.max(np.abs(g - h)) / np.max(np.abs(g))
    assert rel < 5e-3


def test_alpha_duality():
    assert alpha_duality_residual(1.3) < 1e-14


def test_dual_geodesics_are_straight():
    start, end = (0.0, 1.0), (1.1, 0.7)
    exp = dual_straight(start, end, "exp", n=40)
    mix = dual_straight(start, end, "mix", n=40)
    th = np.stack([to_natural(float(m), float(s)) for m, s in exp])
    et = np.stack([to_expectation(float(m), float(s)) for m, s in mix])
    # collinearity: max deviation from the chord
    def chord_dev(pts: np.ndarray) -> float:
        v = pts[-1] - pts[0]
        nrm = np.linalg.norm(v)
        rel = pts - pts[0]
        proj = np.outer((rel @ v) / (nrm * nrm), v)
        return float(np.max(np.linalg.norm(rel - proj, axis=1)))

    assert chord_dev(th) < 1e-10
    assert chord_dev(et) < 1e-10
    shot = geodesic_connect(start, end, alpha=1.0, n_steps=50)
    assert np.linalg.norm(shot[-1] - np.array(end)) < 5e-3


def test_efron_curvature_formula():
    u, a = 0.4, 1.2
    g = parabola_fisher(u, a)
    gam = parabola_gamma2(u, a)
    assert abs(gam - (4 * a * a) / g**3) < 1e-14


def test_sphere_stays_on_sphere():
    rng = np.random.default_rng(2)
    path = sphere_bm(np.array([0.0, 0.0, 1.0]), rng, n_steps=200)
    assert constraint_residual(path) < 1e-12


def test_sphere_generator_z_decays():
    rng = np.random.default_rng(3)
    paths = sphere_bm_many(rng, n_paths=80, t_final=0.8, n_steps=120)
    mz = mean_z_decay(paths)
    theory = np.exp(-np.linspace(0.0, 0.8, mz.size))
    assert mz[-1] < mz[0]
    assert abs(mz[-1] - theory[-1]) < 0.25


def test_parallel_frame_orthonormal():
    rng = np.random.default_rng(5)
    path = sphere_bm(np.array([0.0, 0.0, 1.0]), rng, n_steps=80)
    e1, e2 = parallel_frame(path)
    assert np.max(np.abs(np.sum(e1 * path, axis=1))) < 1e-12
    assert np.max(np.abs(np.sum(e2 * path, axis=1))) < 1e-12
    assert np.max(np.abs(np.sum(e1 * e2, axis=1))) < 1e-12
    assert np.max(np.abs(np.linalg.norm(e1, axis=1) - 1.0)) < 1e-12
