import pytest
import numpy as np
from cebmf.ebnm.ebnm_point_laplace import (
    posterior_mean_laplace, wpost_laplace, ebnm_point_laplace_solver, pl_nllik, optimize_pl_nllik_with_gradient
)

def test_wpost_laplace():
    x = np.array([1.0, 2.0, -1.5])
    s = np.array([1.0, 1.2, .5])
    expected = np.array([0.45573542, 0.57074938, 0.93438632])
    out = wpost_laplace(x, s, 0.5, 1)
    np.testing.assert_allclose(out, expected, atol=1e-6)

def test_posterior_mean_laplace_post_mean_sd():
    x = np.array([1.0, 2.0, -1.5])
    s = np.array([1.0, 1.2, 2.5])
    out = posterior_mean_laplace(x, s, w=.7, a=1, mu=0)
    expected_post_sd = np.array([  0.65301652, 0.92274914, 0.94291949 ])
    expected_post_mean = np.array([0.33285794,  0.73456863, -0.20122616])
    np.testing.assert_allclose(out.post_sd, expected_post_sd, atol=1e-6)
    np.testing.assert_allclose(out.post_mean, expected_post_mean, atol=1e-6)

def test_pl_nllik():
    x = np.array([1.0, 2.0, -1.5])
    s = np.array([1.0, 1.0, 1.0])
    par_init = [0.0, 0.0, 0.0]
    fix_par = [False, False, False]
    par = [0.5, 0.1, 1.0]
    nllik = pl_nllik(par, x, s, par_init, fix_par, calc_grad=False)
    expected = 5.74805874231258
    assert np.isclose(nllik, expected, atol=1e-6)

def test_ebnm_point_laplace_solver_loglik_and_postmean():
    x = np.array([0.0, 1.0, -0.5])
    s = np.array([1.0, 0.2, 1.0])
    ebnm_res = ebnm_point_laplace_solver(x=x, s=s)
    expected_log_lik = -4.161880337595547
    expected_post_mean = np.array([0.0, 0.9326135, -0.15496329])
    assert np.isclose(ebnm_res.log_lik, expected_log_lik, atol=1e-6)
    np.testing.assert_allclose(ebnm_res.post_mean, expected_post_mean, atol=1e-6)

def test_optimize_pl_nllik_with_gradient():
    x = np.array([0.0, 0.0, -0.5])
    s = np.array([1.0, 1.0, 1.0])
    par_init = [0.0, 0.0, 0.0]
    fix_par = [False, False, False]

    # This assumes the function returns an object with nllik and w attributes
    result = optimize_pl_nllik_with_gradient(x, s, par_init, fix_par)
    expected_nllik = 2.8401494330200583
    expected_w = 0.046401624283020415
    assert np.isclose(result.nllik, expected_nllik, atol=1e-6)
    assert np.isclose(result.w, expected_w, atol=1e-6)