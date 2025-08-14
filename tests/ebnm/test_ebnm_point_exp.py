import pytest
import numpy as np
from cebmf.ebnm.ebnm_point_exp import (
    pe_nllik, optimize_pe_nllik_with_gradient
)

def test_pe_nllik():
    x = np.array([1.0, 2.0, -1.5])
    s = np.array([1.0, 1.0, 1.0])
    par_init = [0.0, 0.0, 0.0]
    fix_par = [False, False, False]
    par = [0.5, 0.1, 1.0]
    nllik = pe_nllik(par, x, s, par_init, fix_par, calc_grad=False)
    expected = 7.026976565768523
    assert np.isclose(nllik, expected, atol=1e-6)

def test_optimize_pe_nllik_with_gradient():
    x = np.array([1.0, 1.0, -0.5])
    s = np.array([1.0, 1.0, 1.0])
    par_init = [0.0, 0.1, 0.0]
    fix_par = [False, False, True]
    result = optimize_pe_nllik_with_gradient(x, s, par_init, fix_par)
    assert np.isclose(result.nllik, 3.636553632132083, atol=1e-6)
    assert np.isclose(result.w, 0.9999563044116645, atol=1e-6)
    assert np.isclose(result.a, 3.047337093696241, atol=1e-6)
    assert np.isclose(result.mu, 0.0, atol=1e-6)