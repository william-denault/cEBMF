import pytest
import numpy as np
from cebmf.ebnm.ash import ash
from cebmf.routines.numerical_routine import optimize_pi
from cebmf.routines.posterior_computation import posterior_mean_norm
from cebmf.routines.utils_mix import autoselect_scales_mix_norm
from cebmf.routines.distribution_operation import get_data_loglik_normal

def test_ash_loglik_and_scale():
    betahat = np.array([1, 2, 3, 4, 5])
    sebetahat = np.array([1, 0.4, 5, 1, 1])
    res = ash(betahat, sebetahat, mult=np.sqrt(2))

    # Example log likelihood, should update to actual value if needed
    expected_log_lik = -16.91767637608251
    np.testing.assert_allclose(res.log_lik, expected_log_lik, atol=1e-6)

    # Example scale values, should update to actual values if needed
    expected_scale = np.array([0., 0.03827328, 0.05412659, 0.07654655, 0.10825318, 0.15309311,
                              0.21650635, 0.30618622, 0.4330127, 0.61237244, 0.8660254,
                              1.22474487, 1.73205081, 2.44948974, 3.46410162, 4.89897949,
                              6.92820323, 9.79795897])
    expected_posterior_mean= np.array([0.11126873, 1.97346787, 0.20802628, 3.6663574 , 4.61542534])

    expected_posterior_mean2= np.array([ 0.21398654,  4.05293203,  1.93632865, 14.45535405, 22.22774277])

    np.testing.assert_allclose(res.scale, expected_scale, rtol=1e-5)
    np.testing.assert_allclose(res.post_mean, expected_posterior_mean , rtol=1e-5)
    np.testing.assert_allclose(res.post_mean2, expected_posterior_mean2 , rtol=1e-5)


def test_autoselect_scales_equals_ash_scale():
    betahat = np.array([1, 2, 3, 4, 5])
    sebetahat = np.array([1, 0.4, 5, 1, 1])
    mult = 2

    scale = autoselect_scales_mix_norm(betahat=betahat, sebetahat=sebetahat, mult=mult)
    res = ash(betahat, sebetahat, mult= (2))
    np.testing.assert_allclose(res.scale, scale, rtol=1e-3)

def test_optimize_pi_and_posterior_mean_norm_shape():
    betahat = np.array([1, 2, 3, 4, 5])
    sebetahat = np.array([1, 0.4, 5, 1, 1])
    mult = 2
    scale = autoselect_scales_mix_norm(betahat=betahat, sebetahat=sebetahat, mult=mult)
    L = get_data_loglik_normal(betahat=betahat, sebetahat=sebetahat, location=0 * scale, scale=scale)
    exp_L = np.exp(L)
    optimal_pi = optimize_pi(exp_L, penalty=10, verbose=False)
    out = posterior_mean_norm(
        betahat,
        sebetahat,
        log_pi=np.log(optimal_pi + 1e-32),
        scale=scale
    )
    result = exp_L * np.exp(optimal_pi)

    assert result.shape == (5, 10)

    # Optional: assert numerical closeness with provided matrix
    expected_result = np.array([
        [5.20259948e-01, 2.41970595e-01, 2.41968664e-01, 2.41938509e-01,
         2.41499598e-01, 2.36501255e-01, 2.06576619e-01, 1.43080328e-01,
         9.70177435e-02, 4.02981925e-02],
        [7.99146900e-06, 4.14419530e-06, 5.67727637e-06, 1.71434569e-05,
         2.99017321e-04, 1.29778396e-02, 9.28131811e-02, 1.18400878e-01,
         9.26844908e-02, 3.98455744e-02],
        [1.43292884e-01, 6.66436710e-02, 6.66399227e-02, 6.66249342e-02,
         6.65650602e-02, 6.63268389e-02, 6.53941666e-02, 6.31577118e-02,
         6.44948202e-02, 3.49434622e-02],
        [2.87747646e-04, 1.35306388e-04, 1.39805592e-04, 1.58887214e-04,
         2.54037078e-04, 1.01147338e-03, 1.02848443e-02, 4.90077101e-02,
         7.18725121e-02, 3.72997651e-02],
        [3.19658760e-06, 1.51304459e-06, 1.59435046e-06, 1.95667780e-06,
         4.15035767e-06, 3.83377890e-05, 1.70007332e-03, 2.57676671e-02,
         6.00329684e-02, 3.56088883e-02]
    ])
    np.testing.assert_allclose(result, expected_result, atol=1e-6)