
import pytest
import numpy as np
from cebmf.ebnm.ash import ash
from cebmf.routines.numerical_routine import optimize_pi
from cebmf.routines.posterior_computation import posterior_mean_exp
from cebmf.routines.utils_mix import autoselect_scales_mix_exp
from cebmf.routines.distribution_operation import get_data_loglik_exp




betahat=  np.array([1,2,3,4,5])
sebetahat=np.array([1,0.4,5,1,1])

res= ash(betahat, sebetahat,   prior="exp")







def test_ash_loglik_and_scale():
    betahat = np.array([1, 2, 3, 4, 5])
    sebetahat = np.array([1, 0.4, 5, 1, 1])
    res = ash(betahat, sebetahat,   prior="exp")

    # Example log likelihood, should update to actual value if needed
    expected_log_lik = -15.244064765169643
    np.testing.assert_allclose(res.log_lik, expected_log_lik, atol=1e-6)

    # Example scale values, should update to actual values if needed
    expected_scale = np.array([ 0.        , 0.02929687, 0.04143204, 0.05859375, 0.08286408,
                                0.1171875 , 0.16572815, 0.234375  , 0.3314563 , 0.46875   ,
                                0.66291261, 0.9375    , 1.32582521, 1.875     , 2.65165043,
                                3.75      , 5.30330086, 7.5       ])
    expected_posterior_mean= np.array([0.21361966, 1.95003336, 0.68125659, 3.67773689, 4.69599116])

    expected_posterior_mean2= np.array([ 0.34021317,  3.96295414,  3.17114472, 14.58222929, 23.05642398])
    expected_pi=np.array([7.54197100e-01, 1.76702938e-11, 1.78160344e-11, 1.80223138e-11,
       1.83146990e-11, 1.87375353e-11, 1.94253344e-11, 2.09441353e-11,
       2.51185263e-11, 3.58721148e-11, 6.04197486e-11, 1.15894072e-10,
       2.60987406e-10, 2.24338764e-06, 9.20140421e-02, 1.53746415e-01,
       4.01978226e-05, 4.19178644e-10])
    np.testing.assert_allclose(res.scale, expected_scale, rtol=1e-5)
    np.testing.assert_allclose(res.post_mean, expected_posterior_mean , rtol=1e-5)
    np.testing.assert_allclose(res.post_mean2, expected_posterior_mean2 , rtol=1e-5)
    np.testing.assert_allclose(res.pi, expected_pi , rtol=1e-5)

def test_autoselect_scales_equals_ash_scale():
    betahat = np.array([1, 2, 3, 4, 5])
    sebetahat = np.array([1, 0.4, 5, 1, 1])
  

    scale = autoselect_scales_mix_exp(betahat=betahat, sebetahat=sebetahat )
    res = ash(betahat, sebetahat, prior="exp")
    np.testing.assert_allclose(res.scale, scale, rtol=1e-3)

def test_optimize_pi_and_posterior_mean_norm_shape():
    betahat = np.array([1, 2, 3, 4, 5])
    sebetahat = np.array([1, 0.4, 5, 1, 1])
 
    scale = autoselect_scales_mix_exp(betahat=betahat, sebetahat=sebetahat )
    L = get_data_loglik_exp(betahat=betahat, sebetahat=sebetahat,   scale=scale)
    exp_L = np.exp(L)
    optimal_pi = optimize_pi(exp_L, penalty=10, verbose=False)
    out = posterior_mean_exp(
        betahat,
        sebetahat,
        log_pi=np.log(optimal_pi + 1e-32),
        scale=scale
    )
    result = exp_L * np.exp(optimal_pi)
    
    assert result.shape == (5, 10)

    # Optional: assert numerical closeness with provided matrix
    expected_result = np.array([[5.14448019e-01, 2.51266858e-01, 2.55860056e-01, 2.62617097e-01,
        2.72306144e-01, 2.85408925e-01, 3.00955180e-01, 3.14609872e-01,
        3.18224769e-01, 3.04017535e-01, 2.70796954e-01, 2.25296176e-01,
        2.25033292e-01, 1.31687642e-01, 9.50020960e-02],
       [7.90219468e-06, 6.94554999e-06, 1.11708453e-05, 3.01319916e-05,
        1.77620574e-04, 1.47812540e-03, 9.35800140e-03, 3.62349677e-02,
        8.75898768e-02, 1.45065902e-01, 1.81482924e-01, 1.86138341e-01,
        2.11763856e-01, 1.34519079e-01, 1.02269086e-01],
       [1.41692130e-01, 6.69505321e-02, 6.71013478e-02, 6.73249846e-02,
        6.76543894e-02, 6.81341330e-02, 6.88190590e-02, 6.97613770e-02,
        7.09656123e-02, 7.22727727e-02, 7.31514193e-02, 7.26157400e-02,
        8.83034777e-02, 6.23692536e-02, 5.28421736e-02],
       [2.84533158e-04, 1.57892848e-04, 1.73113440e-04, 2.01458758e-04,
        2.61844431e-04, 4.21684195e-04, 9.78133319e-04, 3.22217658e-03,
        1.09942900e-02, 2.94104069e-02, 5.69448355e-02, 8.23804579e-02,
        1.20682664e-01, 9.16919072e-02, 7.89135260e-02],
       [3.16087787e-06, 1.83738982e-06, 2.07754359e-06, 2.56696292e-06,
        3.81619740e-06, 8.66325751e-06, 4.20378818e-05, 3.43749825e-04,
        2.42284899e-03, 1.06997731e-02, 2.90064310e-02, 5.25380779e-02,
        8.94134392e-02, 7.50763563e-02, 6.90667835e-02]])
    np.testing.assert_allclose(result, expected_result, atol=1e-6)