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
    res= ash(betahat, sebetahat, mult=np.sqrt(2),mode=2)
    mult=2
    scale=autoselect_scales_mix_norm(betahat  = betahat,
                                         sebetahat= sebetahat,
                                         mult=np.sqrt(2))
 
    # Example log likelihood, should update to actual value if needed
    expected_log_lik = -11.772394073433777
    np.testing.assert_allclose(res.log_lik, expected_log_lik, atol=1e-6)

    # Example scale values, should update to actual values if needed
    expected_scale = res.scale
    expected_posterior_mean= np. array([1.97808877, 2.        , 2.00759421, 2.14860704, 3.15034247])

    expected_posterior_mean2=np.array([ 3.95305034,  4.00112716,  4.22170576,  4.92353133, 11.94345543])

    expected_pi=np.array([9.56721745e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 1.58603289e-17, 1.58603289e-17,
       7.08349778e-07, 4.31668575e-02, 1.10688684e-04, 9.85782884e-13,
       1.58603289e-17, 0.00000000e+00])

    np.testing.assert_allclose(res.scale, expected_scale, rtol=1e-5)
    np.testing.assert_allclose(res.post_mean, expected_posterior_mean , rtol=1e-5)
    np.testing.assert_allclose(res.post_mean2, expected_posterior_mean2 , rtol=1e-5) 
    L= get_data_loglik_normal(betahat=betahat ,
                                 sebetahat=sebetahat ,
                                 location=location,
                                 scale=scale)
    optimal_pi = optimize_pi( np.exp(L),
                                 penalty=10,
                                 verbose=True) 
    np.testing.assert_allclose(optimal_pi, expected_pi , rtol=1e-5)
 







betahat=  np.array([1,2,3,4,5])
sebetahat=np.array([1,0.4,5,1,1])

res= ash(betahat, sebetahat, mult=np.sqrt(2),mode=2)
print(res.post_mean)
mult=2

print(res.log_lik)

scale=autoselect_scales_mix_norm(betahat  = betahat,
                                         sebetahat= sebetahat,
                                         mult=np.sqrt(2))
location=0*scale+2
print(len(location.shape))