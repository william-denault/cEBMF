import pytest
import numpy as np 
from cebmf.routines.numerical_routine import apply_log_sum_exp , my_e2truncnorm, my_etruncnorm
    
from cebmf.routines.posterior_computation import posterior_mean_exp, wpost_exp
from cebmf.routines.utils_mix import autoselect_scales_mix_exp
from cebmf.routines.distribution_operation import get_data_loglik_exp, convolved_logpdf_exp


def test_trucnorm():

    assert np.isclose(my_etruncnorm(0,2,3,1), 1.48995049, atol=1e-7)
    assert np.isclose(my_e2truncnorm(0,2,3,1), 2.39340536, atol=1e-7)

def test_convolved_loglik_postmean():


    betahat=  np.array([1,2,3,4,5])
    sebetahat=np.array([1,0.4,5,1,1])
    scale = autoselect_scales_mix_exp ( np.array([1,2,3,4,5]),  np.array([1,0.4,5,1,1]))

    non_informativ = np.full( scale.shape[0], 1/ scale.shape[0])
    n=betahat.shape[0]
    log_pi =  np.log( np.tile(non_informativ, (n, 1)))
    assignment = np.exp(log_pi)[0]
    assignment = assignment /   sum(assignment)
    w=assignment
    x=betahat[1]
    s=sebetahat[1]

    obs_wpost= wpost_exp ( x, s, w, scale)
    expected_wpost=np.array([3.53987758e-06, 6.61493885e-06, 1.06391083e-05, 2.86976960e-05,
       1.69165759e-04, 1.40776600e-03, 8.91255655e-03, 3.45101678e-02,
       8.34205613e-02, 1.38160703e-01, 1.72844260e-01, 1.77093219e-01,
       1.57939387e-01, 1.28091688e-01, 9.74010338e-02])
    
    post_assign =   np.zeros ( (betahat.shape[0], scale.shape[0]))
    for i in range(betahat.shape[0]):
        post_assign[i,] = wpost_exp ( x=betahat[i], s=sebetahat[i], w=np.exp(log_pi)[i,], scale=scale) 
    
    post_mean = np.zeros(betahat.shape[0])
    post_mean2 = np.zeros(betahat.shape[0])
 

    for i in range(post_mean.shape[0]):
        post_mean[i]=  sum(    post_assign[i,1:] *  my_etruncnorm(0,
                                             np.inf,
                                             betahat[i]- sebetahat[i]** 2 *(1/scale[1:]) , 
                                             sebetahat[i] 
                                             )
                       )
        post_mean2[i] =  sum ( post_assign[i,1:] *my_e2truncnorm(0,
                                             99999, #some weird warning for inf so just use something large enought for b
                                             betahat[i]- sebetahat[i]** 2 *(1/scale[1:]) , 
                                             sebetahat[i] 
                                             )
                       )
    

    expected_post_mean= np.array([0.4836384 , 1.89328341, 1.05670973, 3.57009763, 4.66379651])
    expected_post_mean2= np.array([ 0.59674218,  3.75372025,  4.34337321, 13.89299102, 22.81107216])
    
    n=betahat.shape[0]
    p= scale.shape[0]
 
    log_pi =  np.log( np.full( (n, p), 1/scale.shape[0]))
 
    res = posterior_mean_exp(betahat, sebetahat, log_pi, scale)
    np.testing.assert_allclose(obs_wpost, expected_wpost, atol=1e-7)
    np.testing.assert_allclose(post_mean, expected_post_mean, atol=1e-7)
    np.testing.assert_allclose(post_mean2, expected_post_mean2, atol=1e-7)
    np.testing.assert_allclose(res.post_mean, expected_post_mean, atol=1e-7)
    np.testing.assert_allclose(res.post_mean2, expected_post_mean2, atol=1e-7)





