import numpy as np
from cebmf.routines.numerical_routine import optimize_pi_logL
from cebmf.routines.utils_mix import autoselect_scales_mix_norm, autoselect_scales_mix_exp
from cebmf.routines.distribution_operation import get_data_loglik_normal, get_data_loglik_exp
from cebmf.routines.posterior_computation import posterior_mean_norm, posterior_mean_exp
from cebmf.routines.R_import import PiOptim, choose_pi_optimizer
from rpy2.rinterface_lib.sexp import NULLType  # safe to import; does not require an R session


class ash_object:
    def __init__(
        self,
        post_mean,
        post_mean2,
        post_sd,
        scale,
        pi,
        prior,
        log_lik=0.0,
        mode=0.0,
    ):
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.scale = scale
        self.pi = pi
        self.prior = prior
        self.log_lik = log_lik
        self.mode = mode


def ash(
    betahat,
    sebetahat,
    prior="norm",
    mult=np.sqrt(2),
    penalty=10.0,
    verbose=True,
    threshold_loglikelihood=-300.0,
    mode=0.0,
    *,
    optmode: PiOptim = PiOptim.AUTO,
    prefer_mixsqp: bool = True,
):
    """
    Adaptive shrinkage with mixture priors (normal or exponential) using either EM or mixsqp
    for mixture-weight optimization, depending on `optmode`.

    Parameters
    ----------
    prior : {"norm","exp"}
    penalty : float
        Dirichlet prior α_1 on the first component (EM), approximated via anchor-row weight for mixsqp.
    optmode : PiOptim
        AUTO: use mixsqp if available, else EM; MIXSQP: require mixsqp; EM: force EM.
    prefer_mixsqp : bool
        Only used when optmode==AUTO: whether to prefer mixsqp if available.
    """
    betahat = np.asarray(betahat)
    sebetahat = np.asarray(sebetahat)

    # Decide optimizer once and pass the indicator down
    mode_pi = choose_pi_optimizer(prefer_mixsqp) if optmode is PiOptim.AUTO else optmode

    if prior == "norm":
        scale = autoselect_scales_mix_norm(
            betahat=betahat, sebetahat=sebetahat, mult=mult
        )
        L = get_data_loglik_normal(
            betahat=betahat,
            sebetahat=sebetahat,
            location=np.zeros_like(scale) + mode,
            scale=scale,
        )
        optimal_pi = optimize_pi_logL(
            logL=L, penalty=penalty, verbose=verbose, optmode=mode_pi
        )
        out = posterior_mean_norm(
            betahat,
            sebetahat,
            log_pi=np.log(optimal_pi + 1e-32),
            location=np.zeros_like(scale) + mode,
            scale=scale,
        )

    elif prior == "exp":
        scale = autoselect_scales_mix_exp(
            betahat=betahat, sebetahat=sebetahat, mult=mult
        )
        L = get_data_loglik_exp(
            betahat=betahat, sebetahat=sebetahat, scale=scale
        )
        optimal_pi = optimize_pi_logL(
            logL=L, penalty=penalty, verbose=verbose, optmode=mode_pi
        )
        log_pi = np.tile(np.log(optimal_pi + 1e-32), (betahat.shape[0], 1))
        out = posterior_mean_exp(
            betahat, sebetahat, log_pi=log_pi, scale=scale
        )

    else:
        raise ValueError("prior must be either 'norm' or 'exp'.")

    # For logging only: clamp L before computing total log-likelihood
    L = np.maximum(L, threshold_loglikelihood)
    # log p(data | pi, mixture) = sum_j log sum_k exp( L_{jk} + log pi_k )
    L_max = np.max(L, axis=1, keepdims=True)
    exp_term = np.exp(L - L_max) * optimal_pi  # broadcasting pi over rows
    exp_term = np.maximum(exp_term, 1e-300)
    log_lik = np.sum(L_max + np.log(np.sum(exp_term, axis=1, keepdims=True)))

    return ash_object(
        post_mean=out.post_mean,
        post_mean2=out.post_mean2,
        post_sd=out.post_sd,
        scale=scale,
        pi=optimal_pi,
        prior=prior,
        log_lik=float(log_lik),
        mode=mode,
    )


# REMOVE this from top-level:
# from rpy2.rinterface_lib.sexp import NULLType

def call_r_ash_fit_all_with_postmean(beta, sigma):
    try:
        from rpy2 import robjects as ro
        from rpy2.robjects import numpy2ri, packages
        from rpy2.rinterface_lib.sexp import NULLType   # <-- move here
        numpy2ri.activate()
        ashr = packages.importr("ashr")
    except Exception:
        raise ImportError(
            "Optional dependency `rpy2` and/or R package `ashr` are not installed.\n"
            "Install rpy2 and ashr to use this function."
        )
    ...
