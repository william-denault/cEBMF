# cebmf/__init__.py

from .ebnm.ash import ash, ebnm_point_laplace        # lightweight
from .main import (                                  # keep exporting these
    cEBMF, cEBMF_object,
    compute_hat_l_and_s_l, compute_hat_f_and_s_f,
)
from .routines.R_import import PiOptim, choose_pi_optimizer

__all__ = [
    "ash",
    "ebnm_point_laplace",
    "cEBMF",
    "cEBMF_object",
    "compute_hat_l_and_s_l",
    "compute_hat_f_and_s_f",
    "PiOptim",
    "choose_pi_optimizer",
    "EmdnPosteriorMeanNorm",  # expose name, load lazily below
]

def __getattr__(name):
    # Lazy import to avoid pulling torch / tensorflow at import time
    if name == "EmdnPosteriorMeanNorm":
        from .cebnm_solver.empirical_mdn import EmdnPosteriorMeanNorm
        return EmdnPosteriorMeanNorm
    raise AttributeError(f"module 'cebmf' has no attribute '{name}'")
