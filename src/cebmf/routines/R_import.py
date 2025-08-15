 # pi_optimizer.py
import numpy as np
from enum import Enum
from typing import Optional, Tuple

# ====== MODE / INDICATOR ======

class PiOptim(Enum):
    AUTO = "auto"     # decide per-call
    MIXSQP = "mixsqp" # force mixsqp (error if unavailable)
    EM = "em"         # force EM

# Cache for availability so we don't probe R every time
_MIXSQP_AVAILABLE: Optional[bool] = None

def _probe_mixsqp_available() -> bool:
    """Return True iff rpy2 and R package 'mixsqp' are importable."""
    global _MIXSQP_AVAILABLE
    if _MIXSQP_AVAILABLE is not None:
        return _MIXSQP_AVAILABLE
    try:
        from rpy2.robjects import packages  # noqa: F401
        packages.importr('mixsqp')          # raises if not installed
        _MIXSQP_AVAILABLE = True
    except Exception:
        _MIXSQP_AVAILABLE = False
        raise ImportError(
            "Optional dependency `rpy2` is not installed or the R package `mixsqp` is missing.\n"
            "To use fast optimization via mixsqp for ebnm solvers \n"
            " This is usefull if you have prior without covariate for columns or rows\n"
            "espicially when the number of columns /rows is large (over 10,0000)\n"
            "Make sure R and the R package `mixsqp` are installed. For example, in R:\n"
            "  install.packages('remotes')\n"
            "  remotes::install_github('stephenslab/mixsqp')\n"
            "Or equivalently:\n"
            "  install.packages('mixsqp')\n"
        )
    return _MIXSQP_AVAILABLE




 


def choose_pi_optimizer(prefer_mixsqp: bool = True) -> PiOptim:
    from .R_import import _has_mixsqp_silent  # or move function above
    return PiOptim.MIXSQP if (prefer_mixsqp and _has_mixsqp_silent()) else PiOptim.EM

 

def _has_mixsqp_silent() -> bool:
    """Return True iff rpy2 + R package 'mixsqp' are importable. Never raises."""
    global _MIXSQP_AVAILABLE
    if _MIXSQP_AVAILABLE is not None:
        return _MIXSQP_AVAILABLE
    try:
        from rpy2.robjects import packages  # noqa: F401
        packages.importr('mixsqp')
        _MIXSQP_AVAILABLE = True
    except Exception:
        _MIXSQP_AVAILABLE = False
    return _MIXSQP_AVAILABLE