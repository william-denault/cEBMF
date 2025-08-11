try:
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False

def run_mixsqp(X, y):
    if not RPY2_AVAILABLE:
        raise ImportError(
            "Optional dependency `rpy2` is not installed.\n"
            "To use fast optimization via mixsqp, install:\n"
            "  pip install cEBMF[r]\n"
            "Also make sure R and the R package `mixsqp` are installed. For example, in R:\n"
            "  install.packages('remotes')\n"
            "  remotes::install_github('stephenslab/mixsqp')\n"
            "Or equivalently:\n"
            "  install.packages('mixsqp')\n"
        )
    try:
        mixsqp = importr('mixsqp')
    except Exception as e:
        raise ImportError(
            "R package `mixsqp` is not available or failed to import.\n"
            "Make sure it is installed in your R environment. Original error:\n"
            f"{e}"
        )

    # TODO: adapt the argument passing to mixsqp based on your needs
    result = mixsqp.mixsqp(X, y)
    return result