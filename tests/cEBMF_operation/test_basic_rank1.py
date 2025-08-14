# tests/test_cebmf_single.py
import numpy as np
import random
import pytest
import torch

# --------------------- Global seeding ---------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
try:
    torch.cuda.manual_seed_all(SEED)
except Exception:
    pass


# --------------------- Fixtures --------------------------
@pytest.fixture(scope="session")
def rank1_data():
    """Synthetic rank-1 matrix with homoscedastic Gaussian noise."""
    n, p = 50, 40
    noise_std = 0.1
    rng = np.random.default_rng(SEED)
    u = rng.random(n)
    v = rng.random(p)
    rank_1_matrix = np.outer(u, v)
    noise = rng.normal(0, noise_std, size=(n, p))
    noisy_matrix = rank_1_matrix + noise
    return {
        "n": n,
        "p": p,
        "noise_std": noise_std,
        "u": u,
        "v": v,
        "rank_1_matrix": rank_1_matrix,
        "noisy_matrix": noisy_matrix,
    }


# --------------------- Helpers ---------------------------
def rmse(A, B):
    return np.sqrt(np.mean((A - B) ** 2))


# --------------------- Tests -----------------------------
def test_import_and_init(rank1_data):
    try:
        from cebmf import cEBMF
    except Exception as e:
        pytest.skip(f"Cannot import cebmf.cEBMF: {e}")

    X = rank1_data["noisy_matrix"]
    m = cEBMF(data=X)

    cebmf_object = getattr(cEBMF, "cEBMF_object", None)
    if cebmf_object is not None:
        assert isinstance(m, cebmf_object), "model is not a cEBMF_object"

    for name in ["init_LF", "update_fitted_val", "update_tau", "cal_partial_residuals", "iter"]:
        assert hasattr(m, name), f"Missing method: {name}"


def test_yfit_approximates_truncated_svd(rank1_data):
    from numpy.linalg import svd
    from cebmf import cEBMF

    X = rank1_data["noisy_matrix"]
    K = 5

    U, s, Vt = svd(X, full_matrices=False)
    K_eff = min(K, U.shape[1])
    Uk = U[:, :K_eff]
    Dk = np.diag(s[:K_eff])
    Vk = Vt[:K_eff, :]
    L = Uk @ Dk
    F = Vk.T
    X_svdK = sum(np.outer(L[:, k], F[:, k]) for k in range(K_eff))

    m = cEBMF(data=X)
    m.init_LF()
    m.update_fitted_val()
    assert hasattr(m, "Y_fit"), "Y_fit missing after update_fitted_val()"
    assert np.allclose(m.Y_fit, X_svdK, atol=1e-3, rtol=1e-3), \
        "Y_fit should approximate the K-truncated SVD reconstruction"


def test_partial_residuals_consistency(rank1_data):
    from cebmf import cEBMF

    X = rank1_data["noisy_matrix"]
    m = cEBMF(data=X)
    m.init_LF()
    m.update_fitted_val()

    assert hasattr(m, "L") and hasattr(m, "F"), "Model must expose L and F"
    L = np.asarray(m.L)
    F = np.asarray(m.F)
    assert L.ndim == 2 and F.ndim == 2 and L.shape[1] == F.shape[1], "L/F shape mismatch"

    k = 0
    m.cal_partial_residuals(k=k)
    assert hasattr(m, "Rk"), "cal_partial_residuals(k) must set Rk"

    idx_others = [j for j in range(F.shape[1]) if j != k]
    manual_Rk = X - sum(np.outer(L[:, j], F[:, j]) for j in idx_others)
    assert np.allclose(m.Rk, manual_Rk, atol=1e-6), "Rk mismatch with manual computation"


def test_update_tau_positive_and_homoscedastic(rank1_data):
    from cebmf import cEBMF

    X = rank1_data["noisy_matrix"]
    m = cEBMF(data=X)
    m.init_LF()
    m.update_tau()
    assert hasattr(m, "tau"), "tau missing after update_tau()"

    tau = np.asarray(m.tau, dtype=float)
    assert np.all(np.isfinite(tau)), "tau must be finite"
    assert np.all(tau > 0), "tau must be positive"
    if tau.ndim >= 2:
        assert np.allclose(tau, tau.flat[0]), "tau should be constant across entries"


def test_prior_F_callable_if_exposed(rank1_data):
    from cebmf import cEBMF

    X = rank1_data["noisy_matrix"]
    m = cEBMF(data=X)
    prior_F = getattr(m, "prior_F", None)
    if prior_F is None:
        pytest.skip("Model does not expose prior_F")
    assert callable(prior_F), "prior_F should be callable"


def test_compute_hat_functions_if_available(rank1_data):
    import numpy as np
    import pytest

    try:
        import cebmf
        from cebmf import cEBMF
    except Exception as e:
        pytest.skip(f"Cannot import cebmf/cEBMF: {e}")

    X = rank1_data["noisy_matrix"]
    m = cEBMF(data=X)
    m.init_LF()
    k = 0
    m.cal_partial_residuals(k=k)

    # Try instance methods first, then package top-level, then submodule
    fn_l = getattr(m, "compute_hat_l_and_s_l", None) \
        or getattr(cebmf, "compute_hat_l_and_s_l", None)
    fn_f = getattr(m, "compute_hat_f_and_s_f", None) \
        or getattr(cebmf, "compute_hat_f_and_s_f", None)

    if fn_l is None or fn_f is None:
        pytest.skip("compute_hat_* functions not available on instance or package")

    for attr in ["F", "F2", "L", "L2", "tau", "Rk"]:
        assert hasattr(m, attr), f"Missing attribute: {attr}"

    res_l = fn_l(Z=m.Rk, nu=m.F[:, k], omega=m.F2[:, k], tau=m.tau)
    res_f = fn_f(Z=m.Rk, nu=m.L[:, k], omega=m.L2[:, k], tau=m.tau)

    for res in (res_l, res_f):
        arrs = res if isinstance(res, tuple) else (res,)
        for arr in arrs:
            arr = np.asarray(arr)
            assert np.all(np.isfinite(arr)), "Non-finite values from compute_hat_*"



def test_iter_improves_fit_and_elbo_monotone(rank1_data):
    """Check RMSE improves and ELBO/objective is monotone non-decreasing."""
    from cebmf import cEBMF

    X = rank1_data["noisy_matrix"]
    X_true = rank1_data["rank_1_matrix"]

    m = cEBMF(data=X)
    m.init_LF()
    m.update_fitted_val()
    assert hasattr(m, "Y_fit"), "Y_fit missing"

    base = rmse(m.Y_fit, X_true)

    # Run several iterations, updating Y_fit each time
    for _ in range(40):
        m.iter()
        m.update_fitted_val()

    improved = rmse(m.Y_fit, X_true)
    assert improved <= base + 1e-8, "Fit worsened after iterations"
    assert improved < base/2, f"RMSE still high after iterations: {improved:.4f}"

    # --- ELBO strict monotonicity check (non-decreasing) ---
    obj = getattr(m, "obj", None) or getattr(m, "elbo_history", None)
    obj=obj[1:]
    if obj is None:
        pytest.skip("Model does not track an objective/ELBO (obj or elbo_history)")
    obj = np.asarray(obj, dtype=float)
    assert np.all(np.isfinite(obj)), "Objective contains non-finite values"
    # Allow exactly non-decreasing (ties OK)
    assert all(b >= a for a, b in zip(obj, obj[1:])), \
        "Objective/ELBO should be monotone non-decreasing across iterations"


def test_L_other_columns_small_on_rank1(rank1_data):
    """For rank-1 data, non-first columns of L should be small compared to the first."""
    from cebmf import cEBMF

    X = rank1_data["noisy_matrix"]
    m = cEBMF(data=X)
    m.init_LF()
    for _ in range(40):
        m.iter()

    if not hasattr(m, "L"):
        pytest.skip("Model does not expose L")

    L = np.asarray(m.L)
    if L.ndim != 2 or L.shape[1] <= 1:
        pytest.skip("Not enough factors to test sparsity on other columns of L")

    col0 = np.linalg.norm(L[:, 0])
    other = np.linalg.norm(L[:, 1:]) if L.shape[1] > 1 else 0.0
    assert other <= 0.1 * col0 + 1e-8, "Other L columns should be near zero on rank-1 data"

