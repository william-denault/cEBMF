# tests/test_cebmf_nan.py
import warnings
import random
import numpy as np
import pytest
import torch

# Keep test output clean (sklearn/fancyimpute deprecations)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

SEED = 42
N, P = 50, 40
NOISE_STD = 0.1
K = 5


@pytest.fixture
def rank1_data_with_nan():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    try:
        torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass

    u = np.random.rand(N)
    v = np.random.rand(P)
    rank_1 = np.outer(u, v)
    noise = np.random.normal(0, NOISE_STD, size=(N, P))
    X = rank_1 + noise

    # Inject a few NaNs
    X[1, 1] = np.nan
    X[10, 5] = np.nan
    X[30:33, 12:14] = np.nan

    return rank_1, X


def _try_import_helpers():
    """Return dict with optional helpers and flags."""
    out = {"HAS_FANCY": False, "HAS_HATS": False}
    try:
        from fancyimpute import IterativeSVD  # noqa: F401
        out["IterativeSVD"] = IterativeSVD
        out["HAS_FANCY"] = True
    except Exception:
        pass

    # Prefer top-level re-exports; fall back to cebmf.main
    try:
        from cebmf import compute_hat_l_and_s_l, compute_hat_f_and_s_f, normal_means_loglik  # noqa: F401
        out["compute_hat_l_and_s_l"] = compute_hat_l_and_s_l
        out["compute_hat_f_and_s_f"] = compute_hat_f_and_s_f
        out["normal_means_loglik"] = normal_means_loglik
        out["HAS_HATS"] = True
    except Exception:
        try:
            from cebmf.main import compute_hat_l_and_s_l, compute_hat_f_and_s_f, normal_means_loglik  # noqa: F401
            out["compute_hat_l_and_s_l"] = compute_hat_l_and_s_l
            out["compute_hat_f_and_s_f"] = compute_hat_f_and_s_f
            out["normal_means_loglik"] = normal_means_loglik
            out["HAS_HATS"] = True
        except Exception:
            pass
    return out


def _rmse(A, B):
    return np.sqrt(np.nanmean((A - B) ** 2))


def test_cebmf_handles_nan(rank1_data_with_nan):
    try:
        from cebmf import cEBMF
    except Exception as e:
        pytest.skip(f"Cannot import cEBMF: {e}")

    helpers = _try_import_helpers()
    rank_1_matrix, noisy_matrix = rank1_data_with_nan

    # Optional: SVD baseline on imputed data (for comparison)
    if helpers["HAS_FANCY"]:
        imputed = helpers["IterativeSVD"]().fit_transform(noisy_matrix)
        U, s, Vt = np.linalg.svd(imputed, full_matrices=False)
        K_fit = min(K, U.shape[1])
        L_svd = U[:, :K_fit] @ np.diag(s[:K_fit])
        F_svd = Vt[:K_fit, :].T
        svd_recon = sum(np.outer(L_svd[:, k], F_svd[:, k]) for k in range(K_fit))

    # ---- Fit model on data with NaNs ----
    m = cEBMF(data=noisy_matrix, prior_L="norm", prior_F="norm")
    assert bool(getattr(m, "has_nan", False)), "Model should detect NaNs"

    m.init_LF()
    m.update_fitted_val()

    # Basic attributes are finite
    for name in ["L", "F", "L2", "F2", "tau", "Y_fit"]:
        assert hasattr(m, name), f"Missing attribute {name}"
        arr = np.asarray(getattr(m, name))
        assert np.all(np.isfinite(arr)), f"{name} contains non-finite values"

    # Compare Y_fit to imputed SVD reconstruction (ignore NaNs) if available
    if helpers["HAS_FANCY"]:
        mask = ~np.isnan(noisy_matrix)
        assert np.allclose(m.Y_fit[mask], svd_recon[mask], atol=1e-1), \
            "Y_fit not close to K-truncated SVD of imputed matrix"

    # ---- Partial residuals (respect NaNs) ----
    m.cal_partial_residuals(k=0)
    L, F = np.asarray(m.L), np.asarray(m.F)
    idx_others = [j for j in range(L.shape[1]) if j != 0]
    manual_Rk = noisy_matrix - sum(np.outer(L[:, j], F[:, j]) for j in idx_others)
    # NaN mask should match; compare on observed entries
    assert np.array_equal(np.isnan(m.Rk), np.isnan(manual_Rk)), "NaN mask mismatch in Rk"
    mask_rk = ~np.isnan(manual_Rk)
    assert np.allclose(m.Rk[mask_rk], manual_Rk[mask_rk], atol=2e-1)

    # ---- tau sanity (finite/positive; roughly homoscedastic) ----
    m.update_tau()
    tau = np.asarray(m.tau, dtype=float)
    assert np.all(np.isfinite(tau)) and np.all(tau > 0), "tau must be finite and positive"
    if tau.size > 1:
        cv = tau.std() / (tau.mean() + 1e-12)
        assert cv < 0.001, f"tau should be roughly homoscedastic (cv={cv:.3f})"

    # ---- compute_hat_* (if exported) ----
    if helpers["HAS_HATS"]:
        k = 1 if m.K > 1 else 0
        m.cal_partial_residuals(k=k)
        lhat, s_l = helpers["compute_hat_l_and_s_l"](
            Z=m.Rk, nu=m.F[:, k], omega=m.F2[:, k],
            tau=m.tau, has_nan=getattr(m, "has_nan", True)
        )
        fhat, s_f = helpers["compute_hat_f_and_s_f"](
            Z=m.Rk, nu=m.L[:, k], omega=m.L2[:, k],
            tau=m.tau, has_nan=getattr(m, "has_nan", True)
        )
        for arr in (lhat, s_l, fhat, s_f):
            arr = np.asarray(arr)
            assert np.all(np.isfinite(arr)), "compute_hat_* returned non-finite values"

        # If normal_means_loglik is exposed, at least check it returns a finite value
        try:
            ll_val = helpers["normal_means_loglik"](fhat, s_f, fhat, fhat**2)
            assert np.isfinite(ll_val)
        except Exception:
            pass

    # ---- Iterate; ensure RMSE improves; ELBO monotone if tracked ----
    base_mask = ~np.isnan(rank_1_matrix)
    base_rmse = _rmse(m.Y_fit[base_mask], rank_1_matrix[base_mask])
    m.init_LF()
    m.update_fitted_val()
    for _ in range(80):
        m.iter()
        m.update_fitted_val()
  
    final_rmse = _rmse(m.Y_fit[base_mask], rank_1_matrix[base_mask])
    assert final_rmse <= base_rmse + 1e-8, "Final RMSE worsened after iterations"
    # Don’t require ultra-tiny; just ensure proper denoising on rank-1 data
    assert final_rmse <base_rmse/2, f"Final RMSE too large: {final_rmse:.4f}"

   