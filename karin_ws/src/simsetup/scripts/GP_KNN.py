# Data-adaptive lengthscale with Paciorek–Schervish kernel
# EXACT ORIGINAL LOGIC (square grid_size + rectangular support via padding)
# Adapted I/O for lawnmower simulation CSV (meters already)

# Imports
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular
from scipy.special import kv, gamma as gamma_fn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch

# =========================
# CUDA setup (unchanged)
# =========================
USE_CUDA = torch.cuda.is_available()
if USE_CUDA and torch.cuda.get_device_capability()[0] < 7:
    TORCH_DTYPE = torch.float32
    print("Using float32 for older GPU (compute capability < 7.0)")
else:
    TORCH_DTYPE = torch.float64
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Torch device:", DEVICE)

# =========================
# Utilities (unchanged)
# =========================
def sanitize(txt: str) -> str:
    """Safe folder/file names (letters, digits, underscore)"""
    return re.sub(r'[^0-9A-Za-z_]+', '_', str(txt))

def fit_linear_trend(X, y):
    """Fit linear trend for universal kriging (GP models residuals)"""
    import numpy.linalg as npl
    A = np.c_[np.ones(len(X)), X]      # [1, x, y]
    beta, *_ = npl.lstsq(A, y, rcond=None)
    def trend_fn(Xq):
        Aq = np.c_[np.ones(len(Xq)), Xq]
        return Aq @ beta
    return beta, trend_fn

# =========================
# I/O: lawnmower CSV (meters)
# ONLY CHANGE: read x_meters,y_meters,temperature_celsius; no GPS→UTM
# =========================
def process_lawnmower_csv(csv_path):
    """Process lawnmower simulation CSV file (meters already)"""
    csv_file = os.path.basename(csv_path)
    print(f"\n{'='*60}")
    print(f"Processing: {csv_file}")
    print(f"{'='*60}")

    data = pd.read_csv(csv_path)
    print(f"Loaded {len(data)} samples from: {csv_path}")

    # Required columns for this adapter
    required = ['x_meters', 'y_meters', 'temperature_celsius']
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(data.columns)}")

    print("\nHead:")
    print(data.head())
    print(f"X range (m): {data['x_meters'].min():.2f} → {data['x_meters'].max():.2f}")
    print(f"Y range (m): {data['y_meters'].min():.2f} → {data['y_meters'].max():.2f}")
    print(f"T range (°C): {data['temperature_celsius'].min():.2f} → {data['temperature_celsius'].max():.2f}")

    # Use meters directly (no UTM)
    data['X_coord'] = data['x_meters'].values
    data['Y_coord'] = data['y_meters'].values

    # Scale coordinates (same as original)
    scaler = StandardScaler()
    XY = data[['X_coord','Y_coord']].values
    XYs = scaler.fit_transform(XY)
    data['X_scaled'] = XYs[:,0]
    data['Y_scaled'] = XYs[:,1]
    print(f"Scaled coords mean={XYs.mean(axis=0)}, std={XYs.std(axis=0)}")

    # Prepare training
    X_train = data[['X_scaled','Y_scaled']].values
    y_train = data['temperature_celsius'].values

    # Linear trend + residuals (universal kriging)
    _, trend = fit_linear_trend(X_train, y_train)
    y_train_centered = y_train - trend(X_train)
    print(f"Trend fitted; residual mean ≈ {y_train_centered.mean():.4f}")

    date_tag = os.path.splitext(csv_file)[0]
    sigma_n = 0.1   # same noise default as original

    return data, X_train, y_train, y_train_centered, trend, "Temperature (°C)", date_tag, sigma_n

# =========================================================
# Hyperparameters / local anisotropy learning (unchanged)
# =========================================================
ell_min, ell_max = 0.30, 2.00        # scaled-units bounds for ℓ
k_nn_fit   = 20
k_nn_query = 30
idw_power  = 2.0
ratio_min, ratio_max = 1.2, 3.0      # ℓ_perp/ℓ_par bounds

def fit_local_anisotropic_field(X, y, k=k_nn_fit):
    """Learn direction-aware field (ℓ∥, ℓ⊥, u) from data — identical logic"""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(X)
    dists, idxs = nbrs.kneighbors(X, return_distance=True)
    N = X.shape[0]

    grads = np.zeros((N,2))
    for i in range(N):
        Xi = X[idxs[i]]
        yi = y[idxs[i]]
        A  = np.c_[np.ones(k), Xi - X[i]]      # centered
        beta, *_ = np.linalg.lstsq(A, yi, rcond=None)
        grads[i] = beta[1:3]

    r = np.linalg.norm(grads, axis=1)
    ql, qh = np.quantile(r, [0.05, 0.95])
    rhat = np.clip((r - ql) / max(qh-ql, 1e-6), 0.0, 1.0)

    # shorter across sharp fronts, longer in smooth areas
    ell_par_train  = ell_max - (ell_max-ell_min)*rhat
    ratio_train    = ratio_min + (ratio_max-ratio_min)*(1.0 - rhat)
    ell_perp_train = ell_par_train * ratio_train

    # directions (unit); fall back to x-axis if tiny gradient
    u_train = grads.copy()
    norms = np.linalg.norm(u_train, axis=1, keepdims=True)
    u_train = np.where(norms>1e-8, u_train/norms, np.array([1.0,0.0])[None,:])

    def aniso_params_of(Xq, kq=k_nn_query):
        dq, jq = nbrs.kneighbors(Xq, n_neighbors=kq, return_distance=True)
        w = 1.0 / (np.power(dq, idw_power) + 1e-6)
        # weighted avg of grads → direction
        gq = (w[...,None] * grads[jq]).sum(axis=1) / w.sum(axis=1, keepdims=True)
        gn = np.linalg.norm(gq, axis=1, keepdims=True)
        uq = np.where(gn>1e-8, gq/gn, np.array([[1.0,0.0]]))
        # ℓ fields
        lpar_q  = (w * ell_par_train[jq]).sum(axis=1) / w.sum(axis=1)
        ratio_q = (w * ratio_train[jq]).sum(axis=1) / w.sum(axis=1)
        lperp_q = lpar_q * ratio_q
        lpar_q  = np.clip(lpar_q,  ell_min, ell_max)
        lperp_q = np.clip(lperp_q, ell_min, ell_max*ratio_max)
        return lpar_q, lperp_q, uq

    return (ell_par_train, ell_perp_train, u_train, aniso_params_of)

# =========================================================
# Vectorized isotropic PS kernel (CPU fallback) — unchanged
# =========================================================
def build_cov_iso_ps(XA, XB, lA, lB, nu=1.5, sigma_f=1.0):
    """
    Paciorek–Schervish NS Matérn with Σ(x)=ℓ(x)^2 I (2D).
      prefactor = (2 ℓ_i ℓ_j) / (ℓ_i^2 + ℓ_j^2)
      Q_ij      = 2 ||x_i - x_j||^2 / (ℓ_i^2 + ℓ_j^2)
      k         = σ_f^2 * prefactor * [ (√(2νQ))^ν K_ν(√(2νQ)) ] / [ Γ(ν) 2^{ν-1} ]
    """
    XA = np.asarray(XA, dtype=np.float64); XB = np.asarray(XB, dtype=np.float64)
    lA = np.asarray(lA, dtype=np.float64); lB = np.asarray(lB, dtype=np.float64)

    XA2 = (XA**2).sum(axis=1)[:, None]
    XB2 = (XB**2).sum(axis=1)[None, :]
    D2  = XA2 + XB2 - 2.0 * XA @ XB.T

    LA2 = (lA**2)[:, None]
    LB2 = (lB**2)[None, :]
    Lsum = LA2 + LB2
    pref = (2.0 * (lA[:, None] * lB[None, :])) / np.clip(Lsum, 1e-12, np.inf)

    Q    = 2.0 * D2 / np.clip(Lsum, 1e-12, np.inf)
    Q    = np.clip(Q, 1e-12, np.inf)

    arg  = np.sqrt(2.0 * nu * Q)
    matern_part = (arg**nu) * kv(nu, arg)
    norm_const  = 1.0 / (gamma_fn(nu) * (2.0**(nu - 1.0)))

    K = (sigma_f**2) * pref * norm_const * matern_part
    return K

def ns_build_K_train_cpu(X_train, l_train, nu, sigma_f, sigma_n, jitter):
    K = build_cov_iso_ps(X_train, X_train, l_train, l_train, nu=nu, sigma_f=sigma_f)
    np.fill_diagonal(K, sigma_f**2)
    K[np.diag_indices_from(K)] += sigma_n**2 + jitter
    return K

def ns_build_K_star_cpu(X_star, X_train, l_star, l_train, nu, sigma_f):
    return build_cov_iso_ps(X_star, X_train, l_star, l_train, nu=nu, sigma_f=sigma_f)

# =========================================================
# CUDA isotropic PS kernel (unchanged)
# =========================================================
def _matern_shape_closed_form_torch(arg: torch.Tensor, nu: float) -> torch.Tensor:
    if nu == 0.5:
        return torch.exp(-arg)
    elif nu == 1.5:
        return (1.0 + arg) * torch.exp(-arg)
    elif nu == 2.5:
        return (1.0 + arg + (arg * arg) / 3.0) * torch.exp(-arg)
    else:
        raise NotImplementedError("CUDA path supports nu in {0.5, 1.5, 2.5} (closed form).")

def build_cov_iso_ps_torch(XA, XB, lA, lB, nu=1.5, sigma_f=1.0, eps=1e-12, device=DEVICE, dtype=TORCH_DTYPE):
    XA = torch.as_tensor(XA, device=device, dtype=dtype)
    XB = torch.as_tensor(XB, device=device, dtype=dtype)
    lA = torch.as_tensor(lA, device=device, dtype=dtype)
    lB = torch.as_tensor(lB, device=device, dtype=dtype)

    XA2 = (XA * XA).sum(dim=1).unsqueeze(1)
    XB2 = (XB * XB).sum(dim=1).unsqueeze(0)
    D2  = XA2 + XB2 - 2.0 * (XA @ XB.T)

    LA2 = (lA * lA).unsqueeze(1)
    LB2 = (lB * lB).unsqueeze(0)
    Lsum = (LA2 + LB2).clamp_min(eps)

    pref = (2.0 * (lA.unsqueeze(1) * lB.unsqueeze(0))) / Lsum
    Q    = 2.0 * D2 / Lsum
    Q    = Q.clamp_min(eps)

    arg  = torch.sqrt(torch.tensor(2.0*nu, device=device, dtype=dtype) * Q)
    shape = _matern_shape_closed_form_torch(arg, float(nu))
    K = (sigma_f**2) * pref * shape
    return K

# =========================================================
# CUDA anisotropic PS kernel (unchanged)
# =========================================================
def _sigma_from_u_l(ux, uy, lpar, lperp):
    lp2 = lpar*lpar; lt2 = lperp*lperp; d = (lp2 - lt2)
    s11 = lt2 + d*(ux*ux)
    s22 = lt2 + d*(uy*uy)
    s12 = d*(ux*uy)
    return s11, s12, s22

def build_cov_ps_aniso_torch(XA, XB, uA, uB, lparA, lperpA, lparB, lperpB,
                             nu=1.5, sigma_f=1.0, eps=1e-12,
                             device=DEVICE, dtype=TORCH_DTYPE):
    XA = torch.as_tensor(XA, device=device, dtype=dtype)
    XB = torch.as_tensor(XB, device=device, dtype=dtype)
    uA = torch.as_tensor(uA, device=device, dtype=dtype)
    uB = torch.as_tensor(uB, device=device, dtype=dtype)
    lparA = torch.as_tensor(lparA, device=device, dtype=dtype)
    lperpA= torch.as_tensor(lperpA,device=device, dtype=dtype)
    lparB = torch.as_tensor(lparB, device=device, dtype=dtype)
    lperpB= torch.as_tensor(lperpB,device=device, dtype=dtype)

    # Σ components
    s11A,s12A,s22A = _sigma_from_u_l(uA[:,0], uA[:,1], lparA, lperpA)
    s11B,s12B,s22B = _sigma_from_u_l(uB[:,0], uB[:,1], lparB, lperpB)

    # dets / prefactor
    detA = (s11A*s22A - s12A*s12A).clamp_min(eps)
    detB = (s11B*s22B - s12B*s12B).clamp_min(eps)
    logdetA = torch.log(detA); logdetB = torch.log(detB)

    M11 = 0.5*(s11A.unsqueeze(1) + s11B.unsqueeze(0))
    M22 = 0.5*(s22A.unsqueeze(1) + s22B.unsqueeze(0))
    M12 = 0.5*(s12A.unsqueeze(1) + s12B.unsqueeze(0))
    detM = (M11*M22 - M12*M12).clamp_min(eps)
    logdetM = torch.log(detM)

    logpref = 0.25*logdetA.unsqueeze(1) + 0.25*logdetB.unsqueeze(0) - 0.5*logdetM
    pref = torch.exp(logpref)

    # Q = d^T M^{-1} d
    dx = XA[:,0].unsqueeze(1) - XB[:,0].unsqueeze(0)
    dy = XA[:,1].unsqueeze(1) - XB[:,1].unsqueeze(0)
    inv_detM = 1.0 / detM
    Q = (M22*dx*dx - 2.0*M12*dx*dy + M11*dy*dy) * inv_detM
    Q = Q.clamp_min(eps)

    arg = torch.sqrt(torch.tensor(2.0*nu, device=device, dtype=dtype) * Q)
    shape = _matern_shape_closed_form_torch(arg, float(nu))
    K = (sigma_f**2) * pref * shape
    return K

# =========================================================
# Diagnostics + CUDA hyperparam fit (unchanged)
# =========================================================
def report_lengthscale_diagnostics(X, ell_par, ell_perp, ell_scale, ell_min=0.30, ell_max=2.00, ratio_max=3.0):
    from sklearn.neighbors import NearestNeighbors
    lp = np.clip(ell_scale*ell_par,  ell_min, ell_max)
    lt = np.clip(ell_scale*ell_perp, ell_min, ell_max*ratio_max)
    pmin_lp = 100.0*np.mean(lp <= ell_min+1e-9)
    pmin_lt = 100.0*np.mean(lt <= ell_min+1e-9)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    d = nn.kneighbors(X, return_distance=True)[0][:,1]
    print(f"[Diag] ℓ_scale={ell_scale:.4f} | ℓ‖ med={np.median(lp):.3f}, ℓ⊥ med={np.median(lt):.3f}")
    print(f"[Diag] % clamped at ℓ_min: ℓ‖={pmin_lp:.1f}%, ℓ⊥={pmin_lt:.1f}%")
    print(f"[Diag] NN dist p50={np.median(d):.3f}")

def fit_hypers_cuda_aniso(X, y_centered, u_train, lpar, lperp, nu=1.5, iters=80, lr=0.08):
    Xt = torch.as_tensor(X, device=DEVICE, dtype=TORCH_DTYPE)
    yt = torch.as_tensor(y_centered, device=DEVICE, dtype=TORCH_DTYPE).unsqueeze(1)
    uA = torch.as_tensor(u_train, device=DEVICE, dtype=TORCH_DTYPE)
    lpar0  = torch.as_tensor(lpar,  device=DEVICE, dtype=TORCH_DTYPE)
    lperp0 = torch.as_tensor(lperp, device=DEVICE, dtype=TORCH_DTYPE)

    log_sigma_f = torch.tensor([np.log(np.std(y_centered)+1e-6)], device=DEVICE,
                               dtype=TORCH_DTYPE, requires_grad=True)
    log_sigma_n = torch.tensor([np.log(0.1)], device=DEVICE,
                               dtype=TORCH_DTYPE, requires_grad=True)
    log_lscale  = torch.tensor([0.0], device=DEVICE,
                               dtype=TORCH_DTYPE, requires_grad=True)

    opt = torch.optim.Adam([log_sigma_f, log_sigma_n, log_lscale], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        sf = torch.exp(log_sigma_f)
        sn = torch.exp(log_sigma_n)
        ls = torch.exp(log_lscale)
        lpar_use  = (ls * lpar0).clamp(min=ell_min, max=ell_max)
        lperp_use = (ls * lperp0).clamp(min=ell_min, max=ell_max*ratio_max)

        K = build_cov_ps_aniso_torch(Xt, Xt, uA, uA, lpar_use, lperp_use, lpar_use, lperp_use,
                                     nu=float(nu), sigma_f=sf)
        K.diagonal().add_(sn*sn + 1e-6)
        K = 0.5 * (K + K.T)
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            K.diagonal().add_(1e-4)
            L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(yt, L)
        n = Xt.shape[0]
        nll = 0.5*(yt.T @ alpha) + torch.log(torch.diag(L)).sum() + \
              0.5*n*torch.log(torch.tensor(2*np.pi, device=DEVICE, dtype=TORCH_DTYPE))
        # weak priors
        prior_l = 0.5*(log_lscale/0.7)**2
        prior_n = 0.5*((log_sigma_n - np.log(0.03))/0.7)**2
        (nll + prior_l + prior_n).squeeze().backward()
        opt.step()
        with torch.no_grad():
            log_lscale.clamp_(min=-1.5, max=0.5)

    with torch.no_grad():
        return float(torch.exp(log_sigma_f)), float(torch.exp(log_sigma_n)), float(torch.exp(log_lscale))

# =========================================================
# CPU fallback NS-Matérn predictor (unchanged logic)
# (kept for parity; we still return square grid_size + padding)
# =========================================================
def ns_matern_gp_predict_adaptive(
    X_train, y_train_centered, trend_fn,
    aniso_params_of, ell_par_train, ell_perp_train,
    grid_size=80, pad=0.12, sigma_n=0.1, jitter=1e-6, out_dir=None,
    var_label="Temperature (°C)", date_tag="sim",
    nu=1.5, sigma_f=None
):
    if sigma_f is None:
        sigma_f = float(np.std(y_train_centered)) or 1.0

    # training K with isotropic average ℓ (original CPU logic)
    ell_train_cpu = (ell_par_train + ell_perp_train) / 2.0
    K = ns_build_K_train_cpu(X_train, ell_train_cpu, nu, sigma_f, sigma_n, jitter)
    K = 0.5 * (K + K.T)
    try:
        L = cholesky(K, lower=True, overwrite_a=False, check_finite=False)
    except np.linalg.LinAlgError:
        print(f"[NS-Matern] Cholesky failed, increase jitter to 1e-4")
        K[np.diag_indices_from(K)] += 1e-4
        L = cholesky(K, lower=True, overwrite_a=False, check_finite=False)
    alpha = solve_triangular(L, y_train_centered, lower=True, check_finite=False)
    alpha = solve_triangular(L.T, alpha, lower=False, check_finite=False)

    # ORIGINAL GRID LOGIC: same grid_size on X and Y; rectangular handled by pad
    (x_min, y_min) = X_train.min(axis=0); (x_max, y_max) = X_train.max(axis=0)
    dx, dy = (x_max-x_min), (y_max-y_min)
    gx = np.linspace(x_min - pad*dx, x_max + pad*dx, grid_size)
    gy = np.linspace(y_min - pad*dy, y_max + pad*dy, grid_size)
    Xg, Yg = np.meshgrid(gx, gy, indexing="ij")
    Xstar  = np.stack([Xg.ravel(), Yg.ravel()], axis=1)

    # interpolate ℓ at queries (IDW), then isotropic average
    lpar_q, lperp_q, _ = aniso_params_of(Xstar)
    ell_star = (lpar_q + lperp_q) / 2.0

    # predictions
    K_star = ns_build_K_star_cpu(Xstar, X_train, ell_star, ell_train_cpu, nu, sigma_f)
    mu_centered = K_star @ alpha
    mu = mu_centered + trend_fn(Xstar)

    # variance / entropy
    K_ss_diag = (sigma_f**2) * np.ones(Xstar.shape[0], dtype=np.float64)
    v = solve_triangular(L, K_star.T, lower=True, check_finite=False)
    var_latent = np.maximum(K_ss_diag - np.einsum('ij,ij->j', v, v), 1e-12)
    ent = 0.5 * np.log(2 * np.pi * np.e * var_latent)

    Zmu  = mu.reshape(grid_size, grid_size)
    Zent = ent.reshape(grid_size, grid_size)

    # crop to data bbox for plotting
    ix = (gx>=x_min) & (gx<=x_max)
    iy = (gy>=y_min) & (gy<=y_max)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(6.5,5.6))
        plt.contourf(gy[iy], gx[ix], Zmu[np.ix_(ix,iy)].T, 20, cmap='viridis')
        plt.colorbar(label=var_label)
        plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
        plt.title(f"{date_tag} – {var_label} mean (NS-Matérn, CPU)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sanitize(var_label)}_NSMatern_cpu_mean.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(6.5,5.6))
        plt.contourf(gy[iy], gx[ix], Zent[np.ix_(ix,iy)].T, 20, cmap='inferno')
        plt.colorbar(label="Entropy (latent)")
        plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
        plt.title(f"{date_tag} – {var_label} uncertainty (NS-Matérn, CPU)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sanitize(var_label)}_NSMatern_cpu_uncert.png"), dpi=160)
        plt.close()

    print(f"CPU fallback: ℓ bounds [{ell_min},{ell_max}] | ℓ_train median={np.median(ell_train_cpu):.3f}")
    return gx, gy, Zmu, Zent

# =========================================================
# CUDA anisotropic predictor — ORIGINAL GRID LOGIC
# (square grid_size; rectangular via padding only)
# =========================================================
@torch.no_grad()
def ns_matern_gp_predict_aniso_cuda(
    X_train, y_train_centered, trend_fn,
    aniso_params_of, u_train, ell_par_train, ell_perp_train,
    grid_size=100, pad=0.12,
    nu=1.5, sigma_f=None, sigma_n=0.1, ell_scale=1.0,
    batch_G=8192, out_dir=None, var_label="Temperature (°C)", date_tag="sim"
):
    # scale ℓ-fields (unchanged)
    lpar_tr  = np.clip(ell_scale*ell_par_train,  ell_min, ell_max)
    lperp_tr = np.clip(ell_scale*ell_perp_train, ell_min, ell_max*ratio_max)

    # move to GPU & build K
    Xt  = torch.as_tensor(X_train, device=DEVICE, dtype=TORCH_DTYPE)
    yt  = torch.as_tensor(y_train_centered, device=DEVICE, dtype=TORCH_DTYPE).unsqueeze(1)
    u_t = torch.as_tensor(u_train, device=DEVICE, dtype=TORCH_DTYPE)
    lp_t  = torch.as_tensor(lpar_tr,  device=DEVICE, dtype=TORCH_DTYPE)
    lt_t  = torch.as_tensor(lperp_tr, device=DEVICE, dtype=TORCH_DTYPE)
    if sigma_f is None: sigma_f = float(np.std(y_train_centered) or 1.0)

    K = build_cov_ps_aniso_torch(Xt, Xt, u_t, u_t, lp_t, lt_t, lp_t, lt_t, nu=float(nu), sigma_f=float(sigma_f))
    K.diagonal().add_(sigma_n**2 + 1e-6)
    K = 0.5 * (K + K.T)
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(yt, L)[:,0]

    # ORIGINAL padded grid (same grid_size both axes)
    (x_min, y_min) = X_train.min(axis=0); (x_max, y_max) = X_train.max(axis=0)
    dx, dy = (x_max-x_min), (y_max-y_min)
    gx = np.linspace(x_min - pad*dx, x_max + pad*dx, grid_size)
    gy = np.linspace(y_min - pad*dy, y_max + pad*dy, grid_size)
    Xg, Yg = np.meshgrid(gx, gy, indexing="ij")
    Xstar  = np.stack([Xg.ravel(), Yg.ravel()], axis=1)

    # interpolate anisotropic params at queries
    lpar_q, lperp_q, u_q = aniso_params_of(Xstar)
    Xst  = torch.as_tensor(Xstar, device=DEVICE, dtype=TORCH_DTYPE)
    u_qt = torch.as_tensor(u_q,   device=DEVICE, dtype=TORCH_DTYPE)
    lp_qt= torch.as_tensor(np.clip(ell_scale*lpar_q,  ell_min, ell_max), device=DEVICE, dtype=TORCH_DTYPE)
    lt_qt= torch.as_tensor(np.clip(ell_scale*lperp_q, ell_min, ell_max*ratio_max), device=DEVICE, dtype=TORCH_DTYPE)

    G = Xstar.shape[0]
    mu_resid = torch.empty(G, device=DEVICE, dtype=TORCH_DTYPE)
    var_lat  = torch.empty(G, device=DEVICE, dtype=TORCH_DTYPE)
    s2 = sigma_f**2

    for beg in range(0, G, batch_G):
        end = min(G, beg+batch_G)
        Ks = build_cov_ps_aniso_torch(Xst[beg:end], Xt, u_qt[beg:end], u_t, lp_qt[beg:end], lt_qt[beg:end], lp_t, lt_t,
                                      nu=float(nu), sigma_f=float(sigma_f))
        mu_resid[beg:end] = Ks @ alpha
        v = torch.linalg.solve_triangular(L, Ks.T, upper=False)
        var_lat[beg:end] = torch.maximum(s2 - (v*v).sum(dim=0), torch.tensor(1e-12, device=DEVICE, dtype=TORCH_DTYPE))

    # add back trend
    mu = (mu_resid.cpu().numpy() + trend_fn(Xstar))
    ent = 0.5*np.log(2*np.pi*np.e*var_lat.cpu().numpy())

    Zmu, Zent = mu.reshape(grid_size, grid_size), ent.reshape(grid_size, grid_size)

    # crop to data bbox for plotting
    ix = (gx>=x_min) & (gx<=x_max)
    iy = (gy>=y_min) & (gy<=y_max)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(6.5,5.6))
        plt.contourf(gy[iy], gx[ix], Zmu[np.ix_(ix,iy)].T, 20, cmap='viridis')
        plt.colorbar(label=var_label)
        plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
        plt.title(f"{date_tag} – {var_label} (KNN-Based Non-Stationary GP)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sanitize(var_label)}_KNN_NSGP_mean.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(6.5,5.6))
        plt.contourf(gy[iy], gx[ix], Zent[np.ix_(ix,iy)].T, 20, cmap='inferno')
        plt.colorbar(label="Uncertainty")
        plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
        plt.title(f"{date_tag} – {var_label} Uncertainty (KNN-Based Non-Stationary GP)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sanitize(var_label)}_KNN_NSGP_uncertainty.png"), dpi=160)
        plt.close()

    print(f"[CUDA Aniso] ν={nu}, σ_f≈{sigma_f:.3f}, σ_n={sigma_n:.3f}, ℓ_scale={ell_scale:.3f}")
    return gx, gy, Zmu, Zent

# =========================================================
# Driver (unchanged logic; only the CSV reader differs)
# =========================================================
def run_reconstruction(csv_path, grid_size=100, pad=0.12, nu=1.5):
    print("\nProcessing lawnmower data with KNN Non-Stationary GP (original logic)")
    (data, X_train, y_train, y_train_centered,
     trend, var_label, date_tag, sigma_n) = process_lawnmower_csv(csv_path)

    # Fit anisotropic field
    ell_par_train, ell_perp_train, u_train, aniso_params_of = fit_local_anisotropic_field(
        X_train.astype(np.float64), y_train_centered.astype(np.float64), k=k_nn_fit
    )
    print(f"ℓ∥ stats: min={ell_par_train.min():.2f}, med={np.median(ell_par_train):.2f}, max={ell_par_train.max():.2f}")
    print(f"ℓ⊥/ℓ∥ stats: min={(ell_perp_train/ell_par_train).min():.2f}, med={np.median(ell_perp_train/ell_par_train):.2f}, max={(ell_perp_train/ell_par_train).max():.2f}")

    # Output dir
    out_dir = os.path.join(os.path.dirname(csv_path), "gp_reconstruction")
    os.makedirs(out_dir, exist_ok=True)

    # Diagnostics
    plt.figure(figsize=(8,6))
    plt.scatter(X_train[:,0], X_train[:,1], c=np.log10(ell_perp_train/ell_par_train),
                s=6, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='log10 anisotropy ratio (ℓ⊥/ℓ∥)')
    plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
    plt.title(f"{date_tag} - KNN-Based Anisotropy Field")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{date_tag}_anisotropy_field.png"), dpi=160)
    plt.close()

    if USE_CUDA:
        print("Using CUDA-accelerated anisotropic implementation...")
        sigma_f_hat, sigma_n_hat, ell_scale = fit_hypers_cuda_aniso(
            X_train, y_train_centered, u_train, ell_par_train, ell_perp_train, nu=nu, iters=80, lr=0.08
        )
        print(f"Fitted hyperparams: σ_f={sigma_f_hat:.3f}, σ_n={sigma_n_hat:.3f}, ℓ_scale={ell_scale:.3f}")

        report_lengthscale_diagnostics(X_train, ell_par_train, ell_perp_train, ell_scale,
                                       ell_min=ell_min, ell_max=ell_max, ratio_max=ratio_max)

        gx, gy, Zmu, Zent = ns_matern_gp_predict_aniso_cuda(
            X_train=X_train.astype(np.float64),
            y_train_centered=y_train_centered.astype(np.float64),
            trend_fn=trend,
            aniso_params_of=aniso_params_of,
            u_train=u_train,
            ell_par_train=ell_par_train,
            ell_perp_train=ell_perp_train,
            grid_size=grid_size, pad=pad,
            nu=nu, sigma_f=sigma_f_hat, sigma_n=sigma_n_hat, ell_scale=ell_scale,
            batch_G=8192, out_dir=out_dir, var_label=var_label, date_tag=date_tag
        )
    else:
        print("CUDA not available — using CPU isotropic fallback (original logic)")
        gx, gy, Zmu, Zent = ns_matern_gp_predict_adaptive(
            X_train=X_train.astype(np.float64),
            y_train_centered=y_train_centered.astype(np.float64),
            trend_fn=trend,
            aniso_params_of=aniso_params_of,
            ell_par_train=ell_par_train,
            ell_perp_train=ell_perp_train,
            grid_size=grid_size, pad=pad,
            sigma_n=sigma_n, jitter=1e-5,
            out_dir=out_dir, var_label=var_label, date_tag=date_tag,
            nu=nu, sigma_f=None
        )

    print("\n" + "="*60)
    print("Reconstruction complete (original padding/grid logic).")
    print(f"Results saved to: {out_dir}")
    print("="*60 + "\n")
    return gx, gy, Zmu, Zent

if __name__ == "__main__":
    # Point to your lawnmower sim data
    # Example:
    # csv_path = "/home/blazar/karin_ws/src/simsetup/scripts/data/sim_data/lawnmower_samples_20251006_121620.csv"
    csv_path = "/home/blazar/karin_ws/src/simsetup/scripts/data/sim_data/sim1.csv"

    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        print("Please update csv_path to your data file.")
    else:
        run_reconstruction(csv_path, grid_size=100, pad=0.12, nu=1.5)
