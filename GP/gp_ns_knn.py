# Data-adaptive lengthscale with Paciorek-Schervish kernel

# Imports
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular
from scipy.special import kv, gamma as gamma_fn
import pyproj
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch

# CUDA setup
USE_CUDA = torch.cuda.is_available()
# Double precision on some GPUs is slow - use float32 for older GPUs
if USE_CUDA and torch.cuda.get_device_capability()[0] < 7:
    TORCH_DTYPE = torch.float32
    print("Using float32 for older GPU (compute capability < 7.0)")
else:
    TORCH_DTYPE = torch.float64     # keep high precision for stability
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Torch device:", DEVICE)

# Utility functions
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

def process_single_csv(csv_path, csv_file):
    """Process a single CSV file and generate GP predictions"""
    print(f"\n{'='*60}")
    print(f"Processing: {csv_file}")
    print(f"{'='*60}")
    
    data = pd.read_csv(csv_path)
    print(f"Loaded {len(data)} data points from: {csv_path}")

    # Display dataset info
    print("First few rows of raw data:")
    print(data.head())
    print(f"Raw Latitude range: {data['Latitude'].min()} to {data['Latitude'].max()}")
    print(f"Raw Longitude range: {data['Longitude'].min()} to {data['Longitude'].max()}")

    # Convert lat/lon to UTM coordinates
    utm_crs = pyproj.CRS("EPSG:32612")  # UTM Zone 12N for Tempe, Arizona
    wgs84_crs = pyproj.CRS("EPSG:4326")  # WGS84 Latitude/Longitude
    transformer = pyproj.Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)

    # Apply transformation to convert Longitude/Latitude to UTM (X, Y)
    data["X_coord"], data["Y_coord"] = transformer.transform(
        data["Longitude"].values,  # Input longitude (degrees)
        data["Latitude"].values    # Input latitude (degrees)
    )

    print("Sample processed data (in meters):")
    print(data[["Latitude", "Longitude", "X_coord", "Y_coord"]].head())
    print(f"X_coord range (meters): {data['X_coord'].min()} to {data['X_coord'].max()}")
    print(f"Y_coord range (meters): {data['Y_coord'].min()} to {data['Y_coord'].max()}")

    # Scale the UTM coordinates
    scaler = StandardScaler()
    utm_vals = data[['X_coord','Y_coord']].values
    utm_scaled = scaler.fit_transform(utm_vals)

    # Store back into the DataFrame
    data['X_scaled'], data['Y_scaled'] = utm_scaled[:,0], utm_scaled[:,1]
    print(f"UTM coords scaled: mean={utm_scaled.mean(axis=0)}, std={utm_scaled.std(axis=0)}")

    # Prepare training data
    max_points_per_file = 1000000000  # Use all data points
    n = len(data)
    sampled_indices = data.sample(min(n, max_points_per_file), random_state=42).index
    sampled_indices = sorted(sampled_indices)

    # Extract target variable and features
    target_var = "Temperature (°C)"
    y = data[target_var].values
    X_features = data[["X_scaled", "Y_scaled"]].values

    # Select training data
    X_train = X_features[sampled_indices]
    y_train = y[sampled_indices]

    print(f"Total data points: {len(data)}, Training subset size: {X_train.shape[0]}")
    print("Example training point (2D):", X_train[0])

    # Re-fit scaler on training subset to avoid data leakage
    scaler = StandardScaler()
    utm_train = data.loc[sampled_indices, ['X_coord', 'Y_coord']].values
    utm_train_scaled = scaler.fit_transform(utm_train)

    # Overwrite scaled columns for sampled rows
    data.loc[sampled_indices, 'X_scaled'] = utm_train_scaled[:, 0]
    data.loc[sampled_indices, 'Y_scaled'] = utm_train_scaled[:, 1]

    # Refresh features
    X_features = data[['X_scaled', 'Y_scaled']].values
    X_train = X_features[sampled_indices]

    # Set hyperparameters
    nu_default = 0.5  # Default Matérn smoothness (can be overridden in function calls)
    sigma_f = np.std(y_train)  # Signal standard deviation
    sigma_n = 0.1  # Noise standard deviation

    print("\nHyperparameters:")
    print(f"nu_default = {nu_default}, sigma_f = {sigma_f:.2f}, sigma_n = {sigma_n}")
    print("Note: nu=1.5 will be used in the adaptive GP for better behavior")
    print("CUDA closed-form Matérn supports ν ∈ {0.5, 1.5, 2.5}")

    # Fit linear trend and center residuals (universal kriging)
    _, trend = fit_linear_trend(X_train, y_train)
    y_train_centered = y_train - trend(X_train)   # residuals (mean=~0)
    y_mean = 0.0  # residuals are centered
    print(f"Linear trend fitted, modeling residuals (mean≈{y_train_centered.mean():.3f})")
    
    return data, X_train, y_train, y_train_centered, trend, target_var, csv_file, sigma_n

# Load and preprocess data - process all temperature CSV files
data_dir = r"C:\Workspace\spacerobotics\GP\data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'depth' not in f.lower()]
print(f"Found {len(csv_files)} temperature CSV files: {csv_files}")

# =============================================================================
# DIRECTION-AWARE ANISOTROPIC LENGTHSCALE LEARNING
# =============================================================================

# Choose reasonable, non-cheaty bounds in *scaled* units
ell_min, ell_max = 0.30, 2.00        # bounds for ℓ_parallel (scaled units)
k_nn_fit   = 20
k_nn_query = 30
idw_power  = 2.0
ratio_min, ratio_max = 1.2, 3.0      # bounds for anisotropy ℓ_perp/ℓ_parallel

def fit_local_anisotropic_field(X, y, k=k_nn_fit):
    """Learn direction-aware field (ℓ∥, ℓ⊥, u) from data"""
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

def run_all_csvs():
    data_dir = r"C:\Workspace\spacerobotics\GP\data"
    # Process only dec6.csv
    csv_file = "dec6.csv"
    csv_path = os.path.join(data_dir, csv_file)
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_file} not found in {data_dir}")
        return
    
    print(f"Processing {csv_file} with KNN-based Non-Stationary Gaussian Process")
    
    (data, X_train, y_train, y_train_centered,
     trend, target_var, date_tag, sigma_n) = process_single_csv(csv_path, csv_file)
    
    # Fit anisotropic field for this dataset
    ell_par_train, ell_perp_train, u_train, aniso_params_of = fit_local_anisotropic_field(
        X_train.astype(np.float64), y_train_centered.astype(np.float64), k=k_nn_fit
    )
    print(f"ℓ∥ stats: min={ell_par_train.min():.2f}, med={np.median(ell_par_train):.2f}, max={ell_par_train.max():.2f}")
    print(f"ℓ⊥/ℓ∥ stats: min={(ell_perp_train/ell_par_train).min():.2f}, med={np.median(ell_perp_train/ell_par_train):.2f}, max={(ell_perp_train/ell_par_train).max():.2f}")

    # Create output directory for this date
    date_tag_clean = os.path.splitext(date_tag)[0]  # Remove .csv extension
    out_dir_ns = os.path.join(r"C:\ASU\paper", date_tag_clean, "KNN_NonStationary")
    os.makedirs(out_dir_ns, exist_ok=True)

    # Sanity plot for learned geometry
    plt.figure(figsize=(8,6))
    plt.scatter(X_train[:,0], X_train[:,1], c=np.log10(ell_perp_train/ell_par_train), 
                s=6, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='log10 anisotropy ratio (ℓ⊥/ℓ∥)')
    plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
    plt.title(f"{date_tag_clean} - KNN-Based Anisotropy Field")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_ns, f"{date_tag_clean}_KNN_anisotropy_field.png"), dpi=160)
    plt.close()

    # Visual sanity on directions: quiver overlay to ensure u aligns with elongated features
    step = max(1, len(X_train)//400)  # thin arrows
    plt.figure(figsize=(7,6))
    plt.scatter(X_train[:,0], X_train[:,1], s=5, c='lightgray', alpha=0.5)
    plt.quiver(X_train[::step,0], X_train[::step,1],
               u_train[::step,0], u_train[::step,1], 
               angles='xy', scale_units='xy', scale=20, alpha=0.7)
    plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
    plt.title(f"{date_tag_clean} - KNN-Based Direction Field")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_ns, f"{date_tag_clean}_KNN_direction_field.png"), dpi=160)
    plt.close()

    # =============================================================================
    # RUN THE DIRECTION-AWARE ANISOTROPIC NON-STATIONARY MATÉRN GP
    # =============================================================================
    
    if USE_CUDA:
        print("Using CUDA-accelerated anisotropic implementation...")
        
        # --- hyperparameter calibration on GPU (optional but recommended)
        print("Fitting hyperparameters...")
        sigma_f_hat, sigma_n_hat, ell_scale = fit_hypers_cuda_aniso(
            X_train, y_train_centered, u_train, ell_par_train, ell_perp_train, nu=1.5, iters=80, lr=0.08
        )
        print(f"Fitted hyperparams: σ_f={sigma_f_hat:.3f}, σ_n={sigma_n_hat:.3f}, ℓ_scale={ell_scale:.3f}")
        
        # Report lengthscale diagnostics
        report_lengthscale_diagnostics(X_train, ell_par_train, ell_perp_train, ell_scale,
                                       ell_min=ell_min, ell_max=ell_max, ratio_max=ratio_max)
        
        # --- run anisotropic prediction
        gx, gy, Zmu, Zent = ns_matern_gp_predict_aniso_cuda(
            X_train=X_train.astype(np.float64),
            y_train_centered=y_train_centered.astype(np.float64),
            trend_fn=trend,
            aniso_params_of=aniso_params_of,
            u_train=u_train,
            ell_par_train=ell_par_train,
            ell_perp_train=ell_perp_train,
            grid_size=100, pad=0.12,
            nu=1.5, sigma_f=sigma_f_hat, sigma_n=sigma_n_hat, ell_scale=ell_scale,
            batch_G=8192, out_dir=out_dir_ns, var_label=target_var, date_tag=date_tag_clean
        )
    else:
        print("CUDA not available, using CPU isotropic implementation...")
        ell_scale = 1.0  # CPU fallback
        gx, gy, Zmu, Zent = ns_matern_gp_predict_adaptive(
            X_train=X_train.astype(np.float64),
            y_train_centered=y_train_centered.astype(np.float64),
            trend_fn=trend,
            aniso_params_of=aniso_params_of,
            ell_par_train=ell_par_train,
            ell_perp_train=ell_perp_train,
            grid_size=100, pad=0.12,
            sigma_n=sigma_n,         # now defined
            jitter=1e-5,                # Increased jitter for better numerical stability
            out_dir=out_dir_ns,
            var_label=target_var,
            date_tag=date_tag_clean,
            nu=1.5,                     # Use Matérn-3/2 for better behavior
            sigma_f=None
        )

    print(f"KNN-based Non-Stationary Gaussian Process completed for {date_tag_clean}!")
    print(f"Results saved to: {out_dir_ns}")

    print("\n" + "="*60)
    print("KNN-based Non-Stationary Gaussian Process completed successfully!")
    print("="*60)

# =============================================================================
# VECTORIZED ISOTROPIC PACIOREK-SCHERVISH MATÉRN KERNEL
# =============================================================================

def build_cov_iso_ps(XA, XB, lA, lB, nu=1.5, sigma_f=1.0):
    """
    Vectorized Paciorek–Schervish NS Matérn with Σ(x)=ℓ(x)^2 I (2D).
      prefactor = (2 ℓ_i ℓ_j) / (ℓ_i^2 + ℓ_j^2)
      Q_ij      = 2 ||x_i - x_j||^2 / (ℓ_i^2 + ℓ_j^2)
      k         = σ_f^2 * prefactor * [ (√(2νQ))^ν K_ν(√(2νQ)) ] / [ Γ(ν) 2^{ν-1} ]
    """
    XA = np.asarray(XA, dtype=np.float64); XB = np.asarray(XB, dtype=np.float64)
    lA = np.asarray(lA, dtype=np.float64); lB = np.asarray(lB, dtype=np.float64)

    XA2 = (XA**2).sum(axis=1)[:, None]         # (NA,1)
    XB2 = (XB**2).sum(axis=1)[None, :]         # (1,NB)
    D2  = XA2 + XB2 - 2.0 * XA @ XB.T          # (NA,NB), >= 0 numerically

    LA2 = (lA**2)[:, None]
    LB2 = (lB**2)[None, :]
    Lsum = LA2 + LB2                             # (NA,NB)
    pref = (2.0 * (lA[:, None] * lB[None, :])) / np.clip(Lsum, 1e-12, np.inf)

    Q    = 2.0 * D2 / np.clip(Lsum, 1e-12, np.inf)
    Q    = np.clip(Q, 1e-12, np.inf)            # safety

    arg  = np.sqrt(2.0 * nu * Q)
    matern_part = (arg**nu) * kv(nu, arg)
    norm_const  = 1.0 / (gamma_fn(nu) * (2.0**(nu - 1.0)))

    K = (sigma_f**2) * pref * norm_const * matern_part
    return K

def ns_build_K_train_cpu(X_train, l_train, nu, sigma_f, sigma_n, jitter):
    """Build training covariance matrix with learned ℓ(x) - CPU fallback"""
    K = build_cov_iso_ps(X_train, X_train, l_train, l_train, nu=nu, sigma_f=sigma_f)
    # Ensure exact diagonal σ_f^2 (limit Q→0) and add noise/jitter
    np.fill_diagonal(K, sigma_f**2)
    K[np.diag_indices_from(K)] += sigma_n**2 + jitter
    return K

def ns_build_K_star_cpu(X_star, X_train, l_star, l_train, nu, sigma_f):
    """Build cross-covariance matrix for predictions - CPU fallback"""
    K_star = build_cov_iso_ps(X_star, X_train, l_star, l_train, nu=nu, sigma_f=sigma_f)
    return K_star

# =============================================================================
# CUDA-ACCELERATED PACIOREK-SCHERVISH MATÉRN KERNEL
# =============================================================================

def _matern_shape_closed_form_torch(arg: torch.Tensor, nu: float) -> torch.Tensor:
    """Closed-form Matérn shape functions for ν ∈ {0.5, 1.5, 2.5} on GPU"""
    # arg >= 0, shape: (...,)
    if nu == 0.5:
        return torch.exp(-arg)
    elif nu == 1.5:
        return (1.0 + arg) * torch.exp(-arg)
    elif nu == 2.5:
        return (1.0 + arg + (arg * arg) / 3.0) * torch.exp(-arg)
    else:
        raise NotImplementedError("CUDA path supports nu in {0.5, 1.5, 2.5} (closed form).")

def build_cov_iso_ps_torch(XA, XB, lA, lB, nu=1.5, sigma_f=1.0, eps=1e-12, device=DEVICE, dtype=TORCH_DTYPE):
    """
    Paciorek–Schervish NS Matérn in 2D with Σ(x)=ℓ(x)^2 I, vectorized on GPU.
      pref = (2 ℓ_i ℓ_j) / (ℓ_i^2 + ℓ_j^2)
      Q    = 2 ||x_i - x_j||^2 / (ℓ_i^2 + ℓ_j^2)
      arg  = sqrt(2ν Q)
      k    = σ_f^2 * pref * shape(arg)   with closed-form shape per ν
    """
    XA = torch.as_tensor(XA, device=device, dtype=dtype)  # (NA,2)
    XB = torch.as_tensor(XB, device=device, dtype=dtype)  # (NB,2)
    lA = torch.as_tensor(lA, device=device, dtype=dtype)  # (NA,)
    lB = torch.as_tensor(lB, device=device, dtype=dtype)  # (NB,)

    XA2 = (XA * XA).sum(dim=1).unsqueeze(1)               # (NA,1)
    XB2 = (XB * XB).sum(dim=1).unsqueeze(0)               # (1,NB)
    D2  = XA2 + XB2 - 2.0 * (XA @ XB.T)                   # (NA,NB), ≥ 0

    LA2 = (lA * lA).unsqueeze(1)                           # (NA,1)
    LB2 = (lB * lB).unsqueeze(0)                           # (1,NB)
    Lsum = (LA2 + LB2).clamp_min(eps)                      # (NA,NB)

    pref = (2.0 * (lA.unsqueeze(1) * lB.unsqueeze(0))) / Lsum
    Q    = 2.0 * D2 / Lsum
    Q    = Q.clamp_min(eps)

    arg  = torch.sqrt(torch.tensor(2.0*nu, device=device, dtype=dtype) * Q)
    shape = _matern_shape_closed_form_torch(arg, float(nu))
    K = (sigma_f**2) * pref * shape
    return K

# =============================================================================
# CUDA ANISOTROPIC PACIOREK-SCHERVISH MATÉRN KERNEL
# =============================================================================

def _sigma_from_u_l(ux, uy, lpar, lperp):
    """Build Σ = l_perp^2 I + (l_par^2 - l_perp^2) u u^T"""
    lp2 = lpar*lpar; lt2 = lperp*lperp; d = (lp2 - lt2)
    s11 = lt2 + d*(ux*ux)
    s22 = lt2 + d*(uy*uy)
    s12 = d*(ux*uy)
    return s11, s12, s22

def build_cov_ps_aniso_torch(XA, XB, uA, uB, lparA, lperpA, lparB, lperpB,
                             nu=1.5, sigma_f=1.0, eps=1e-12,
                             device=DEVICE, dtype=TORCH_DTYPE):
    """
    CUDA anisotropic Paciorek–Schervish NS Matérn with full Σ(x) matrices.
    """
    XA = torch.as_tensor(XA, device=device, dtype=dtype)       # (NA,2)
    XB = torch.as_tensor(XB, device=device, dtype=dtype)       # (NB,2)
    uA = torch.as_tensor(uA, device=device, dtype=dtype)       # (NA,2)
    uB = torch.as_tensor(uB, device=device, dtype=dtype)       # (NB,2)
    lparA = torch.as_tensor(lparA, device=device, dtype=dtype) # (NA,)
    lperpA= torch.as_tensor(lperpA,device=device, dtype=dtype) # (NA,)
    lparB = torch.as_tensor(lparB, device=device, dtype=dtype) # (NB,)
    lperpB= torch.as_tensor(lperpB,device=device, dtype=dtype) # (NB,)

    # Σ components for A, B
    s11A,s12A,s22A = _sigma_from_u_l(uA[:,0], uA[:,1], lparA, lperpA)
    s11B,s12B,s22B = _sigma_from_u_l(uB[:,0], uB[:,1], lparB, lperpB)

    # determinants (per point)
    detA = (s11A*s22A - s12A*s12A).clamp_min(eps)
    detB = (s11B*s22B - s12B*s12B).clamp_min(eps)
    logdetA = torch.log(detA); logdetB = torch.log(detB)

    # pairwise M = (ΣA + ΣB)/2 components
    M11 = 0.5*(s11A.unsqueeze(1) + s11B.unsqueeze(0))
    M22 = 0.5*(s22A.unsqueeze(1) + s22B.unsqueeze(0))
    M12 = 0.5*(s12A.unsqueeze(1) + s12B.unsqueeze(0))
    detM = (M11*M22 - M12*M12).clamp_min(eps)
    logdetM = torch.log(detM)

    # prefactor
    logpref = 0.25*logdetA.unsqueeze(1) + 0.25*logdetB.unsqueeze(0) - 0.5*logdetM
    pref = torch.exp(logpref)

    # Q = d^T M^{-1} d using 2x2 inverse
    dx = XA[:,0].unsqueeze(1) - XB[:,0].unsqueeze(0)
    dy = XA[:,1].unsqueeze(1) - XB[:,1].unsqueeze(0)
    inv_detM = 1.0 / detM
    Q = (M22*dx*dx - 2.0*M12*dx*dy + M11*dy*dy) * inv_detM
    Q = Q.clamp_min(eps)

    arg = torch.sqrt(torch.tensor(2.0*nu, device=device, dtype=dtype) * Q)
    shape = _matern_shape_closed_form_torch(arg, float(nu))
    K = (sigma_f**2) * pref * shape
    return K

# =============================================================================
# DIAGNOSTICS AND UTILITIES
# =============================================================================

def report_lengthscale_diagnostics(X, ell_par, ell_perp, ell_scale, ell_min=0.30, ell_max=2.00, ratio_max=3.0):
    """Report lengthscale diagnostics to help tune hyperparameters"""
    from sklearn.neighbors import NearestNeighbors
    # Scaled, then clamped
    lp = np.clip(ell_scale*ell_par,  ell_min, ell_max)
    lt = np.clip(ell_scale*ell_perp, ell_min, ell_max*ratio_max)

    # Percent clamped at lower bound
    pmin_lp = 100.0*np.mean(lp <= ell_min+1e-9)
    pmin_lt = 100.0*np.mean(lt <= ell_min+1e-9)

    # Nearest-neighbor distance
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    d = nn.kneighbors(X, return_distance=True)[0][:,1]
    print(f"[Diag] ℓ_scale={ell_scale:.4f} | ℓ‖ med={np.median(lp):.3f}, ℓ⊥ med={np.median(lt):.3f}")
    print(f"[Diag] % clamped at ℓ_min: ℓ‖={pmin_lp:.1f}%, ℓ⊥={pmin_lt:.1f}%")
    print(f"[Diag] NN dist p50={np.median(d):.3f}, compare ℓ med above.")

# =============================================================================
# CUDA HYPERPARAMETER OPTIMIZATION
# =============================================================================

def fit_hypers_cuda_aniso(X, y_centered, u_train, lpar, lperp, nu=1.5, iters=80, lr=0.08):
    """Fit hyperparameters σf, σn, ℓ-scale using CUDA optimization"""
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
                               dtype=TORCH_DTYPE, requires_grad=True)  # scales both ℓ

    opt = torch.optim.Adam([log_sigma_f, log_sigma_n, log_lscale], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        sf = torch.exp(log_sigma_f)
        sn = torch.exp(log_sigma_n)
        ls = torch.exp(log_lscale)
        lpar_use  = (ls * lpar0).clamp(min=ell_min, max=ell_max)
        lperp_use = (ls * lperp0).clamp(min=ell_min, max=ell_max*ratio_max)

        K = build_cov_ps_aniso_torch(Xt, Xt, uA, uA, lpar_use, lperp_use, lpar_use, lperp_use,
                                     nu=float(nu), sigma_f=sf)  # tensor, not float
        K.diagonal().add_(sn*sn + 1e-6)
        K = 0.5 * (K + K.T)  # symmetrize
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            K.diagonal().add_(1e-4)
            L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(yt, L)
        n = Xt.shape[0]
        nll = 0.5*(yt.T @ alpha) + torch.log(torch.diag(L)).sum() + 0.5*n*torch.log(torch.tensor(2*np.pi, device=DEVICE, dtype=TORCH_DTYPE))
        
        # weak priors to avoid pathological tiny noise/lengthscale
        prior_l = 0.5*(log_lscale/0.7)**2         # log ℓ_scale ~ N(0, 0.7^2)
        prior_n = 0.5*((log_sigma_n - np.log(0.03))/0.7)**2  # prefer σ_n ~ 0.03 (std), weakly
        nll = nll + prior_l + prior_n
        
        nll = nll.squeeze()
        nll.backward()
        opt.step()
        
        # Constrain ℓ_scale search window
        with torch.no_grad():
            log_lscale.clamp_(min=-1.5, max=0.5)

    with torch.no_grad():
        return float(torch.exp(log_sigma_f)), float(torch.exp(log_sigma_n)), float(torch.exp(log_lscale))

# =============================================================================
# ADAPTIVE NON-STATIONARY MATÉRN GP PREDICTION
# =============================================================================

def ns_matern_gp_predict_adaptive(
    X_train, y_train_centered, trend_fn,
    aniso_params_of, ell_par_train, ell_perp_train,
    grid_size=80, pad=0.12, sigma_n=0.1, jitter=1e-6, out_dir=None,
    var_label="Temperature (°C)", date_tag="dec6",
    nu=1.5, sigma_f=None
):
    """
    Exact GP with data-adaptive non-stationary Matérn kernel.
    Returns (gx, gy, Zmu, Zent).
    """
    if sigma_f is None:
        sigma_f = float(np.std(y_train_centered)) or 1.0

    # 1) Build K_train with learned ℓ(x) - CPU fallback using average of anisotropic lengths
    ell_train_cpu = (ell_par_train + ell_perp_train) / 2.0
    K = ns_build_K_train_cpu(X_train, ell_train_cpu, nu, sigma_f, sigma_n, jitter)
    K = 0.5 * (K + K.T)  # symmetrize
    
    # 2) Cholesky + alpha (with fallback for numerical stability)
    try:
        L = cholesky(K, lower=True, overwrite_a=False, check_finite=False)
    except np.linalg.LinAlgError:
        print(f"[NS-Matern] Cholesky failed, increasing jitter from {jitter} to 1e-4")
        K[np.diag_indices_from(K)] += 1e-4 - jitter
        L = cholesky(K, lower=True, overwrite_a=False, check_finite=False)

    alpha = solve_triangular(L, y_train_centered, lower=True, check_finite=False)
    alpha = solve_triangular(L.T, alpha, lower=False, check_finite=False)

    # 3) Grid and ℓ(x*) via IDW interpolation (with padding)
    (x_min, y_min) = X_train.min(axis=0); (x_max, y_max) = X_train.max(axis=0)
    dx, dy = (x_max-x_min), (y_max-y_min)
    gx = np.linspace(x_min - pad*dx, x_max + pad*dx, grid_size)
    gy = np.linspace(y_min - pad*dy, y_max + pad*dy, grid_size)
    Xg, Yg = np.meshgrid(gx, gy, indexing="ij")
    Xstar  = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    lpar_q, lperp_q, _ = aniso_params_of(Xstar)
    ell_star = (lpar_q + lperp_q) / 2.0  # average for isotropic fallback

    # 4) K_star and predictions
    K_star = ns_build_K_star_cpu(Xstar, X_train, ell_star, ell_train_cpu, nu, sigma_f)
    mu_centered = K_star @ alpha
    mu = mu_centered + trend_fn(Xstar)  # add trend back

    # 5) Variance (latent) and entropy
    # For PS NS Matérn with Σ(x)=ℓ(x)^2 I, k(x*,x*) = σ_f^2
    K_ss_diag = (sigma_f**2) * np.ones(Xstar.shape[0], dtype=np.float64)
    v = solve_triangular(L, K_star.T, lower=True, check_finite=False)
    var_latent = np.maximum(K_ss_diag - np.einsum('ij,ij->j', v, v), 1e-12)  # safety clamp
    ent = 0.5 * np.log(2 * np.pi * np.e * var_latent)

    Zmu  = mu.reshape(grid_size, grid_size)
    Zent = ent.reshape(grid_size, grid_size)

    # crop to original bbox for plotting
    ix = (gx>=x_min) & (gx<=x_max)
    iy = (gy>=y_min) & (gy<=y_max)

    # Save figures
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(6.5,5.6))
        plt.contourf(gy[iy], gx[ix], Zmu[np.ix_(ix,iy)].T, 20, cmap='viridis')
        plt.colorbar(label=var_label)
        plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
        plt.title(f"{date_tag} – {var_label} mean (NS-Matérn, CPU fallback)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sanitize(var_label)}_NSMatern_cpu_mean.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(6.5,5.6))
        plt.contourf(gy[iy], gx[ix], Zent[np.ix_(ix,iy)].T, 20, cmap='inferno')
        plt.colorbar(label="Entropy (latent)")
        plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
        plt.title(f"{date_tag} – {var_label} uncertainty (NS-Matérn, CPU fallback)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sanitize(var_label)}_NSMatern_cpu_uncert.png"), dpi=160)
        plt.close()

    print(f"CPU fallback: ℓ(x) bounds: [{ell_min}, {ell_max}] "
          f" median ℓ_train={np.median(ell_train_cpu):.3f}")
    return gx, gy, Zmu, Zent

@torch.no_grad()
def ns_matern_gp_predict_adaptive_cuda(
    X_train, y_train_centered, trend_fn,
    aniso_params_of, ell_par_train, ell_perp_train,
    grid_size=80, pad=0.12, sigma_n=0.1, jitter=1e-6, out_dir=None,
    var_label="Temperature (°C)", date_tag="dec6",
    nu=1.5, sigma_f=None, batch_G=8192
):
    """
    Exact GP with PS non-stationary Matérn on GPU (ν∈{0.5,1.5,2.5} closed-form).
    KNN-based ℓ(x) is already fitted on CPU → we reuse (ell_train, lengthscale_of).
    """
    if sigma_f is None:
        sigma_f = float(np.std(y_train_centered)) or 1.0

    # ---- Move training to GPU
    Xtr_t = torch.as_tensor(X_train, device=DEVICE, dtype=TORCH_DTYPE)  # (N,2)
    y_t   = torch.as_tensor(y_train_centered, device=DEVICE, dtype=TORCH_DTYPE)  # (N,)
    ell_tr_t = torch.as_tensor((ell_par_train + ell_perp_train) / 2.0, device=DEVICE, dtype=TORCH_DTYPE)  # (N,)

    # ---- K(X,X) + noise/jitter
    K = build_cov_iso_ps_torch(Xtr_t, Xtr_t, ell_tr_t, ell_tr_t, nu=nu, sigma_f=sigma_f)
    K.diagonal().add_(sigma_n**2 + jitter)
    K = 0.5 * (K + K.T)  # symmetrize

    # ---- Cholesky & alpha
    try:
        L = torch.linalg.cholesky(K)  # (N,N)
    except RuntimeError:
        K.diagonal().add_(1e-4)       # gentle bump
        L = torch.linalg.cholesky(K)

    # Solve K alpha = y using Cholesky (L L^T)
    # alpha = (L^T)^{-1} (L^{-1} y)
    alpha = torch.cholesky_solve(y_t.unsqueeze(1), L).squeeze(1)  # (N,)

    # ---- Prediction grid (CPU → ℓ(x*) via IDW → GPU) with padding
    (x_min, y_min) = X_train.min(axis=0); (x_max, y_max) = X_train.max(axis=0)
    dx, dy = (x_max-x_min), (y_max-y_min)
    gx = np.linspace(x_min - pad*dx, x_max + pad*dx, grid_size)
    gy = np.linspace(y_min - pad*dy, y_max + pad*dy, grid_size)
    Xg, Yg = np.meshgrid(gx, gy, indexing="ij")
    Xstar  = np.stack([Xg.ravel(), Yg.ravel()], axis=1)           # (G,2), CPU
    lpar_q, lperp_q, _ = aniso_params_of(Xstar)
    ell_star = (lpar_q + lperp_q) / 2.0  # average for isotropic fallback

    Xst_t  = torch.as_tensor(Xstar,     device=DEVICE, dtype=TORCH_DTYPE)  # (G,2)
    ell_st = torch.as_tensor(ell_star,  device=DEVICE, dtype=TORCH_DTYPE)  # (G,)

    # ---- K_star in batches to control GPU memory
    G = Xstar.shape[0]
    mu_centered = torch.empty(G, device=DEVICE, dtype=TORCH_DTYPE)
    var_latent  = torch.empty(G, device=DEVICE, dtype=TORCH_DTYPE)
    s2 = sigma_f**2

    for beg in range(0, G, batch_G):
        end = min(G, beg + batch_G)
        Ks = build_cov_iso_ps_torch(Xst_t[beg:end], Xtr_t, ell_st[beg:end], ell_tr_t, nu=nu, sigma_f=sigma_f)
        # mean: K_* @ alpha
        mu_centered[beg:end] = Ks @ alpha

        # variance: σ_f^2 - ||L^{-1} K_*^T||^2 (column-wise)
        # v = solve(L, K_*^T)  → triangular solve (lower)
        v = torch.linalg.solve_triangular(L, Ks.T, upper=False)  # (N, batch)
        var_latent[beg:end] = torch.maximum(s2 - (v*v).sum(dim=0), torch.tensor(1e-12, device=DEVICE, dtype=TORCH_DTYPE))

    mu = mu_centered.detach().cpu().numpy() + trend_fn(Xstar)
    ent = (0.5 * torch.log(2 * torch.pi * torch.e * var_latent)).detach().cpu().numpy()

    Zmu  = mu.reshape(grid_size, grid_size)
    Zent = ent.reshape(grid_size, grid_size)
    
    # crop to original bbox for plotting
    ix = (gx>=x_min) & (gx<=x_max)
    iy = (gy>=y_min) & (gy<=y_max)

    # ---- Save figs (CPU numpy)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(6.5,5.6))
        plt.contourf(gy[iy], gx[ix], Zmu[np.ix_(ix,iy)].T, 20, cmap='viridis')
        plt.colorbar(label=var_label)
        plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
        plt.title(f"{date_tag} – {var_label} mean (NS-Matérn, CUDA)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sanitize(var_label)}_NSMatern_adapt_mean_CUDA.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(6.5,5.6))
        plt.contourf(gy[iy], gx[ix], Zent[np.ix_(ix,iy)].T, 20, cmap='inferno')
        plt.colorbar(label="Entropy (latent)")
        plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
        plt.title(f"{date_tag} – {var_label} uncertainty (NS-Matérn, CUDA)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sanitize(var_label)}_NSMatern_adapt_uncert_CUDA.png"), dpi=160)
        plt.close()

    print(f"[CUDA] Used ν={nu}, σ_f≈{sigma_f:.3f}, jitter={jitter}, batch_G={batch_G}")
    return gx, gy, Zmu, Zent

@torch.no_grad()
def ns_matern_gp_predict_aniso_cuda(
    X_train, y_train_centered, trend_fn,
    aniso_params_of, u_train, ell_par_train, ell_perp_train,
    grid_size=100, pad=0.12,
    nu=1.5, sigma_f=None, sigma_n=0.1, ell_scale=1.0,
    batch_G=8192, out_dir=None, var_label="Temperature (°C)", date_tag="dec6"
):
    """CUDA anisotropic NS-Matérn GP with trend and padding"""
    # scale ℓ-fields
    lpar_tr  = np.clip(ell_scale*ell_par_train,  ell_min, ell_max)
    lperp_tr = np.clip(ell_scale*ell_perp_train, ell_min, ell_max*ratio_max)

    # --- move to GPU & build K
    Xt  = torch.as_tensor(X_train, device=DEVICE, dtype=TORCH_DTYPE)
    yt  = torch.as_tensor(y_train_centered, device=DEVICE, dtype=TORCH_DTYPE).unsqueeze(1)
    u_t = torch.as_tensor(u_train, device=DEVICE, dtype=TORCH_DTYPE)
    lp_t  = torch.as_tensor(lpar_tr,  device=DEVICE, dtype=TORCH_DTYPE)
    lt_t  = torch.as_tensor(lperp_tr, device=DEVICE, dtype=TORCH_DTYPE)
    if sigma_f is None: sigma_f = float(np.std(y_train_centered) or 1.0)

    K = build_cov_ps_aniso_torch(Xt, Xt, u_t, u_t, lp_t, lt_t, lp_t, lt_t, nu=float(nu), sigma_f=float(sigma_f))
    K.diagonal().add_(sigma_n**2 + 1e-6)
    K = 0.5 * (K + K.T)  # symmetrize
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(yt, L)[:,0]

    # --- padded grid
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

    # add back the trend
    mu = (mu_resid.cpu().numpy() + trend_fn(Xstar))
    ent = 0.5*np.log(2*np.pi*np.e*var_lat.cpu().numpy())

    Zmu, Zent = mu.reshape(grid_size, grid_size), ent.reshape(grid_size, grid_size)

    # crop to original bbox for plotting
    ix = (gx>=x_min) & (gx<=x_max)
    iy = (gy>=y_min) & (gy<=y_max)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(6.5,5.6))
        plt.contourf(gy[iy], gx[ix], Zmu[np.ix_(ix,iy)].T, 20, cmap='viridis')
        plt.colorbar(label=var_label)
        plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
        plt.title(f"{date_tag} – {var_label} (KNN-Based Non-Stationary Gaussian Process)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sanitize(var_label)}_KNN_NSGP_mean.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(6.5,5.6))
        plt.contourf(gy[iy], gx[ix], Zent[np.ix_(ix,iy)].T, 20, cmap='inferno')
        plt.colorbar(label="Uncertainty")
        plt.xlabel("Easting (scaled)"); plt.ylabel("Northing (scaled)")
        plt.title(f"{date_tag} – {var_label} Uncertainty (KNN-Based Non-Stationary Gaussian Process)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sanitize(var_label)}_KNN_NSGP_uncertainty.png"), dpi=160)
        plt.close()

    print(f"[CUDA Aniso] Used ν={nu}, σ_f≈{sigma_f:.3f}, σ_n={sigma_n:.3f}, ℓ_scale={ell_scale:.3f}")
    return gx, gy, Zmu, Zent

if __name__ == "__main__":
    run_all_csvs()
