# compare.py — ground-truth vs KNN-NSGP, extra metrics + 6 images
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular
from scipy.special import kv, gamma as gamma_fn
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch

# ========= CUDA setup =========
USE_CUDA = torch.cuda.is_available()
TORCH_DTYPE = torch.float64 if (not USE_CUDA or torch.cuda.get_device_capability()[0] >= 7) else torch.float32
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Torch device:", DEVICE, "| dtype:", TORCH_DTYPE)

# ========= small utils =========
def sanitize(txt: str) -> str:
    return re.sub(r'[^0-9A-Za-z_]+', '_', str(txt))

def fit_linear_trend(X, y):
    import numpy.linalg as npl
    A = np.c_[np.ones(len(X)), X]
    beta, *_ = npl.lstsq(A, y, rcond=None)
    def trend_fn(Xq):
        Aq = np.c_[np.ones(len(Xq)), Xq]
        return Aq @ beta
    return beta, trend_fn

def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def mae(a,b):  return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))

def stats(name, arr):
    arr = np.asarray(arr)
    d = dict(
        min=float(np.nanmin(arr)),
        max=float(np.nanmax(arr)),
        rng=lambda d: d["max"]-d["min"],
        mean=float(np.nanmean(arr)),
        std=float(np.nanstd(arr)),
        median=float(np.nanmedian(arr)),
        p05=float(np.nanpercentile(arr,5)),
        p95=float(np.nanpercentile(arr,95))
    )
    d["range"] = d["rng"](d); del d["rng"]
    return name, d

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float('nan')

def pearson_r(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    if a.size != b.size or a.size == 0: return float('nan')
    a = a - a.mean(); b = b - b.mean()
    denom = (np.linalg.norm(a)*np.linalg.norm(b))
    return float(a.dot(b)/denom) if denom>0 else float('nan')

def sigma_from_entropy(ent):
    # H = 0.5 * ln(2π e σ²) → σ² = exp(2H)/(2π e)
    return np.sqrt(np.exp(2.0*ent)/(2.0*np.pi*np.e))

def convex_hull_mask(sample_xy, grid_xy):
    # sample_xy: (N,2), grid_xy: (M,2)
    if sample_xy.shape[0] < 3:
        return np.zeros(len(grid_xy), dtype=bool)
    hull = ConvexHull(sample_xy)
    poly = Path(sample_xy[hull.vertices])
    return poly.contains_points(grid_xy)

# ========= ground-truth (metres) – same recipe as your ROS node =========
def generate_ground_truth(bounds_x=(0.0,150.0), bounds_y=(0.0,50.0), res=1.0,
                          seed=42, n_bumps=8, n_aniso=4, n_fronts=2,
                          base_T=20.0, bg_gx=0.08, bg_gy=0.05):
    np.random.seed(seed)
    x_grid = np.arange(bounds_x[0], bounds_x[1], res)
    y_grid = np.arange(bounds_y[0], bounds_y[1], res)
    X, Y = np.meshgrid(x_grid, y_grid)  # (Ny,Nx)

    T = np.zeros_like(X, dtype=float)
    # background
    T += base_T
    T += bg_gx * (X - X.min())
    T += bg_gy * (Y - Y.min())

    # isotropic bumps
    for _ in range(n_bumps):
        cx = np.random.uniform(bounds_x[0] + 15, bounds_x[1] - 15)
        cy = np.random.uniform(bounds_y[0] + 10, bounds_y[1] - 10)
        sigma = np.random.uniform(4, 12)
        amp   = np.random.uniform(-6, 12)
        T += amp * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*sigma**2))

    # anisotropic bumps
    for _ in range(n_aniso):
        cx = np.random.uniform(bounds_x[0] + 15, bounds_x[1] - 15)
        cy = np.random.uniform(bounds_y[0] + 10, bounds_y[1] - 10)
        sig_maj = np.random.uniform(15, 25)
        sig_min = np.random.uniform(3,  8)
        theta   = np.random.uniform(0, np.pi)
        amp     = np.random.uniform(-5, 8)
        Xr =  (X-cx)*np.cos(theta) + (Y-cy)*np.sin(theta)
        Yr = -(X-cx)*np.sin(theta) + (Y-cy)*np.cos(theta)
        T += amp * np.exp(-(Xr**2/(2*sig_maj**2) + Yr**2/(2*sig_min**2)))

    # sharp fronts
    for _ in range(n_fronts):
        cx = np.random.uniform(bounds_x[0] + 20, bounds_x[1] - 20)
        cy = np.random.uniform(bounds_y[0] + 10, bounds_y[1] - 10)
        theta = np.random.uniform(0, np.pi)
        amp   = np.random.uniform(4, 8)
        sharp = np.random.uniform(0.1, 0.3)
        d = (X-cx)*np.cos(theta) + (Y-cy)*np.sin(theta)
        T += amp * np.tanh(d * sharp)

    # spatially varying noise
    noise_base = 0.3
    noise_var  = 0.4 * np.exp(-((X - X.mean())**2 + (Y - Y.mean())**2) / (30**2))
    T += np.random.normal(0, noise_base + noise_var, size=T.shape)

    return x_grid, y_grid, T

def bilinear_sample(xg, yg, Z, xs, ys):
    xs = np.asarray(xs); ys = np.asarray(ys)
    i = np.searchsorted(xg, xs) - 1
    j = np.searchsorted(yg, ys) - 1
    valid = (i>=0)&(i<len(xg)-1)&(j>=0)&(j<len(yg)-1)
    out = np.full(xs.shape, np.nan)
    if not np.any(valid): return out, valid
    x0,x1 = xg[i[valid]], xg[i[valid]+1]
    y0,y1 = yg[j[valid]], yg[j[valid]+1]
    wx = (xs[valid]-x0)/(x1-x0); wy=(ys[valid]-y0)/(y1-y0)
    z00=Z[j[valid], i[valid]]; z10=Z[j[valid], i[valid]+1]
    z01=Z[j[valid]+1, i[valid]]; z11=Z[j[valid]+1, i[valid]+1]
    out[valid] = (1-wx)*(1-wy)*z00 + wx*(1-wy)*z10 + (1-wx)*wy*z01 + wx*wy*z11
    return out, valid

# ========= CSV (metres already) =========
def load_lawnmower_csv(csv_path):
    import pyproj

    data = pd.read_csv(csv_path)

    # normalize header names for matching
    norm = {c.lower().strip(): c for c in data.columns}

    def first_exists(keys):
        for k in keys:
            if k in norm: 
                return norm[k]
        return None

    # find columns (meters spelling tolerant)
    col_x = first_exists(["x_meters","x_metres","x_m","x"])
    col_y = first_exists(["y_meters","y_metres","y_m","y"])
    col_t = first_exists(["temperature_celsius","temperature","temp_c","t"])

    # if meters not found, try lat/lon → project to meters (local ENU around first sample)
    if (col_x is None or col_y is None):
        col_lat = first_exists(["latitude","lat"])
        col_lon = first_exists(["longitude","lon","lng"])
        if col_lat is None or col_lon is None:
            raise ValueError(
                f"CSV missing position columns. Found: {list(data.columns)}. "
                f"Need x/y in meters (meters/metres) or latitude/longitude."
            )
        # simple local projection around first point (WGS84 → Azimuthal Equidistant)
        lat0 = float(data[col_lat].iloc[0]); lon0 = float(data[col_lon].iloc[0])
        enu = pyproj.Proj(proj="aeqd", lat_0=lat0, lon_0=lon0, datum="WGS84")
        to_enu = pyproj.Transformer.from_proj(pyproj.CRS("EPSG:4326"), enu, always_xy=True)
        x_m, y_m = to_enu.transform(data[col_lon].values, data[col_lat].values)
        data["X_m"], data["Y_m"] = x_m, y_m
    else:
        # use provided meters
        data["X_m"] = pd.to_numeric(data[col_x], errors="coerce")
        data["Y_m"] = pd.to_numeric(data[col_y], errors="coerce")

    if col_t is None:
        raise ValueError(
            f"CSV missing temperature column. Found: {list(data.columns)}. "
            f"Expected one of ['temperature_celsius','temperature','temp_c','t']."
        )
    data["T_meas"] = pd.to_numeric(data[col_t], errors="coerce")

    # drop any rows with NaNs from coercion
    data = data.dropna(subset=["X_m","Y_m","T_meas"]).reset_index(drop=True)

    # scale (for GP / KNN geometry learning)
    scaler = StandardScaler()
    XYs = scaler.fit_transform(data[["X_m","Y_m"]].values)
    data["X_s"], data["Y_s"] = XYs[:,0], XYs[:,1]
    X_train = data[["X_s","Y_s"]].values
    y_train = data["T_meas"].values

    # linear trend for universal kriging
    _, trend = fit_linear_trend(X_train, y_train)
    y_train_centered = y_train - trend(X_train)

    # small echo so you see what was used
    print(f"Loaded CSV with {len(data)} rows | using columns: "
          f"X='{col_x or 'ENU_from_latlon'}', Y='{col_y or 'ENU_from_latlon'}', T='{col_t}'")
    return data, X_train, y_train, y_train_centered, trend, scaler

# ========= KNN anisotropy learning =========
ell_min, ell_max = 0.30, 2.00
k_nn_fit, k_nn_query = 20, 30
idw_power = 2.0
ratio_min, ratio_max = 1.2, 3.0

def fit_local_anisotropic_field(X, y, k=k_nn_fit):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(X)
    dists, idxs = nbrs.kneighbors(X, return_distance=True)
    N = X.shape[0]
    grads = np.zeros((N,2))
    for i in range(N):
        Xi = X[idxs[i]]
        yi = y[idxs[i]]
        A  = np.c_[np.ones(k), Xi - X[i]]
        beta, *_ = np.linalg.lstsq(A, yi, rcond=None)
        grads[i] = beta[1:3]
    r = np.linalg.norm(grads, axis=1)
    ql,qh = np.quantile(r,[0.05,0.95])
    rhat = np.clip((r-ql)/max(qh-ql,1e-6), 0.0, 1.0)
    ell_par  = ell_max - (ell_max-ell_min)*rhat
    ratio    = ratio_min + (ratio_max-ratio_min)*(1.0 - rhat)
    ell_perp = ell_par*ratio
    u = grads.copy()
    nrm = np.linalg.norm(u, axis=1, keepdims=True)
    u = np.where(nrm>1e-8, u/nrm, np.array([1.0,0.0])[None,:])

    def aniso_params_of(Xq, kq=k_nn_query):
        dq, jq = nbrs.kneighbors(Xq, n_neighbors=kq, return_distance=True)
        w = 1.0/(np.power(dq, idw_power)+1e-6)
        gq = (w[...,None]*grads[jq]).sum(axis=1)/w.sum(axis=1, keepdims=True)
        gn = np.linalg.norm(gq, axis=1, keepdims=True)
        uq = np.where(gn>1e-8, gq/gn, np.array([[1.0,0.0]]))
        lpar_q  = (w*ell_par[jq]).sum(axis=1)/w.sum(axis=1)
        ratio_q = (w*ratio[jq]).sum(axis=1)/w.sum(axis=1)
        lperp_q = lpar_q*ratio_q
        lpar_q  = np.clip(lpar_q,  ell_min, ell_max)
        lperp_q = np.clip(lperp_q, ell_min, ell_max*ratio_max)
        return lpar_q, lperp_q, uq

    return ell_par, ell_perp, u, aniso_params_of

# ========= Paciorek–Schervish kernels (GPU + CPU) =========
def _matern_shape_closed_form_torch(arg: torch.Tensor, nu: float) -> torch.Tensor:
    if nu == 0.5: return torch.exp(-arg)
    if nu == 1.5: return (1.0 + arg)*torch.exp(-arg)
    if nu == 2.5: return (1.0 + arg + (arg*arg)/3.0)*torch.exp(-arg)
    raise NotImplementedError("nu must be 0.5, 1.5, or 2.5 in CUDA path")

def _sigma_from_u_l(ux, uy, lpar, lperp):
    lp2 = lpar*lpar; lt2 = lperp*lperp; d = (lp2 - lt2)
    s11 = lt2 + d*(ux*ux); s22 = lt2 + d*(uy*uy); s12 = d*(ux*uy)
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

    s11A,s12A,s22A = _sigma_from_u_l(uA[:,0], uA[:,1], lparA, lperpA)
    s11B,s12B,s22B = _sigma_from_u_l(uB[:,0], uB[:,1], lparB, lperpB)
    detA = (s11A*s22A - s12A*s12A).clamp_min(eps)
    detB = (s11B*s22B - s12B*s12B).clamp_min(eps)
    M11 = 0.5*(s11A.unsqueeze(1) + s11B.unsqueeze(0))
    M22 = 0.5*(s22A.unsqueeze(1) + s22B.unsqueeze(0))
    M12 = 0.5*(s12A.unsqueeze(1) + s12B.unsqueeze(0))
    detM= (M11*M22 - M12*M12).clamp_min(eps)
    pref = torch.exp(0.25*torch.log(detA).unsqueeze(1) + 0.25*torch.log(detB).unsqueeze(0) - 0.5*torch.log(detM))
    dx = XA[:,0].unsqueeze(1) - XB[:,0].unsqueeze(0)
    dy = XA[:,1].unsqueeze(1) - XB[:,1].unsqueeze(0)
    inv_detM = 1.0/detM
    Q = (M22*dx*dx - 2.0*M12*dx*dy + M11*dy*dy) * inv_detM
    Q = Q.clamp_min(eps)
    arg = torch.sqrt(torch.tensor(2.0*nu, device=device, dtype=dtype) * Q)
    shape = _matern_shape_closed_form_torch(arg, float(nu))
    return (sigma_f**2) * pref * shape

def build_cov_iso_ps(XA, XB, lA, lB, nu=1.5, sigma_f=1.0):
    XA = np.asarray(XA, dtype=np.float64); XB = np.asarray(XB, dtype=np.float64)
    lA = np.asarray(lA, dtype=np.float64); lB = np.asarray(lB, dtype=np.float64)
    XA2 = (XA**2).sum(axis=1)[:,None]; XB2=(XB**2).sum(axis=1)[None,:]
    D2 = XA2 + XB2 - 2.0 * XA @ XB.T
    LA2=(lA**2)[:,None]; LB2=(lB**2)[None,:]
    Lsum = LA2 + LB2
    pref = (2.0*(lA[:,None]*lB[None,:]))/np.clip(Lsum,1e-12,np.inf)
    Q = 2.0*D2/np.clip(Lsum,1e-12,np.inf); Q=np.clip(Q,1e-12,np.inf)
    arg = np.sqrt(2.0*nu*Q)
    matern_part = (arg**nu)*kv(nu,arg)
    norm_const = 1.0/(gamma_fn(nu)*(2.0**(nu-1.0)))
    return (sigma_f**2)*pref*norm_const*matern_part

# ========= hyperparam fit (CUDA) =========
def fit_hypers_cuda_aniso(X, y_centered, u_train, lpar, lperp, nu=1.5, iters=80, lr=0.08):
    Xt = torch.as_tensor(X, device=DEVICE, dtype=TORCH_DTYPE)
    yt = torch.as_tensor(y_centered, device=DEVICE, dtype=TORCH_DTYPE).unsqueeze(1)
    uA = torch.as_tensor(u_train, device=DEVICE, dtype=TORCH_DTYPE)
    lpar0 = torch.as_tensor(lpar, device=DEVICE, dtype=TORCH_DTYPE)
    lperp0= torch.as_tensor(lperp,device=DEVICE, dtype=TORCH_DTYPE)
    log_sigma_f = torch.tensor([np.log(np.std(y_centered)+1e-6)], device=DEVICE, dtype=TORCH_DTYPE, requires_grad=True)
    log_sigma_n = torch.tensor([np.log(0.1)], device=DEVICE, dtype=TORCH_DTYPE, requires_grad=True)
    log_lscale  = torch.tensor([0.0], device=DEVICE, dtype=TORCH_DTYPE, requires_grad=True)
    opt = torch.optim.Adam([log_sigma_f, log_sigma_n, log_lscale], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        sf = torch.exp(log_sigma_f); sn = torch.exp(log_sigma_n); ls = torch.exp(log_lscale)
        lpar_use  = (ls*lpar0).clamp(min=ell_min, max=ell_max)
        lperp_use = (ls*lperp0).clamp(min=ell_min, max=ell_max*ratio_max)
        K = build_cov_ps_aniso_torch(Xt, Xt, uA, uA, lpar_use, lperp_use, lpar_use, lperp_use, nu=float(nu), sigma_f=sf)
        K.diagonal().add_(sn*sn + 1e-6); K = 0.5*(K+K.T)
        try: L = torch.linalg.cholesky(K)
        except RuntimeError:
            K.diagonal().add_(1e-4); L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(yt, L)
        n = Xt.shape[0]
        nll = 0.5*(yt.T@alpha) + torch.log(torch.diag(L)).sum() + 0.5*n*torch.log(torch.tensor(2*np.pi, device=DEVICE, dtype=TORCH_DTYPE))
        prior_l = 0.5*(log_lscale/0.7)**2
        prior_n = 0.5*((log_sigma_n - np.log(0.03))/0.7)**2
        (nll + prior_l + prior_n).squeeze().backward()
        opt.step()
        with torch.no_grad(): log_lscale.clamp_(min=-1.5, max=0.5)
    return float(torch.exp(log_sigma_f)), float(torch.exp(log_sigma_n)), float(torch.exp(log_lscale))

# ========= model fit (CUDA or CPU) and predict =========
def fit_model(X_s, y_centered, u_train, ell_par, ell_perp, nu=1.5):
    if USE_CUDA:
        sf, sn, lscale = fit_hypers_cuda_aniso(X_s, y_centered, u_train, ell_par, ell_perp, nu=nu, iters=80, lr=0.08)
        Xt = torch.as_tensor(X_s, device=DEVICE, dtype=TORCH_DTYPE)
        u_t= torch.as_tensor(u_train, device=DEVICE, dtype=TORCH_DTYPE)
        lp = torch.as_tensor(np.clip(lscale*ell_par,  ell_min, ell_max), device=DEVICE, dtype=TORCH_DTYPE)
        lt = torch.as_tensor(np.clip(lscale*ell_perp, ell_min, ell_max*ratio_max), device=DEVICE, dtype=TORCH_DTYPE)
        K  = build_cov_ps_aniso_torch(Xt, Xt, u_t, u_t, lp, lt, lp, lt, nu=nu, sigma_f=sf)
        K.diagonal().add_(sn**2 + 1e-6); K = 0.5*(K+K.T)
        L  = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(torch.as_tensor(y_centered, device=DEVICE, dtype=TORCH_DTYPE).unsqueeze(1), L).squeeze(1)
        return {"mode":"cuda","nu":nu, "sf":sf,"sn":sn,"lscale":lscale,
                "Xt":Xt,"u_t":u_t,"lp":lp,"lt":lt,"L":L,"alpha":alpha}
    else:
        sf = float(np.std(y_centered)) or 1.0
        sn = 0.1
        lscale = 1.0
        ell_train_cpu = (ell_par + ell_perp)/2.0
        K = build_cov_iso_ps(X_s, X_s, ell_train_cpu, ell_train_cpu, nu=nu, sigma_f=sf)
        np.fill_diagonal(K, sf**2 + sn**2 + 1e-6)
        K = 0.5*(K+K.T)
        L = cholesky(K, lower=True, overwrite_a=False, check_finite=False)
        al = solve_triangular(L, y_centered, lower=True, check_finite=False)
        al = solve_triangular(L.T, al, lower=False, check_finite=False)
        return {"mode":"cpu","nu":nu, "sf":sf,"sn":sn,"lscale":lscale,
                "X_s":X_s,"ell_avg":ell_train_cpu,"L":L,"alpha":al}

def predict_on_points(model, Xq_s, aniso_params_of, trend_fn, return_var=True):
    if model["mode"] == "cuda":
        lpar_q, lperp_q, u_q = aniso_params_of(Xq_s)
        lpq = torch.as_tensor(np.clip(model["lscale"]*lpar_q,  ell_min, ell_max), device=DEVICE, dtype=TORCH_DTYPE)
        ltq = torch.as_tensor(np.clip(model["lscale"]*lperp_q, ell_min, ell_max*ratio_max), device=DEVICE, dtype=TORCH_DTYPE)
        uq  = torch.as_tensor(u_q, device=DEVICE, dtype=TORCH_DTYPE)
        Xq  = torch.as_tensor(Xq_s, device=DEVICE, dtype=TORCH_DTYPE)
        K_star = build_cov_ps_aniso_torch(Xq, model["Xt"], uq, model["u_t"], lpq, ltq, model["lp"], model["lt"],
                                          nu=model["nu"], sigma_f=model["sf"])
        mu_resid = K_star @ model["alpha"]
        v = torch.linalg.solve_triangular(model["L"], K_star.T, upper=False)
        var_lat = torch.clamp(model["sf"]**2 - (v*v).sum(dim=0), min=1e-12)
        mu = mu_resid.detach().cpu().numpy() + trend_fn(Xq_s)
        if return_var:
            return mu, var_lat.detach().cpu().numpy(), lpar_q, lperp_q, u_q
        else:
            return mu, None, lpar_q, lperp_q, u_q
    else:
        lpar_q, lperp_q, _ = aniso_params_of(Xq_s)
        ell_q = (lpar_q + lperp_q)/2.0
        K_star = build_cov_iso_ps(Xq_s, model["X_s"], ell_q, model["ell_avg"], nu=model["nu"], sigma_f=model["sf"])
        mu_resid = K_star @ model["alpha"]
        v = solve_triangular(model["L"], K_star.T, lower=True, check_finite=False)
        var_lat = np.maximum(model["sf"]**2 - np.einsum('ij,ij->j', v, v), 1e-12)
        mu = mu_resid + trend_fn(Xq_s)
        if return_var:
            return mu, var_lat, lpar_q, lperp_q, None
        else:
            return mu, None, lpar_q, lperp_q, None

def u_scaled_to_meters(u_xy, scaler):
    sx, sy = scaler.scale_
    u_m = u_xy.copy().astype(float)
    u_m[:,0] = u_m[:,0]/sx
    u_m[:,1] = u_m[:,1]/sy
    n = np.linalg.norm(u_m, axis=1, keepdims=True)
    u_m = np.where(n>1e-12, u_m/n, u_m)
    return u_m

# ========= pipeline =========
def main():
    CSV_PATH = "/home/blazar/karin_ws/src/simsetup/scripts/data/sim_data/sim1.csv"
    OUT_DIR  = "./compare_out"
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) ground truth (metres)
    xg, yg, T_gt = generate_ground_truth(bounds_x=(0,150), bounds_y=(0,50), res=1.0)
    Xg_m, Yg_m = np.meshgrid(xg, yg, indexing='xy')  # (Ny,Nx)
    XY_grid = np.stack([Xg_m.ravel(), Yg_m.ravel()], axis=1)
    extent = [xg.min(), xg.max(), yg.min(), yg.max()]

    # 2) CSV & prep
    data, X_train, y_train, y_center, trend, scaler = load_lawnmower_csv(CSV_PATH)

    # 3) KNN anisotropy (scaled)
    ell_par, ell_perp, u_train, aniso_params_of = fit_local_anisotropic_field(
        X_train.astype(np.float64), y_center.astype(np.float64), k=k_nn_fit
    )

    # 4) GP fit
    model = fit_model(X_train, y_center, u_train, ell_par, ell_perp, nu=1.5)
    sn_val = model.get("sn", 0.1)
    print(f"Model: {model['mode'].upper()} | nu={model['nu']} | sf≈{model['sf']:.3f} | sn≈{sn_val:.3f} | lscale≈{model['lscale']:.3f}")

    # 5) Predict on GT grid and on CSV points
    XY_s = scaler.transform(XY_grid)
    mu_flat, var_flat, lpar_q, lperp_q, u_q = predict_on_points(model, XY_s, aniso_params_of, trend_fn=trend, return_var=True)
    Zmu  = mu_flat.reshape(Yg_m.shape)
    Zvar = var_flat.reshape(Yg_m.shape)
    Zstd = np.sqrt(np.maximum(Zvar, 1e-12))

    mu_csv, var_csv, _, _, _ = predict_on_points(model, X_train, aniso_params_of, trend_fn=trend, return_var=True)
    std_csv = np.sqrt(np.maximum(var_csv, 1e-12))

    # 6) Metrics
    # global
    grid_rmse = rmse(Zmu, T_gt)
    grid_mae  = mae(Zmu, T_gt)
    r2_global = r2_score(T_gt.ravel(), Zmu.ravel())
    r_global  = pearson_r(T_gt.ravel(), Zmu.ravel())
    gt_name, gt_stats = stats("GT(grid)", T_gt)
    pr_name, pr_stats = stats("Pred(grid)", Zmu)
    nrmse_rng = grid_rmse / max(gt_stats["range"], 1e-12)
    nrmse_std = grid_rmse / max(gt_stats["std"],   1e-12)

    # convex-hull vs all-domain
    sample_xy = data[["X_m","Y_m"]].values
    inside = convex_hull_mask(sample_xy, XY_grid).reshape(Yg_m.shape)
    rmse_hull = rmse(Zmu[inside], T_gt[inside]) if inside.any() else float('nan')
    mae_hull  = mae(Zmu[inside],  T_gt[inside]) if inside.any() else float('nan')

    # at CSV points
    gt_at_csv, valid = bilinear_sample(xg, yg, T_gt, data['X_m'].values, data['Y_m'].values)
    mask = valid & np.isfinite(gt_at_csv)
    rmse_meas = rmse(mu_csv, y_train)
    mae_meas  = mae(mu_csv, y_train)
    rmse_gt   = rmse(mu_csv[mask], gt_at_csv[mask]); mae_gt = mae(mu_csv[mask], gt_at_csv[mask])

    # calibration: |error| within ±1σ, ±2σ on grid
    abs_err_grid = np.abs(Zmu - T_gt)
    p_within_1s = float(np.mean(abs_err_grid <= Zstd))
    p_within_2s = float(np.mean(abs_err_grid <= 2.0*Zstd))

    # measured/predicted range at samples
    sm_name, sm_stats = stats("Samples(Meas)", y_train)
    sp_name, sp_stats = stats("Samples(Pred μ)", mu_csv)
    sg_name, sg_stats = stats("Samples(GT@CSV)", gt_at_csv[mask])

    # 7) Print report
    print("\n==== Metrics ====")
    print(f"Grid   vs GT:    RMSE={grid_rmse:.3f}  MAE={grid_mae:.3f}  | R²={r2_global:.3f}  r={r_global:.3f}")
    print(f"           NRMSE(range)={100*nrmse_rng:.1f}%  NRMSE(std)={100*nrmse_std:.1f}%")
    if inside.any():
        print(f"Inside hull:     RMSE={rmse_hull:.3f}  MAE={mae_hull:.3f}  (covers sampled region)")
    print(f"CSV μ vs Meas:   RMSE={rmse_meas:.3f}  MAE={mae_meas:.3f}")
    print(f"CSV μ vs GT:     RMSE={rmse_gt:.3f}   MAE={mae_gt:.3f}  (N={mask.sum()} in-bounds)")
    print(f"Uncertainty calibration: P(|err|≤1σ)={100*p_within_1s:.1f}%  P(|err|≤2σ)={100*p_within_2s:.1f}%")
    print("-----------------")
    for nm, st in [ (gt_name,gt_stats), (pr_name,pr_stats), (sm_name,sm_stats), (sp_name,sp_stats), (sg_name,sg_stats) ]:
        print(f"{nm}: min={st['min']:.2f}  max={st['max']:.2f}  range={st['range']:.2f}  mean={st['mean']:.2f}  std={st['std']:.2f}")
    print("=================\n")

    # 8) Plots — 6 images
    def save_img(fig, fname): fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, fname), dpi=160); plt.close(fig)

    # (1) GT only
    fig = plt.figure(figsize=(7.2,4.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(T_gt, extent=extent, origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax, label="Temperature (°C)")
    ax.set_title("Ground Truth"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    save_img(fig, "01_gt.png")

    # (2) GT + samples
    fig = plt.figure(figsize=(7.2,4.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(T_gt, extent=extent, origin='lower', aspect='equal')
    ax.scatter(data['X_m'], data['Y_m'], s=6, alpha=0.6, edgecolors='k', linewidths=0.2)
    plt.colorbar(im, ax=ax, label="Temperature (°C)")
    ax.set_title("Ground Truth + Sampled Points"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    save_img(fig, "02_gt_samples.png")

    # (3) Pred mean only
    fig = plt.figure(figsize=(7.2,4.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(Zmu, extent=extent, origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax, label="Temperature (°C)")
    ax.set_title("Recreated Field (GP Mean)"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    save_img(fig, "03_pred_mean.png")

    # (4) Pred mean + samples
    fig = plt.figure(figsize=(7.2,4.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(Zmu, extent=extent, origin='lower', aspect='equal')
    ax.scatter(data['X_m'], data['Y_m'], s=6, alpha=0.6, edgecolors='k', linewidths=0.2)
    plt.colorbar(im, ax=ax, label="Temperature (°C)")
    ax.set_title("Recreated Field + Sampled Points"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    save_img(fig, "04_pred_mean_samples.png")

    # (5) Uncertainty (std) only
    fig = plt.figure(figsize=(7.2,4.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(Zstd, extent=extent, origin='lower', aspect='equal')
    plt.colorbar(im, ax=ax, label="Std (°C) — latent")
    ax.set_title("Uncertainty (Std)"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    save_img(fig, "05_uncert_std.png")

    # (6) Uncertainty (std) + samples
    fig = plt.figure(figsize=(7.2,4.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(Zstd, extent=extent, origin='lower', aspect='equal')
    ax.scatter(data['X_m'], data['Y_m'], s=6, alpha=0.6, edgecolors='k', linewidths=0.2)
    plt.colorbar(im, ax=ax, label="Std (°C) — latent")
    ax.set_title("Uncertainty (Std) + Sampled Points"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    save_img(fig, "06_uncert_std_samples.png")

    print(f"Saved 6 images to: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
