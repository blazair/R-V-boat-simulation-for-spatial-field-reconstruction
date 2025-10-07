#!/usr/bin/env python3
"""
Live field reconstruction visualization with KNN-based Non-Stationary GP
Uses CUDA-accelerated implementation from the original model
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.linalg import cholesky, solve_triangular
from scipy.special import kv, gamma as gamma_fn
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
import threading
import queue
from datetime import datetime

# ========= CUDA setup (EXACT SAME AS YOUR MODEL) =========
USE_CUDA = torch.cuda.is_available()
TORCH_DTYPE = torch.float64 if (not USE_CUDA or torch.cuda.get_device_capability()[0] >= 7) else torch.float32
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# ========= KNN-NSGP Model Parameters (EXACT SAME AS YOUR MODEL) =========
ell_min, ell_max = 0.30, 2.00
k_nn_fit, k_nn_query = 20, 30
idw_power = 2.0
ratio_min, ratio_max = 1.2, 3.0

class LiveFieldReconstructor(Node):
    def __init__(self):
        super().__init__('live_field_reconstructor')
        
        self.get_logger().info("="*60)
        self.get_logger().info("LIVE FIELD RECONSTRUCTION WITH KNN-NSGP")
        self.get_logger().info(f"CUDA: {USE_CUDA} | Device: {DEVICE} | Dtype: {TORCH_DTYPE}")
        self.get_logger().info("="*60)
        
        # Data storage
        self.samples = []  # [(x,y,temp), ...]
        self.current_pose = None
        self.update_queue = queue.Queue()
        
        # Ground truth field (same as your generator)
        self.generate_ground_truth()
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/wamv/odom', self.odom_callback, 10
        )
        
        # Service client for sampling
        self.field_client = self.create_client(Trigger, '/sample_field')
        
        # Check for new samples periodically
        self.sample_timer = self.create_timer(0.5, self.check_for_samples)
        self.last_sample_pos = None
        self.SAMPLE_INTERVAL = 2.0  # meters
        
        # Visualization will be started from main thread
        self.output_dir = "/home/blazar/karin_ws/src/simsetup/scripts/compare_out/sim2"
        self.frame_count = 0
        
    def generate_ground_truth(self):
        """Generate ground truth field (exact same as your ground_truth_field_node.py)"""
        np.random.seed(42)
        bounds_x = (0.0, 150.0)
        bounds_y = (0.0, 50.0)
        res = 1.0
        
        x_grid = np.arange(bounds_x[0], bounds_x[1], res)
        y_grid = np.arange(bounds_y[0], bounds_y[1], res)
        self.X_gt, self.Y_gt = np.meshgrid(x_grid, y_grid)
        
        # Generate field with same parameters
        T = np.zeros_like(self.X_gt)
        T += 20.0  # base temperature
        T += 0.08 * (self.X_gt - self.X_gt.min())  # gradient x
        T += 0.05 * (self.Y_gt - self.Y_gt.min())  # gradient y
        
        # Add features (same random seed ensures same field)
        np.random.seed(42)
        
        # Gaussian bumps
        for _ in range(8):
            cx = np.random.uniform(15, 135)
            cy = np.random.uniform(10, 40)
            sigma = np.random.uniform(4, 12)
            amp = np.random.uniform(-6, 12)
            T += amp * np.exp(-((self.X_gt-cx)**2 + (self.Y_gt-cy)**2) / (2*sigma**2))
        
        # Anisotropic features
        for _ in range(4):
            cx = np.random.uniform(15, 135)
            cy = np.random.uniform(10, 40)
            sig_maj = np.random.uniform(15, 25)
            sig_min = np.random.uniform(3, 8)
            theta = np.random.uniform(0, np.pi)
            amp = np.random.uniform(-5, 8)
            Xr = (self.X_gt-cx)*np.cos(theta) + (self.Y_gt-cy)*np.sin(theta)
            Yr = -(self.X_gt-cx)*np.sin(theta) + (self.Y_gt-cy)*np.cos(theta)
            T += amp * np.exp(-(Xr**2/(2*sig_maj**2) + Yr**2/(2*sig_min**2)))
        
        # Sharp fronts
        for _ in range(2):
            cx = np.random.uniform(20, 130)
            cy = np.random.uniform(10, 40)
            theta = np.random.uniform(0, np.pi)
            amp = np.random.uniform(4, 8)
            sharp = np.random.uniform(0.1, 0.3)
            d = (self.X_gt-cx)*np.cos(theta) + (self.Y_gt-cy)*np.sin(theta)
            T += amp * np.tanh(d * sharp)
        
        # Noise
        noise_base = 0.3
        noise_var = 0.4 * np.exp(-((self.X_gt - self.X_gt.mean())**2 + 
                                 (self.Y_gt - self.Y_gt.mean())**2) / (30**2))
        T += np.random.normal(0, noise_base + noise_var, size=T.shape)
        
        self.T_gt = T
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
        
        self.get_logger().info(f"Ground truth generated: {T.shape}")
        self.get_logger().info(f"Temperature range: [{T.min():.2f}, {T.max():.2f}] °C")
        
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        
    def check_for_samples(self):
        """Check if we should take a new sample"""
        if self.current_pose is None:
            return
            
        if not self.field_client.service_is_ready():
            return
            
        pos = self.current_pose.position
        
        # Check distance from last sample
        if self.last_sample_pos is not None:
            dx = pos.x - self.last_sample_pos[0]
            dy = pos.y - self.last_sample_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < self.SAMPLE_INTERVAL:
                return
        
        # Take sample
        request = Trigger.Request()
        future = self.field_client.call_async(request)
        future.add_done_callback(lambda f: self.sample_callback(f, pos))
        
    def sample_callback(self, future, pos):
        """Process new sample"""
        try:
            response = future.result()
            if response.success:
                temp = float(response.message)
                self.last_sample_pos = (pos.x, pos.y)
                new_sample = (pos.x, pos.y, temp)
                self.samples.append(new_sample)
                self.update_queue.put(new_sample)
                
                self.get_logger().info(
                    f"Sample #{len(self.samples)}: pos=({pos.x:.1f}, {pos.y:.1f}) temp={temp:.2f}°C"
                )
        except Exception as e:
            self.get_logger().error(f"Sample error: {e}")
    
    # ========= EXACT SAME MODEL FUNCTIONS FROM YOUR CODE =========
    
    def fit_linear_trend(self, X, y):
        """Fit linear trend for universal kriging"""
        A = np.c_[np.ones(len(X)), X]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        def trend_fn(Xq):
            Aq = np.c_[np.ones(len(Xq)), Xq]
            return Aq @ beta
        return beta, trend_fn
    
    def fit_local_anisotropic_field(self, X, y):
        """KNN-based anisotropic lengthscale learning (EXACT FROM YOUR compare.py)"""
        k = min(k_nn_fit, len(X))
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(X)
        dists, idxs = nbrs.kneighbors(X, return_distance=True)
        N = X.shape[0]
        
        grads = np.zeros((N,2))
        for i in range(N):
            Xi = X[idxs[i]]
            yi = y[idxs[i]]
            A = np.c_[np.ones(k), Xi - X[i]]
            beta, *_ = np.linalg.lstsq(A, yi, rcond=None)
            grads[i] = beta[1:3]
        
        r = np.linalg.norm(grads, axis=1)
        ql, qh = np.quantile(r, [0.05, 0.95]) if len(r) > 1 else (r[0], r[0])
        rhat = np.clip((r-ql)/max(qh-ql,1e-6), 0.0, 1.0)
        
        ell_par = ell_max - (ell_max-ell_min)*rhat
        ratio = ratio_min + (ratio_max-ratio_min)*(1.0 - rhat)
        ell_perp = ell_par * ratio
        
        u = grads.copy()
        nrm = np.linalg.norm(u, axis=1, keepdims=True)
        u = np.where(nrm>1e-8, u/nrm, np.array([1.0,0.0])[None,:])
        
        def aniso_params_of(Xq):
            kq = min(k_nn_query, len(X))
            dq, jq = nbrs.kneighbors(Xq, n_neighbors=kq, return_distance=True)
            w = 1.0/(np.power(dq, idw_power)+1e-6)
            gq = (w[...,None]*grads[jq]).sum(axis=1)/w.sum(axis=1, keepdims=True)
            gn = np.linalg.norm(gq, axis=1, keepdims=True)
            uq = np.where(gn>1e-8, gq/gn, np.array([[1.0,0.0]]))
            lpar_q = (w*ell_par[jq]).sum(axis=1)/w.sum(axis=1)
            ratio_q = (w*ratio[jq]).sum(axis=1)/w.sum(axis=1)
            lperp_q = lpar_q*ratio_q
            lpar_q = np.clip(lpar_q, ell_min, ell_max)
            lperp_q = np.clip(lperp_q, ell_min, ell_max*ratio_max)
            return lpar_q, lperp_q, uq
        
        return ell_par, ell_perp, u, aniso_params_of
    
    def _matern_shape_closed_form_torch(self, arg, nu):
        """Closed-form Matérn (EXACT FROM YOUR MODEL)"""
        if nu == 0.5: return torch.exp(-arg)
        if nu == 1.5: return (1.0 + arg)*torch.exp(-arg)
        if nu == 2.5: return (1.0 + arg + (arg*arg)/3.0)*torch.exp(-arg)
        raise NotImplementedError("nu must be 0.5, 1.5, or 2.5 in CUDA path")
    
    def _sigma_from_u_l(self, ux, uy, lpar, lperp):
        """Build Σ matrix components (EXACT FROM YOUR MODEL)"""
        lp2 = lpar*lpar; lt2 = lperp*lperp; d = (lp2 - lt2)
        s11 = lt2 + d*(ux*ux)
        s22 = lt2 + d*(uy*uy)
        s12 = d*(ux*uy)
        return s11, s12, s22
    
    def build_cov_ps_aniso_torch(self, XA, XB, uA, uB, lparA, lperpA, lparB, lperpB,
                                 nu=1.5, sigma_f=1.0, eps=1e-12):
        """CUDA anisotropic Paciorek-Schervish kernel (EXACT FROM YOUR MODEL)"""
        XA = torch.as_tensor(XA, device=DEVICE, dtype=TORCH_DTYPE)
        XB = torch.as_tensor(XB, device=DEVICE, dtype=TORCH_DTYPE)
        uA = torch.as_tensor(uA, device=DEVICE, dtype=TORCH_DTYPE)
        uB = torch.as_tensor(uB, device=DEVICE, dtype=TORCH_DTYPE)
        lparA = torch.as_tensor(lparA, device=DEVICE, dtype=TORCH_DTYPE)
        lperpA= torch.as_tensor(lperpA,device=DEVICE, dtype=TORCH_DTYPE)
        lparB = torch.as_tensor(lparB, device=DEVICE, dtype=TORCH_DTYPE)
        lperpB= torch.as_tensor(lperpB,device=DEVICE, dtype=TORCH_DTYPE)

        s11A,s12A,s22A = self._sigma_from_u_l(uA[:,0], uA[:,1], lparA, lperpA)
        s11B,s12B,s22B = self._sigma_from_u_l(uB[:,0], uB[:,1], lparB, lperpB)
        
        detA = (s11A*s22A - s12A*s12A).clamp_min(eps)
        detB = (s11B*s22B - s12B*s12B).clamp_min(eps)
        
        M11 = 0.5*(s11A.unsqueeze(1) + s11B.unsqueeze(0))
        M22 = 0.5*(s22A.unsqueeze(1) + s22B.unsqueeze(0))
        M12 = 0.5*(s12A.unsqueeze(1) + s12B.unsqueeze(0))
        detM= (M11*M22 - M12*M12).clamp_min(eps)
        
        pref = torch.exp(0.25*torch.log(detA).unsqueeze(1) + 
                        0.25*torch.log(detB).unsqueeze(0) - 0.5*torch.log(detM))
        
        dx = XA[:,0].unsqueeze(1) - XB[:,0].unsqueeze(0)
        dy = XA[:,1].unsqueeze(1) - XB[:,1].unsqueeze(0)
        inv_detM = 1.0/detM
        Q = (M22*dx*dx - 2.0*M12*dx*dy + M11*dy*dy) * inv_detM
        Q = Q.clamp_min(eps)
        
        arg = torch.sqrt(torch.tensor(2.0*nu, device=DEVICE, dtype=TORCH_DTYPE) * Q)
        shape = self._matern_shape_closed_form_torch(arg, float(nu))
        
        return (sigma_f**2) * pref * shape
    
    def build_cov_iso_ps(self, XA, XB, lA, lB, nu=1.5, sigma_f=1.0):
        """CPU fallback isotropic PS kernel (EXACT FROM YOUR MODEL)"""
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
    
    def fit_hypers_cuda_aniso(self, X, y_centered, u_train, lpar, lperp, nu=1.5, iters=80, lr=0.08):
        """Fit hyperparameters on GPU (EXACT FROM YOUR MODEL)"""
        Xt = torch.as_tensor(X, device=DEVICE, dtype=TORCH_DTYPE)
        yt = torch.as_tensor(y_centered, device=DEVICE, dtype=TORCH_DTYPE).unsqueeze(1)
        uA = torch.as_tensor(u_train, device=DEVICE, dtype=TORCH_DTYPE)
        lpar0 = torch.as_tensor(lpar, device=DEVICE, dtype=TORCH_DTYPE)
        lperp0= torch.as_tensor(lperp,device=DEVICE, dtype=TORCH_DTYPE)
        
        log_sigma_f = torch.tensor([np.log(np.std(y_centered)+1e-6)], device=DEVICE, dtype=TORCH_DTYPE, requires_grad=True)
        log_sigma_n = torch.tensor([np.log(0.1)], device=DEVICE, dtype=TORCH_DTYPE, requires_grad=True)
        log_lscale = torch.tensor([0.0], device=DEVICE, dtype=TORCH_DTYPE, requires_grad=True)
        
        opt = torch.optim.Adam([log_sigma_f, log_sigma_n, log_lscale], lr=lr)
        
        for _ in range(iters):
            opt.zero_grad()
            sf = torch.exp(log_sigma_f); sn = torch.exp(log_sigma_n); ls = torch.exp(log_lscale)
            lpar_use = (ls*lpar0).clamp(min=ell_min, max=ell_max)
            lperp_use = (ls*lperp0).clamp(min=ell_min, max=ell_max*ratio_max)
            
            K = self.build_cov_ps_aniso_torch(Xt, Xt, uA, uA, lpar_use, lperp_use, lpar_use, lperp_use, nu=float(nu), sigma_f=sf)
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
    
    def fit_model(self, X_s, y_centered, u_train, ell_par, ell_perp, nu=1.5):
        """Fit GP model using CUDA or CPU (EXACT FROM YOUR MODEL)"""
        if USE_CUDA:
            sf, sn, lscale = self.fit_hypers_cuda_aniso(X_s, y_centered, u_train, ell_par, ell_perp, nu=nu, iters=80, lr=0.08)
            Xt = torch.as_tensor(X_s, device=DEVICE, dtype=TORCH_DTYPE)
            u_t= torch.as_tensor(u_train, device=DEVICE, dtype=TORCH_DTYPE)
            lp = torch.as_tensor(np.clip(lscale*ell_par, ell_min, ell_max), device=DEVICE, dtype=TORCH_DTYPE)
            lt = torch.as_tensor(np.clip(lscale*ell_perp, ell_min, ell_max*ratio_max), device=DEVICE, dtype=TORCH_DTYPE)
            K = self.build_cov_ps_aniso_torch(Xt, Xt, u_t, u_t, lp, lt, lp, lt, nu=nu, sigma_f=sf)
            K.diagonal().add_(sn**2 + 1e-6); K = 0.5*(K+K.T)
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(torch.as_tensor(y_centered, device=DEVICE, dtype=TORCH_DTYPE).unsqueeze(1), L).squeeze(1)
            return {"mode":"cuda","nu":nu, "sf":sf,"sn":sn,"lscale":lscale,
                   "Xt":Xt,"u_t":u_t,"lp":lp,"lt":lt,"L":L,"alpha":alpha}
        else:
            sf = float(np.std(y_centered)) or 1.0
            sn = 0.1; lscale = 1.0
            ell_train_cpu = (ell_par + ell_perp)/2.0
            K = self.build_cov_iso_ps(X_s, X_s, ell_train_cpu, ell_train_cpu, nu=nu, sigma_f=sf)
            np.fill_diagonal(K, sf**2 + sn**2 + 1e-6)
            K = 0.5*(K+K.T)
            L = cholesky(K, lower=True, overwrite_a=False, check_finite=False)
            al = solve_triangular(L, y_centered, lower=True, check_finite=False)
            al = solve_triangular(L.T, al, lower=False, check_finite=False)
            return {"mode":"cpu","nu":nu, "sf":sf,"sn":sn,"lscale":lscale,
                   "X_s":X_s,"ell_avg":ell_train_cpu,"L":L,"alpha":al}
    
    def predict_on_points(self, model, Xq_s, aniso_params_of, trend_fn, return_var=True):
        """Predict using fitted model (EXACT FROM YOUR MODEL)"""
        if model["mode"] == "cuda":
            lpar_q, lperp_q, u_q = aniso_params_of(Xq_s)
            lpq = torch.as_tensor(np.clip(model["lscale"]*lpar_q, ell_min, ell_max), device=DEVICE, dtype=TORCH_DTYPE)
            ltq = torch.as_tensor(np.clip(model["lscale"]*lperp_q, ell_min, ell_max*ratio_max), device=DEVICE, dtype=TORCH_DTYPE)
            uq = torch.as_tensor(u_q, device=DEVICE, dtype=TORCH_DTYPE)
            Xq = torch.as_tensor(Xq_s, device=DEVICE, dtype=TORCH_DTYPE)
            
            K_star = self.build_cov_ps_aniso_torch(Xq, model["Xt"], uq, model["u_t"], lpq, ltq, model["lp"], model["lt"],
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
            K_star = self.build_cov_iso_ps(Xq_s, model["X_s"], ell_q, model["ell_avg"], nu=model["nu"], sigma_f=model["sf"])
            mu_resid = K_star @ model["alpha"]
            v = solve_triangular(model["L"], K_star.T, lower=True, check_finite=False)
            var_lat = np.maximum(model["sf"]**2 - np.einsum('ij,ij->j', v, v), 1e-12)
            mu = mu_resid + trend_fn(Xq_s)
            if return_var:
                return mu, var_lat, lpar_q, lperp_q, None
            else:
                return mu, None, lpar_q, lperp_q, None
    
    def run_knn_nsgp(self, samples):
        """Run KNN-NSGP model on current samples using CUDA"""
        if len(samples) < 10:  # Need minimum samples for KNN
            return None
        
        # Extract data
        X_m = np.array([(s[0], s[1]) for s in samples])
        y = np.array([s[2] for s in samples])
        
        # Scale coordinates
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_m)
        
        # Fit trend and center
        _, trend = self.fit_linear_trend(X_s, y)
        y_centered = y - trend(X_s)
        
        # Learn anisotropic field
        ell_par, ell_perp, u_train, aniso_params_of = self.fit_local_anisotropic_field(X_s, y_centered)
        
        # Fit model (GPU or CPU)
        model = self.fit_model(X_s, y_centered, u_train, ell_par, ell_perp, nu=1.5)
        
        # Prediction grid
        XY_grid = np.stack([self.X_gt.ravel(), self.Y_gt.ravel()], axis=1)
        XY_s = scaler.transform(XY_grid)
        
        # Predict using GPU
        mu_flat, var_flat, _, _, _ = self.predict_on_points(model, XY_s, aniso_params_of, trend, return_var=True)
        
        mu = mu_flat.reshape(self.T_gt.shape)
        std = np.sqrt(np.maximum(var_flat, 1e-12)).reshape(self.T_gt.shape)
        
        return {
            'mu': mu,
            'std': std,
            'X_m': X_m,
            'scaler': scaler,
            'model_info': f"{model['mode'].upper()} | σf={model['sf']:.3f} | σn={model['sn']:.3f} | ℓ_scale={model['lscale']:.3f}"
        }
    
    def compute_metrics(self, pred):
        """Compute real-time metrics"""
        if pred is None:
            return {}
        
        # Global metrics
        rmse = np.sqrt(np.mean((pred['mu'] - self.T_gt)**2))
        mae = np.mean(np.abs(pred['mu'] - self.T_gt))
        
        # R-squared
        ss_res = np.sum((self.T_gt - pred['mu'])**2)
        ss_tot = np.sum((self.T_gt - self.T_gt.mean())**2)
        r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
        
        # Convex hull metrics
        if len(pred['X_m']) >= 3:
            try:
                hull = ConvexHull(pred['X_m'])
                poly = Path(pred['X_m'][hull.vertices])
                XY_grid = np.stack([self.X_gt.ravel(), self.Y_gt.ravel()], axis=1)
                inside = poly.contains_points(XY_grid).reshape(self.T_gt.shape)
                
                if inside.any():
                    rmse_hull = np.sqrt(np.mean((pred['mu'][inside] - self.T_gt[inside])**2))
                    mae_hull = np.mean(np.abs(pred['mu'][inside] - self.T_gt[inside]))
                else:
                    rmse_hull = mae_hull = np.nan
            except:
                rmse_hull = mae_hull = np.nan
        else:
            rmse_hull = mae_hull = np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'rmse_hull': rmse_hull,
            'mae_hull': mae_hull,
            'n_samples': len(pred['X_m']),
            'model_info': pred.get('model_info', '')
        }
    
    def save_individual_plots(self, pred, metrics, metrics_history):
        """Save individual visualization components as separate images"""
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create figures with Agg backend (no display windows)
        # 1. Ground Truth Field
        fig_gt = Figure(figsize=(8, 6))
        ax_gt = fig_gt.add_subplot(111)
        im_gt = ax_gt.imshow(self.T_gt, extent=self.extent, origin='lower', aspect='equal',
                           vmin=self.T_gt.min(), vmax=self.T_gt.max())
        ax_gt.set_title('Ground Truth Temperature Field')
        ax_gt.set_xlabel('X (m)')
        ax_gt.set_ylabel('Y (m)')
        fig_gt.colorbar(im_gt, ax=ax_gt, label='Temperature (°C)')
        fig_gt.tight_layout()
        fig_gt.savefig(f'{self.output_dir}/ground_truth_{self.frame_count:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_gt)
        
        # 2. Predicted Field (clean, without sampling points)
        fig_pred = Figure(figsize=(8, 6))
        ax_pred = fig_pred.add_subplot(111)
        im_pred = ax_pred.imshow(pred['mu'], extent=self.extent, origin='lower', aspect='equal',
                               vmin=self.T_gt.min(), vmax=self.T_gt.max())
        ax_pred.set_title('Predicted Temperature Field (Clean)')
        ax_pred.set_xlabel('X (m)')
        ax_pred.set_ylabel('Y (m)')
        fig_pred.colorbar(im_pred, ax=ax_pred, label='Temperature (°C)')
        fig_pred.tight_layout()
        fig_pred.savefig(f'{self.output_dir}/predicted_clean_{self.frame_count:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_pred)
        
        # 3. Predicted Field with Sampling Points
        fig_pred_samples = Figure(figsize=(8, 6))
        ax_pred_samples = fig_pred_samples.add_subplot(111)
        im_pred_samples = ax_pred_samples.imshow(pred['mu'], extent=self.extent, origin='lower', aspect='equal',
                                               vmin=self.T_gt.min(), vmax=self.T_gt.max())
        ax_pred_samples.scatter(pred['X_m'][:, 0], pred['X_m'][:, 1], 
                              c='red', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax_pred_samples.set_title('Predicted Field with Sampling Points')
        ax_pred_samples.set_xlabel('X (m)')
        ax_pred_samples.set_ylabel('Y (m)')
        fig_pred_samples.colorbar(im_pred_samples, ax=ax_pred_samples, label='Temperature (°C)')
        fig_pred_samples.tight_layout()
        fig_pred_samples.savefig(f'{self.output_dir}/predicted_with_samples_{self.frame_count:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_pred_samples)
        
        # 4. Uncertainty Field
        fig_std = Figure(figsize=(8, 6))
        ax_std = fig_std.add_subplot(111)
        im_std = ax_std.imshow(pred['std'], extent=self.extent, origin='lower', aspect='equal')
        ax_std.set_title('Prediction Uncertainty')
        ax_std.set_xlabel('X (m)')
        ax_std.set_ylabel('Y (m)')
        fig_std.colorbar(im_std, ax=ax_std, label='Standard Deviation (°C)')
        fig_std.tight_layout()
        fig_std.savefig(f'{self.output_dir}/uncertainty_{self.frame_count:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_std)
        
        # 5. Error Field
        fig_err = Figure(figsize=(8, 6))
        ax_err = fig_err.add_subplot(111)
        error = np.abs(pred['mu'] - self.T_gt)
        im_err = ax_err.imshow(error, extent=self.extent, origin='lower', aspect='equal')
        ax_err.set_title('Absolute Error')
        ax_err.set_xlabel('X (m)')
        ax_err.set_ylabel('Y (m)')
        fig_err.colorbar(im_err, ax=ax_err, label='Error (°C)')
        fig_err.tight_layout()
        fig_err.savefig(f'{self.output_dir}/error_{self.frame_count:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_err)
        
        # 6. Metrics Text
        fig_metrics = Figure(figsize=(10, 8))
        ax_metrics = fig_metrics.add_subplot(111)
        ax_metrics.axis('off')
        metrics_text = f"""
CURRENT METRICS ({metrics.get('model_info', 'N/A')}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Global RMSE:        {metrics['rmse']:.3f} °C
Global MAE:         {metrics['mae']:.3f} °C
R-squared:          {metrics['r2']:.3f}
Hull RMSE:          {metrics['rmse_hull']:.3f} °C
Hull MAE:           {metrics['mae_hull']:.3f} °C
Samples collected:  {metrics['n_samples']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temperature range:
  Ground Truth: [{self.T_gt.min():.1f}, {self.T_gt.max():.1f}] °C
  Predicted:    [{pred['mu'].min():.1f}, {pred['mu'].max():.1f}] °C
        """
        # Replace ℓ with mathtext to avoid font warnings
        metrics_text = metrics_text.replace('ℓ', r'$\ell$')
        ax_metrics.text(0.05, 0.5, metrics_text, transform=ax_metrics.transAxes,
                      fontsize=12, fontfamily='sans-serif', verticalalignment='center')
        fig_metrics.tight_layout()
        fig_metrics.savefig(f'{self.output_dir}/metrics_{self.frame_count:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_metrics)
        
        # 7. Progress Plots
        if len(metrics_history['n']) > 1:
            fig_progress = Figure(figsize=(12, 6))
            ax_progress = fig_progress.add_subplot(111)
            ax_progress.plot(metrics_history['n'], metrics_history['rmse'], 'b-', label='RMSE Global', linewidth=2)
            ax_progress.plot(metrics_history['n'], metrics_history['mae'], 'g-', label='MAE Global', linewidth=2)
            if len(metrics_history['rmse_hull']) > 0:
                ax_progress.plot(metrics_history['n'][-len(metrics_history['rmse_hull']):], 
                               metrics_history['rmse_hull'], 'b--', label='RMSE Hull', linewidth=2)
                ax_progress.plot(metrics_history['n'][-len(metrics_history['mae_hull']):], 
                               metrics_history['mae_hull'], 'g--', label='MAE Hull', linewidth=2)
            ax_progress.set_xlabel('Number of Samples')
            ax_progress.set_ylabel('Error (°C)')
            ax_progress.set_title('Error Evolution Over Time')
            ax_progress.legend()
            ax_progress.grid(True, alpha=0.3)
            fig_progress.tight_layout()
            fig_progress.savefig(f'{self.output_dir}/progress_{self.frame_count:04d}.png', dpi=150, bbox_inches='tight')
            plt.close(fig_progress)
        
        # 8. Comparison Scatter Plot
        fig_comparison = Figure(figsize=(8, 6))
        ax_comparison = fig_comparison.add_subplot(111)
        ax_comparison.scatter(self.T_gt.ravel(), pred['mu'].ravel(), s=1, alpha=0.3)
        ax_comparison.plot([self.T_gt.min(), self.T_gt.max()], 
                         [self.T_gt.min(), self.T_gt.max()], 'r--', linewidth=2)
        ax_comparison.set_xlabel('Ground Truth (°C)')
        ax_comparison.set_ylabel('Predicted (°C)')
        ax_comparison.set_title(f'Prediction vs Truth (R²={metrics["r2"]:.3f})')
        ax_comparison.grid(True, alpha=0.3)
        fig_comparison.tight_layout()
        fig_comparison.savefig(f'{self.output_dir}/comparison_{self.frame_count:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_comparison)
        
        # 9. Error Histogram
        fig_histogram = Figure(figsize=(8, 6))
        ax_histogram = fig_histogram.add_subplot(111)
        errors = (pred['mu'] - self.T_gt).ravel()
        ax_histogram.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax_histogram.axvline(0, color='red', linestyle='--', linewidth=2)
        ax_histogram.set_xlabel('Error (°C)')
        ax_histogram.set_ylabel('Frequency')
        ax_histogram.set_title(f'Error Distribution (μ={np.mean(errors):.3f}, σ={np.std(errors):.3f})')
        ax_histogram.grid(True, alpha=0.3)
        fig_histogram.tight_layout()
        fig_histogram.savefig(f'{self.output_dir}/histogram_{self.frame_count:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig_histogram)
        
        self.frame_count += 1
    
    def run_visualization(self):
        """Main visualization loop"""
        plt.ion()
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Subplots
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_pred = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[0, 2])
        ax_std = fig.add_subplot(gs[0, 3])
        ax_metrics = fig.add_subplot(gs[1, :2])
        ax_progress = fig.add_subplot(gs[1, 2:])
        ax_comparison = fig.add_subplot(gs[2, :2])
        ax_histogram = fig.add_subplot(gs[2, 2:])
        
        # Initialize plots
        im_gt = ax_gt.imshow(self.T_gt, extent=self.extent, origin='lower', aspect='equal')
        ax_gt.set_title("Ground Truth")
        ax_gt.set_xlabel("X (m)")
        ax_gt.set_ylabel("Y (m)")
        plt.colorbar(im_gt, ax=ax_gt, label="°C")
        
        # Storage for metrics history
        metrics_history = {'rmse': [], 'mae': [], 'r2': [], 'rmse_hull': [], 'mae_hull': [], 'n': []}
        
        def update(frame):
            # Check for new samples
            new_samples = []
            while not self.update_queue.empty():
                new_samples.append(self.update_queue.get())
            
            if len(self.samples) < 10:
                return
            
            # Run model
            pred = self.run_knn_nsgp(self.samples)
            if pred is None:
                return
            
            # Compute metrics
            metrics = self.compute_metrics(pred)
            
            # Update metrics history
            for key in ['rmse', 'mae', 'r2', 'rmse_hull', 'mae_hull']:
                if key in metrics and not np.isnan(metrics[key]):
                    metrics_history[key].append(metrics[key])
            metrics_history['n'].append(len(self.samples))
            
            # Clear and update plots
            ax_pred.clear()
            im_pred = ax_pred.imshow(pred['mu'], extent=self.extent, origin='lower', aspect='equal',
                                     vmin=self.T_gt.min(), vmax=self.T_gt.max())
            ax_pred.scatter(pred['X_m'][:,0], pred['X_m'][:,1], s=10, c='red', alpha=0.5)
            ax_pred.set_title(f"KNN-NSGP Prediction (n={len(self.samples)})")
            ax_pred.set_xlabel("X (m)")
            ax_pred.set_ylabel("Y (m)")
            
            ax_err.clear()
            error = np.abs(pred['mu'] - self.T_gt)
            im_err = ax_err.imshow(error, extent=self.extent, origin='lower', aspect='equal')
            ax_err.set_title("Absolute Error")
            ax_err.set_xlabel("X (m)")
            ax_err.set_ylabel("Y (m)")
            
            ax_std.clear()
            im_std = ax_std.imshow(pred['std'], extent=self.extent, origin='lower', aspect='equal')
            ax_std.scatter(pred['X_m'][:,0], pred['X_m'][:,1], s=10, c='red', alpha=0.5)
            ax_std.set_title("Uncertainty (Std)")
            ax_std.set_xlabel("X (m)")
            ax_std.set_ylabel("Y (m)")
            
            # Metrics text
            ax_metrics.clear()
            ax_metrics.axis('off')
            metrics_text = f"""
CURRENT METRICS ({metrics.get('model_info', 'N/A')}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Global RMSE:        {metrics['rmse']:.3f} °C
Global MAE:         {metrics['mae']:.3f} °C
R-squared:          {metrics['r2']:.3f}
Hull RMSE:          {metrics['rmse_hull']:.3f} °C
Hull MAE:           {metrics['mae_hull']:.3f} °C
Samples collected:  {metrics['n_samples']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temperature range:
  Ground Truth: [{self.T_gt.min():.1f}, {self.T_gt.max():.1f}] °C
  Predicted:    [{pred['mu'].min():.1f}, {pred['mu'].max():.1f}] °C
            """
            # Replace ℓ with mathtext to avoid font warnings
            metrics_text = metrics_text.replace('ℓ', r'$\ell$')
            ax_metrics.text(0.05, 0.5, metrics_text, transform=ax_metrics.transAxes,
                          fontsize=10, fontfamily='sans-serif', verticalalignment='center')
            
            # Progress plots
            ax_progress.clear()
            if len(metrics_history['n']) > 1:
                ax_progress.plot(metrics_history['n'], metrics_history['rmse'], 'b-', label='RMSE Global', linewidth=2)
                ax_progress.plot(metrics_history['n'], metrics_history['mae'], 'g-', label='MAE Global', linewidth=2)
                if len(metrics_history['rmse_hull']) > 0:
                    ax_progress.plot(metrics_history['n'][-len(metrics_history['rmse_hull']):], 
                                   metrics_history['rmse_hull'], 'b--', label='RMSE Hull', linewidth=2)
                    ax_progress.plot(metrics_history['n'][-len(metrics_history['mae_hull']):], 
                                   metrics_history['mae_hull'], 'g--', label='MAE Hull', linewidth=2)
                ax_progress.legend()  # Add legend inside the if block
            ax_progress.set_xlabel("Number of Samples")
            ax_progress.set_ylabel("Error (°C)")
            ax_progress.set_title("Error Evolution")
            ax_progress.grid(True, alpha=0.3)
            
            # Comparison scatter
            ax_comparison.clear()
            ax_comparison.scatter(self.T_gt.ravel(), pred['mu'].ravel(), s=1, alpha=0.3)
            ax_comparison.plot([self.T_gt.min(), self.T_gt.max()], 
                             [self.T_gt.min(), self.T_gt.max()], 'r--', linewidth=2)
            ax_comparison.set_xlabel("Ground Truth (°C)")
            ax_comparison.set_ylabel("Predicted (°C)")
            ax_comparison.set_title(f"Prediction vs Truth (R²={metrics['r2']:.3f})")
            ax_comparison.grid(True, alpha=0.3)
            
            # Error histogram
            ax_histogram.clear()
            errors = (pred['mu'] - self.T_gt).ravel()
            ax_histogram.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            ax_histogram.axvline(0, color='red', linestyle='--', linewidth=2)
            ax_histogram.set_xlabel("Error (°C)")
            ax_histogram.set_ylabel("Frequency")
            ax_histogram.set_title(f"Error Distribution (μ={np.mean(errors):.3f}, σ={np.std(errors):.3f})")
            ax_histogram.grid(True, alpha=0.3)
            
            # Update the display first
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            # Save individual plots in a non-blocking way
            try:
                self.save_individual_plots(pred, metrics, metrics_history)
            except Exception as e:
                self.get_logger().warn(f"Failed to save plots: {e}")
            
            # Terminal output
            self.get_logger().info(
                f"[LIVE GPU] n={len(self.samples)} | RMSE={metrics['rmse']:.3f} | "
                f"MAE={metrics['mae']:.3f} | R²={metrics['r2']:.3f} | "
                f"Hull_RMSE={metrics['rmse_hull']:.3f} | {metrics.get('model_info', '')} | "
                f"Saved frame {self.frame_count-1:04d}"
            )
        
        # Animation
        ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
        plt.show()
        
        # Keep running
        while rclpy.ok():
            plt.pause(0.1)

def main():
    rclpy.init()
    node = LiveFieldReconstructor()
    
    # Create executor for ROS in background thread
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    
    # Start ROS in background thread
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()
    
    try:
        # Run visualization on main thread
        node.run_visualization()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()