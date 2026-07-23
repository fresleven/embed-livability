"""
Utilities for Conditional VAE (C-VAE) based deconfounder for spatial confounding.
Implements the loss functions and posterior predictive checks as described in
the spatial deconfounder methodology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
from scipy.stats import binom
import logging

logger = logging.getLogger(__name__)


class CVAELoss(nn.Module):
    """
    Computes the C-VAE loss: L_A = E_q[-log p(A|X,Z)] + beta * KL(q||p)
    where:
    - q_phi is the encoder (inference network)
    - p_psi is the decoder (likelihood network)
    - Z ~ p_theta is the prior (GMRF with spatial smoothness via Laplacian)
    """
    
    def __init__(self, beta=1.0, prior_type='laplacian_gmrf', latent_h=4, latent_w=4, tau=1.0, connectivity=8, treatment = "alphaearth"):
        super().__init__()
        self.beta = beta
        self.prior_type = prior_type
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.tau = tau
        self.n_pixels = latent_h * latent_w
        self.connectivity = connectivity
        self.treatment = treatment            
        
        # Compute Laplacian matrix for spatial smoothness
        self.laplacian = self._compute_laplacian(latent_h, latent_w, connectivity)
    
    def _compute_laplacian(self, h, w, connectivity=8):
        """
        Compute the graph Laplacian for a 2D grid.
        L = D - A where D is degree matrix and A is adjacency matrix.
        
        Args:
            h, w: height and width of spatial grid
            connectivity: 4 or 8 (4=4-connected, 8=8-connected including diagonals)
        
        Returns:
            laplacian: (n_pixels, n_pixels) Laplacian matrix
        """
        if connectivity not in [4, 8]:
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
        
        n_pixels = h * w
        # Create adjacency matrix (4-connected or 8-connected grid)
        A = torch.zeros(n_pixels, n_pixels)
        
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                
                # 4-connected neighbors (up, down, left, right)
                # Right neighbor
                if j < w - 1:
                    A[idx, idx + 1] = 1
                    A[idx + 1, idx] = 1
                # Bottom neighbor
                if i < h - 1:
                    A[idx, idx + w] = 1
                    A[idx + w, idx] = 1
                
                # 8-connected: add diagonal neighbors
                if connectivity == 8:
                    # Bottom-right diagonal
                    if i < h - 1 and j < w - 1:
                        A[idx, idx + w + 1] = 1
                        A[idx + w + 1, idx] = 1
                    # Bottom-left diagonal
                    if i < h - 1 and j > 0:
                        A[idx, idx + w - 1] = 1
                        A[idx + w - 1, idx] = 1
        
        # Degree matrix
        D = torch.diag(A.sum(dim=1))
        
        # Laplacian
        L = D - A
        return L
    
    def set_beta(self, beta):
        """Update KL weight (for KL warmup scheduling)"""
        self.beta = beta
    
    def forward(self, reconstructed_mean, reconstructed_kappa, original_treatments, mu, logvar):
        """
        Args:
            reconstructed_mean: (B, C, H, W) - mean of reconstructed treatments
            reconstructed_kappa: (B, 1, H, W) - kappa parameter of reconstructed treatments
            original_treatments: (B, C, H, W) - ground truth treatments
            mu: (B, latent_d, latent_h, latent_w) - mean of posterior q
            logvar: (B, latent_d, latent_h, latent_w) - log variance of posterior q
        
        Returns:
            loss: scalar loss
            reconstruction_loss: scalar
            kl_loss: scalar
        """
        # Get the middle fifth of the spatial dimensions
        _, _, H, W = original_treatments.shape

        # Calculate the middle fifth boundaries
        h_start = H * 2 // 5  # Start at 40% (2/5)
        h_end = H * 3 // 5    # End at 60% (3/5)
        w_start = W * 2 // 5
        w_end = W * 3 // 5
        h_end -= 1 if H != 50 else 0
        w_end -= 1 if W != 50 else 0

        # Extract middle fifth
        original_treatments = original_treatments[:, :, h_start:h_end, w_start:w_end]

        if self.treatment == "alphaearth":
            # Dot product (cosine similarity for unit vectors)
            dim = original_treatments.size(1)
            dot = (reconstructed_mean * original_treatments).sum(dim=1, keepdim=True)
            if reconstructed_kappa is None:
                b = 0.05
                recon_loss = -b * dot.mean()
            else:
                # Clamp kappa safely and compute log-normalizer approximation
                # reconstructed_kappa = torch.clamp(reconstructed_kappa, min=1e-2, max=50.0)
                logC_approx = 0.5 * (dim - 1) * torch.log(2 * torch.pi / reconstructed_kappa)  # dim = your vector size

                # Full vMF NLL: logC(κ) - κ * μ^T x
                recon_loss = (logC_approx - reconstructed_kappa * dot).mean()
        else:
            rs_true, dsm_true, giu_true = torch.chunk(original_treatments, 3, dim=1)
            rs_mean, dsm_mean, giu_mean = torch.chunk(reconstructed_mean, 3, dim=1)

            #print(f"rs_true: min={rs_true.min()}, max={rs_true.max()}, has_nan={torch.isnan(rs_true).any()}")
            #print(f"dsm_true: min={dsm_true.min()}, max={dsm_true.max()}, has_nan={torch.isnan(dsm_true).any()}")
            #print(f"giu_true: min={giu_true.min()}, max={giu_true.max()}, has_nan={torch.isnan(giu_true).any()}")
            #print(f"rs_mean: min={rs_mean.min()}, max={rs_mean.max()}, has_nan={torch.isnan(rs_mean).any()}")
            #print(f"dsm_mean: min={dsm_mean.min()}, max={dsm_mean.max()}, has_nan={torch.isnan(dsm_mean).any()}")
            #print(f"giu_mean: min={giu_mean.min()}, max={giu_mean.max()}, has_nan={torch.isnan(giu_mean).any()}")
            

            if reconstructed_kappa is not None:
                # --- RS: Zero-Inflated Normal ---
                rs_logit_p = reconstructed_kappa['rs']['logit_p']
                rs_logvar = reconstructed_kappa['rs']['logvar']

                #print(f"rs_logit_p: min={rs_logit_p.min()}, max={rs_logit_p.max()}, has_nan={torch.isnan(rs_logit_p).any()}")
                #print(f"rs_logvar: min={rs_logvar.min()}, max={rs_logvar.max()}, has_nan={torch.isnan(rs_logvar).any()}")
      
                rs_p_zero = torch.sigmoid(rs_logit_p)
                #print(f"rs_p_zero: min={rs_p_zero.min()}, max={rs_p_zero.max()}, has_nan={torch.isnan(rs_p_zero).any()}")

                rs_zero_mask = (rs_true == 0).float()
                rs_non_zero_mask = (rs_true > 0).float()
                
                # Log-likelihood for zeros
                rs_zero_ll = rs_zero_mask * torch.log(rs_p_zero + 1e-8)
                #print(f"rs_zero_ll: min={rs_zero_ll.min()}, max={rs_zero_ll.max()}, has_nan={torch.isnan(rs_zero_ll).any()}")
                if not torch.isfinite(rs_zero_ll).all():
                    raise ValueError(f"rs_zero_ll contains non-finite values!")
                
                # Log-likelihood for non-zeros (Gaussian)
                rs_std = torch.exp(0.5 * rs_logvar)
                #print(f"rs_std: min={rs_std.min()}, max={rs_std.max()}, has_nan={torch.isnan(rs_std).any()}")

                rs_non_zero_ll = rs_non_zero_mask * (
                    torch.log(1 - rs_p_zero + 1e-8) - 
                    0.5 * torch.log(2 * torch.pi * torch.exp(rs_logvar) + 1e-8) -
                    0.5 * (rs_true - rs_mean) ** 2 / (torch.exp(rs_logvar) + 1e-8)
                )

                #print(f"rs_non_zero_ll: min={rs_non_zero_ll.min()}, max={rs_non_zero_ll.max()}, has_nan={torch.isnan(rs_non_zero_ll).any()}")
                if not torch.isfinite(rs_non_zero_ll).all():
                    raise ValueError(f"rs_non_zero_ll contains non-finite values!")

                rs_loss = -(rs_zero_ll + rs_non_zero_ll).mean()
                #print(f"rs_loss: {rs_loss}, has_nan={torch.isnan(rs_loss).any()}")
                if not torch.isfinite(rs_loss):
                    raise ValueError(f"rs_loss is not finite: {rs_loss}")
                
                # --- DSM: Zero-Inflated Gamma ---
                dsm_logit_p = reconstructed_kappa['dsm']['logit_p']
                dsm_log_shape = reconstructed_kappa['dsm']['log_shape']
                dsm_log_scale = reconstructed_kappa['dsm']['log_scale']

                #print(f"dsm_logit_p: min={dsm_logit_p.min()}, max={dsm_logit_p.max()}, has_nan={torch.isnan(dsm_logit_p).any()}")
                #print(f"dsm_log_shape: min={dsm_log_shape.min()}, max={dsm_log_shape.max()}, has_nan={torch.isnan(dsm_log_shape).any()}")
                #print(f"dsm_log_scale: min={dsm_log_scale.min()}, max={dsm_log_scale.max()}, has_nan={torch.isnan(dsm_log_scale).any()}")

                
                dsm_p_zero = torch.sigmoid(dsm_logit_p)
                dsm_shape = torch.exp(dsm_log_shape)
                dsm_scale = torch.exp(dsm_log_scale)

                #print(f"dsm_p_zero: min={dsm_p_zero.min()}, max={dsm_p_zero.max()}, has_nan={torch.isnan(dsm_p_zero).any()}")
                #print(f"dsm_shape: min={dsm_shape.min()}, max={dsm_shape.max()}, has_nan={torch.isnan(dsm_shape).any()}")
                #print(f"dsm_scale: min={dsm_scale.min()}, max={dsm_scale.max()}, has_nan={torch.isnan(dsm_scale).any()}")

                
                dsm_zero_mask = (dsm_true == 0).float()
                dsm_non_zero_mask = (dsm_true > 0).float()
                
                # Log-likelihood for zeros
                dsm_zero_ll = dsm_zero_mask * torch.log(dsm_p_zero + 1e-8)

                #print(f"dsm_zero_ll: min={dsm_zero_ll.min()}, max={dsm_zero_ll.max()}, has_nan={torch.isnan(dsm_zero_ll).any()}")
                if not torch.isfinite(dsm_zero_ll).all():
                    raise ValueError(f"dsm_zero_ll contains non-finite values!")
            
                            
                # Log-likelihood for non-zeros (Gamma)
                dsm_lgamma = torch.lgamma(dsm_shape)
                #print(f"dsm_lgamma: min={dsm_lgamma.min()}, max={dsm_lgamma.max()}, has_nan={torch.isnan(dsm_lgamma).any()}")
                if not torch.isfinite(dsm_lgamma).all():
                    print(f"dsm_shape values causing lgamma issues: {dsm_shape[~torch.isfinite(dsm_lgamma)]}")
                    raise ValueError(f"dsm_lgamma contains non-finite values!")
                
                dsm_non_zero_ll = dsm_non_zero_mask * (
                    torch.log(1 - dsm_p_zero + 1e-8) -
                    dsm_lgamma -
                    dsm_shape * torch.log(dsm_scale + 1e-8) +
                    (dsm_shape - 1) * torch.log(dsm_true + 1e-8) -
                    dsm_true / (dsm_scale + 1e-8)
                )
                #t1 = torch.log(1 - dsm_p_zero + 1e-8)
                #t2 = dsm_lgamma
                #t3 = dsm_shape * torch.log(dsm_scale + 1e-8)
                #t4 = (dsm_shape - 1) * torch.log(dsm_true + 1e-8)
                #t5 = dsm_true / (dsm_scale + 1e-8)
                #assert (dsm_true >= 0).all(), f"Negative targets: min={dsm_true.min()}"

                #for name, t in [("log(1-p_zero)", t1), ("lgamma", t2), ("shape*log(scale)", t3),
                #                ("(shape-1)*log(true)", t4), ("true/scale", t5)]:
                #    print(f"{name}: has_nan={t.isnan().any()}, min={t.min():.4f}, max={t.max():.4f}")

                #print(f"dsm_non_zero_ll: min={dsm_non_zero_ll.min()}, max={dsm_non_zero_ll.max()}, has_nan={torch.isnan(dsm_non_zero_ll).any()}")
                if not torch.isfinite(dsm_non_zero_ll).all():
                    raise ValueError(f"dsm_non_zero_ll contains non-finite values!")
                
                dsm_loss = -(dsm_zero_ll + dsm_non_zero_ll).mean()
                #print(f"dsm_loss: {dsm_loss}, has_nan={torch.isnan(dsm_loss).any()}")
                if not torch.isfinite(dsm_loss):
                    raise ValueError(f"dsm_loss is not finite: {dsm_loss}")

                
                # --- GIU: Zero-Inflated Log-Normal ---
                giu_logit_p = reconstructed_kappa['giu']['logit_p']
                giu_logvar = reconstructed_kappa['giu']['logvar']

                #print(f"giu_logit_p: min={giu_logit_p.min()}, max={giu_logit_p.max()}, has_nan={torch.isnan(giu_logit_p).any()}")
                #print(f"giu_logvar: min={giu_logvar.min()}, max={giu_logvar.max()}, has_nan={torch.isnan(giu_logvar).any()}")

                
                giu_p_zero = torch.sigmoid(giu_logit_p)
                #print(f"giu_p_zero: min={giu_p_zero.min()}, max={giu_p_zero.max()}, has_nan={torch.isnan(giu_p_zero).any()}")

                giu_zero_mask = (giu_true == 0).float()
                giu_non_zero_mask = (giu_true > 0).float()
                
                # Log-likelihood for zeros
                giu_zero_ll = giu_zero_mask * torch.log(giu_p_zero + 1e-8)

                #print(f"giu_zero_ll: min={giu_zero_ll.min()}, max={giu_zero_ll.max()}, has_nan={torch.isnan(giu_zero_ll).any()}")
                if not torch.isfinite(giu_zero_ll).all():
                    raise ValueError(f"giu_zero_ll contains non-finite values!")

                
                # Log-likelihood for non-zeros (Log-Normal)
                # Log-Normal PDF: (1/(x*σ*√(2π))) * exp(-(log(x)-μ)²/(2σ²))
                # Log PDF: -log(x) - 0.5*log(2π) - 0.5*logvar - (log(x)-μ)²/(2*exp(logvar))
                giu_non_zero_ll = giu_non_zero_mask * (
                    torch.log(1 - giu_p_zero + 1e-8) -
                    torch.log(giu_true + 1e-8) -
                    0.5 * torch.log(2 * torch.pi * torch.exp(giu_logvar) + 1e-8) -
                    0.5 * (torch.log(giu_true + 1e-8) - giu_mean) ** 2 / (torch.exp(giu_logvar) + 1e-8)
                )

                #print(f"giu_non_zero_ll: min={giu_non_zero_ll.min()}, max={giu_non_zero_ll.max()}, has_nan={torch.isnan(giu_non_zero_ll).any()}")
                if not torch.isfinite(giu_non_zero_ll).all():
                    raise ValueError(f"giu_non_zero_ll contains non-finite values!")

                
                giu_loss = -(giu_zero_ll + giu_non_zero_ll).mean()

                #print(f"giu_loss: {giu_loss}, has_nan={torch.isnan(giu_loss).any()}")
                if not torch.isfinite(giu_loss):
                    raise ValueError(f"giu_loss is not finite: {giu_loss}")

                
                # Total loss
                recon_loss = rs_loss + dsm_loss + giu_loss

                #print(f"recon_loss: {recon_loss}, has_nan={torch.isnan(recon_loss).any()}")
                if not torch.isfinite(recon_loss):
                    raise ValueError(f"recon_loss is not finite: {recon_loss}")

                
            else:
                # Fallback: MSE loss
                recon_loss = F.mse_loss(reconstructed_mean, original_treatments, reduction='mean')
                
            # reconstructed_logvar = reconstructed_kappa
            # # Gaussian negative log-likelihood
            # # NLL = 0.5 * (log(2π) + logvar + (x - μ)² / exp(logvar))
            # # Simplified: 0.5 * (logvar + (x - μ)² / exp(logvar)) + constant
            # if reconstructed_logvar is not None:
            #     recon_loss = 0.5 * (
            #         reconstructed_logvar + 
            #         (original_treatments - reconstructed_mean) ** 2 / torch.exp(reconstructed_logvar)
            #     ).mean()
            # else:
            #     recon_loss = F.mse_loss(reconstructed_mean, original_treatments, reduction='mean')
        
        # KL divergence with spatial smoothness prior
        if self.prior_type == 'laplacian_gmrf':
            kl_loss = self._kldiv_laplacian(mu, logvar)
        else:
            kl_loss = self._kldiv_gaussian(mu, logvar)
        
        # Total C-VAE loss
        loss = recon_loss + self.beta * kl_loss
        
        return loss, recon_loss, kl_loss
    
    def _kldiv_gaussian(self, mu, logvar):
        """
        Standard Gaussian KL divergence: KL(N(mu, sigma) || N(0, I))
        
        Args:
            mu: (B, latent_d, latent_h, latent_w)
            logvar: (B, latent_d, latent_h, latent_w)
        
        Returns:
            kl_loss: scalar
        """
        # Flatten spatial dimensions
        B, D, H, W = mu.shape
        mu_flat = mu.view(B, D, -1)  # (B, latent_d, n_pixels)
        logvar_flat = logvar.view(B, D, -1)
        
        # KL = 0.5 * sum(-1 - logvar + mu^2 + exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar_flat - mu_flat.pow(2) - logvar_flat.exp(), dim=[1, 2])
        return kl.mean()
    
    def _kldiv_laplacian(self, mu, logvar):
        """
        Spatial smoothness KL divergence using Laplacian matrix.
        KL with GMRF prior: tau/2 * E[Z^T L Z]
        
        Args:
            mu: (B, latent_d, latent_h, latent_w)
            logvar: (B, latent_d, latent_h, latent_w)
        
        Returns:
            kl_loss: scalar
        """
        B, D, H, W = mu.shape
        
        # Reshape: (B, latent_d, n_pixels)
        mu_ = mu.view(B, D, -1)
        logvar_ = logvar.view(B, D, -1)
        
        # Move Laplacian to same device as mu
        L = self.laplacian.to(mu.device)
        
        # Quadratic term: mu^T L mu for each batch and latent dimension
        # L: (n_pixels, n_pixels)
        # mu_: (B, latent_d, n_pixels)
        # We compute: for each (b,d): sum_ij mu[b,d,i] * L[i,j] * mu[b,d,j]
        # This is: mu[b,d] . (L @ mu[b,d])
        
        # L @ mu^T: (n_pixels, n_pixels) @ (B, n_pixels, latent_d) = (B, n_pixels, latent_d)
        L_mu = torch.matmul(L.unsqueeze(0), mu_.transpose(1, 2))  # (B, n_pixels, latent_d)
        L_mu = L_mu.transpose(1, 2)  # (B, latent_d, n_pixels)
        
        # Element-wise product and sum over pixels
        quadratic_term = torch.sum(mu_ * L_mu, dim=2)  # (B, latent_d)
        
        # Trace term: tr(L * diag(exp(logvar))) = sum(diag(L) * exp(logvar))
        L_diag = torch.diag(L)  # (n_pixels,)
        trace_term = torch.sum(L_diag.unsqueeze(0).unsqueeze(0) * torch.exp(logvar_), dim=2)  # (B, latent_d)
        
        # KL divergence: tau/2 * sum(quadratic_term + trace_term)
        kl_loss = (self.tau / 2) * torch.sum(quadratic_term + trace_term, dim=1)  # (B,)
        
        # Normalize by number of pixels and latent dims
        kl_loss = kl_loss / (self.n_pixels * D)
        
        return kl_loss.mean()
    
    def sample_z(self, mu, logvar, num_samples=1):
        """
        Sample from posterior distribution q(Z|A,X)
        Args:
            mu: (B, latent_size)
            logvar: (B, latent_size)
            num_samples: number of samples per batch element
        
        Returns:
            z_samples: (B*num_samples, latent_size)
        """
        batch_size = mu.size(0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).repeat(num_samples, 1, 1)
        
        z_samples = mu.unsqueeze(1) + eps * std.unsqueeze(1)  # (B, num_samples, latent_size)
        z_samples = z_samples.view(batch_size * num_samples, -1)
        
        return z_samples


class PosteriorPredictiveCheck:
    """
    Implements posterior predictive checks as described in Rubin (1984).
    Assesses whether the learned C-VAE adequately explains treatment assignment.
    """
    
    def __init__(self, discrepancy_type='marginal_loglikelihood'):
        self.discrepancy_type = discrepancy_type
    
    def compute_discrepancy(self, treatment_means, treatments, treatment_logvar=None, sigma=1.0, kappa=None):
        """
        Compute discrepancy statistic T(a) for continuous treatments.
        
        Args:
            treatment_means: (B, C, H, W) - mean predictions from decoder
            treatments: (B, C, H, W) - observed continuous treatments (unit vectors per pixel)
            treatment_logvar: (B, C, H, W) - log variance (optional, uses sigma if not provided)
            sigma: scalar - standard deviation for Gaussian likelihood (default 1.0)
            kappa: (B, 1, H, W) - concentration parameter for vMF (optional)
        
        Returns:
            discrepancy: scalar or (B,) tensor depending on discrepancy type
        """
        if self.discrepancy_type == 'marginal_loglikelihood':
            # # Gaussian log-likelihood for continuous unbounded data
            # # log p(a) = -0.5 * ||a - mu||^2 / sigma^2 - 0.5 * log(2*pi*sigma^2)
            # if treatment_logvar is not None:
            #     var = torch.exp(treatment_logvar)
            #     log_probs = -0.5 * ((treatments - treatment_means) ** 2 / var) - 0.5 * torch.log(2 * np.pi * var)
            # else:
            #     log_probs = -0.5 * ((treatments - treatment_means) ** 2 / (sigma ** 2)) - 0.5 * np.log(2 * np.pi * sigma ** 2)
            # discrepancy = log_probs.mean()
            # Zero-Inflated log-likelihood for bounded data with excess zeros
            # Split treatments by channel and compute appropriate log-likelihood for each
            if treatment_logvar is not None:
                eps = 1e-8
                
                # Split treatments into separate channels: [RS, DSM, GIU]
                treatments_rs, treatments_dsm, treatments_giu = torch.chunk(treatments, 3, dim=1)
                treatment_means_rs, treatment_means_dsm, treatment_means_giu = torch.chunk(treatment_means, 3, dim=1)
                
                # --- RS: Zero-Inflated Normal ---
                rs_logit_p = treatment_logvar['rs']['logit_p']
                rs_logvar = treatment_logvar['rs']['logvar']
                
                rs_p_zero = torch.sigmoid(rs_logit_p).clamp(eps, 1-eps)
                rs_var = torch.exp(rs_logvar).clamp(min=eps)
                
                rs_zero_mask = (treatments_rs == 0).float()
                rs_non_zero_mask = (treatments_rs > 0).float()
                
                rs_zero_log_prob = rs_zero_mask * torch.log(rs_p_zero)
                rs_non_zero_log_prob = rs_non_zero_mask * (
                    torch.log(1 - rs_p_zero) -
                    0.5 * ((treatments_rs - treatment_means_rs) ** 2 / rs_var) -
                    0.5 * torch.log(2 * np.pi * rs_var)
                )
                rs_log_probs = rs_zero_log_prob + rs_non_zero_log_prob
                
                # --- DSM: Zero-Inflated Gamma ---
                dsm_logit_p = treatment_logvar['dsm']['logit_p']
                dsm_log_shape = treatment_logvar['dsm']['log_shape']
                dsm_log_scale = treatment_logvar['dsm']['log_scale']
                
                dsm_p_zero = torch.sigmoid(dsm_logit_p).clamp(eps, 1-eps)
                dsm_shape = torch.exp(dsm_log_shape).clamp(eps, 100)
                dsm_scale = torch.exp(dsm_log_scale).clamp(eps, 1000)
                
                dsm_zero_mask = (treatments_dsm == 0).float()
                dsm_non_zero_mask = (treatments_dsm > 0).float()
                
                dsm_zero_log_prob = dsm_zero_mask * torch.log(dsm_p_zero)
                dsm_non_zero_log_prob = dsm_non_zero_mask * (
                    torch.log(1 - dsm_p_zero) -
                    torch.lgamma(dsm_shape) -
                    dsm_shape * torch.log(dsm_scale) +
                    (dsm_shape - 1) * torch.log(treatments_dsm.clamp(min=eps)) -
                    treatments_dsm / dsm_scale
                )
                dsm_log_probs = dsm_zero_log_prob + dsm_non_zero_log_prob
                
                # --- GIU: Zero-Inflated Log-Normal ---
                giu_logit_p = treatment_logvar['giu']['logit_p']
                giu_logvar = treatment_logvar['giu']['logvar']
                
                giu_p_zero = torch.sigmoid(giu_logit_p).clamp(eps, 1-eps)
                giu_var = torch.exp(giu_logvar).clamp(min=eps)
                
                giu_zero_mask = (treatments_giu == 0).float()
                giu_non_zero_mask = (treatments_giu > 0).float()
                
                giu_zero_log_prob = giu_zero_mask * torch.log(giu_p_zero)
                giu_log_x = torch.log(treatments_giu.clamp(min=eps))
                giu_non_zero_log_prob = giu_non_zero_mask * (
                    torch.log(1 - giu_p_zero) -
                    giu_log_x -
                    0.5 * torch.log(2 * np.pi * giu_var) -
                    0.5 * ((giu_log_x - treatment_means_giu) ** 2 / giu_var)
                )
                giu_log_probs = giu_zero_log_prob + giu_non_zero_log_prob
                
                # Combine all log probabilities
                log_probs = torch.cat([rs_log_probs, dsm_log_probs, giu_log_probs], dim=1)
                
            else:
                log_probs = -0.5 * ((treatments - treatment_means) ** 2 / (sigma ** 2)) - 0.5 * np.log(2 * np.pi * sigma ** 2)
            
            discrepancy = log_probs.mean()

            return discrepancy
        
        elif self.discrepancy_type == 'vmf_loglikelihood':
            # von Mises-Fisher log-likelihood for directional data
            # log p(a|mu, kappa) = log C(kappa) + kappa * (mu · a)
            # where C(kappa) is the normalization constant
            
            # # Ensure unit norm
            # treatment_means = F.normalize(treatment_means, dim=1, eps=1e-8)
            # treatments = F.normalize(treatments, dim=1, eps=1e-8)
            
            # Dot product (cosine similarity)
            dim = treatments.size(1)
            dot = (treatment_means * treatments).sum(dim=1, keepdim=True)
            
            # Clamp kappa and compute log-normalizer approximation
            if kappa is None:
                b = 0.05
                discrepancy = -b * dot.mean()
                return discrepancy
            kappa = torch.clamp(kappa, min=1e-4, max=100.0)
            logC_approx = 0.5 * (dim - 1) * torch.log(2 * torch.pi / kappa)
            
            # vMF log-likelihood: log C(κ) + κ * (μ · a)
            log_probs = -logC_approx + kappa * dot
            discrepancy = log_probs.mean()
            return discrepancy
        
        elif self.discrepancy_type == 'mse':
            # Mean squared error between observed and predicted
            discrepancy = F.mse_loss(treatment_means, treatments, reduction='mean')
            return discrepancy
        
        else:
            raise ValueError(f"Unknown discrepancy type: {self.discrepancy_type}")
    
    def posterior_predictive_check(
        self,
        encoder,
        decoder,
        treatments,
        covariates,
        treatments_raw,
        num_samples=1,
        num_mc_samples=1,
        alpha=0.05,
    ):
        """
        Posterior predictive check for directional treatments using
        vMF-style test statistic (no Gaussian likelihood).
        """

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            B = treatments.size(0)

            # Ensure unit-norm treatments
            #treatments = F.normalize(treatments, dim=1, eps=1e-8)

            # Encoder posterior
            mu, logvar = encoder(treatments, covariates)

            # Get the middle fifth of the spatial dimensions
            _, _, H, W = treatments_raw.shape

            # Calculate the middle fifth boundaries
            h_start = H * 2 // 5  # Start at 40% (2/5)
            h_end = H * 3 // 5    # End at 60% (3/5)
            w_start = W * 2 // 5
            w_end = W * 3 // 5
            h_end -= 1 if H != 50 else 0
            w_end -= 1 if W != 50 else 0

            # Extract middle fifth
            treatments = treatments_raw[:, :, h_start:h_end, w_start:w_end]

            # Expand for latent sampling
            mu_exp = mu.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
            logvar_exp = logvar.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)

            # Sample latent Z
            z_samples = reparameterize(mu_exp, logvar_exp)
            z_flat = z_samples.reshape(num_samples * B, *z_samples.shape[2:])

            # Repeat covariates
            cov_exp = covariates.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
            cov_flat = cov_exp.reshape(num_samples * B, *covariates.shape[1:])

            # Decoder outputs: mean direction + concentration
            a_mean, a_kappa = decoder(z_flat, cov_flat, normalize_output=True)
            a_mean = a_mean.reshape(num_samples, B, *a_mean.shape[1:])
            if a_kappa is not None:
                # Reshape each parameter in the nested dictionary
                a_kappa_reshaped = {}
                for dataset_name in ['rs', 'dsm', 'giu']:
                    a_kappa_reshaped[dataset_name] = {}
                    for param_name, param_tensor in a_kappa[dataset_name].items():
                        a_kappa_reshaped[dataset_name][param_name] = param_tensor.reshape(
                            num_samples, B, *param_tensor.shape[1:]
                        )
                a_kappa = a_kappa_reshaped
                # a_kappa = a_kappa.reshape(num_samples, B, *a_kappa.shape[1:])

            # Normalize decoder mean directions
            # a_mean = F.normalize(a_mean, dim=2, eps=1e-8)
            # a_kappa = torch.clamp(a_kappa, min=1e-4, max=100.0)

            # Expand true treatments
            a_true = treatments.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)

            # ---------- Observed test statistic ----------
            # Use compute_discrepancy for consistency
            test_stat_true_list = []
            for s in range(num_samples):
                a_kappa_s = None
                if a_kappa is not None:
                    a_kappa_s = {
                        dataset: {
                            param: tensor[s]
                            for param, tensor in params.items()
                        }
                        for dataset, params in a_kappa.items()
                    }
                disc = self.compute_discrepancy(
                    a_mean[s], a_true[s], treatment_logvar=a_kappa_s, kappa=a_kappa_s
                )
                test_stat_true_list.append(disc)
            test_stat_true = torch.stack(test_stat_true_list).mean()  # [B]

            # ---------- Monte Carlo test statistics ----------
            test_stat_mc_list = []

            for _ in range(num_mc_samples):
                z_gen = reparameterize(mu_exp, logvar_exp)
                z_gen_flat = z_gen.reshape(num_samples * B, *z_gen.shape[2:])

                a_mean_gen, a_kappa_gen = decoder(z_gen_flat, cov_flat, normalize_output=True)
                a_mean_gen = a_mean_gen.reshape(num_samples, B, *a_mean_gen.shape[1:])
                if a_kappa_gen is not None:
                    # Reshape each parameter in the nested dictionary
                    a_kappa_gen_reshaped = {}
                    for dataset_name in ['rs', 'dsm', 'giu']:
                        a_kappa_gen_reshaped[dataset_name] = {}
                        for param_name, param_tensor in a_kappa_gen[dataset_name].items():
                            a_kappa_gen_reshaped[dataset_name][param_name] = param_tensor.reshape(
                                num_samples, B, *param_tensor.shape[1:]
                            )
                    a_kappa_gen = a_kappa_gen_reshaped
                    # a_kappa_gen = a_kappa_gen.reshape(num_samples, B, *a_kappa_gen.shape[1:])

                # a_mean_gen = F.normalize(a_mean_gen, dim=2, eps=1e-8)
                # a_kappa_gen = torch.clamp(a_kappa_gen, min=1e-4, max=100.0)

                # Compute discrepancy for each sample
                disc_list = []
                for s in range(num_samples):
                    a_kappa_gen_s = None
                    if a_kappa_gen is not None:
                        a_kappa_gen_s = {
                            dataset: {
                                param: tensor[s]
                                for param, tensor in params.items()
                            }
                            for dataset, params in a_kappa.items()
                        }
                    disc = self.compute_discrepancy(
                        a_mean_gen[s], a_true[s], treatment_logvar=a_kappa_gen_s, kappa=a_kappa_gen_s
                    )
                    disc_list.append(disc)
                test_stat_mc = torch.stack(disc_list).mean()
                test_stat_mc_list.append(test_stat_mc)

            test_stat_mc_array = torch.stack(test_stat_mc_list)  # (num_mc_samples,)

            # ---------- Bayesian p-values ----------
            p_values = (test_stat_mc_array < test_stat_true.unsqueeze(0)).float().mean(dim=0)
            is_valid = (p_values > 0.25) & (p_values < 0.75)

        return {
            "p_value": p_values.cpu().numpy(),
            "mean_p_value": p_values.mean().item(),
            "is_valid": is_valid.cpu().numpy(),
            "test_stat_true": test_stat_true.cpu().numpy(),
            "test_stat_mc": test_stat_mc_array.cpu().numpy(),
            "num_replications": num_mc_samples,
            "num_samples": num_samples,
        }



def kl_warmup_schedule(current_epoch, total_epochs, warmup_epochs):
    """
    Compute KL weight with linear warmup schedule.
    
    Args:
        current_epoch: current training epoch
        total_epochs: total number of epochs
        warmup_epochs: number of epochs for warmup
    
    Returns:
        beta: KL weight (increases from 0 to 1 during warmup)
    """
    if current_epoch < warmup_epochs:
        beta = current_epoch / warmup_epochs
    else:
        beta = 1.0
    return beta


def reparameterize(mu, logvar):
    """
    Reparameterization trick for sampling from N(mu, exp(logvar))
    
    Args:
        mu: (B, latent_size)
        logvar: (B, latent_size)
    
    Returns:
        z: (B, latent_size) - sample from q
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z


def get_treatment_and_covariates(batch, args, idx):
    """
    Extract treatments and covariates from batch based on configuration.
    
    Args:
        batch: tuple of batch tensors from collate_fn
        args: arguments containing treatment/covariate source specifications
    
    Returns:
        treatments: tensor representing treatments
        covariates: tensor representing covariates
    """
    # batch structure from collate_fn:
    # [input_ids, attention_mask, input_modal, modal_start_tokens, modal_end_tokens, labels]
    
    input_modal = batch[idx]  # (B, channels, H, W) - can be 9-channel or 9+64-channel
    # RAW
    tensor_first = input_modal[:, :9, :, :]
    tensor_ae = input_modal[:, 9:, :50, :50]
    
    if args.cvae_treatment_source == 'alphaearth':
        # First 3 channels from alphaearth (e.g., RS data)
        treatments = tensor_ae
    else:  # 'image'
        # First 3 channels from task-specific images
        treatments = tensor_first
    
    if args.cvae_covariate_source == 'alphaearth':
        # AlphaEarth data as covariates
        covariates = tensor_ae
    else:  # 'image'
        # Task-specific images as covariates
        covariates = tensor_first

    return treatments, covariates
