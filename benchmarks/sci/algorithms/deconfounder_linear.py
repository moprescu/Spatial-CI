import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from .utils import UNet, get_k_hop_neighbors, DoubleConvMultiChannel
import networkx as nx

from sci import SpaceDataset
from spacebench.algorithms import SpaceAlgo
from spacebench.log import LOGGER
import os
from .spatialplus import SpatialPlus
# from .pysal_spreg import GMLag
from copy import deepcopy

total_batch_size = 64

class CVAE_Grid(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        radius: float,
        encoder_conv1: int = 32,
        encoder_conv2: int = 64,
        kernel_size: int = 3,
        encoder_pool: str = "avg",
        connectivity: int = 4,
        tau: float = 10.0,
        eps: float = 1e-5,
    ):
        """
        CVAE for interference-aware deconfounding on gridded datasets
        
        Args:
            latent_dim: Dimension of latent space Z_s
            radius: Radius for patch-based convolution
            tau: Precision parameter for GMRF prior
            eps: Small constant for numerical stability
        """
        super(CVAE_Grid, self).__init__()
        
        self.feature_dim = feature_dim
        self.radius = radius
        self.tau = tau
        self.eps = eps
        self.patch_size = 2 * radius + 1
        self.kernel_size = kernel_size
        # self.per_location_latent_dim = latent_dim
        self.per_location_latent_dim = encoder_conv2 // 2
        self.latent_dim = self.patch_size * self.patch_size * self.per_location_latent_dim
        self.connectivity = connectivity
        
        # Calculate input channels: A_s + A_{N_s} (treatments + neighbor treatments)
        # Patch of treatments size self.patch_size x self.patch_size
        self.encoder_input_dim = 1 + self.feature_dim
        
        # Encoder: Two 3x3 convolutions over radius-r patch
        self.enc_mid_chan = encoder_conv1
        self.enc_out_chan = encoder_conv2
        self.encoder_conv = DoubleConvMultiChannel(self.encoder_input_dim, self.enc_mid_chan, self.enc_out_chan, radius=self.radius)
        self.encoder_pool = encoder_pool
        
        # Encoder output processing including covariates
        self.encoder_flatten_dim = self.enc_out_chan
        self.fc_mu = nn.Linear(self.encoder_flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_flatten_dim, self.latent_dim)
        
        # Decoder: One-layer MLP p_ψ(A_s=1|x_s,Z_s) = σ(f_ψ(x_s,Z_s))
        self.decoder = nn.Linear( self.patch_size * self.patch_size * self.feature_dim + self.latent_dim, 1)
        
        # Precompute grid Laplacian matrix L for GMRF prior
        self.register_buffer('laplacian', self._build_patch_laplacian())
    
    
    def _build_patch_laplacian(self):
        """
        Build patch Laplacian matrix L for 2D grid
        """
        n = self.patch_size ** 2
        L = torch.zeros(n, n)
        
        def coord_to_idx(i, j):
            return i * self.patch_size + j
        
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                idx = coord_to_idx(i, j)
                degree = 0
                
                # Add edges to neighbors
                if self.connectivity == 4:
                    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                else:
                    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                for di, dj in nbrs:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.patch_size and 0 <= nj < self.patch_size:
                        neighbor_idx = coord_to_idx(ni, nj)
                        L[idx, neighbor_idx] = -1
                        degree += 1
                
                L[idx, idx] = degree
        
        return L    
    
    def encode(self, A, x):
        """
        Encoder: q_φ(Z_s|A_s, A_{N_s}, x_s)
        Two 3×3 convolutions over radius-r patch
        
        Args:
            A: Current and Neighbor treatments [batch_size, patch_size, patch_size, 1]
            x: Current and Neighbor features [batch_size, patch_size, patch_size, feature_dim]
        
        Returns:
            mu, logvar: Parameters of latent distribution
        """
        # Rearrange for convolution: [batch, channels, height, width]
        encoder_input = torch.cat([A, x], dim=3).permute(0, 3, 1, 2)
                
        # Two 3×3 convolutions
        h = self.encoder_conv(encoder_input)
        
        if self.encoder_pool == "max":
            h = F.adaptive_max_pool2d(h, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.encoder_pool == "avg":
            # Global average pooling to get fixed-size representation
            h = F.adaptive_avg_pool2d(h, (1, 1)).squeeze(-1).squeeze(-1)
                
        # # Output latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # mu = h[:, 0:self.enc_out_chan//2].flatten(start_dim=1)
        # logvar =h[:, self.enc_out_chan//2:].flatten(start_dim=1)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, x_s, z_s):
        """
        Decoder: p_ψ(A_s=1|x_s, Z_s) = σ(f_ψ(x_s, Z_s))
        One-layer MLP
        
        Args:
            x_s: Features at location [batch_size, patch_size *  patch_size *  feature_dim]
            z_s: Latent variables [batch_size, patch_size *  patch_size *  latent_dim]
        
        Returns:
            Treatment probability at location s
        """
        # Concatenate features and latent variables
        combined = torch.cat([x_s, z_s], dim=-1)
        
        # One-layer MLP with sigmoid activation
        logits = self.decoder(combined)
        probs = torch.sigmoid(logits)
        
        return probs.squeeze(-1)
    
    def forward(self, treatments, covariates):
        """
        Forward pass through interference-aware CVAE
        
        Args:
            treatments: Current and neighboring treatments [batch_size, patch_size, patch_size, 1]
            covariates: Features at location [batch_size, patch_size, patch_size, feature_dim]
        
        Returns:
            treatment_probs: Treatment probabilities
            z_s: Latent distribution
            out: Outcome prediction
        """
        # Encode
        mu, logvar = self.encode(treatments, covariates)
        
        # Reparameterize
        z_s = self.reparameterize(mu, logvar)
        
        # Decode
        treatment_probs = self.decode(covariates.reshape(covariates.size(0), -1), z_s)
        
        return treatment_probs, mu, logvar
    
    def get_latent(self, treatments, covariates):
        """
        Return latent values interference-aware CVAE
        
        Args:
            treatments: Current and neighboring treatments [batch_size, patch_size, patch_size, 1]
            covariates: Features at location [batch_size, patch_size, patch_size, feature_dim]
        
        Returns:
            treatment_probs: Treatment probabilities
            z_s: Latent distribution
            out: Outcome prediction
        """
        # Encode
        mu, logvar = self.encode(treatments, covariates)
        
        # Reparameterize
        z_s = self.reparameterize(mu, logvar)
        
        return z_s.view(-1, self.patch_size, self.patch_size, self.per_location_latent_dim)


class CVAE(pl.LightningModule):
    def __init__(
        self,
        feature_dim: int,
        radius: float,
        encoder_conv1: int = 32,
        encoder_conv2: int = 64,
        kernel_size: int = 3,
        encoder_pool: str = "avg",
        connectivity: int = 4,
        tau: float = 10.0,
        eps: float = 1e-5,
        weight_decay: float = 1e-5,
        lr: float = 1e-3,
        beta_max: float = 1e-1,
        beta_epoch_max: int = 10,
        epochs: int = 10,
        datatype: str = "grid",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.radius = radius
        self.encoder_conv1 = encoder_conv1
        self.encoder_conv2 = encoder_conv2
        self.kernel_size = kernel_size
        self.encoder_pool = encoder_pool
        self.connectivity = connectivity
        self.tau = tau
        self.eps = eps
        self.datatype = datatype
        
        if self.datatype == "grid":
            self.model = CVAE_Grid(
                feature_dim=self.feature_dim,
                radius=self.radius,
                encoder_conv1=self.encoder_conv1,
                encoder_conv2=self.encoder_conv2,
                kernel_size=self.kernel_size,
                encoder_pool=self.encoder_pool,
                connectivity=self.connectivity,
                tau=self.tau,
                eps=self.eps,
            )
        else:
            raise ValueError(f"Unsupported dataset type: {datatype}")
        
        # self.model = torch.compile(self.model, mode="reduce-overhead")
        
        self.weight_decay = weight_decay
        self.lr = lr
        self.beta_max = beta_max
        self.beta_epoch_max = beta_epoch_max
        self.epochs = epochs
        self.num_samples = 100
        self.n_pixels = self.model.patch_size * self.model.patch_size
        self.current_val_p_value = None
        
    def predict_step(self, batch, batch_idx):
        treatments, covariates, true_treatments = batch
        z_s = self.model.get_latent(treatments, covariates)
        return z_s
        
    def training_step(self, batch, batch_idx):
        treatments, covariates, true_treatments = batch
        treatment_probs, mu, logvar = self.model(treatments, covariates)
        
        treatment_loss, kldiv_loss = self.loss(treatment_probs, mu, logvar, true_treatments)
        # Combined objective
        if self.beta_epoch_max == 0:
            beta = self.beta_max
        else:
            beta = min(self.beta_max, self.current_epoch * (self.beta_max / self.beta_epoch_max))
        loss = treatment_loss + beta * kldiv_loss
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_treatment_loss", treatment_loss, on_epoch=True, prog_bar=True)
        self.log("train_kldiv_loss", kldiv_loss, on_epoch=True, prog_bar=True)
        self.log("beta", beta, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        treatments, covariates, true_treatments = batch
        treatment_probs, mu, logvar = self.model(treatments, covariates)
        
        treatment_loss, kldiv_loss = self.loss(treatment_probs, mu, logvar, true_treatments)
        # Combined objective
        if self.beta_epoch_max == 0:
            beta = self.beta_max
        else:
            beta = min(self.beta_max, self.current_epoch * (self.beta_max / self.beta_epoch_max))
        loss = treatment_loss + beta * kldiv_loss
        max_loss = treatment_loss + self.beta_max * kldiv_loss
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_max_loss", max_loss, on_epoch=True, prog_bar=True)
        self.log("val_treatment_loss", treatment_loss, on_epoch=True, prog_bar=True)
        self.log("val_kldiv_loss", kldiv_loss, on_epoch=True, prog_bar=True)
        
        # Predictive checks implementation - vectorized
        batch_size = covariates.size(0)

        # Compute test statistic for true treatments: T(a) = E_Z[log p(a|X,Z)]
        # Sample all z_s at once: [num_samples, batch_size, z_dim]
        z_s = self.model.reparameterize(
            mu.unsqueeze(0).expand(self.num_samples, -1, -1),
            logvar.unsqueeze(0).expand(self.num_samples, -1, -1)
        ).view(-1, mu.size(-1))  # [num_samples * batch_size, z_dim]

        # Expand covariates to match: [num_samples * batch_size, cov_dim]
        covariates_expanded = covariates.reshape(batch_size, -1).unsqueeze(0).expand(
            self.num_samples, -1, -1
        ).reshape(-1, covariates.reshape(batch_size, -1).size(-1))

        # Get logits for all samples at once
        a_logits = self.model.decode(covariates_expanded, z_s)  # [num_samples * batch_size, treatment_dim]

        # Reshape back and expand true_treatments
        a_logits = a_logits.view(self.num_samples, batch_size, -1)  # [num_samples, batch_size, treatment_dim]
        true_treatments_expanded = true_treatments.unsqueeze(-1).unsqueeze(0).expand(self.num_samples, -1, -1)

        # Compute log probabilities for all samples
        log_probs = -F.binary_cross_entropy_with_logits(
            a_logits, true_treatments_expanded, reduction='none'
        ).sum(dim=-1)  # [num_samples, batch_size]

        test_stat_true = log_probs.mean(dim=0)  # T(a_true) for each sample in batch

        # Sample M treatment vectors and compute their test statistics
        # Sample z for MC sampling: [num_samples, batch_size, z_dim]
        z_s_mc = self.model.reparameterize(
            mu.unsqueeze(0).expand(self.num_samples, -1, -1),
            logvar.unsqueeze(0).expand(self.num_samples, -1, -1)
        ).view(-1, mu.size(-1))  # [num_samples * batch_size, z_dim]

        # Get MC logits and sample treatments
        a_logits_mc = self.model.decode(covariates_expanded, z_s_mc)
        a_logits_mc = a_logits_mc.view(self.num_samples, batch_size, -1)
        a_mc = torch.bernoulli(torch.sigmoid(a_logits_mc))  # [num_samples, batch_size, treatment_dim]

        # For each MC sample, compute test statistic using vectorized inner sampling
        # Sample z for inner loop: [num_samples, num_samples, batch_size, z_dim]
        z_s_inner = self.model.reparameterize(
            mu.unsqueeze(0).unsqueeze(0).expand(self.num_samples, self.num_samples, -1, -1),
            logvar.unsqueeze(0).unsqueeze(0).expand(self.num_samples, self.num_samples, -1, -1)
        ).view(-1, mu.size(-1))  # [num_samples^2 * batch_size, z_dim]

        # Expand covariates for inner sampling
        covariates_inner = covariates.reshape(batch_size, -1).unsqueeze(0).unsqueeze(0).expand(
            self.num_samples, self.num_samples, -1, -1
        ).reshape(-1, covariates.reshape(batch_size, -1).size(-1))

        # Get inner logits
        a_logits_inner = self.model.decode(covariates_inner, z_s_inner)
        a_logits_inner = a_logits_inner.view(self.num_samples, self.num_samples, batch_size, -1)

        # Expand a_mc for broadcasting
        a_mc_expanded = a_mc.unsqueeze(1).expand(-1, self.num_samples, -1, -1)

        # Compute MC log probabilities
        mc_log_probs = -F.binary_cross_entropy_with_logits(
            a_logits_inner, a_mc_expanded, reduction='none'
        ).sum(dim=-1)  # [num_samples, num_samples, batch_size]

        # Average over inner samples to get test statistics
        mc_test_stats = mc_log_probs.mean(dim=1)  # [num_samples, batch_size]

        # Compute predictive p-values
        comparisons = (mc_test_stats < test_stat_true.unsqueeze(0)).float()
        p_values = comparisons.mean(dim=0)  # Average over M samples

        # Log the mean p-value across the batch
        self.log("val_p_value", p_values.mean(), on_epoch=True, prog_bar=True)
        # self.log("val_p_value_diff", torch.abs(0.5-p_values.mean()), on_epoch=True, prog_bar=True)
        # self.log("val_loss_and_p", loss if 0.45 <= p_values.mean() <= 0.55 else 100, on_epoch=True, prog_bar=True)
        self.current_val_p_value = p_values.mean()
        self.current_val_treatment_loss = treatment_loss.detach().cpu().item()
        self.current_val_loss = loss.detach().cpu().item()
        
        return loss
    
    def on_validation_epoch_end(self):
        # average over logged values from the epoch
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_treatment_loss = self.trainer.callback_metrics.get("val_treatment_loss")
        val_p_value = getattr(self, "current_val_p_value", None)

        if val_loss is not None:
            self.current_val_loss = float(val_loss.cpu().item())
        if val_treatment_loss is not None:
            self.current_val_treatment_loss = float(val_treatment_loss.cpu().item())
        if val_p_value is not None:
            self.current_val_p_value = float(val_p_value.cpu().item())
    
    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "current_val_loss"):
            checkpoint["val_loss"] = self.current_val_loss
        if hasattr(self, "current_val_treatment_loss"):
            checkpoint["val_treatment_loss"] = self.current_val_treatment_loss
        if hasattr(self, "current_val_p_value"):
            checkpoint["val_p_value"] = self.current_val_p_value
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
            
    def kldiv(self, mu, logvar):
        mu_ = mu.view(-1, self.model.patch_size * self.model.patch_size, self.model.per_location_latent_dim)
        logvar_ = logvar.view(-1, self.model.patch_size * self.model.patch_size, self.model.per_location_latent_dim)
        
        # Spatial smoothness term: tau/2 * E[Z^T L Z] (vectorized)
        # E[Z^T L Z] = mu^T L mu + tr(L * Sigma) where Sigma = diag(exp(logvar))
        
        laplacian = self.model.laplacian.to(mu.device)
        # Quadratic term: mu^T L mu for all latent dims simultaneously
        # mu_flat: [batch_size, n_pixels, latent_dim]
        # L: [n_pixels, n_pixels]
        mu_L = torch.matmul(mu_.transpose(1, 2), laplacian)  # [batch_size, latent_dim, n_pixels]
        quadratic_term = torch.sum(mu_L * mu_.transpose(1, 2), dim=2)  # [batch_size, latent_dim]
        
        # Trace term: tr(L * diag(exp(logvar))) = sum(diag(L) * exp(logvar))
        L_diag = torch.diag(laplacian)  # [n_pixels]
        trace_term = torch.sum(L_diag.unsqueeze(0).unsqueeze(-1) * torch.exp(logvar_), dim=1)  # [batch_size, latent_dim]
        
        # Sum over latent dimensions and apply tau scaling
        kldiv_loss = self.model.tau / 2 * torch.sum(quadratic_term + trace_term, dim=1)  # [batch_size]
        
        return kldiv_loss.mean() / (self.n_pixels * self.model.per_location_latent_dim)
    
    def loss(self, treatment_probs, mu, logvar, true_treatments):
        # Term 1: -log p_ψ(A_s | x_s, Z_s) - Treatment reconstruction
        treatment_loss = F.binary_cross_entropy(treatment_probs, true_treatments.float())

        # Term 2: β * KL(q_φ || p_θ) - Spatial KL divergence
        kldiv_loss = self.kldiv(mu, logvar)
        
        return treatment_loss, kldiv_loss     

    
class UNetHead(pl.LightningModule):
    def __init__(
        self,
        feature_dim: int,
        radius: float,
        bilinear: bool = False,
        unet_base_chan: int = 16,
        weight_decay: float = 1e-5,
        lr: float = 1e-3,
        epochs: int = 10,
        datatype: str = "grid",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.radius = radius
        self.bilinear = bilinear
        self.unet_base_chan = unet_base_chan
        self.datatype = datatype
        
        if self.datatype != "grid":
            raise ValueError(f"Unsupported dataset type: {datatype}")
        
        self.model = UNet(n_channels=1 + self.feature_dim, n_classes=1, bilinear=self.bilinear, radius=self.radius, base_channels=self.unet_base_chan)
        self.model = torch.compile(self.model, mode="reduce-overhead")
        
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        
    def predict_step(self, batch, batch_idx):
        treatments, features, true_outcomes = batch
        
        treat = treatments.permute(0, 3, 1, 2)
        feat = features.permute(0, 3, 1, 2)
        head_input = torch.cat([treat, feat], dim=1)
        
        pred = self.model(head_input)
        return pred
        
    def training_step(self, batch, batch_idx):
        treatments, features, true_outcomes = batch
        treat = treatments.permute(0, 3, 1, 2)
        feat = features.permute(0, 3, 1, 2)
        head_input = torch.cat([treat, feat], dim=1)
        
        pred = self.model(head_input)
        
        loss = self.loss(pred, true_outcomes)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)        
        return loss
        
    def validation_step(self, batch, batch_idx):
        treatments, features, true_outcomes = batch
        treat = treatments.permute(0, 3, 1, 2)
        feat = features.permute(0, 3, 1, 2)
        head_input = torch.cat([treat, feat], dim=1)
        
        pred = self.model(head_input)
        
        loss = self.loss(pred, true_outcomes)        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
            
    
    def loss(self, pred, true_outcomes):        
        # Term 3: γ * E_q[(Y_s - h_θ(A_s,A_{N_s},x_s,Z_s))²] - Outcome prediction
        outcome_loss = F.mse_loss(pred, true_outcomes)
        return outcome_loss
     
        
def pad_center(tensor, target_size):
    """Pad 4D tensor (B,H,W,C) to target_size x target_size, centered."""
    b, h, w, c = tensor.shape
    pad_h = target_size - h
    pad_w = target_size - w
    assert pad_h >= 0 and pad_w >= 0
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    # for shape (B,H,W,C), need (left,right,top,bottom) padding order
    # F.pad applies padding from the last dimension backwards
    return F.pad(tensor, (0, 0, pad_left, pad_right, pad_top, pad_bottom))


class CVAEDataset(Dataset):
    def __init__(self, dataset, nodes, coords2id, radius, treat_scaler=None, feat_scaler=None, output_scaler=None, a=None, change=None, datatype="grid", dataset_radius=None):
        
        if datatype != "grid":
            raise ValueError(f"Unsupported dataset type: {datatype}")
        if dataset_radius is None:
            dataset_radius = radius
        
        treat = dataset.full_treatment.reshape(-1, 1)  # shape (N,), np.array
        cov = dataset.full_covariates   # shape (N, covariate_dim)
        out = dataset.full_outcome.reshape(-1, 1)      # shape (N,)

        # Identify non-binary columns
        nonbinary_treat_cols = [i for i in range(treat.shape[1]) if not np.all(np.isin(treat[:, i], [0, 1]))]
        nonbinary_cov_cols = [i for i in range(cov.shape[1]) if not np.all(np.isin(cov[:, i], [0, 1]))]
        nonbinary_out_cols = [i for i in range(out.shape[1]) if not np.all(np.isin(out[:, i], [0, 1]))]

        # Initialize or fit scalers
        if treat_scaler is None and nonbinary_treat_cols:
            treat_scaler = StandardScaler().fit(treat[:, nonbinary_treat_cols])
        if feat_scaler is None and nonbinary_cov_cols:
            feat_scaler = StandardScaler().fit(cov[:, nonbinary_cov_cols])
        if output_scaler is None and nonbinary_out_cols:
            output_scaler = StandardScaler().fit(out[:, nonbinary_out_cols])

        self.treat_scaler = treat_scaler
        self.feat_scaler = feat_scaler
        self.output_scaler = output_scaler

        # Transform non-binary columns
        self.treat_scaled = treat.copy()
        if nonbinary_treat_cols and treat_scaler:
            self.treat_scaled[:, nonbinary_treat_cols] = treat_scaler.transform(treat[:, nonbinary_treat_cols])

        self.cov_scaled = cov.copy()
        if nonbinary_cov_cols and feat_scaler:
            self.cov_scaled[:, nonbinary_cov_cols] = feat_scaler.transform(cov[:, nonbinary_cov_cols])

        self.out_scaled = out.copy()
        if nonbinary_out_cols and output_scaler:
            self.out_scaled[:, nonbinary_out_cols] = output_scaler.transform(out[:, nonbinary_out_cols])        
        
        true_treatment = torch.from_numpy(self.treat_scaled).float()
        covariates = torch.from_numpy(self.cov_scaled).float()
        outcomes = torch.from_numpy(self.out_scaled).float()
        ids = nodes
        id2coords = {v: k for k, v in coords2id.items()}
        
        self.treatments = torch.zeros(len(ids), 2*radius+1, 2*radius+1, 1)
        
        for ii, n in enumerate(nodes):
            center_coords = id2coords[n]
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    cur_coords = (center_coords[0] + i, center_coords[1] + j)
                    if change == "center" and (i, j) == (0, 0):
                        self.treatments[ii, i + radius, j + radius, 0] = a
                    elif change == "nbr" and (i, j) != (0, 0):
                        self.treatments[ii, i + radius, j + radius, 0] = a
                    else:
                        self.treatments[ii, i + radius, j + radius, 0] = true_treatment[coords2id[cur_coords]]
                    
        self.covariates = torch.zeros(len(ids), 2*dataset_radius+1, 2*dataset_radius+1, cov.shape[1])
        
        for ii, n in enumerate(nodes):
            center_coords = id2coords[n]
            for i in range(-dataset_radius, dataset_radius + 1):
                for j in range(-dataset_radius, dataset_radius + 1):
                    cur_coords = (center_coords[0] + i, center_coords[1] + j)
                    self.covariates[ii, i + dataset_radius, j + dataset_radius, :] = covariates[coords2id[cur_coords]]
        
        treat_size = 2*radius + 1
        cov_size = 2*dataset_radius + 1
        target_size = max(treat_size, cov_size)

        self.treatments = pad_center(self.treatments, target_size)
        self.covariates = pad_center(self.covariates, target_size)

        self.true_treatment = true_treatment[ids].squeeze()
        self.outcomes = outcomes[ids]
        
    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        treatments = self.treatments[idx]
        covariates = self.covariates[idx]
        true_treatments = self.true_treatment[idx]
        # true_outcomes = self.outcomes[idx]
        
        return treatments, covariates, true_treatments


class UNetDataset(Dataset):
    def __init__(self, dataset, nodes, coords2id, radius, treat_scaler=None, feat_scaler=None, output_scaler=None, a=None, change=None, datatype="grid", latent=None, dataset_radius=None):
        
        if datatype != "grid":
            raise ValueError(f"Unsupported dataset type: {datatype}")
        if dataset_radius is None:
            dataset_radius = radius
        
        treat = dataset.full_treatment.reshape(-1, 1)  # shape (N,), np.array
        cov = dataset.full_covariates   # shape (N, covariate_dim)
        out = dataset.full_outcome.reshape(-1, 1)      # shape (N,)

        # Identify non-binary columns
        nonbinary_treat_cols = [i for i in range(treat.shape[1]) if not np.all(np.isin(treat[:, i], [0, 1]))]
        nonbinary_cov_cols = [i for i in range(cov.shape[1]) if not np.all(np.isin(cov[:, i], [0, 1]))]
        nonbinary_out_cols = [i for i in range(out.shape[1]) if not np.all(np.isin(out[:, i], [0, 1]))]

        # Initialize or fit scalers
        if treat_scaler is None and nonbinary_treat_cols:
            treat_scaler = StandardScaler().fit(treat[:, nonbinary_treat_cols])
        if feat_scaler is None and nonbinary_cov_cols:
            feat_scaler = StandardScaler().fit(cov[:, nonbinary_cov_cols])
        if output_scaler is None and nonbinary_out_cols:
            output_scaler = StandardScaler().fit(out[:, nonbinary_out_cols])

        self.treat_scaler = treat_scaler
        self.feat_scaler = feat_scaler
        self.output_scaler = output_scaler

        # Transform non-binary columns
        self.treat_scaled = treat.copy()
        if nonbinary_treat_cols and treat_scaler:
            self.treat_scaled[:, nonbinary_treat_cols] = treat_scaler.transform(treat[:, nonbinary_treat_cols])

        self.cov_scaled = cov.copy()
        if nonbinary_cov_cols and feat_scaler:
            self.cov_scaled[:, nonbinary_cov_cols] = feat_scaler.transform(cov[:, nonbinary_cov_cols])

        self.out_scaled = out.copy()
        if nonbinary_out_cols and output_scaler:
            self.out_scaled[:, nonbinary_out_cols] = output_scaler.transform(out[:, nonbinary_out_cols])        
        
        true_treatment = torch.from_numpy(self.treat_scaled).float()
        covariates = torch.from_numpy(self.cov_scaled).float()
        outcomes = torch.from_numpy(self.out_scaled).float()
        ids = nodes
        id2coords = {v: k for k, v in coords2id.items()}
        
        self.treatments = torch.zeros(len(ids), 2*radius+1, 2*radius+1, 1)
        
        for ii, n in enumerate(nodes):
            center_coords = id2coords[n]
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    cur_coords = (center_coords[0] + i, center_coords[1] + j)
                    if change == "center" and (i, j) == (0, 0):
                        self.treatments[ii, i + radius, j + radius, 0] = a
                    elif change == "nbr" and (i, j) != (0, 0):
                        self.treatments[ii, i + radius, j + radius, 0] = a
                    else:
                        self.treatments[ii, i + radius, j + radius, 0] = true_treatment[coords2id[cur_coords]]
                    
        self.covariates = torch.zeros(len(ids), 2*dataset_radius+1, 2*dataset_radius+1, cov.shape[1])
        
        for ii, n in enumerate(nodes):
            center_coords = id2coords[n]
            for i in range(-dataset_radius, dataset_radius + 1):
                for j in range(-dataset_radius, dataset_radius + 1):
                    cur_coords = (center_coords[0] + i, center_coords[1] + j)
                    self.covariates[ii, i + dataset_radius, j + dataset_radius, :] = covariates[coords2id[cur_coords]]
        
        treat_size = 2*radius + 1
        cov_size = 2*dataset_radius + 1
        target_size = max(treat_size, cov_size)
        
        self.treatments = pad_center(self.treatments, target_size)
        self.covariates = pad_center(self.covariates, target_size)
        self.latents = torch.from_numpy(latent)
        self.latents = pad_center(self.latents, target_size)
        self.covariates = torch.cat([self.covariates, self.latents], dim=-1)
        
        self.true_treatment = true_treatment[ids].squeeze()
        self.outcomes = outcomes[ids]
        
    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        treatments = self.treatments[idx]
        covariates = self.covariates[idx]
        # true_treatments = self.true_treatment[idx]
        true_outcomes = self.outcomes[idx]
        
        return treatments, covariates, true_outcomes


class Deconfounder(SpaceAlgo):
    """
    Wrapper of Interference-Aware Deconfounder with GPU acceleration support.
    """
    supports_binary = True
    supports_continuous = True
    
    def __init__(self,
        radius: float,
        head: str = "unet",
        encoder_conv1: int = 32,
        encoder_conv2: int = 64,
        kernel_size: int = 3,
        encoder_pool: str = "avg",
        connectivity: int = 4,
        tau: float = 10.0,
        eps: float = 1e-5,
        bilinear: bool = False,
        unet_base_chan: int = 16,
        weight_decay_cvae: float = 1e-5,
        lr_cvae: float = 1e-3,
        weight_decay_head: float = 1e-5,
        lr_head: float = 1e-3,
        beta_max: float = 1e-1,
        beta_epoch_max: int = 10,
        device: str = None,
        epochs_cvae: int = 50,
        epochs_head: int = 50,
        k: int = 100,
        max_iter: int = 20_000,
        lam_t: float = 0.001,
        lam_y: float = 0.001,
        spatial_split_kwargs: dict | None = None,
        batch_size = None,
        w_lags: int = 1,
    ):
        """
        Initialize Interference-Aware Deconfounder with optional device specification.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        
        self.cvae_kwargs = {
            "radius": radius,
            "encoder_conv1": encoder_conv1,
            "encoder_conv2": encoder_conv2,
            "kernel_size": kernel_size,
            "encoder_pool": encoder_pool,
            "connectivity": connectivity,
            "tau": tau,
            "eps": eps,
            "weight_decay": weight_decay_cvae,
            "lr": lr_cvae,
            "beta_max": beta_max,
            "beta_epoch_max": beta_epoch_max,
            "epochs": epochs_cvae,
        }
        
        self.unet_kwargs = {
            "bilinear": bilinear,
            "weight_decay": weight_decay_head,
            "lr": lr_head,
            "epochs": epochs_head,
            "unet_base_chan": unet_base_chan,
        }
        
        self.spatialplus_kwargs = {
            "k": k,
            "max_iter": max_iter,
            "lam_t": lam_t,
            "lam_y": lam_y,
            "spatial_split_kwargs": spatial_split_kwargs,
        }
        
        self.s2sls_kwargs = {
            "w_lags": w_lags,
        }
        
        
        self.epochs_cvae = epochs_cvae
        self.epochs_head = epochs_head
        self.head = head
        self.cvae_radius = self.cvae_kwargs["radius"]
        
        if batch_size is not None:
            global total_batch_size
            total_batch_size = batch_size
        
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Store whether we're using GPU
        self.use_gpu = self.device.type == 'cuda'
        
        if self.use_gpu:
            LOGGER.debug(f"Using GPU acceleration on device: {self.device}")
        else:
            LOGGER.debug("Using CPU computation")
            
    
    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor on appropriate device."""
        return torch.from_numpy(array).float().to(self.device)
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to numpy array."""
        return tensor.detach().cpu().numpy()
    
    def fit(self, dataset: SpaceDataset):
        import wandb
        os.environ["WANDB_START_METHOD"] = "thread"
        os.environ["PYTORCH_LIGHTNING_DEBUG"] = "1"
        
        LOGGER.debug("Building cvae model...")
        
        self.cvae_kwargs["feature_dim"] = dataset.covariates.shape[1]
        self.cvae_kwargs["datatype"] = dataset.datatype
        self.cvae_kwargs["radius"] = max(self.cvae_radius, dataset.conf_radius)
                
        self.cvae_model = CVAE(**self.cvae_kwargs)
        
        LOGGER.debug("Building dataset and dataloader...")
        graph = nx.from_edgelist(dataset.full_edge_list)
        train_ix, test_ix, _ = spatial_train_test_split_radius(
            graph, 
            init_frac= 0.02, 
            levels = 1, 
            buffer = 1 + max(self.cvae_radius, dataset.radius), 
            radius = max(self.cvae_radius, dataset.radius),
        )
        self.train_ix = train_ix
        self.test_ix = test_ix
        
        graph = nx.from_edgelist(dataset.full_edge_list)        
        node_list = list(graph.nodes())
        nbrs = {node: get_k_hop_neighbors(graph, node,  max(dataset.radius, self.cvae_radius)) for node in node_list}
        nbr_counts = {node: len(neigh) for node, neigh in nbrs.items()}
        max_count = max(nbr_counts.values())
        
        self.max_nodes = [node for node, cnt in nbr_counts.items() if cnt == max_count]
        self.max_coords2id = {tuple(coord): i for i, coord in enumerate(dataset.full_coordinates)}
                
        self.cvae_traindata = CVAEDataset(dataset, self.train_ix, self.max_coords2id, self.cvae_radius, datatype=dataset.datatype, dataset_radius=dataset.conf_radius)
        self.cvae_valdata = CVAEDataset(dataset, self.test_ix, self.max_coords2id, self.cvae_radius, self.cvae_traindata.treat_scaler, self.cvae_traindata.feat_scaler, self.cvae_traindata.output_scaler, datatype=dataset.datatype, dataset_radius=dataset.conf_radius)
        
        loader = DataLoader(self.cvae_traindata, batch_size=total_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(self.cvae_valdata, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        
        LOGGER.debug("Preparing trainer...")
        callbacks = [
            ModelCheckpoint(
                dirpath="checkpoints/",
                filename="{epoch}-{val_max_loss:.2f}",
                monitor="val_max_loss",
                mode="min",
                save_top_k=3
            ),
            EarlyStopping(
                monitor="val_max_loss",
                patience=5,
                mode="min"
            ),
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar()
        ]
        
        # wandb_logger = WandbLogger(
        #     project="cvae-treatment-model",  # Your project name
        #     name="cvae-experiment",          # Run name
        #     log_model=False,                  # Log model checkpoints
        #     save_dir="./wandb_logs",
        #     config=self.cvae_kwargs,
        #     offline=True,
        # )
        self.cvae_trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            enable_checkpointing=True,
            logger=None,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            callbacks=callbacks,
            max_epochs=self.epochs_cvae,
            deterministic=True,
            enable_model_summary=True,
        )

        LOGGER.debug("Training cvae model...")
        self.cvae_trainer.fit(self.cvae_model, train_dataloaders=loader, val_dataloaders=val_loader)
        
        # Load the best checkpoint state dict
        best_checkpoint_path = self.cvae_trainer.checkpoint_callback.best_model_path
        if best_checkpoint_path:
            LOGGER.debug(f"Loading best model from: {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=self.cvae_model.device)
            self.cvae_model.load_state_dict(checkpoint['state_dict'])
            
            self.cur_val_p_value = checkpoint["val_p_value"]
            self.val_treatment_loss = checkpoint["val_treatment_loss"]
            
            # if val_p_value < 0.3:
            #     raise ValueError(f"Validation p_value too low: {val_p_value:.3f}")
        else:
            LOGGER.warning("No best checkpoint found, using final epoch model")

        LOGGER.debug("Finished training cvae model.")
        
        
        LOGGER.debug("Getting latent dims...")
        
        latent_loader = DataLoader(self.cvae_traindata, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        latent_val_loader = DataLoader(self.cvae_valdata, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        train_latents = torch.cat(self.cvae_trainer.predict(self.cvae_model, latent_loader))
        train_latents = train_latents.cpu().numpy()
        
        val_latents = torch.cat(self.cvae_trainer.predict(self.cvae_model, latent_val_loader))
        val_latents = val_latents.cpu().numpy()        
        
        LOGGER.debug(f"Building outcome head {self.head}...")
        
        if self.head == "spatialplus" or self.head == "s2sls-lag1":
            
            spatialplusdata = CVAEDataset(dataset, self.max_nodes, self.max_coords2id, self.cvae_radius, datatype=dataset.datatype, dataset_radius=dataset.conf_radius)
            latent_loader = DataLoader(spatialplusdata, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
            B = spatialplusdata.treatments.shape[0]
            if self.cvae_radius != 0:
                B, H, W, C = spatialplusdata.treatments.shape
                flat = spatialplusdata.treatments.view(B, -1, C)              # [B, (2r+1)^2, 1]
                center_idx = (H * W) // 2
                flat_wo_center = torch.cat(
                    [flat[:, :center_idx], flat[:, center_idx+1:]], dim=1
                )  # [B, (2r+1)^2 - 1, 1]

                flat_wo_center = flat_wo_center.squeeze(-1).cpu().numpy()
            
            val_latents = torch.cat(self.cvae_trainer.predict(self.cvae_model, latent_loader))
            val_latents = val_latents.view(B, -1).cpu().numpy() 
            
            
            new_dataset = deepcopy(dataset)
            if self.cvae_radius != 0:
                new_dataset.covariates = np.concatenate([new_dataset.covariates, flat_wo_center, val_latents], axis=1)
            else:
                new_dataset.covariates = np.concatenate([new_dataset.covariates, val_latents], axis=1)
            
            if self.head == "spatialplus":
                self.head_model = SpatialPlus(**self.spatialplus_kwargs)
                self.head_model.fit(new_dataset)
            elif self.head == "s2sls-lag1":
                self.head_model = GMLag(**self.s2sls_kwargs)        
        
            
        if self.head == "unet":
            
            self.unet_kwargs["feature_dim"] = dataset.covariates.shape[1] + (self.cvae_kwargs["encoder_conv2"] // 2)
            self.unet_kwargs["datatype"] = dataset.datatype
            self.unet_kwargs["radius"] = max(self.cvae_radius, dataset.conf_radius)
            
            self.head_model = UNetHead(**self.unet_kwargs)
            
            self.head_traindata = UNetDataset(dataset, self.train_ix, self.max_coords2id, self.cvae_radius, datatype=dataset.datatype, latent=train_latents, dataset_radius=dataset.conf_radius)
            self.head_valdata = UNetDataset(dataset, self.test_ix, self.max_coords2id, self.cvae_radius, self.head_traindata.treat_scaler, self.head_traindata.feat_scaler, self.head_traindata.output_scaler, datatype=dataset.datatype, latent=val_latents, dataset_radius=dataset.conf_radius)

            loader = DataLoader(self.head_traindata, batch_size=total_batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(self.head_valdata, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)


            LOGGER.debug("Preparing trainer...")
            callbacks = [
                ModelCheckpoint(
                    dirpath="checkpoints/",
                    filename="{epoch}-{val_loss:.2f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=3
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    mode="min"
                ),
                LearningRateMonitor(logging_interval="step"),
                RichProgressBar()
            ]

            # wandb_logger = WandbLogger(
            #     project="cvae-treatment-model",  # Your project name
            #     name="outcome",          # Run name
            #     log_model=False,                  # Log model checkpoints
            #     save_dir="./wandb_logs",
            #     config=self.unet_kwargs,
            #     offline=True,
            # )

            self.head_trainer = pl.Trainer(
                accelerator="auto",
                devices=1,
                enable_checkpointing=True,
                logger=None,
                gradient_clip_val=1.0,
                enable_progress_bar=True,
                callbacks=callbacks,
                max_epochs=self.epochs_head,
                deterministic=True,
                enable_model_summary=True,
            )

            LOGGER.debug("Training outcome model...")
            self.head_trainer.fit(self.head_model, train_dataloaders=loader, val_dataloaders=val_loader)

            # Load the best checkpoint state dict
            best_checkpoint_path = self.head_trainer.checkpoint_callback.best_model_path
            if best_checkpoint_path:
                LOGGER.debug(f"Loading best model from: {best_checkpoint_path}")
                checkpoint = torch.load(best_checkpoint_path, map_location=self.head_model.device)
                self.head_model.load_state_dict(checkpoint['state_dict'])
            else:
                LOGGER.warning("No best checkpoint found, using final epoch model")

            LOGGER.debug("Finished training outcome model.")

        
        
        
    def eval(self, dataset: SpaceDataset) -> dict:
        """
        Evaluate the model with GPU acceleration for large datasets.
        """
        
        LOGGER.debug("Computing counterfactuals...")
        ite = []
        for a in dataset.treatment_values:
            preds_a = self.predict(dataset, self.max_nodes, a=a, change="center")
            ite.append(preds_a)
        ite = np.concatenate(ite, axis=1)

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = effects["erf"][1] - effects["erf"][0]

            spill = []
            for a in dataset.treatment_values:
                preds_a = self.predict(dataset, self.max_nodes, a=a, change="nbr")
                spill.append(preds_a)
            spill = np.concatenate(spill, axis=1)

            s = spill.mean(0)
            effects["spill"] = s[1] - s[0]
        

        return effects
    
    def tune_metric(self, dataset: SpaceDataset) -> float:
    
        if self.head == "unet":
            preds = self.predict(dataset, self.max_nodes, a=None, change=None)[:, 0]
            return np.mean((dataset.full_outcome[self.max_nodes] - preds) ** 2)
        elif self.head == "spatialplus" or self.head == "s2sls-lag1":
            predict_data = CVAEDataset(dataset, self.max_nodes, self.max_coords2id, self.cvae_radius, self.cvae_traindata.treat_scaler, self.cvae_traindata.feat_scaler, self.cvae_traindata.output_scaler, datatype=dataset.datatype, dataset_radius=dataset.conf_radius)
        
            loader = DataLoader(predict_data, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
            latents = torch.cat(self.cvae_trainer.predict(self.cvae_model, loader))
            latents = latents.view(predict_data.treatments.shape[0], -1).cpu().numpy()
        
            new_dataset = deepcopy(dataset)
            B = predict_data.treatments.shape[0]
            if self.cvae_radius != 0:
                B, H, W, C = predict_data.treatments.shape
                flat = predict_data.treatments.view(B, -1, C)              # [B, (2r+1)^2, 1]
                center_idx = (H * W) // 2
                flat_wo_center = torch.cat(
                    [flat[:, :center_idx], flat[:, center_idx+1:]], dim=1
                )  # [B, (2r+1)^2 - 1, 1]

                flat_wo_center = flat_wo_center.squeeze(-1).cpu().numpy()            

                new_dataset.covariates = np.concatenate([new_dataset.covariates, flat_wo_center, latents], axis=1)
            else:
                new_dataset.covariates = np.concatenate([new_dataset.covariates, latents], axis=1)
            if self.head == "spatialplus":
                return self.head_model.tune_metric(new_dataset)
            if self.head == "s2sls-lag1":
                return self.val_treatment_loss
                # preds = self.head_model.predict(new_dataset).flatten()
                # return np.mean((dataset.outcome - preds) ** 2)
                
                
    
    
    def predict(self, dataset: SpaceDataset, nodes, a, change) -> dict:
        """
        Get outcome predictions with GPU acceleration for large datasets.
        """
        self.cvae_model.eval()
        if self.head == "unet":
            self.head_model.eval()
        
        
        predict_data = CVAEDataset(dataset, nodes, self.max_coords2id, self.cvae_radius, self.cvae_traindata.treat_scaler, self.cvae_traindata.feat_scaler, self.cvae_traindata.output_scaler, datatype=dataset.datatype, a=a, change=change, dataset_radius=dataset.conf_radius)
        
        loader = DataLoader(predict_data, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        latents = torch.cat(self.cvae_trainer.predict(self.cvae_model, loader))
        if self.head == "unet":
            latents = latents.cpu().numpy()
        elif self.head == "spatialplus" or self.head == "s2sls-lag1":
            latents = latents.view(predict_data.treatments.shape[0], -1).cpu().numpy()
        
        if self.head == "unet":
            predict_head_data = UNetDataset(dataset, nodes, self.max_coords2id, self.cvae_radius, self.head_traindata.treat_scaler, self.head_traindata.feat_scaler, self.head_traindata.output_scaler, a=a, change=change, datatype=dataset.datatype, latent=latents, dataset_radius=dataset.conf_radius)

            loader = DataLoader(predict_head_data, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
            preds = torch.cat(self.head_trainer.predict(self.head_model, loader))
            preds = preds.cpu().numpy()
            # scale back the preds
            preds = self.head_traindata.output_scaler.inverse_transform(preds)
        
        elif self.head == "spatialplus" or self.head == "s2sls-lag1":
            new_dataset = deepcopy(dataset)
            if self.cvae_radius != 0:
                B, H, W, C = predict_data.treatments.shape
                flat = predict_data.treatments.view(B, -1, C)              # [B, (2r+1)^2, 1]
                center_idx = (H * W) // 2
                flat_wo_center = torch.cat(
                    [flat[:, :center_idx], flat[:, center_idx+1:]], dim=1
                )  # [B, (2r+1)^2 - 1, 1]

                flat_wo_center = flat_wo_center.squeeze(-1).cpu().numpy()            

                new_dataset.covariates = np.concatenate([new_dataset.covariates, flat_wo_center, latents], axis=1)
            else:
                new_dataset.covariates = np.concatenate([new_dataset.covariates, latents], axis=1)
            
            return self.head_model.predict(new_dataset)
        
        
        return preds
    
    
    
def spatial_train_test_split_radius(
    graph: nx.Graph,
    init_frac: float,
    levels: int,
    buffer: int = 0,
    radius: int = 1,
) -> tuple[list, list, list]:
    """Split restricted to nodes with max neighbors."""
    LOGGER.debug(
        f"Selecting tuning split with {levels} levels and {buffer} buffer, restricted to max-degree nodes."
    )
    node_list = list(graph.nodes())
    n = len(node_list)

    # dict of neighbors
    nbrs = {node: get_k_hop_neighbors(graph, node, radius) for node in node_list}
    
    # find nodes with max neighbors
    nbr_counts = {node: len(neigh) for node, neigh in nbrs.items()}
    max_count = max(nbr_counts.values())
    max_nodes = [node for node, cnt in nbr_counts.items() if cnt == max_count]
    
    # restrict neighbor sets to only max-degree nodes
    nbrs = {node: set(graph.neighbors(node)) for node in node_list}
    nbrs = {node: neigh & set(max_nodes) for node, neigh in nbrs.items()}

    # restrict everything to only these nodes
    node_list = max_nodes
    n = len(node_list)

    # pick tuning centroids from max nodes
    num_tuning_centroids = int(init_frac * n)
    rng = np.random
    tuning_nodes = rng.choice(node_list, size=num_tuning_centroids, replace=False)
    tuning_nodes = set(tuning_nodes)

    # expand tuning set
    for _ in range(levels):
        tmp = tuning_nodes.copy()
        for node in tmp:
            tuning_nodes.update(nbrs[node])
    tuning_nodes = list(tuning_nodes)

    # buffer set
    buffer_nodes = set(tuning_nodes)
    for _ in range(buffer):
        tmp = buffer_nodes.copy()
        for node in tmp:
            buffer_nodes.update(nbrs[node])
    buffer_nodes = list(buffer_nodes - set(tuning_nodes))

    # training = remaining max-degree nodes not in tuning or buffer
    training_nodes = list(set(node_list) - set(tuning_nodes) - set(buffer_nodes))
    LOGGER.debug(
        f"Length of training, tuning and buffer: {len(training_nodes)} and {len(tuning_nodes)} and {len(buffer_nodes)}"
    )
    return training_nodes, tuning_nodes, buffer_nodes
