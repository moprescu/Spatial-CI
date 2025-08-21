import libpysal as lp
import numpy as np
from pysal.model import spreg
from typing import Optional
import torch
from .utils import UNet

from sci import SpaceDataset
from spacebench.algorithms import SpaceAlgo
from spacebench.log import LOGGER

class InterferenceAwareDeconfounder_Grid(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        radius: float,
        latent_dim: int,
        head: str = "unet",
        encoder_conv1: int = 32,
        encoder_conv2: int = 64,
        kernel_size: int = 3,
        encoder_pool: str = "avg",
        encoder_fc: str = "separate",
        connectivity: int = 4,
        tau: float = 10.0,
        eps: float = 1e-5,
        bilinear: bool = False,
    ):
        """
        Interference-Aware Deconfounder for spatial causal inference
        
        Args:
            latent_dim: Dimension of latent space Z_s
            radius: Radius for patch-based convolution
            tau: Precision parameter for GMRF prior
            eps: Small constant for numerical stability
        """
        super(InterferenceAwareDeconfounder, self).__init__()
        
        self.feature_dim = feature_dim
        self.radius = radius
        self.tau = tau
        self.eps = eps
        self.patch_size = 2 * radius + 1
        self.kernel_size = kernel_size
        self.per_location_latent_dim = latent_dim
        self.latent_dim = self.patch_size * self.patch_size * self.per_location_latent_dim
        self.connectivity = connectivity
        self.bilinear = bilinear
        
        # Calculate input channels: A_s + A_{N_s} (treatments + neighbor treatments)
        # Patch of treatments size self.patch_size x self.patch_size
        self.encoder_fc = encoder_fc
        self.encoder_input_dim = 1
        if self.encoder_fc == "expand":
            self.encoder_input_dim += self.feature_dim
        
        # Encoder: Two 3x3 convolutions over radius-r patch
        self.enc_mid_chan = encoder_conv1
        self.enc_out_chan = encoder_conv2
        self.encoder_conv = DoubleConvMultiChannel(self.encoder_input_dim, enc_mid_chan, enc_out_chan)
        self.encoder_pool = encoder_pool
        
        # Encoder output processing including covariates
        self.encoder_flatten_dim = self.enc_out_chan
        if self.encoder_fc == "separate":
            self.encoder_flatten_dim += self.feature_dim
        self.fc_mu = nn.Linear(self.encoder_flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_flatten_dim, self.latent_dim)
        
        # Decoder: One-layer MLP p_ψ(A_s=1|x_s,Z_s) = σ(f_ψ(x_s,Z_s))
        self.decoder = nn.Linear(self.feature_dim + self.latent_dim, 1)
        
        # Precompute grid Laplacian matrix L for GMRF prior
        self.register_buffer('laplacian', self._build_patch_laplacian())
        
        if head == "unet":
            self.head = UNet(n_channels=1 + self.per_location_latent_dim + self.feature_dim, n_classes=1, bilinear=self.bilinear)
        elif head == "vit":
            raise ValueError(f"Unsupported outcome head type: {head}")
            # self.head = 
        else:
            raise ValueError(f"Unsupported outcome head type: {head}")
    
    
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
            x: Features [batch_size, feature_dim]
        
        Returns:
            mu, logvar: Parameters of latent distribution
        """
        # Rearrange for convolution: [batch, channels, height, width]
        encoder_input = A.permute(0, 3, 1, 2)
        if self.encoder_fc == "expand":
            # Combine features and neighbor treatments
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = x.expand(-1, -1, self.patch_size, self.patch_size)
            encoder_input = torch.cat([encoder_input, x], dim=1)
        
        # Two 3×3 convolutions
        h = self.encoder_conv(encoder_input)
        
        if self.encoder_pool == "max":
            h = F.adaptive_max_pool2d(h2, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.encoder_pool == "avg":
            # Global average pooling to get fixed-size representation
            h = F.adaptive_avg_pool2d(h2, (1, 1)).squeeze(-1).squeeze(-1)
        
        if self.encoder_fc == "separate":
            # Combine features and neighbor treatments
            h = torch.cat([h, x], dim=-1)
        
        # Output latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
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
            x_s: Features at location s [batch_size, feature_dim]
            z_s: Latent variables [batch_size, latent_dim]
        
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
        Forward pass through interference-aware CVAE and outcome head
        
        Args:
            treatments: Current and neighboring treatments [batch_size, patch_size, patch_size, 1]
            covariates: Features at location [batch_size, feature_dim]
        
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
        treatment_probs = self.decode(covariates, z_s)
        
        treat = treatments.permute(0, 3, 1, 2)
        cov = covariates.unsqueeze(-1).unsqueeze(-1)
        cov = cov.expand(-1, -1, self.patch_size, self.patch_size)
        z_s_grid = z_s.view(-1, self.patch_size, self.patch_size, self.per_location_latent_dim).permute(0, 3, 1, 2)
        
        head_input = torch.cat([treat, cov, z_s_grid], dim=1)
            
        out = self.head(head_input)
        return treatment_probs, mu, logvar, out
    
class deconfounder_grid(pl.LightningModule):
    def __init__(
        self,
        feature_dim: int,
        radius: float,
        latent_dim: int,
        head: str = "unet",
        encoder_conv1: int = 32,
        encoder_conv2: int = 64,
        kernel_size: int = 3,
        encoder_pool: str = "avg",
        encoder_fc: str = "separate",
        connectivity: int = 4,
        tau: float = 10.0,
        eps: float = 1e-5,
        bilinear: bool = False,
        weight_decay: float = 1e-5,
        lr: float = 1e-3,
        beta_max: float = 1e-1,
        beta_epoch_max: int = 10,
        gamma: float = 1e-1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.radius = radius
        self.latent_dim = latent_dim
        self.head = head
        self.encoder_conv1 = encoder_conv1
        self.encoder_conv2 = encoder_conv2
        self.kernel_size = kernel_size
        self.encoder_pool = encoder_pool
        self.encoder_fc = encoder_fc
        self.connectivity = connectivity
        self.tau = tau
        self.eps = eps
        self.bilinear = bilinear
        
        self.model = InterferenceAwareDeconfounder_Grid(
            feature_dim=self.feature_dim,
            radius=self.radius,
            latent_dim=self.latent_dim,
            head=self.head,
            encoder_conv1=self.encoder_conv1,
            encoder_conv2=self.encoder_conv2,
            kernel_size=self.kernel_size,
            encoder_pool=self.encoder_pool,
            encoder_fc=self.encoder_fc,
            connectivity=self.connectivity,
            tau=self.tau,
            eps=self.eps,
            bilinear=self.bilinear,
        )
        
        self.weight_decay = weight_decay
        self.lr = lr
        self.beta_max = beta_max
        self.beta_epoch_max = beta_epoch_max
        self.gamma = gamma
        
    def forward(treatments, covariates):
        treatment_probs, mu, logvar, pred = self.model(treatments, covariates)
        return pred
        
    def training_step(self, treatments, covariates, true_treatments, true_outcomes):
        treatment_probs, mu, logvar, pred = self.model(treatments, covariates)
        
        treatment_loss, kldiv_loss, outcome_loss = self.loss(treatment_probs, mu, logvar, pred, true_treatments, true_outcomes)
        # Combined objective
        beta = min(self.beta_max, self.current_epoch * (self.beta_max / self.beta_epoch_max))
        loss = treatment_loss + beta * kldiv_loss + self.gamma * outcome_loss
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_treatment_loss", treatment_loss, on_epoch=True, prog_bar=True)
        self.log("train_kldiv_loss", kldiv_loss, on_epoch=True, prog_bar=True)
        self.log("train_outcome_loss", outcome_loss, on_epoch=True, prog_bar=True)
        
    def validation_step(self, treatments, covariates, true_treatments, true_outcomes):
        treatment_probs, mu, logvar, pred = self.model(treatments, covariates)
        
        treatment_loss, kldiv_loss, outcome_loss = self.loss(treatment_probs, mu, logvar, pred, true_treatments, true_outcomes)
        # Combined objective
        beta = min(self.beta_max, self.current_epoch * (self.beta_max / self.beta_epoch_max))
        loss = treatment_loss + beta * kldiv_loss + self.gamma * outcome_loss
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_treatment_loss", treatment_loss, on_epoch=True, prog_bar=True)
        self.log("val_kldiv_loss", kldiv_loss, on_epoch=True, prog_bar=True)
        self.log("val_outcome_loss", outcome_loss, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        return optimizer
            
    def kldiv(mu, logvar):
        mu_ = mu.view(-1, self.model.patch_size * self.model.patch_size, self.model.per_location_latent_dim)
        logvar_ = logvar.view(-1, self.model.patch_size * self.model.patch_size, self.model.per_location_latent_dim)
        
        # Spatial smoothness term: tau/2 * E[Z^T L Z] (vectorized)
        # E[Z^T L Z] = mu^T L mu + tr(L * Sigma) where Sigma = diag(exp(logvar))
        
        # Quadratic term: mu^T L mu for all latent dims simultaneously
        # mu_flat: [batch_size, n_pixels, latent_dim]
        # L: [n_pixels, n_pixels]
        mu_L = torch.matmul(mu_.transpose(1, 2), self.model.laplacian)  # [batch_size, latent_dim, n_pixels]
        quadratic_term = torch.sum(mu_L * mu_.transpose(1, 2), dim=2)  # [batch_size, latent_dim]
        
        # Trace term: tr(L * diag(exp(logvar))) = sum(diag(L) * exp(logvar))
        L_diag = torch.diag(self.model.laplacian)  # [n_pixels]
        trace_term = torch.sum(L_diag.unsqueeze(0).unsqueeze(-1) * torch.exp(lovar_), dim=1)  # [batch_size, latent_dim]
        
        # Sum over latent dimensions and apply tau scaling
        kldiv_loss = self.model.tau / 2 * torch.sum(quadratic_term + trace_term, dim=1)  # [batch_size]
        
        return kldiv_loss.mean()
        
        
    
    def loss(treatment_probs, mu, logvar, pred, true_treatments, true_outcomes):
        # Term 1: -log p_ψ(A_s | x_s, Z_s) - Treatment reconstruction
        treatment_loss = F.binary_cross_entropy(treatment_probs, true_treatments.float())

        # Term 2: β * KL(q_φ || p_θ) - Spatial KL divergence
        kldiv_loss = self.kldiv(mu, logvar)
        
        # Term 3: γ * E_q[(Y_s - h_θ(A_s,A_{N_s},x_s,Z_s))²] - Outcome prediction
        outcome_loss = F.mse_loss(outcome_pred, true_outcomes)
        
        return treatment_loss, kldiv_loss, outcome_loss       
        
        
        



class Deconfounder(SpaceAlgo):
    """
    Wrapper of Interference-Aware Deconfounder with GPU acceleration support.
    """
    supports_binary = True
    supports_continuous = True
    
    def __init__(self,
        feature_dim: int,
        radius: float,
        latent_dim: int,
        head: str = "unet",
        encoder_conv1: int = 32,
        encoder_conv2: int = 64,
        kernel_size: int = 3,
        encoder_pool: str = "avg",
        encoder_fc: str = "separate",
        connectivity: int = 4,
        tau: float = 10.0,
        eps: float = 1e-5,
        bilinear: bool = False,
        weight_decay: float = 1e-5,
        lr: float = 1e-3,
        beta_max: float = 1e-1,
        beta_epoch_max: int = 10,
        gamma: float = 1e-1,
        device: Optional[str] = None,
    ):
        """
        Initialize Interference-Aware Deconfounder with optional device specification.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
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
        
        
        # Convert data to tensors for GPU computation
        treatment_tensor = self._to_tensor(dataset.treatment)
        covariates_tensor = self._to_tensor(dataset.covariates)
        outcome_tensor = self._to_tensor(dataset.outcome)
        
        
        # Add noise to covariates using GPU
        if self.use_gpu:
            noise = torch.normal(0, 1e-6, size=covariates_tensor.shape, device=self.device)
            noisy_covars_tensor = covariates_tensor + noise
        else:
            # Fallback to numpy for CPU
            noisy_covars = dataset.covariates + np.random.normal(
                scale=1e-6, size=dataset.covariates.shape
            )
            noisy_covars_tensor = self._to_tensor(noisy_covars)
        
        # Concatenate treatment and covariates
        x_tensor = torch.cat([treatment_tensor.unsqueeze(1), noisy_covars_tensor], dim=1)
        
        # Compute standardization parameters on GPU
        self.mu_x_tensor = x_tensor.mean(0)
        self.sig_x_tensor = x_tensor.std(0)
        self.mu_y_tensor = outcome_tensor.mean()
        self.sig_y_tensor = outcome_tensor.std()
        
        # Standardize data
        x_standardized = (x_tensor - self.mu_x_tensor) / self.sig_x_tensor
        y_standardized = (outcome_tensor - self.mu_y_tensor) / self.sig_y_tensor
        
        # Convert back to numpy for PySAL compatibility
        x = self._to_numpy(x_standardized)
        y = self._to_numpy(y_standardized)
        
        # Store standardization parameters as numpy for compatibility
        self.mu_x = self._to_numpy(self.mu_x_tensor)
        self.sig_x = self._to_numpy(self.sig_x_tensor)
        self.mu_y = self._to_numpy(self.mu_y_tensor)
        self.sig_y = self._to_numpy(self.sig_y_tensor)
        
        LOGGER.debug("Computing spatial weights")
        
    def eval(self, dataset: SpaceDataset) -> dict:
        """
        Evaluate the model with GPU acceleration for large datasets.
        """
        if self.use_gpu:
            # GPU-accelerated evaluation
            outcome_tensor = self._to_tensor(dataset.outcome)
            treatment_tensor = self._to_tensor(dataset.treatment)
            
            # Compute ITE for each treatment value
            ite_list = []
            for a in dataset.treatment_values:
                a_tensor = torch.full_like(treatment_tensor, a)
                ite_a = outcome_tensor + self.t_coef * (a_tensor - treatment_tensor)
                ite_list.append(ite_a)
            
            ite_tensor = torch.stack(ite_list, dim=1)
            erf_tensor = ite_tensor.mean(0)
            
            # Convert back to numpy
            ite = self._to_numpy(ite_tensor)
            erf = self._to_numpy(erf_tensor)
            
        else:
            # Original CPU computation
            ite = [
                dataset.outcome + self.t_coef * (a - dataset.treatment)
                for a in dataset.treatment_values
            ]
            ite = np.stack(ite, axis=1)
            erf = ite.mean(0)
        
        effects = {"erf": erf, "ite": ite}
        
        if dataset.has_binary_treatment():
            effects["ate"] = self.t_coef
            
        return effects