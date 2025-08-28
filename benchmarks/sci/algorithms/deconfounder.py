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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from .utils import UNet, get_k_hop_neighbors, DoubleConvMultiChannel
import networkx as nx

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
        connectivity: int = 4,
        tau: float = 10.0,
        eps: float = 1e-5,
        bilinear: bool = False,
        unet_base_chan: int = 16,
    ):
        """
        Interference-Aware Deconfounder for spatial causal inference on gridded datasets
        
        Args:
            latent_dim: Dimension of latent space Z_s
            radius: Radius for patch-based convolution
            tau: Precision parameter for GMRF prior
            eps: Small constant for numerical stability
        """
        super(InterferenceAwareDeconfounder_Grid, self).__init__()
        
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
        self.encoder_input_dim = 1 + self.feature_dim
        
        # Encoder: Two 3x3 convolutions over radius-r patch
        self.enc_mid_chan = encoder_conv1
        self.enc_out_chan = encoder_conv2
        self.encoder_conv = DoubleConvMultiChannel(self.encoder_input_dim, self.enc_mid_chan, self.enc_out_chan)
        self.encoder_pool = encoder_pool
        
        # Encoder output processing including covariates
        self.encoder_flatten_dim = self.enc_out_chan
        self.fc_mu = nn.Linear(self.encoder_flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_flatten_dim, self.latent_dim)
        
        # Decoder: One-layer MLP p_ψ(A_s=1|x_s,Z_s) = σ(f_ψ(x_s,Z_s))
        self.decoder = nn.Linear( self.patch_size * self.patch_size * self.feature_dim + self.latent_dim, 1)
        
        # Precompute grid Laplacian matrix L for GMRF prior
        self.register_buffer('laplacian', self._build_patch_laplacian())
        
        if head == "unet":
            self.head = UNet(n_channels=1 + self.per_location_latent_dim + self.feature_dim, n_classes=1, bilinear=self.bilinear, radius=self.radius, base_channels=unet_base_chan)
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
        Forward pass through interference-aware CVAE and outcome head
        
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
        
        treat = treatments.permute(0, 3, 1, 2)
        cov = covariates.permute(0, 3, 1, 2)
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
        connectivity: int = 4,
        tau: float = 10.0,
        eps: float = 1e-5,
        bilinear: bool = False,
        unet_base_chan: int = 16,
        weight_decay: float = 1e-5,
        lr: float = 1e-3,
        beta_max: float = 1e-1,
        beta_epoch_max: int = 10,
        gamma: float = 1e-1,
        epochs: int = 10,
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
        self.connectivity = connectivity
        self.tau = tau
        self.eps = eps
        self.bilinear = bilinear
        self.unet_base_chan = unet_base_chan
        
        self.model = InterferenceAwareDeconfounder_Grid(
            feature_dim=self.feature_dim,
            radius=self.radius,
            latent_dim=self.latent_dim,
            head=self.head,
            encoder_conv1=self.encoder_conv1,
            encoder_conv2=self.encoder_conv2,
            kernel_size=self.kernel_size,
            encoder_pool=self.encoder_pool,
            connectivity=self.connectivity,
            tau=self.tau,
            eps=self.eps,
            bilinear=self.bilinear,
            unet_base_chan=self.unet_base_chan
        )
        
        self.weight_decay = weight_decay
        self.lr = lr
        self.beta_max = beta_max
        self.beta_epoch_max = beta_epoch_max
        self.gamma = gamma
        self.epochs = epochs
        
    def predict_step(self, batch, batch_idx):
        treatments, covariates, true_treatments, true_outcomes = batch
        treatment_probs, mu, logvar, pred = self.model(treatments, covariates)
        return pred
        
    def training_step(self, batch, batch_idx):
        treatments, covariates, true_treatments, true_outcomes = batch
        treatment_probs, mu, logvar, pred = self.model(treatments, covariates)
        
        treatment_loss, kldiv_loss, outcome_loss = self.loss(treatment_probs, mu, logvar, pred, true_treatments, true_outcomes)
        # Combined objective
        beta = min(self.beta_max, self.current_epoch * (self.beta_max / self.beta_epoch_max))
        loss = treatment_loss + beta * kldiv_loss + self.gamma * outcome_loss
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_treatment_loss", treatment_loss, on_epoch=True, prog_bar=True)
        self.log("train_kldiv_loss", kldiv_loss, on_epoch=True, prog_bar=True)
        self.log("train_outcome_loss", outcome_loss, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        treatments, covariates, true_treatments, true_outcomes = batch
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
        
        # Quadratic term: mu^T L mu for all latent dims simultaneously
        # mu_flat: [batch_size, n_pixels, latent_dim]
        # L: [n_pixels, n_pixels]
        mu_L = torch.matmul(mu_.transpose(1, 2), self.model.laplacian)  # [batch_size, latent_dim, n_pixels]
        quadratic_term = torch.sum(mu_L * mu_.transpose(1, 2), dim=2)  # [batch_size, latent_dim]
        
        # Trace term: tr(L * diag(exp(logvar))) = sum(diag(L) * exp(logvar))
        L_diag = torch.diag(self.model.laplacian)  # [n_pixels]
        trace_term = torch.sum(L_diag.unsqueeze(0).unsqueeze(-1) * torch.exp(logvar_), dim=1)  # [batch_size, latent_dim]
        
        # Sum over latent dimensions and apply tau scaling
        kldiv_loss = self.model.tau / 2 * torch.sum(quadratic_term + trace_term, dim=1)  # [batch_size]
        
        return kldiv_loss.mean()
    
    def loss(self, treatment_probs, mu, logvar, pred, true_treatments, true_outcomes):
        # Term 1: -log p_ψ(A_s | x_s, Z_s) - Treatment reconstruction
        treatment_loss = F.binary_cross_entropy(treatment_probs, true_treatments.float())

        # Term 2: β * KL(q_φ || p_θ) - Spatial KL divergence
        kldiv_loss = self.kldiv(mu, logvar)
        
        # Term 3: γ * E_q[(Y_s - h_θ(A_s,A_{N_s},x_s,Z_s))²] - Outcome prediction
        outcome_loss = F.mse_loss(pred, true_outcomes)
        
        return treatment_loss, kldiv_loss, outcome_loss       
        
        
        

class InterferenceGridDataset(Dataset):
    def __init__(self, dataset, nodes, coords2id, radius, treat_scaler=None, feat_scaler=None, output_scaler=None, a=None, change=None):
        
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
        self.covariates = torch.zeros(len(ids), 2*radius+1, 2*radius+1, cov.shape[1])
        
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
                    self.covariates[ii, i + radius, j + radius, :] = covariates[coords2id[cur_coords]]
        
        
        self.true_treatment = true_treatment[ids].squeeze()
        self.outcomes = outcomes[ids]
        
    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        treatments = self.treatments[idx]
        covariates = self.covariates[idx]
        true_treatments = self.true_treatment[idx]
        true_outcomes = self.outcomes[idx]
        
        return treatments, covariates, true_treatments, true_outcomes


class Deconfounder(SpaceAlgo):
    """
    Wrapper of Interference-Aware Deconfounder with GPU acceleration support.
    """
    supports_binary = True
    supports_continuous = True
    
    def __init__(self,
        radius: float,
        latent_dim: int,
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
        weight_decay: float = 1e-5,
        lr: float = 1e-3,
        beta_max: float = 1e-1,
        beta_epoch_max: int = 10,
        gamma: float = 1e-1,
        device: str = None,
        epochs: int = 50,
    ):
        """
        Initialize Interference-Aware Deconfounder with optional device specification.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        
        self.impl_kwargs = {
            "radius": radius,
            "latent_dim": latent_dim,
            "head": head,
            "encoder_conv1": encoder_conv1,
            "encoder_conv2": encoder_conv2,
            "kernel_size": kernel_size,
            "encoder_pool": encoder_pool,
            "connectivity": connectivity,
            "tau": tau,
            "eps": eps,
            "bilinear": bilinear,
            "weight_decay": weight_decay,
            "lr": lr,
            "beta_max": beta_max,
            "beta_epoch_max": beta_epoch_max,
            "gamma": gamma,
            "epochs": epochs,
            "unet_base_chan": unet_base_chan,
        }
        
        self.epochs = epochs
        
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
        LOGGER.debug("Building deconfounder model...")
        self.impl_kwargs["feature_dim"] = dataset.covariates.shape[1]
        self.model = deconfounder_grid(**self.impl_kwargs)
        
        LOGGER.debug("Building dataset and dataloader...")
        graph = nx.from_edgelist(dataset.full_edge_list)
        train_ix, test_ix, _ = spatial_train_test_split_radius(
            graph, 
            init_frac= 0.02, 
            levels = 1, 
            buffer = 1 + self.impl_kwargs["radius"], 
            seed = 0, 
            radius = self.impl_kwargs["radius"],
        )
        self.train_ix = train_ix
        self.test_ix = test_ix
        
        node_list = list(graph.nodes())
        nbrs = {node: get_k_hop_neighbors(graph, node, self.impl_kwargs["radius"]) for node in node_list}
        coords2id = {tuple(coord): i for i, coord in enumerate(dataset.full_coordinates)}
        
        self.traindata = InterferenceGridDataset(dataset, train_ix, coords2id, self.impl_kwargs["radius"])
        self.valdata = InterferenceGridDataset(dataset, test_ix, coords2id, self.impl_kwargs["radius"], self.traindata.treat_scaler, self.traindata.feat_scaler, self.traindata.output_scaler)
        
        loader = DataLoader(self.traindata, batch_size=4, shuffle=True)
        val_loader = DataLoader(self.valdata, batch_size=4, shuffle=False)
        
        
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
                monitor="val_outcome_loss",
                patience=10,
                mode="min"
            ),
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar()
        ]

        self.trainer = pl.Trainer(
            accelerator="gpu" if self.use_gpu else "cpu",
            devices=1,
            enable_checkpointing=True,
            logger=True,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            callbacks=callbacks,
            max_epochs=self.epochs,
            enable_model_summary=True,
        )

        LOGGER.debug("Training deconfounder model...")
        self.trainer.fit(self.model, train_dataloaders=loader, val_dataloaders=val_loader)
        LOGGER.debug("Finished training deconfounder model.")
        
        
        
    def eval(self, dataset: SpaceDataset) -> dict:
        """
        Evaluate the model with GPU acceleration for large datasets.
        """
        self.model.eval()
        LOGGER.debug("Preparing data loader with existing scaler...")
        
        graph = nx.from_edgelist(dataset.full_edge_list)        
        node_list = list(graph.nodes())
        nbrs = {node: get_k_hop_neighbors(graph, node, self.impl_kwargs["radius"]) for node in node_list}
        nbr_counts = {node: len(neigh) for node, neigh in nbrs.items()}
        max_count = max(nbr_counts.values())
        max_nodes = [node for node, cnt in nbr_counts.items() if cnt == max_count]
    
        coords2id = {tuple(coord): i for i, coord in enumerate(dataset.full_coordinates)}
        
        
        self.evaldata = InterferenceGridDataset(dataset, max_nodes, coords2id, self.impl_kwargs["radius"], self.traindata.treat_scaler, self.traindata.feat_scaler, self.traindata.output_scaler)
        
        loader = DataLoader(self.evaldata, batch_size=4, shuffle=False)
        preds = torch.cat(self.trainer.predict(self.model, loader))
        preds = preds.cpu().numpy()

        # scale back the preds
        preds = self.traindata.output_scaler.inverse_transform(preds)
        resid = dataset.full_outcome[max_nodes] - preds[:, 0]

        LOGGER.debug("Computing counterfactuals...")
        ite = []
        for a in dataset.treatment_values:
            cfdata = InterferenceGridDataset(dataset, max_nodes, coords2id, self.impl_kwargs["radius"], self.traindata.treat_scaler, self.traindata.feat_scaler, self.traindata.output_scaler, a=a, change="center")
            loader = DataLoader(cfdata, batch_size=4, shuffle=False)
            
            preds_a = torch.cat(self.trainer.predict(self.model, loader))
            preds_a = preds_a.cpu().numpy()
            preds_a = self.traindata.output_scaler.inverse_transform(preds_a)
            ite.append(preds_a)
        ite = np.concatenate(ite, axis=1)
        ite += resid[:, None]

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = effects["erf"][1] - effects["erf"][0]
            
            spill = []
            for a in dataset.treatment_values:
                cfdata = InterferenceGridDataset(dataset, max_nodes, coords2id, self.impl_kwargs["radius"], self.traindata.treat_scaler, self.traindata.feat_scaler, self.traindata.output_scaler, a=a, change="nbr")
                loader = DataLoader(cfdata, batch_size=4, shuffle=False)

                preds_a = torch.cat(self.trainer.predict(self.model, loader))
                preds_a = preds_a.cpu().numpy()
                preds_a = self.traindata.output_scaler.inverse_transform(preds_a)
                spill.append(preds_a)
            spill = np.concatenate(spill, axis=1)
            spill += resid[:, None]
            
            s = spill.mean(0)
            effects["spill"] = s[1] - s[0]
        

        return effects
    
    def tune_metric(self, dataset: SpaceDataset) -> float:
        self.model.eval()
        loader = DataLoader(self.valdata, batch_size=4, shuffle=False)
        preds = torch.cat(self.trainer.predict(self.model, loader))
        preds = preds.cpu().numpy()

        # scale back the preds
        preds = self.traindata.output_scaler.inverse_transform(preds)[:, 0]
        return np.mean((dataset.full_outcome[self.test_ix] - preds) ** 2)
    
    
    
def spatial_train_test_split_radius(
    graph: nx.Graph,
    init_frac: float,
    levels: int,
    buffer: int = 0,
    seed: int | None = None,
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
    rng = np.random.default_rng(seed)
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
