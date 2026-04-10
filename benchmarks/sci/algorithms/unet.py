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
        # self.model = torch.compile(self.model, mode="reduce-overhead")
        
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



class UNetDataset(Dataset):
    def __init__(self, dataset, nodes, coords2id, radius, treat_scaler=None, feat_scaler=None, output_scaler=None, a=None, change=None, datatype="grid", dataset_radius=None, nbr_off=None):
        
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

        # Scale the counterfactual value `a` to match scaled treatment space
        a_scaled = a
        if a is not None and treat_scaler is not None and nonbinary_treat_cols:
            a_scaled = treat_scaler.transform(np.array([[a]]))[0, 0]

        self.treatments = torch.zeros(len(ids), 2*radius+1, 2*radius+1, 1)

        for ii, n in enumerate(nodes):
            center_coords = id2coords[n]
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    cur_coords = (center_coords[0] + i, center_coords[1] + j)
                    if change == "center" and (i, j) == (0, 0):
                        self.treatments[ii, i + radius, j + radius, 0] = float(a_scaled)
                    elif change == "nbr" and (i, j) != (0, 0):
                        self.treatments[ii, i + radius, j + radius, 0] = float(a_scaled)
                    elif change == "nbr_one" and nbr_off is not None and (i, j) == tuple(nbr_off):
                        self.treatments[ii, i + radius, j + radius, 0] = float(a_scaled)
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
        # true_treatments = self.true_treatment[idx]
        true_outcomes = self.outcomes[idx]
        
        return treatments, covariates, true_outcomes


class U_Net(SpaceAlgo):
    """
    Wrapper of UNet with GPU acceleration support.
    """
    supports_binary = True
    supports_continuous = True
    
    def __init__(self,
        bilinear: bool = False,
        unet_base_chan: int = 16,
        weight_decay: float = 1e-5,
        lr: float = 1e-3,
        device: str = None,
        epochs: int = 50,
        batch_size = None,
    ):
        """
        Initialize Interference-Aware Deconfounder with optional device specification.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        
        
        self.unet_kwargs = {
            "bilinear": bilinear,
            "weight_decay": weight_decay,
            "lr": lr,
            "epochs": epochs,
            "unet_base_chan": unet_base_chan,
        }
                
        self.epochs = epochs
        
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
                
        
        LOGGER.debug("Building dataset and dataloader...")
        graph = nx.from_edgelist(dataset.full_edge_list)
        train_ix, test_ix, _ = spatial_train_test_split_radius(
            graph, 
            init_frac= 0.02, 
            levels = 1, 
            buffer = 1 + dataset.radius, 
            radius = dataset.radius,
        )
        self.train_ix = train_ix
        self.test_ix = test_ix
        
        graph = nx.from_edgelist(dataset.full_edge_list)        
        node_list = list(graph.nodes())
        nbrs = {node: get_k_hop_neighbors(graph, node,  dataset.radius) for node in node_list}
        nbr_counts = {node: len(neigh) for node, neigh in nbrs.items()}
        max_count = max(nbr_counts.values())
        
        self.max_nodes = [node for node, cnt in nbr_counts.items() if cnt == max_count]
        self.max_coords2id = {tuple(coord): i for i, coord in enumerate(dataset.full_coordinates)}
        

        self.unet_kwargs["feature_dim"] = dataset.covariates.shape[1]
        self.unet_kwargs["datatype"] = dataset.datatype
        self.unet_kwargs["radius"] = dataset.conf_radius

        self.head_model = UNetHead(**self.unet_kwargs)

        self.head_traindata = UNetDataset(dataset, self.train_ix, self.max_coords2id, dataset.conf_radius, datatype=dataset.datatype, dataset_radius=dataset.conf_radius)
        self.head_valdata = UNetDataset(dataset, self.test_ix, self.max_coords2id, dataset.conf_radius, self.head_traindata.treat_scaler, self.head_traindata.feat_scaler, self.head_traindata.output_scaler, datatype=dataset.datatype, dataset_radius=dataset.conf_radius)

        loader = DataLoader(self.head_traindata, batch_size=total_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(self.head_valdata, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)


        LOGGER.debug("Preparing trainer...")
        callbacks = [
            ModelCheckpoint(
                dirpath="new_unet_checkpoints/",
                filename="{epoch}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min"
            ),
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar()
        ]

        self.head_trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            enable_checkpointing=True,
            logger=None,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            callbacks=callbacks,
            max_epochs=self.epochs,
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

            from sci.env import PER_NEIGHBOR_SPILLOVER
            if PER_NEIGHBOR_SPILLOVER:
                model_radius = dataset.conf_radius
                nbr_positions = [
                    (dr, dc)
                    for dr in range(-model_radius, model_radius + 1)
                    for dc in range(-model_radius, model_radius + 1)
                    if not (dr == 0 and dc == 0)
                ]
                LOGGER.debug(
                    f"Per-nbr spillover: averaging over {len(nbr_positions)} "
                    f"model neighbor positions"
                )
                per_nbr_effects = []
                for (dr, dc) in nbr_positions:
                    preds_0 = self.predict(
                        dataset, self.max_nodes,
                        a=dataset.treatment_values[0],
                        change="nbr_one", nbr_off=(dr, dc),
                    )[:, 0]
                    preds_1 = self.predict(
                        dataset, self.max_nodes,
                        a=dataset.treatment_values[1],
                        change="nbr_one", nbr_off=(dr, dc),
                    )[:, 0]
                    per_nbr_effects.append(preds_1 - preds_0)
                effects["spill"] = float(np.mean(per_nbr_effects))
            else:
                spill = []
                for a in dataset.treatment_values:
                    preds_a = self.predict(dataset, self.max_nodes, a=a, change="nbr")
                    spill.append(preds_a)
                spill = np.concatenate(spill, axis=1)
                s = spill.mean(0)
                effects["spill"] = s[1] - s[0]

        # Per-pixel counterfactual: annual and summer % increase in SIC
        # from a 5 % LWDN reduction.
        if hasattr(dataset, "cf_annual_treatment") and dataset.cf_annual_treatment is not None:
            LOGGER.debug("Computing annual / summer counterfactual effects...")

            # --- annual ---
            preds_annual_f = self.predict(
                dataset, self.max_nodes, a=None, change=None,
            )
            preds_annual_cf = self.predict(
                dataset, self.max_nodes, a=None, change=None,
                cf_full_treatment=dataset.cf_annual_treatment,
            )
            annual_diff = preds_annual_cf - preds_annual_f
            effects["cf_annual_pct"] = float(
                annual_diff.mean() / np.abs(preds_annual_f).mean() * 100
            )

            # --- summer (JJA) ---
            preds_summer_f = self.predict(
                dataset, self.max_nodes, a=None, change=None,
                cf_full_treatment=dataset.summer_treatment,
            )
            preds_summer_cf = self.predict(
                dataset, self.max_nodes, a=None, change=None,
                cf_full_treatment=dataset.cf_summer_treatment,
            )
            summer_diff = preds_summer_cf - preds_summer_f
            effects["cf_summer_pct"] = float(
                summer_diff.mean() / np.abs(preds_summer_f).mean() * 100
            )

        return effects
    
    def tune_metric(self, dataset: SpaceDataset) -> float:
        preds = self.predict(dataset, self.max_nodes, a=None, change=None)[:, 0]
        return np.mean((dataset.full_outcome[self.max_nodes] - preds) ** 2)
    
    
    def predict(self, dataset: SpaceDataset, nodes, a, change,
                cf_full_treatment=None, nbr_off=None) -> dict:
        """
        Get outcome predictions with GPU acceleration for large datasets.

        Parameters
        ----------
        cf_full_treatment : np.ndarray | None
            If provided, uses this as ``full_treatment`` for patch
            construction instead of the observed values.
        """
        self.head_model.eval()
        if cf_full_treatment is not None:
            head_dataset = deepcopy(dataset)
            head_dataset.full_treatment = cf_full_treatment
            head_a, head_change = None, None
        else:
            head_dataset = dataset
            head_a, head_change = a, change
        predict_head_data = UNetDataset(head_dataset, nodes, self.max_coords2id, dataset.conf_radius, self.head_traindata.treat_scaler, self.head_traindata.feat_scaler, self.head_traindata.output_scaler, a=head_a, change=head_change, datatype=dataset.datatype, dataset_radius=dataset.conf_radius, nbr_off=nbr_off)
        loader = DataLoader(predict_head_data, batch_size=total_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        preds = torch.cat(self.head_trainer.predict(self.head_model, loader))
        preds = preds.cpu().numpy()
        # scale back the preds
        preds = self.head_traindata.output_scaler.inverse_transform(preds)
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


# ======================================================================
# Temporal UNet — operates on full (C, H, W) spatial maps per timestep
# ======================================================================

from .utils import Down, Up, DoubleConv, OutConv


class ImageUNet(nn.Module):
    """Standard encoder-decoder UNet for image-to-image prediction.

    Takes (B, n_channels, H, W) → (B, 1, H, W).
    Handles non-power-of-2 spatial sizes via reflect-padding.
    """

    def __init__(self, n_channels: int, base_channels: int = 32, n_downs: int = 3, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_downs = n_downs

        self.inc = DoubleConv(n_channels, base_channels, radius=1)

        self.downs = nn.ModuleList()
        ch = base_channels
        self.enc_channels = [ch]
        for _ in range(n_downs):
            out_ch = min(ch * 2, 256)
            self.downs.append(Down(ch, out_ch))
            self.enc_channels.append(out_ch)
            ch = out_ch

        self.ups = nn.ModuleList()
        for i in range(n_downs):
            skip_ch = self.enc_channels[-(i + 2)]
            if bilinear:
                in_ch = ch + skip_ch
                out_ch = skip_ch if i < n_downs - 1 else base_channels
                up = Up.__new__(Up)
                nn.Module.__init__(up)
                up.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                up.conv = DoubleConv(in_ch, out_ch, radius=1)
            else:
                half_ch = ch // 2
                in_ch = half_ch + skip_ch
                out_ch = skip_ch if i < n_downs - 1 else base_channels
                up = Up.__new__(Up)
                nn.Module.__init__(up)
                up.up = nn.ConvTranspose2d(ch, half_ch, kernel_size=2, stride=2)
                up.conv = DoubleConv(in_ch, out_ch, radius=1)
            self.ups.append(up)
            ch = out_ch

        self.outc = OutConv(base_channels, 1)

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]
        divisor = 2 ** self.n_downs
        pad_h = (divisor - orig_h % divisor) % divisor
        pad_w = (divisor - orig_w % divisor) % divisor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        feats = [self.inc(x)]
        for d in self.downs:
            feats.append(d(feats[-1]))

        x = feats[-1]
        for i, up in enumerate(self.ups):
            x = up(x, feats[-(i + 2)])

        x = self.outc(x)
        return x[:, :, :orig_h, :orig_w]


class TemporalUNetHead(pl.LightningModule):
    """PL wrapper for image-to-image UNet on temporal data."""

    def __init__(
        self,
        n_channels: int,
        base_channels: int = 32,
        n_downs: int = 3,
        bilinear: bool = True,
        weight_decay: float = 1e-5,
        lr: float = 1e-3,
        epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ImageUNet(n_channels, base_channels, n_downs, bilinear)
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        x, y, mask = batch
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        pred = self.model(x).squeeze(1)  # (B, H, W)
        loss = F.mse_loss(pred[mask.bool()], y[mask.bool()])
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        pred = self.model(x).squeeze(1)
        loss = F.mse_loss(pred[mask.bool()], y[mask.bool()])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


class TemporalUNetDataset(Dataset):
    """Dataset of (X[t], Y[t], valid_mask) tuples for temporal UNet.

    Each sample is one timestep: X[t] is (C, H, W), Y[t] is (H, W).

    Parameters
    ----------
    X : np.ndarray, shape (N_time, C, H, W)
    Y : np.ndarray, shape (N_time, H, W)
    valid_mask : np.ndarray, shape (H, W), bool
    indices : array-like of ints — which timesteps to include
    """

    def __init__(self, X, Y, valid_mask, indices):
        self.X = torch.from_numpy(X[indices]).float()
        self.Y = torch.from_numpy(Y[indices]).float()
        self.mask = torch.from_numpy(valid_mask).float().unsqueeze(0).expand(len(indices), -1, -1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.mask[idx]


class TemporalU_Net(SpaceAlgo):
    """Standalone temporal UNet for Arctic-style datasets.

    Trains on full spatial maps per timestep: X[t] (15, H, W) → Y[t] (H, W).
    Each example is one timestep — no sequence modeling.
    """
    supports_binary = False
    supports_continuous = True

    def __init__(
        self,
        base_channels: int = 32,
        n_downs: int = 3,
        bilinear: bool = True,
        weight_decay: float = 1e-5,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 8,
        device: str = None,
    ):
        self.model_kwargs = dict(
            base_channels=base_channels,
            n_downs=n_downs,
            bilinear=bilinear,
            weight_decay=weight_decay,
            lr=lr,
            epochs=epochs,
        )
        self.epochs = epochs
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def fit(self, dataset: SpaceDataset):
        os.environ["PYTORCH_LIGHTNING_DEBUG"] = "1"

        X, Y = dataset.X, dataset.Y
        valid_mask = dataset.valid_mask
        n_channels = X.shape[1]

        train_ds = TemporalUNetDataset(X, Y, valid_mask, dataset.train_idx)
        val_ds   = TemporalUNetDataset(X, Y, valid_mask, dataset.val_idx)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        self.head = TemporalUNetHead(n_channels=n_channels, **self.model_kwargs)

        callbacks = [
            ModelCheckpoint(dirpath="temporal_unet_ckpt/", monitor="val_loss", mode="min", save_top_k=3),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ]

        self.trainer = pl.Trainer(
            accelerator="auto", devices=1,
            enable_checkpointing=True, logger=None,
            gradient_clip_val=1.0, enable_progress_bar=False,
            callbacks=callbacks, max_epochs=self.epochs,
            deterministic=True,
        )

        LOGGER.debug("Training temporal UNet...")
        self.trainer.fit(self.head, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best = self.trainer.checkpoint_callback.best_model_path
        if best:
            ckpt = torch.load(best, map_location=self.head.device)
            self.head.load_state_dict(ckpt['state_dict'])
            del ckpt
            torch.cuda.empty_cache()
        LOGGER.debug("Finished training temporal UNet.")

    def _predict_maps(self, X_input, valid_mask):
        """Run UNet on (N, C, H, W) input maps.  Returns (N, H, W) predictions."""
        self.head.eval()
        dummy_Y = np.zeros((X_input.shape[0], X_input.shape[2], X_input.shape[3]), dtype=np.float32)
        ds = TemporalUNetDataset.__new__(TemporalUNetDataset)
        ds.X = torch.from_numpy(X_input).float()
        ds.Y = torch.from_numpy(dummy_Y).float()
        ds.mask = torch.from_numpy(valid_mask).float().unsqueeze(0).expand(len(X_input), -1, -1)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        preds = torch.cat(self.trainer.predict(self.head, loader)).squeeze(1).cpu().numpy()
        return preds  # (N, H, W)

    def eval(self, dataset: SpaceDataset) -> dict:
        effects = {}

        # The dataset has already been subset to max_nodes in env.make(),
        # so dataset.coordinates gives the (row, col) of each node we
        # need to report on.
        node_rows = dataset.coordinates[:, 0]
        node_cols = dataset.coordinates[:, 1]

        # ERF over treatment quantiles — per pixel (averaged over time)
        ite = []
        for a in dataset.treatment_values:
            X_a = dataset.X.copy()
            X_a[:, 1, :, :] = (a - dataset.lwdn_mu) / dataset.lwdn_sd
            X_a[:, 1, :, :][..., ~dataset.valid_mask] = 0.0
            preds_a = self._predict_maps(X_a, dataset.valid_mask)  # (T-1, H, W)
            avg_a = preds_a.mean(axis=0)  # (H, W) — average over time
            per_node = avg_a[node_rows, node_cols]  # (n_nodes,)
            ite.append(per_node.reshape(-1, 1))
        ite = np.concatenate(ite, axis=1)  # (n_nodes, n_treatment_values)
        effects["erf"] = ite.mean(0)
        effects["ite"] = ite

        # Counterfactual: 5% LWDN reduction
        # Predictions are in standardized SIC space — inverse-standardize
        # before computing % change so the baseline is in real units.
        sic_mu, sic_sd = dataset.sic_mu, dataset.sic_sd
        def to_raw(p):
            return p * sic_sd + sic_mu

        if hasattr(dataset, "X_cf"):
            LOGGER.debug("Computing annual / summer counterfactual effects...")
            preds_f  = to_raw(self._predict_maps(dataset.X_factual, dataset.valid_mask))
            preds_cf = to_raw(self._predict_maps(dataset.X_cf, dataset.valid_mask))

            diff_annual = (preds_cf - preds_f)[:, dataset.valid_mask]
            base_annual = np.abs(preds_f[:, dataset.valid_mask])
            effects["cf_annual_pct"] = float(diff_annual.mean() / base_annual.mean() * 100)

            jja = dataset.jja_mask
            diff_summer = (preds_cf[jja] - preds_f[jja])[:, dataset.valid_mask]
            base_summer = np.abs(preds_f[jja][:, dataset.valid_mask])
            effects["cf_summer_pct"] = float(diff_summer.mean() / base_summer.mean() * 100)

        # Counterfactual: +18 W/m² LWDN increase
        if hasattr(dataset, "X_cf_plus18"):
            LOGGER.debug("Computing +18 LWDN counterfactual effects...")
            preds_f18  = to_raw(self._predict_maps(dataset.X_factual, dataset.valid_mask))
            preds_cf18 = to_raw(self._predict_maps(dataset.X_cf_plus18, dataset.valid_mask))

            diff18 = (preds_cf18 - preds_f18)[:, dataset.valid_mask]
            base18 = np.abs(preds_f18[:, dataset.valid_mask])
            effects["cf_plus18_annual_pct"] = float(diff18.mean() / base18.mean() * 100)

            jja = dataset.jja_mask
            diff18_s = (preds_cf18[jja] - preds_f18[jja])[:, dataset.valid_mask]
            base18_s = np.abs(preds_f18[jja][:, dataset.valid_mask])
            effects["cf_plus18_summer_pct"] = float(diff18_s.mean() / base18_s.mean() * 100)

        return effects

    def tune_metric(self, dataset: SpaceDataset) -> float:
        preds = self._predict_maps(dataset.X[dataset.test_idx], dataset.valid_mask)
        Y_true = dataset.Y[dataset.test_idx]
        mask = dataset.valid_mask
        return float(np.mean((preds[:, mask] - Y_true[:, mask]) ** 2))