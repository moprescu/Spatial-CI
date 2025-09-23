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

            spill = []
            for a in dataset.treatment_values:
                preds_a = self.predict(dataset, self.max_nodes, a=a, change="nbr")
                spill.append(preds_a)
            spill = np.concatenate(spill, axis=1)

            s = spill.mean(0)
            effects["spill"] = s[1] - s[0]
        

        return effects
    
    def tune_metric(self, dataset: SpaceDataset) -> float:
        preds = self.predict(dataset, self.max_nodes, a=None, change=None)[:, 0]
        return np.mean((dataset.full_outcome[self.max_nodes] - preds) ** 2)
    
    
    def predict(self, dataset: SpaceDataset, nodes, a, change) -> dict:
        """
        Get outcome predictions with GPU acceleration for large datasets.
        """
        self.head_model.eval()
        predict_head_data = UNetDataset(dataset, nodes, self.max_coords2id, dataset.conf_radius, self.head_traindata.treat_scaler, self.head_traindata.feat_scaler, self.head_traindata.output_scaler, a=a, change=change, datatype=dataset.datatype, dataset_radius=dataset.conf_radius)
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