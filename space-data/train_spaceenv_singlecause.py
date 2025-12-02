import logging
import os
from glob import glob
import tarfile
import math
import sys

import re
import tempfile
import hydra
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from autogluon.multimodal import MultiModalPredictor
from autogluon.tabular import TabularDataset, TabularPredictor

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from scipy.interpolate import BSpline
import plotly.express as px
import utils
from sklearn.metrics import r2_score
import PIL
from io import BytesIO
# from models.gnn import CustomGCN

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateFinder
from torch.optim import Adam
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import k_hop_subgraph

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.common.utils.resource_utils import ResourceManager
import dill

gnn_epochs = 100

def make_predict_with_fixed_index(model):
    """
    Returns a predict function that ensures the index of incoming data
    matches the original baseline index, repeated as necessary.
    Splits predictions based on baseline length and appends them together.
    """
    baseline_index = None  # Will store original index from first call
    baseline_len = None    # Number of rows in baseline data
    
    def predict_with_fixed_index(X_transformed, *args, **kwargs):
        nonlocal baseline_index, baseline_len
        
        if baseline_index is None:
            # First call (baseline)
            baseline_index = X_transformed.index.copy()
            baseline_len = len(baseline_index)
            return model.predict(X_transformed, *args, **kwargs)
        else:
            # Later calls (shuffled / stacked)
            if len(X_transformed) % baseline_len != 0:
                raise ValueError(
                    f"X_transformed length {len(X_transformed)} is not a multiple of baseline length {baseline_len}"
                )
            
            n_repeat = len(X_transformed) // baseline_len
            all_predictions = []
            
            # Split data into chunks of baseline_len and process each
            for i in range(n_repeat):
                start_idx = i * baseline_len
                end_idx = (i + 1) * baseline_len
                
                # Get chunk and fix its index
                chunk = X_transformed.iloc[start_idx:end_idx].copy()
                chunk.index = baseline_index
                
                # Get prediction for this chunk
                chunk_pred = model.predict(chunk, *args, **kwargs)
                all_predictions.append(chunk_pred)
            
            # Concatenate all predictions
            if isinstance(all_predictions[0], np.ndarray):
                return np.concatenate(all_predictions, axis=0)
            elif isinstance(all_predictions[0], pd.Series):
                return pd.concat(all_predictions, ignore_index=True)
            elif isinstance(all_predictions[0], pd.DataFrame):
                return pd.concat(all_predictions, ignore_index=True)
            else:
                # For other types, try to concatenate as list
                result = []
                for pred in all_predictions:
                    if hasattr(pred, '__iter__') and not isinstance(pred, str):
                        result.extend(pred)
                    else:
                        result.append(pred)
                return result
    
    return predict_with_fixed_index

def graph_data_loader(
    X, y=None, edge_list=None,
    feat_scaler: StandardScaler | None = None,
    output_scaler: StandardScaler | None = None,
    treatment_value: float | None = None,
    radius=1,
    task_type="regression",
):
    if edge_list is None:
        raise ValueError("edge_list cannot be None")

    node_ids = set(X.index)
    
    # Filter edges to only valid nodes
    filtered_edge_list = [(s, t) for s, t in edge_list if s in node_ids and t in node_ids]
    if len(filtered_edge_list) == 0:
        raise ValueError(f"No edges left after filtering by node_ids. Check edge_list and X.index consistency.\n"
                f"X.index: {X.index.tolist()}\n"
                f"node_ids: {node_ids}\n"
                f"edge_list: {edge_list}\n"
                f"edge_list type: {type(edge_list)}\n"
                f"edge_list shape/length: {edge_list.shape if hasattr(edge_list, 'shape') else len(edge_list)}")
    
    # Map node IDs to consecutive integer positions
    # Suggest sorting node_ids to keep consistent ordering
    sorted_node_ids = sorted(node_ids)
    node_id_to_pos = {nid: i for i, nid in enumerate(sorted_node_ids)}

    # Check all node_ids in filtered_edge_list appear in mapping
    for s, t in filtered_edge_list:
        if s not in node_id_to_pos or t not in node_id_to_pos:
            raise ValueError(f"Edge node {s} or {t} not found in node_id_to_pos mapping")
    
    # Map edges using this mapping
    try:
        edge_index_mapped = torch.LongTensor([
            [node_id_to_pos[s] for s, t in filtered_edge_list],
            [node_id_to_pos[t] for s, t in filtered_edge_list]
        ])
    except KeyError as e:
        raise KeyError(f"Node ID not found in mapping: {e}")

    # Check min and max indices of edge_index
    min_idx = edge_index_mapped.min().item()
    max_idx = edge_index_mapped.max().item()
    
    if min_idx < 0:
        raise ValueError(f"Edge index contains negative values: min index = {min_idx}")
    
    num_nodes = len(sorted_node_ids)
    if max_idx >= num_nodes:
        raise ValueError(f"Edge index max ({max_idx}) >= number of nodes ({num_nodes})")

    if y is None:
        outcome = np.full((len(sorted_node_ids), 1), 0)
    else:
        # Make sure y's index matches sorted_node_ids order
        try:
            outcome = y.loc[sorted_node_ids].values.reshape(-1, 1)
        except KeyError as e:
            raise KeyError(f"y is missing values for nodes: {e}")

    features = X.loc[sorted_node_ids].values  # ensure numpy array for scaler
    
    if feat_scaler is None:
        feat_scaler = StandardScaler()
        feat_scaler.fit(features)
    
    if output_scaler is None:
        output_scaler = StandardScaler()
        output_scaler.fit(outcome)
    
    x = torch.FloatTensor(feat_scaler.transform(features))
    if task_type == "regression":
        y_tensor = torch.FloatTensor(output_scaler.transform(outcome))
    else:
        y_tensor = torch.FloatTensor(outcome)
    
    # CRITICAL FIX: Make sure data.num_nodes is explicitly set and consistent
    data = Data(x=x, y=y_tensor, edge_index=edge_index_mapped, num_nodes=num_nodes)

    # Add additional validation
    assert edge_index_mapped.max().item() < x.size(0), (
        f"edge_index max {edge_index_mapped.max().item()} is not less than x.size(0)={x.size(0)}"
    )
    assert num_nodes == x.size(0), (
        f"num_nodes ({num_nodes}) must equal number of node features ({x.size(0)})"
    )
    
    # CRITICAL: Validate edge_index doesn't reference non-existent nodes
    if edge_index_mapped.numel() > 0:  # Only check if edges exist
        assert edge_index_mapped.max().item() < num_nodes, (
            f"Maximum edge index ({edge_index_mapped.max().item()}) must be < num_nodes ({num_nodes})"
        )
    
    dataset = EgoGraphDataset(data, radius)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    return loader, feat_scaler, output_scaler


class EgoGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data, radius):
        self.data = data
        self.radius = radius
        
        # CRITICAL FIX: Ensure num_nodes is set correctly and consistently
        if not hasattr(data, 'num_nodes') or data.num_nodes is None:
            if data.x is not None:
                self.num_nodes = data.x.size(0)
                # Update the data object to have consistent num_nodes
                self.data.num_nodes = self.num_nodes
            else:
                raise ValueError("Data object has no num_nodes and no x features to infer from")
        else:
            self.num_nodes = data.num_nodes
            # Ensure consistency between data.num_nodes and actual feature matrix size
            if data.x is not None and self.num_nodes != data.x.size(0):
                raise ValueError(f"Inconsistent num_nodes: data.num_nodes={self.num_nodes}, x.size(0)={data.x.size(0)}")
    
    def __len__(self):
        return self.num_nodes
    
    def __getitem__(self, idx):
        if not (0 <= idx < self.num_nodes):
            raise IndexError(f"Index {idx} out of bounds for num_nodes {self.num_nodes}")

        # CRITICAL: Additional validation before calling k_hop_subgraph
        if self.data.edge_index.numel() > 0:  # Only validate if edges exist
            if self.data.edge_index.min().item() < 0:
                raise ValueError("Edge index contains negative values")
            if self.data.edge_index.max().item() >= self.num_nodes:
                raise ValueError(f"Edge index max {self.data.edge_index.max().item()} >= num_nodes {self.num_nodes}")
        
        # CRITICAL FIX: Pass num_nodes explicitly to k_hop_subgraph
        try:
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                idx, 
                self.radius, 
                self.data.edge_index, 
                relabel_nodes=True,
                num_nodes=self.num_nodes  # CRITICAL: Explicitly pass num_nodes
            )
        except IndexError as e:
            raise IndexError(f"k_hop_subgraph failed for node {idx} with num_nodes={self.num_nodes}: {e}")

        # Validate subset indices
        if subset.numel() > 0:  # Only validate if subset is not empty
            if subset.max().item() >= self.num_nodes or subset.min().item() < 0:
                raise ValueError(f"Subset nodes indices out of bounds: min {subset.min().item()}, max {subset.max().item()} with num_nodes {self.num_nodes}")

        sub_x = self.data.x[subset]
        sub_y = None
        if hasattr(self.data, 'y') and self.data.y is not None:
            sub_y = self.data.y[subset]
        
        sub_data = Data(x=sub_x, edge_index=sub_edge_index)
        if sub_y is not None:
            sub_data.y = sub_y
        
        sub_data.center_node = mapping
        sub_data.center_node_idx = idx
        
        return sub_data



class _GCN_impl(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,  # Increased from 16
        hidden_layers: int = 3,  # Increased from 2
        output_dim: int = 1,
        dropout: float = 0.3,  # Increased from 0.0
        lr: float = 0.01,  # Increased from 0.001
        weight_decay: float = 5e-4,  # Standard GCN weight decay
        act="relu",
        task_type: str = "regression",
        mlp_hidden_dim: int = None,
        mlp_layers: int = 2,
        use_batch_norm: bool = True,  # Add batch norm
        use_residual: bool = True,    # Add residual connections
        trainfeat_scaler = None,
        trainoutput_scaler = None,
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['trainfeat_scaler', 'trainoutput_scaler'])
        
        # GCN layers with batch norm
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        
        self.convh = nn.ModuleList()
        self.bnh = nn.ModuleList()
        
        for i in range(hidden_layers - 1):
            self.convh.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.bnh.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.bnh.append(nn.Identity())
        
        # MLP head
        if mlp_hidden_dim is None:
            mlp_hidden_dim = hidden_dim * 2  # Make MLP wider than GCN
        
        self.mlp_layers = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        # Build MLP
        if mlp_layers == 1:
            # Single linear layer
            self.mlp_layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            # First MLP layer
            self.mlp_layers.append(nn.Linear(hidden_dim, mlp_hidden_dim))
            if use_batch_norm:
                self.mlp_bns.append(nn.BatchNorm1d(mlp_hidden_dim))
            else:
                self.mlp_bns.append(nn.Identity())
            
            # Hidden MLP layers
            for _ in range(mlp_layers - 2):
                self.mlp_layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
                if use_batch_norm:
                    self.mlp_bns.append(nn.BatchNorm1d(mlp_hidden_dim))
                else:
                    self.mlp_bns.append(nn.Identity())
            
            # Final output layer
            self.mlp_layers.append(nn.Linear(mlp_hidden_dim, output_dim))
        
        self.act = getattr(F, act)
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.task_type = task_type
        self.output_dim = output_dim
        self.use_residual = use_residual and hidden_layers > 1
        self.use_batch_norm = use_batch_norm
        self.trainfeat_scaler = trainfeat_scaler
        self.trainoutput_scaler = trainoutput_scaler
        self.trainer = None
        self.edge_list = None
        self.radius = None

    def forward(self, batch: torch_geometric.data.Data):
        x = batch.x
        edge_index = batch.edge_index
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.act(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store for potential residual connection
        if self.use_residual:
            residual = x
            
        # Hidden GCN layers
        for i, (conv, bn) in enumerate(zip(self.convh, self.bnh)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = self.act(x_new)
            
            # Add residual connection every 2 layers
            if self.use_residual and i > 0 and (i + 1) % 2 == 0:
                x_new = x_new + residual
                residual = x_new
            elif self.use_residual and (i + 1) % 2 == 1:
                residual = x_new
            
            if self.dropout > 0:
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            x = x_new
        
        # MLP head
        if len(self.mlp_layers) == 1:
            # Single layer case
            x = self.mlp_layers[0](x)
        else:
            # Multi-layer MLP
            for i, layer in enumerate(self.mlp_layers[:-1]):
                x = layer(x)
                if i < len(self.mlp_bns):
                    x = self.mlp_bns[i](x)
                x = self.act(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Final output layer (no activation yet)
            x = self.mlp_layers[-1](x)
        
        # Apply final activation based on task
        if self.task_type == "classification":
            if self.output_dim == 1:
                x = torch.sigmoid(x)
            else:
                x = F.log_softmax(x, dim=1)
        
        return x

    def training_step(self, batch):
        y_hat = self(batch)
        
        if self.task_type == "classification":
            if self.output_dim == 1:
                loss = F.binary_cross_entropy(y_hat.squeeze(), batch.y.squeeze().float())
            else:
                loss = F.nll_loss(y_hat, batch.y.long().squeeze())
        else:
            loss = F.mse_loss(y_hat.squeeze(), batch.y.squeeze())
            
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        
        if self.task_type == "classification":
            if self.output_dim == 1:
                loss = F.binary_cross_entropy(y_hat.squeeze(), batch.y.squeeze().float())
                # Calculate accuracy
                preds = (y_hat.squeeze() > 0.5).float()
                acc = (preds == batch.y.squeeze().float()).float().mean()
                self.log('val_acc', acc, prog_bar=True)
            else:
                loss = F.nll_loss(y_hat, batch.y.long().squeeze())
                preds = y_hat.argmax(dim=1)
                acc = (preds == batch.y.long().squeeze()).float().mean()
                self.log('val_acc', acc, prog_bar=True)
        else:
            loss = F.mse_loss(y_hat.squeeze(), batch.y.squeeze())
            
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
    
    def predict(self, dataset):
        dataset = dataset.loc[:, ~dataset.columns.str.contains('_nbr')]
        
        loader, *_ = graph_data_loader(X=dataset, edge_list=self.edge_list, feat_scaler=self.trainfeat_scaler, output_scaler=self.trainoutput_scaler, radius=self.radius, task_type=self.task_type)
        preds = self.trainer.predict(self, loader)
        preds = torch.cat([pred[0:1] for pred in preds]).flatten().reshape(-1, 1)
        preds = preds.cpu().numpy()
        if self.task_type != "classification":
            preds = self.trainoutput_scaler.inverse_transform(preds)
        return preds.flatten()
    
    def predict_proba(self, dataset):
        dataset = dataset.loc[:, ~dataset.columns.str.contains('_nbr')]
        
        loader, *_ = graph_data_loader(X=dataset, edge_list=self.edge_list, feat_scaler=self.trainfeat_scaler, output_scaler=self.trainoutput_scaler, radius=self.radius, task_type=self.task_type)
        preds = self.trainer.predict(self, loader)
        preds = torch.cat([pred[0:1] for pred in preds]).flatten().reshape(-1, 1)
        preds = preds.cpu().numpy()
        if self.task_type != "classification":
            preds = self.trainoutput_scaler.inverse_transform(preds)
        return preds.flatten()

                
class CustomGCN(AbstractModel):
    def __init__(self, **kwargs):
        # Simply pass along kwargs to parent, and init our internal `_feature_generator` variable to None
        super().__init__(**kwargs)
        self._feature_generator = None

    # The `_preprocess` method takes the input data and transforms it to the internal representation usable by the model.
    # `_preprocess` is called by `preprocess` and is used during model fit and model inference.
    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)

        if is_train:
            # X will be the training data.
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            # This converts categorical features to numeric via stateful label encoding.
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        # Add a fillna call to handle missing values.
        # Some algorithms will be able to handle NaN values internally (LightGBM).
        # In those cases, you can simply pass the NaN values into the inner model.
        # Finally, convert to numpy for optimized memory usage and because sklearn RF works with raw numpy input.
        return X.fillna(0)

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(self,
             X: pd.DataFrame,  # training data
             y: pd.Series,  # training labels
             X_val: pd.DataFrame,  # val data (unused in RF model)
             y_val: pd.Series,  # val labels (unused in RF model)
             # time_limit=None,  # time limit in seconds (ignored in tutorial)
             **kwargs):  # kwargs includes many other potential inputs, refer to AbstractModel documentation for details
        print('Entering the `_fit` method')
        # Make sure to call preprocess on X near the start of `_fit`.
        # This is necessary because the data is converted via preprocess during predict, and needs to be in the same format as during fit.
        X = self.preprocess(X, is_train=True)
        X_val = self.preprocess(X_val, is_train=False)
        
        params = self._get_model_params()
        self.edge_list = params.pop("edge_list", None)
        self.radius = params.pop("radius", None)
        
        X = X.loc[:, ~X.columns.str.contains('_nbr')]
        X_val = X_val.loc[:, ~X_val.columns.str.contains('_nbr')]

        input_dim = X.shape[1]
        
        trainloader, self.trainfeat_scaler, self.trainoutput_scaler = graph_data_loader(X=X, y=y, edge_list=self.edge_list, radius=self.radius, task_type=self.problem_type)
        valloader = None
        if X_val is not None and y_val is not None:
            print("Using provided validation data")
            valloader, _, _ = graph_data_loader(
                X=X_val, y=y_val, edge_list=self.edge_list,
                feat_scaler=self.trainfeat_scaler,  # Use same scaler as training
                output_scaler=self.trainoutput_scaler,
                radius=self.radius,
                task_type=self.problem_type,
            )
        else:
            print("No validation data provided")



        # Valid self.problem_type values include ['binary', 'multiclass', 'regression', 'quantile', 'softclass']
        if self.problem_type in ['regression']:
            task_type="regression"
        elif self.problem_type in ['binary', 'multiclass']:
            task_type="classification"

        
        # This fetches the user-specified (and default) hyperparameters for the model.
        # params = self._get_model_params()
        print(f'Hyperparameters: {params}')
        # self.model should be set to the trained inner model, so that internally during predict we can call `self.model.predict(...)`
        if self.model is None:
            self.model_init_args = {"input_dim": input_dim, "task_type": task_type, **params, "trainfeat_scaler": self.trainfeat_scaler, "trainoutput_scaler": self.trainoutput_scaler}
            self.model = _GCN_impl(**self.model_init_args)
        self.model.edge_list = self.edge_list
        self.model.preprocess = self.preprocess
        
        self.model.radius = self.radius
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                verbose=True
            ),
            ModelCheckpoint(
                monitor='val_loss',
                save_top_k=1,
                mode='min'
            )
        ]

        self.trainer = pl.Trainer(
            accelerator="gpu",
            enable_checkpointing=True,
            logger=False,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            callbacks=callbacks,
            max_epochs=gnn_epochs,
            enable_model_summary=True,
            deterministic=True,
        )

        print("Training GCN model...")
        if valloader != None:
            self.trainer.fit(self.model, trainloader, valloader)
        else:
            self.trainer.fit(self.model, trainloader)
        self.model.trainer = self.trainer
        print("Finished training GCN model.")

    # The `_set_default_params` method defines the default hyperparameters of the model.
    # User-specified parameters will override these values on a key-by-key basis.
    def _set_default_params(self):
        default_params = {
            "hidden_dim": 128,        # Increased
            "hidden_layers": 3,       # Increased
            "mlp_hidden_dim": 256,    # Wider MLP
            "mlp_layers": 3,          # Deeper MLP
            "dropout": 0.3,           # Increased
            "lr": 1e-3,               # Increased
            "weight_decay": 5e-4,     # Standard GCN value
            "use_batch_norm": False,   # Enable by default
            "use_residual": True,     # Enable by default
            "edge_list": None,
            "radius": None,            
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # The `_get_default_auxiliary_params` method defines various model-agnostic parameters such as maximum memory usage and valid input column dtypes.
    # For most users who build custom models, they will only need to specify the valid/invalid dtypes to the model here.
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            # the total set of raw dtypes are: ['int', 'float', 'category', 'object', 'datetime']
            # object feature dtypes include raw text and image paths, which should only be handled by specialized models
            # datetime raw dtypes are generally converted to int in upstream pre-processing,
            # so models generally shouldn't need to explicitly support datetime dtypes.
            valid_raw_types=['int' , 'float', 'category'],
            # Other options include `valid_special_types`, `ignored_type_group_raw`, and `ignored_type_group_special`.
            # Refer to AbstractModel for more details on available options.
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
    
    def get_minimum_resources(self, is_gpu_available=False):
        minimum_resources = {
            "num_cpus": 4,
        }
        if is_gpu_available:
            minimum_resources["num_gpus"] = 1
        return minimum_resources

    def _get_default_resources(self):
        # only_physical_cores=True is faster in training
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = 1
        return num_cpus, num_gpus
    
    def save(self, path: str | None = None, verbose: bool = True) -> str:
        """Save the entire CustomGCN class instance with special handling for non-serializable components"""
        if path is None:
            path = self.path
        os.makedirs(path, exist_ok=True)

        # Store references to components that can't be serialized
        backup_components = {}
        
        # List of potentially problematic attributes to remove temporarily
        problematic_attrs = [
            'model', 'trainer', 'trainfeat_scaler', 'trainoutput_scaler',
            '_feature_generator'  # This might also have serialization issues
        ]
        
        # Temporarily remove non-serializable components
        for attr in problematic_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                backup_components[attr] = getattr(self, attr)
                setattr(self, attr, None)

        try:
            # Try to save the main class instance (without problematic components)
            main_instance_path = os.path.join(path, "custom_gcn_main.pkl")
            
            # Create a copy of self.__dict__ to save only serializable attributes
            serializable_state = {}
            for key, value in self.__dict__.items():
                try:
                    # Test if the attribute can be serialized
                    dill.dumps(value)
                    serializable_state[key] = value
                except Exception:
                    if verbose:
                        print(f"Skipping non-serializable attribute: {key}")
                    continue
            
            with open(main_instance_path, "wb") as f:
                dill.dump(serializable_state, f)
            
            if verbose:
                print(f"Main CustomGCN state saved to {main_instance_path}")

        except Exception as e:
            if verbose:
                print(f"Error saving main instance: {e}")
        finally:
            # Always restore the components, regardless of success or failure
            for attr, value in backup_components.items():
                setattr(self, attr, value)

        # Save PyTorch model separately
        if 'model' in backup_components and backup_components['model'] is not None:
            model_backup = backup_components['model']
            model_state_path = os.path.join(path, "model_state_dict.pth")
            torch.save(model_backup.state_dict(), model_state_path)
            
            # Save model initialization arguments
            if hasattr(self, 'model_init_args'):
                model_args_path = os.path.join(path, "model_init_args.pkl")
                with open(model_args_path, "wb") as f:
                    dill.dump(self.model_init_args, f)
            
            if verbose:
                print(f"Model state dict saved to {model_state_path}")
                if hasattr(self, 'model_init_args'):
                    print(f"Model init args saved to {model_args_path}")

        # Save trainer configuration (not the trainer itself, just its config)
        if 'trainer' in backup_components and backup_components['trainer'] is not None:
            trainer_backup = backup_components['trainer']
            trainer_config = {
                'accelerator': getattr(trainer_backup, 'accelerator', 'gpu'),
                'max_epochs': getattr(trainer_backup, 'max_epochs', gnn_epochs),
                'gradient_clip_val': getattr(trainer_backup, 'gradient_clip_val', 10.0),
                # Add other relevant trainer configs as needed
            }
            trainer_config_path = os.path.join(path, "trainer_config.pkl")
            with open(trainer_config_path, "wb") as f:
                dill.dump(trainer_config, f)
            
            if verbose:
                print(f"Trainer configuration saved to {trainer_config_path}")

        # Save scalers separately
        scalers_to_save = {}
        for scaler_name in ['trainfeat_scaler', 'trainoutput_scaler']:
            if scaler_name in backup_components and backup_components[scaler_name] is not None:
                scalers_to_save[scaler_name] = backup_components[scaler_name]
            
        if scalers_to_save:
            scalers_path = os.path.join(path, "scalers.pkl")
            try:
                with open(scalers_path, "wb") as f:
                    dill.dump(scalers_to_save, f)
                if verbose:
                    print(f"Scalers saved to {scalers_path}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not save scalers: {e}")

        # Save feature generator separately
        if '_feature_generator' in backup_components and backup_components['_feature_generator'] is not None:
            feature_gen_path = os.path.join(path, "feature_generator.pkl")
            try:
                with open(feature_gen_path, "wb") as f:
                    dill.dump(backup_components['_feature_generator'], f)
                if verbose:
                    print(f"Feature generator saved to {feature_gen_path}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not save feature generator: {e}")
        
        return path
    
    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """Load the entire CustomGCN class instance with special handling for non-serializable components"""
        
        # Create a new instance first
        loaded_instance = cls()
        
        # Load the main class state
        main_instance_path = os.path.join(path, "custom_gcn_main.pkl")
        if os.path.exists(main_instance_path):
            try:
                with open(main_instance_path, "rb") as f:
                    serializable_state = dill.load(f)
                
                # Restore the serializable state to the instance
                for key, value in serializable_state.items():
                    setattr(loaded_instance, key, value)
                
                if verbose:
                    print(f"Main CustomGCN state loaded from {main_instance_path}")
                    
            except Exception as e:
                if verbose:
                    print(f"Failed to load main instance state: {e}")

        # Load PyTorch model if it exists
        model_state_path = os.path.join(path, "model_state_dict.pth")
        model_args_path = os.path.join(path, "model_init_args.pkl")
        
        if os.path.exists(model_state_path) and os.path.exists(model_args_path):
            try:
                # Load model initialization arguments
                with open(model_args_path, "rb") as f:
                    model_init_args = dill.load(f)
                
                # Recreate the model
                loaded_instance.model_init_args = model_init_args
                loaded_instance.model = _GCN_impl(**model_init_args)
                
                # Load the state dict
                state_dict = torch.load(model_state_path, map_location='cpu')
                loaded_instance.model.load_state_dict(state_dict)
                loaded_instance.model.eval()
                
                # Restore model attributes that were set during training
                if hasattr(loaded_instance, 'edge_list'):
                    loaded_instance.model.edge_list = loaded_instance.edge_list
                if hasattr(loaded_instance, 'radius'):
                    loaded_instance.model.radius = loaded_instance.radius
                loaded_instance.model.preprocess = loaded_instance.preprocess
                
                if verbose:
                    print(f"PyTorch model loaded from {model_state_path}")
                    
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load PyTorch model: {e}")
                loaded_instance.model = None

        # Load trainer configuration and recreate trainer if needed
        trainer_config_path = os.path.join(path, "trainer_config.pkl")
        if os.path.exists(trainer_config_path):
            try:
                with open(trainer_config_path, "rb") as f:
                    trainer_config = dill.load(f)
                
                # Recreate trainer with saved configuration
                callbacks = [LearningRateFinder(min_lr=1e-5, max_lr=1.0)]
                
                loaded_instance.trainer = pl.Trainer(
                    accelerator=trainer_config.get('accelerator', 'gpu'),
                    enable_checkpointing=False,
                    logger=False,
                    gradient_clip_val=trainer_config.get('gradient_clip_val', 10.0),
                    enable_progress_bar=True,
                    callbacks=callbacks,
                    max_epochs=trainer_config.get('max_epochs', gnn_epochs),
                    enable_model_summary=True,
                )
                
                # Link trainer to model if model exists
                if hasattr(loaded_instance, 'model') and loaded_instance.model is not None:
                    loaded_instance.model.trainer = loaded_instance.trainer
                
                if verbose:
                    print(f"Trainer recreated from configuration at {trainer_config_path}")
                    
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not recreate trainer: {e}")
                loaded_instance.trainer = None

        # Load scalers if they exist
        scalers_path = os.path.join(path, "scalers.pkl")
        if os.path.exists(scalers_path):
            try:
                with open(scalers_path, "rb") as f:
                    scalers = dill.load(f)
                
                for scaler_name, scaler_obj in scalers.items():
                    setattr(loaded_instance, scaler_name, scaler_obj)
                
                if verbose:
                    print(f"Scalers loaded from {scalers_path}")
                    
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load scalers: {e}")

        # Load feature generator if it exists
        feature_gen_path = os.path.join(path, "feature_generator.pkl")
        if os.path.exists(feature_gen_path):
            try:
                with open(feature_gen_path, "rb") as f:
                    feature_generator = dill.load(f)
                
                loaded_instance._feature_generator = feature_generator
                
                if verbose:
                    print(f"Feature generator loaded from {feature_gen_path}")
                    
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load feature generator: {e}")

        # Reset paths if requested
        if reset_paths:
            loaded_instance.path = path

        return loaded_instance





@hydra.main(config_path="conf", config_name="train_spaceenv", version_base=None)
def main(cfg: DictConfig):
    """
    Trains a model using AutoGluon and save the results.
    """
    
    # == Load config ===
    spaceenv = cfg.spaceenv
    training_cfg = spaceenv.autogluon
    output_dir = "."  # hydra automatically sets this for this script since
    radius = spaceenv.radius # Number of neighbors to consider using Chebyshev distance
    sparsity = spaceenv.sparsity
    
    # the config option hydra.job.chdir is true
    # we need to set the original workind directory to the
    # this is convenient since autogluon saves the intermediate
    # models on the working directory without obvious way to change it
            
    # == models to train == 
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
    hpars = get_hyperparameter_config('multimodal')
    hpars['AG_AUTOMM'] = {"optim.max_epochs": 10, "model.timm_image.checkpoint_name": "../../resnet18_local"}

    # === Data preparation ===
    # set seed
    seed_everything(spaceenv.seed)

    # == Read collection graph/data ==
    original_cwd = hydra.utils.get_original_cwd()
    collection_path = f"{original_cwd}/data_collections/{spaceenv.collection}/"
    graph_path = f"{original_cwd}/data_collections/{spaceenv.collection}/"
    
    

    # data
    logging.info(f"Reading data collection from {collection_path}:")
    data_file = glob(f"{collection_path}/data*")[0]
    
    
    if data_file.endswith("tab"):
        df_read_opts = {
            "sep": "\t",
            "index_col": spaceenv.index_col,
            "dtype": {spaceenv.index_col: str},
        }
        df = pd.read_csv(data_file, **df_read_opts)
        df.index = df.index.astype(str)
    elif data_file.endswith("parquet"):
        df = pd.read_parquet(data_file)

    else:
        raise ValueError(f"Unknown file extension in {data_file}.")

    # remove duplicate indices
    dupl = df.index.duplicated(keep="first")
    if dupl.sum() > 0:
        logging.info(f"Removed {dupl.sum()}/{df.shape[0]} duplicate indices.")
        df = df[~dupl]

    tmp = df.shape[0]
    df = df[df[spaceenv.treatment].notna()]
    logging.info(f"Removed {tmp - df.shape[0]}/{tmp} rows with missing treatment.")

    # graph
    logging.info(f"Reading graph from {graph_path}.")
    graph_file = glob(f"{graph_path}/graph*")[0]
            
     # deal with possible extensions for the graph
    if graph_file.endswith(("graphml", "graphml.gz")):
        graph = nx.read_graphml(graph_file)
        datatype = "graph"
    elif graph_file.endswith("tar.gz"):
        with tarfile.open(graph_file, "r:gz") as tar:
            # list files in tar
            tar_files = tar.getnames()

            edges = pd.read_parquet(tar.extractfile("graph/edges.parquet"))
            coords = pd.read_parquet(tar.extractfile("graph/coords.parquet"))

        graph = nx.Graph()
        graph.add_nodes_from(coords.index)
        graph.add_edges_from(edges.values)
        datatype = "grid"
    else:
        raise ValueError(f"Unknown file extension of file {graph_file}.")
    
    # === Read covariate groups ===
    if spaceenv.covariates is not None:
        # use specified covar groups
        covar_groups = OmegaConf.to_container(spaceenv.covariates)
        covariates = utils.unpack_covariates(covar_groups)
    else:
        # assume all columns are covariates, each covariate is a group
        covar_groups = df.columns.difference([spaceenv.treatment, spaceenv.outcome])
        covar_groups = covar_groups.tolist()
        covariates = covar_groups
        spaceenv.covariates = covariates
    # d = len(covariates)

    # maintain only covariates, treatment, and outcome
    tmp = df.shape[1]
    df = df[[spaceenv.treatment] + covariates + [spaceenv.outcome]]
    logging.info(f"Removed {tmp - df.shape[1]}/{tmp} columns due to covariate choice.")
    
    strength = 0.8
    covariates = covariates + ["single_cause"]
    df["single_cause"] = 0
    n_clusters = max(1, int(np.ceil(sparsity * len(df))))
    
    # Random cluster centers
    centers = np.random.choice(df.index.to_numpy(), size=n_clusters, replace=False)

    # Accumulator for total confounder
    L = np.zeros(len(df), dtype=float)
    decay_scale = 2.0
    
    nodes = df.index.to_numpy()  # list of all nodes
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    intersection = set(df.index).intersection(set(graph.nodes))
    graph = nx.subgraph(graph, intersection)
    
    for c in centers:
        # Max intensity at the cluster center
        peak = np.random.uniform(0.5, 1.0)

        # BFS distances from the center
        dist_map = nx.single_source_shortest_path_length(graph, c)

        # Radial decay
        for node, d in dist_map.items():
            val = peak * np.exp(-d / decay_scale)
            L[node_to_idx[node]] = max(L[node_to_idx[node]], val)
    
    # Assign to df
    df["single_cause"] = L
    
    # Apply effect to treatment and outcome
    for col in [spaceenv.treatment, spaceenv.outcome]:
        df[col] += strength * df[col].std() * df["single_cause"]

    logging.info(
        f"Injected clustered single-cause confounder with radial decay. "
        f"{n_clusters} clusters, sparsity={sparsity}."
    )

    # === Apply data transformations ===
    logging.info(f"Transforming data.")

    for tgt in ["treatment", "outcome", "covariates"]:
        scaling = getattr(spaceenv.scaling, tgt)
        transform = getattr(spaceenv.transforms, tgt)
        varnames = getattr(spaceenv, tgt)
        if tgt != "covariates":
            varnames = [varnames]
        for varname in varnames:
            if scaling is not None:
                if isinstance(scaling, (dict, DictConfig)):
                    scaling_ = scaling[varname]
                else:
                    scaling_ = scaling
                logging.info(f"Scaling {varname} with {scaling_}")
                df[varname] = utils.scale_variable(df[varname].values, scaling_)
            if transform is not None:
                if isinstance(transform, (dict, DictConfig)):
                    transform_ = transform[varname]
                else:
                    transform_ = transform
                logging.info(f"Transforming {varname} with {transform_}")
                df[varname] = utils.transform_variable(df[varname].values, transform_)
           
    
    
    
    # make treatment boolean if only two values
    if df[spaceenv.treatment].nunique() == 2:
        df[spaceenv.treatment] = df[spaceenv.treatment].astype(bool)
        is_binary_treatment = True
    else:
        # if not binary, remove bottom and top for stable training
        if spaceenv.treatment_quantile_valid_range is not None:
            fmin = 100 * spaceenv.treatment_quantile_valid_range[0]
            fmax = 100 * (1 - spaceenv.treatment_quantile_valid_range[1])
            logging.info(
                f"Removing bottom {fmin:.1f}% and top {fmax:.1f}% of treat. values for stability."
            )
            t = df[spaceenv.treatment].values
            quants = np.nanquantile(t, spaceenv.treatment_quantile_valid_range)
            df = df[(t >= quants[0]) & (t <= quants[1])]
            is_binary_treatment = False
    
    
    # also remove extreme values for outcome
    if spaceenv.outcome_quantile_valid_range is not None:
        fmin = 100 * spaceenv.outcome_quantile_valid_range[0]
        fmax = 100 * (1 - spaceenv.outcome_quantile_valid_range[1])
        logging.info(
            f"Removing bottom {fmin:.1f}% and top {fmax:.1f}% of outcome values for stability."
        )
        y = df[spaceenv.outcome].values
        quants = np.nanquantile(y, spaceenv.outcome_quantile_valid_range)
        df = df[(y >= quants[0]) & (y <= quants[1])]

    # === Add extra columns / techniques for better causal effect estimation
    # based on increasing attention to the treatment ===
    if not is_binary_treatment and spaceenv.bsplines:
        logging.info(f"Boosting treatment with b-splines of pctile (cont. treatment).")
        b_deg = spaceenv.bsplines_degree
        b_df = spaceenv.bsplines_df

        t = df[spaceenv.treatment].values
        t_vals = np.sort(np.unique(t))

        def get_t_pct(t):
            return np.searchsorted(t_vals, t) / len(t_vals)

        knots = np.linspace(0, 1, b_df)[1:-1].tolist()
        knots = [0] * b_deg + knots + [1] * b_deg
        spline_basis = [
            BSpline.basis_element(knots[i : (i + b_deg + 2)])
            for i in range(len(knots) - b_deg - 1)
        ]
        extra_colnames = [f"splines_{i}" for i in range(len(spline_basis))]
        extra_cols = np.stack([s(get_t_pct(t)) for s in spline_basis], axis=1)
        extra_cols = pd.DataFrame(extra_cols, columns=extra_colnames, index=df.index)
        df = pd.concat([df, extra_cols], axis=1)
    elif is_binary_treatment and spaceenv.binary_treatment_iteractions:
        logging.info(f"Boosting treatment adding interactions with all covariates.")
        t_ind = df[spaceenv.treatment].values[:, None].astype(float)
        interacted = df[covariates].values * t_ind
        extra_colnames = [f"{c}_interact" for c in covariates]
        extra_cols = pd.DataFrame(interacted, columns=extra_colnames, index=df.index)
        df = pd.concat([df, extra_cols], axis=1)
        
        def get_t_pct(t):
            return t
        spline_basis = []
    else:
        def get_t_pct(t):
            return t  # identity, pass-through

        spline_basis = []
        extra_colnames = []

    # test with a subset of the data
    if cfg.debug_subsample is not None:
        logging.info(f"Subsampling since debug_subsample={cfg.debug_subsample}.")
        ix = np.random.choice(range(df.shape[0]), cfg.debug_subsample, replace=False)
        df = df.iloc[ix]
        
    
    # === Harmonize data and graph ===
    intersection = set(df.index).intersection(set(graph.nodes))
    n = len(intersection)
    perc = 100 * n / len(df)
    logging.info(f"Homegenizing data and graph")
    logging.info(f"...{perc:.2f}% of the data rows (n={n}) found in graph nodes.")
    graph = nx.subgraph(graph, intersection)
    df = df.loc[list(intersection)]

    # obtain final edge list
    node2ix = {n: i for i, n in enumerate(df.index)}
    edge_list = np.array([(node2ix[e[0]], node2ix[e[1]]) for e in graph.edges])

    # fill missing if needed
    if spaceenv.fill_missing_covariate_values:
        for c in covariates:
            col_vals = df[c].values
            frac_missing = np.isnan(col_vals).mean()
            logging.info(f"Filling {100 * frac_missing:.2f}% missing values for {c}.")
            nbrs_means = utils.get_nbrs_means(col_vals, edge_list)
            col_vals[np.isnan(col_vals)] = nbrs_means[np.isnan(col_vals)]
            df[c] = col_vals
    
    if "nonlocal" in spaceenv.base_name:
        local_confounding=False
    else:
        local_confounding=True
    
    if datatype == "graph":
        if local_confounding:
            raise ValueError(f"Only supported for nonlocal confounding as of right now.")
        hpars.pop('AG_AUTOMM')
        # Create graph features
        geojson_file = glob(f"{graph_path}/geojson*")[0]
        gdf = gpd.read_file(geojson_file)
        geoid_col = next(col for col in gdf.columns if col in ["GEOID", "GEOID10", "GEOID20"])
        gdf[geoid_col] = gdf[geoid_col].astype(str)
        
        map_df = df.copy()
        map_df.index = map_df.index.astype(str)
        map_df = map_df.join(gdf.set_index(geoid_col), how="left")
        map_df = gpd.GeoDataFrame(map_df, geometry="geometry")
        
        lat_col = next((col for col in map_df.columns if 'INTPTLAT' in col), None)
        lon_col = next((col for col in map_df.columns if 'INTPTLON' in col), None)

        map_df['latitude'] = map_df[lat_col].astype(str).str.lstrip('+').astype(float)
        map_df['longitude'] = map_df[lon_col].astype(str).str.lstrip('+').astype(float)
        
        node_list = np.array(graph.nodes())
        nbrs = {node: utils.get_k_hop_neighbors(graph, node, radius) for node in node_list}
        neighbor_counts = {node: len(neighbors) for node, neighbors in nbrs.items()}
        neighbor_count_values = np.array(list(neighbor_counts.values()))
        max_nbrs = int(np.quantile(neighbor_count_values, 0.95))        
        
        
        logging.info(f"Adding {max_nbrs} closest neighbors to df")
        feat_cols = [spaceenv.treatment] + covariates + extra_colnames
        df_graph = utils.add_neighbor_columns(df, nbrs, feat_cols, max_nbrs, map_df)
        
        nbr_cols = []
        for feature in feat_cols:
            for i in range(1, max_nbrs + 1):
                nbr_cols.append(f"{feature}_nbr_{i}")
                
        treat_nbr_cols = []
        for feature in covariates:
            for i in range(1, max_nbrs + 1):
                treat_nbr_cols.append(f"{feature}_nbr_{i}")
        
        # remove nans in outcome
        outcome_nans = np.isnan(df_graph[spaceenv.outcome])
        logging.info(f"Removing {outcome_nans.sum()} for training since missing outcome.")
        dftrain = df_graph[~np.isnan(df_graph[spaceenv.outcome])]

        # == Spatial Train/Test Split ===
        tuning_nodes, buffer_nodes = utils.spatial_train_test_split(
            graph,
            init_frac=spaceenv.spatial_tuning.init_frac,
            levels=spaceenv.spatial_tuning.levels,
            buffer=spaceenv.spatial_tuning.buffer + radius,
        )        
        
        all_edges = []
        raw_edge_list = np.array([(e[0], e[1]) for e in graph.edges])
        
        for s, t in raw_edge_list:
            all_edges.append((s, t))
            all_edges.append((t, s))  # add reverse edge

        # Optional: remove duplicates by using a set, then convert back to list
        all_edges = list(set(all_edges))
        hpars[CustomGCN] = {"edge_list": all_edges, "radius": radius}


        # === Outcome Model fitting ===
        cols = [spaceenv.treatment] + covariates + extra_colnames + nbr_cols + [spaceenv.outcome]
        tuning_data = dftrain[dftrain.index.isin(tuning_nodes)][cols]
        train_data = dftrain[~dftrain.index.isin(buffer_nodes)][cols]
        tunefrac = 100 * len(tuning_nodes) / df.shape[0]
        trainfrac = 100 * len(train_data) / df.shape[0]
        logging.info(f"...{tunefrac:.2f}% of the rows used for tuning split.")
        logging.info(f"...{trainfrac:.2f}% of the rows used for training.")

        logging.info(f"Fitting model to outcome variable on train split.")
        trainer = TabularPredictor(label=spaceenv.outcome)

        predictor = trainer.fit(
            train_data,
            **training_cfg.fit,
            tuning_data=tuning_data,
            use_bag_holdout=True,
            hyperparameters=hpars,
        )
        results = predictor.fit_summary()
        logging.info(f"Model fit summary:\n{results['leaderboard']}")
        results["leaderboard"].to_csv(f"{output_dir}/leaderboard.csv", index=False)

        # === Retrain on full data for the final model
        logging.info(f"Fitting to full data.")
        predictor.refit_full()

        mu = predictor.predict(df_graph)
        mu_synth = mu.copy()
        mu.name = mu.name + "_pred"

        logging.info(f"Outcome Prediction R² score: {r2_score(df_graph[spaceenv.outcome], mu_synth):.4f}")

        # sythetic outcome
        logging.info(f"Generating synthetic residuals for synthetic outcome.")
        residuals = (df_graph[spaceenv.outcome] - mu_synth).values
        synth_residuals = utils.generate_noise_like(residuals, edge_list)
        Y_synth = mu_synth + synth_residuals
        Y_synth.name = "Y_synth"

        scale = np.std(Y_synth)
        
        residual_smoothness = utils.moran_I(residuals, edge_list)
        synth_residual_smoothness = utils.moran_I(synth_residuals, edge_list)
        residual_nbrs_corr = utils.get_nbrs_corr(residuals, edge_list)
        synth_residual_nbrs_corr = utils.get_nbrs_corr(synth_residuals, edge_list)

        # === Counterfactual generation ===
        logging.info(f"Generating counterfactual predictions and adding residuals")
        A = df[spaceenv.treatment]
        amin, amax = np.nanmin(A), np.nanmax(A)
        n_treatment_values = len(np.unique(A))
        n_bins = min(spaceenv.treatment_max_bins, n_treatment_values)
        avals = np.linspace(amin, amax, n_bins)

        mu_cf = []
        for a in avals:
            cfdata = df.copy()
            cfdata_graph = utils.add_neighbor_columns(cfdata, nbrs, feat_cols, max_nbrs, map_df,
                                                     a=a, change="center", treatment=spaceenv.treatment, 
                                 is_binary_treatment=is_binary_treatment, spaceenv=spaceenv, get_t_pct=get_t_pct, spline_basis=spline_basis, 
                                 extra_colnames=extra_colnames, covariates=covariates)
            predicted = predictor.predict(cfdata_graph)
            mu_cf.append(predicted)
        mu_cf = pd.concat(mu_cf, axis=1)
        mu_cf.columns = [
            f"{spaceenv.outcome}_pred_{i:02d}" for i in range(len(mu_cf.columns))
        ]
        Y_cf = mu_cf + synth_residuals[:, None]
        Y_cf.columns = [f"Y_synth_{i:02d}" for i in range(len(mu_cf.columns))]
        
        spill_mu_cf = []
        for a in avals:
            cfdata = df.copy()
            cfdata_graph = utils.add_neighbor_columns(cfdata, nbrs, feat_cols, max_nbrs, map_df,
                                                     a=a, change="nbr", treatment=spaceenv.treatment, 
                                 is_binary_treatment=is_binary_treatment, spaceenv=spaceenv, get_t_pct=get_t_pct, spline_basis=spline_basis, 
                                 extra_colnames=extra_colnames, covariates=covariates)
            predicted = predictor.predict(cfdata_graph)
            spill_mu_cf.append(predicted)
        spill_mu_cf = pd.concat(spill_mu_cf, axis=1)
        spill_mu_cf.columns = [
            f"spill_{spaceenv.outcome}_pred_{i:02d}" for i in range(len(spill_mu_cf.columns))
        ]
        spill_Y_cf = spill_mu_cf + synth_residuals[:, None]
        spill_Y_cf.columns = [f"spill_Y_synth_{i:02d}" for i in range(len(spill_mu_cf.columns))]

        logging.info("Plotting counterfactuals and residuals.")
        ix = np.random.choice(len(df), cfg.num_plot_samples)
        cfpred_sample = mu_cf.iloc[ix].values
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(avals, cfpred_sample.T, color="gray", alpha=0.2)
        ax.scatter(A.iloc[ix], mu.iloc[ix], color="red")

        # Draw a line for the ATE
        ax.plot(
            avals,
            mu_cf.mean(),
            color="red",
            linestyle="--",
            label="Average Treatment Effect",
            alpha=0.5,
        )
        ax.legend()

        ax.set_xlabel(spaceenv.treatment)
        ax.set_ylabel(spaceenv.outcome)
        ax.set_title("Counterfactuals")
        fig.savefig(f"{output_dir}/counterfactuals.png", dpi=300, bbox_inches="tight")

        logging.info("Plotting histogram of true and synthetic residuals.")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(residuals, bins=20, density=True, alpha=0.5, label="True")
        ax.hist(synth_residuals, bins=20, density=True, alpha=0.5, label="Synthetic")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
        ax.set_title("Residuals")
        ax.legend()
        fig.savefig(f"{output_dir}/residuals.png", dpi=300, bbox_inches="tight")

        # === Compute outcome feature importance ===
        logging.info(f"Computing feature importance.")            
        label = spaceenv.outcome
        featimp = utils.feature_importance(
            data=train_data.drop(columns=[label]),
            label=train_data[label],
            predict_func=make_predict_with_fixed_index(predictor),
            eval_metric=predictor.eval_metric,
            **training_cfg.feat_importance,
        )
        # featimp = predictor.feature_importance(
        #     train_data,
        #     **training_cfg.feat_importance,
        # )


        # convert the .importance column to dict
        featimp = dict(featimp.importance)
        group_featimp = utils.group_by_feat_graph(featimp)
        logging.info("Group Featimp: %s", group_featimp)
        
        featimp = {}
        for metric, subdict in group_featimp.items():
            valid_items = {k: v for k, v in subdict.items() if not math.isnan(v)}
            if valid_items:
                max_key = max(valid_items, key=valid_items.get)
                featimp[metric] = valid_items[max_key]
            else:
                featimp[metric] = float('-inf')


        if not is_binary_treatment and spaceenv.bsplines:
            # this is the case when we want to merge the scores of all splines
            # of the treat  ment into a single score. We can use max aggregation
            tname = spaceenv.treatment
            for c in extra_colnames:
                featimp[tname] = max(featimp.get(tname, 0.0), featimp.get(c, 0.0))
                if c in featimp:
                    featimp.pop(c)
        elif is_binary_treatment and spaceenv.binary_treatment_iteractions:
            # this is the case when we want to merge interacted covariates
            # with the treatment. We can use max aggregation strategy.
            for c in covariates:
                featimp[c] = max(featimp.get(c, 0.0), featimp.get(c + "_interact", 0.0))
                if c + "_interact" in featimp:
                    featimp.pop(c + "_interact")

        treat_imp = featimp[spaceenv.treatment]
        featimp = {c: float(featimp.get(c, 0.0)) / scale for c in covariates}
        featimp["treatment"] = treat_imp
        logging.info("Featimp: %s", featimp)
        
        
        # === Fitting model to treatment variable for confounding score ===
        logging.info(f"Fitting model to treatment variable for importance score.")
        treat_trainer = TabularPredictor(label=spaceenv.treatment)
        cols = covariates + [spaceenv.treatment] + treat_nbr_cols
        treat_tuning_data = dftrain[dftrain.index.isin(tuning_nodes)][cols]
        treat_train_data = dftrain[~dftrain.index.isin(buffer_nodes)][cols]
        treat_predictor = treat_trainer.fit(
            treat_train_data,
            **training_cfg.fit,
            tuning_data=treat_tuning_data,
            use_bag_holdout=True,
            hyperparameters=hpars,
        )
        treat_predictor.refit_full()
        results = treat_predictor.fit_summary()

        # normalize feature importance by scale
        tscale = np.nanstd(df[spaceenv.treatment])
        label = spaceenv.treatment
        treat_featimp = utils.feature_importance(
            data=treat_train_data.drop(columns=[label]),
            label=treat_train_data[label],
            predict_func=make_predict_with_fixed_index(treat_predictor),
            eval_metric=treat_predictor.eval_metric,
            **training_cfg.feat_importance,
        )
        treat_featimp = dict(treat_featimp.importance)
        group_treat_featimp = utils.group_by_feat_graph(treat_featimp)
        logging.info("Group Treat Featimp: %s", group_treat_featimp)
        
        treat_featimp = {}
        for metric, subdict in group_treat_featimp.items():
            valid_items = {k: v for k, v in subdict.items() if not math.isnan(v)}
            if valid_items:
                max_key = max(valid_items, key=valid_items.get)
                treat_featimp[metric] = valid_items[max_key]
            else:
                treat_featimp[metric] = float('-inf')

        # do the reduction for the case of interactions
        if is_binary_treatment and spaceenv.binary_treatment_iteractions:
            for c in covariates:
                treat_featimp[c] = max(
                    treat_featimp.get(c, 0.0), treat_featimp.get(c + "_interact", 0.0)
                )
                if c + "_interact" in treat_featimp:
                    treat_featimp.pop(c + "_interact")

        treat_featimp = {c: float(treat_featimp.get(c, 0.0)) / tscale for c in covariates}
        logging.info("Treat Featimp: %s", treat_featimp)
        # legacy confounding score by minimum
        cs_minimum = {k: min(treat_featimp[k], featimp[k]) for k in covariates}
        logging.info(f"Legacy conf. score by minimum:\n{cs_minimum}")

        # Top top_k features that are important to both treatment and outcome
        top_k = 10
        top_feat = dict(sorted(cs_minimum.items(), key=lambda x: x[1], reverse=True)[:top_k])
        logging.info(f"Got top {len(top_feat)} features for outcome and treatment (from graph dataset): {top_feat}.")
        
        # loading from grid dataset because graph dataset doesn't have enough values to calculate feature importance
        grid_path = f"{original_cwd}/trained_spaceenvs/{spaceenv.base_name.replace('graph', 'grid')}/metadata.yaml"
        with open(grid_path, 'r') as file:
            data = yaml.safe_load(file)
        d = data["confounding_score"]
        top_feat = dict(list(d.items())[:top_k])

        top_featcols = list(top_feat.keys())
        logging.info(f"Got top {len(top_feat)} features for outcome and treatment (from grid dataset with same radius): {top_feat}.")

        # === Compute confounding scores ===
        # The strategy for confounding scores is to compute various types
        # using the baseline model.

        # For continous treatment compute the ERF and ITE scores
        # For categorical treatmetn additionally compute the ATE score
        # For both also use the minimum of the treatment and outcome model
        # As in the first version of the paper.

        # For comparability across environments, we divide the scores by the
        # variance of the synthetic outcome.

        # Obtain counterfactuals for the others
        cs_erf = {}
        cs_ite = {}
        cs_ate = {}  # will be empty if not binary

        for i, feat in enumerate(top_featcols):
            key_ = feat
            value_ = [feat] + ([feat + "_interact"] if feat + "_interact" in extra_colnames else [])
            
            leave_out_nbr_cols = []
            for col in covariates + extra_colnames + [spaceenv.treatment]:
                if col == feat or feat + "_interact" == col:
                    continue
                for j in range(1, max_nbrs + 1):
                    leave_out_nbr_cols.append(f"{col}_nbr_{j}")

            cols = [cov for cov in covariates if feat != cov] + [spaceenv.treatment] + [ex for ex in extra_colnames if feat + "_interact" != ex] + [spaceenv.outcome] + leave_out_nbr_cols

            leave_out_predictor = TabularPredictor(label=spaceenv.outcome)
            leave_out_predictor = leave_out_predictor.fit(
                train_data[cols],
                **spaceenv.autogluon.leave_out_fit,
                tuning_data=tuning_data[cols],
                use_bag_holdout=True,
                hyperparameters=hpars,
            )
            leave_out_predictor.refit_full()
            results = leave_out_predictor.fit_summary()

            leave_out_mu_cf = []        
            for a in avals:
                dfcols = [cov for cov in covariates if feat != cov] + [spaceenv.treatment] + [ex for ex in extra_colnames if feat + "_interact" != ex]
                feat_cols = [spaceenv.treatment] + [cov for cov in covariates if feat != cov] + [ex for ex in extra_colnames if feat + "_interact" != ex]
                cfdata = df[dfcols].copy()
                cfdata_graph = utils.add_neighbor_columns(cfdata, nbrs, feat_cols, max_nbrs, map_df,
                                                         a=a, change="center", treatment=spaceenv.treatment, 
                                     is_binary_treatment=is_binary_treatment, spaceenv=spaceenv, get_t_pct=get_t_pct, spline_basis=spline_basis, 
                                     extra_colnames=[ex for ex in extra_colnames if feat + "_interact" != ex], covariates=[cov for cov in covariates if feat != cov])
                predicted = leave_out_predictor.predict(cfdata_graph)
                leave_out_mu_cf.append(predicted)
            leave_out_mu_cf = pd.concat(leave_out_mu_cf, axis=1)

            logging.info(f"[{i + 1} / {len(top_featcols)}]: {key_}")

            # compute loss normalized by the variance of the outcome
            cf_err = (leave_out_mu_cf.values - mu_cf.values) / scale
            cs_ite[key_] = float(np.sqrt((cf_err**2).mean(0)).mean())
            logging.info(f"ITE: {cs_ite[key_]:.3f}")

            erf_err = (leave_out_mu_cf.values - mu_cf.values).mean(0) / scale
            cs_erf[key_] = float(np.abs(erf_err).mean())
            logging.info(f"ERF: {cs_erf[key_]:.3f}")

            if n_treatment_values == 2:
                cs_ate[key_] = np.abs(erf_err[1] - erf_err[0])
                logging.info(f"ATE: {cs_ate[key_]:.3f}")

        # === Compute the spatial smoothness of each covariate
        logging.info(f"Computing spatial smoothness of each covariate.")
        moran_I_values = {}
        for c in covariates:
            moran_I_values[c] = utils.moran_I(df[c].values, edge_list)

        # === Save results ===
        logging.info(f"Saving synthetic data, graph, and metadata")
        X = df[df.columns.difference([spaceenv.outcome, spaceenv.treatment])]
        dfout = pd.concat([A, X, mu, mu_cf, Y_synth, Y_cf, spill_mu_cf, spill_Y_cf], axis=1)

        # whens saving synthetic data, respect the original data format
        if data_file.endswith("tab"):
            dfout.to_csv(f"{output_dir}/synthetic_data.tab", sep="\t", index=True)
        elif data_file.endswith("parquet"):
            dfout.to_parquet(f"{output_dir}/synthetic_data.parquet")
            
        map_df.to_file("map_df.geojson", driver="GeoJSON")
        
        # save subgraph in the right format
        if graph_file.endswith(("graphml", "graphml.gz")):
            ext = "graphml.gz" if graph_file.endswith("graphml.gz") else "graphml"
            tgt_graph_path = f"{output_dir}/graph.{ext}"
            nx.write_graphml(graph, tgt_graph_path)
            
        elif graph_file.endswith("tar.gz"):
            # save edges and coords
            edges = pd.DataFrame(np.array(list(graph.edges)), columns=["source", "target"])
            coords = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
            # save again as a tar.gz
            with tarfile.open(f"{output_dir}/graph.tar.gz", "w:gz") as tar:
                os.makedirs("graph", exist_ok=True)
                edges.to_parquet("graph/edges.parquet")
                coords.to_parquet("graph/coords.parquet")
                tar.add("graph/")

        metadata = {
            "base_name": f"{spaceenv.base_name}",
            "treatment": spaceenv.treatment,
            "predicted_outcome": spaceenv.outcome,
            "synthetic_outcome": "Y_synth",
            "confounding_score": utils.sort_dict(cs_minimum),
            "confounding_score_erf": utils.sort_dict(cs_erf),
            "confounding_score_ite": utils.sort_dict(cs_ite),
            "confounding_score_ate": utils.sort_dict(cs_ate),
            "spatial_scores": utils.sort_dict(moran_I_values),
            "outcome_importance": utils.sort_dict(featimp),
            "treatment_importance": utils.sort_dict(treat_featimp),
            "covariates": list(covariates),
            "treatment_values": avals.tolist(),
            "covariate_groups": covar_groups,
            "original_residual_spatial_score": float(residual_smoothness),
            "synthetic_residual_spatial_score": float(synth_residual_smoothness),
            "original_nbrs_corr": float(residual_nbrs_corr),
            "synthetic_nbrs_corr": float(synth_residual_nbrs_corr),
            "radius": radius,
        }

        # save metadata and resolved config
        with open(f"{output_dir}/metadata.yaml", "w") as f:
            yaml.dump(metadata, f, sort_keys=False)
                
        
    elif datatype == "grid":
        logging.info(f"Creating grids for each feature")
        df_grid, temp_dir = utils.create_grid_features_compact(df, radius=radius)
        
        if not local_confounding:
            nbr_cols = []
            for col in covariates + extra_colnames + [spaceenv.treatment]:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nbr_cols.append(f"{col}_{dr}_{dc}")

            treat_nbr_cols = []
            for col in covariates:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        treat_nbr_cols.append(f"{col}_{dr}_{dc}")
        else:
            nbr_cols = []
            for col in [spaceenv.treatment]:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nbr_cols.append(f"{col}_{dr}_{dc}")
            
            for col in covariates + extra_colnames:
                for dr in range(0, 1):
                    for dc in range(0, 1):
                        nbr_cols.append(f"{col}_{dr}_{dc}")
            
            treat_nbr_cols = []
            for col in covariates:
                for dr in range(0, 1):
                    for dc in range(0, 1):
                        treat_nbr_cols.append(f"{col}_{dr}_{dc}")
            
            

        # remove nans in outcome
        outcome_nans = np.isnan(df_grid[spaceenv.outcome])
        logging.info(f"Removing {outcome_nans.sum()} for training since missing outcome.")
        dftrain = df_grid[~np.isnan(df_grid[spaceenv.outcome])]

        # == Spatial Train/Test Split ===
        training_nodes, tuning_nodes, buffer_nodes = utils.spatial_train_test_split_radius(
            graph,
            init_frac=spaceenv.spatial_tuning.init_frac,
            levels=spaceenv.spatial_tuning.levels,
            buffer=spaceenv.spatial_tuning.buffer + radius,
            radius=radius,
        )

        from autogluon.tabular import FeatureMetadata
        feature_metadata = FeatureMetadata.from_df(dftrain)
        if not local_confounding:
            special_types_dict = {col: ['image_path'] for col in [cov + "_grid" for cov in covariates] + [spaceenv.treatment + "_grid"] + [extra_col + "_grid" for extra_col in extra_colnames]}
        else:
            special_types_dict = {col: ['image_path'] for col in [spaceenv.treatment + "_grid"]}
        feature_metadata = feature_metadata.add_special_types(special_types_dict)

        # === Outcome Model fitting ===
        if not local_confounding:
            cols = [cov + "_grid" for cov in covariates] + [spaceenv.treatment + "_grid"] + [ex + "_grid" for ex in extra_colnames] + [spaceenv.outcome] + nbr_cols
        else:
            cols = [spaceenv.treatment + "_grid"] + [spaceenv.outcome] + nbr_cols
        tuning_data = dftrain[dftrain.index.isin(tuning_nodes)][cols]
        train_data = dftrain[dftrain.index.isin(training_nodes)][cols]
        tunefrac = 100 * len(tuning_nodes) / df.shape[0]
        trainfrac = 100 * len(train_data) / df.shape[0]
        logging.info(f"...{tunefrac:.2f}% of the rows used for tuning split.")
        logging.info(f"...{trainfrac:.2f}% of the rows used for training.")

        logging.info(f"Fitting model to outcome variable on train split.")
        trainer = TabularPredictor(label=spaceenv.outcome)

        predictor = trainer.fit(
            train_data,
            **training_cfg.fit,
            tuning_data=tuning_data,
            use_bag_holdout=True,
            hyperparameters=hpars,
            feature_metadata=feature_metadata,
        )
        results = predictor.fit_summary()
        logging.info(f"Model fit summary:\n{results['leaderboard']}")
        results["leaderboard"].to_csv(f"{output_dir}/leaderboard.csv", index=False)

        # === Retrain on full data for the final model
        logging.info(f"Fitting to full data.")
        predictor.refit_full()

        mu = predictor.predict(df_grid)
        mu_synth = mu.copy()
        mu.name = mu.name + "_pred"

        logging.info(f"Outcome Prediction R² score: {r2_score(df_grid[spaceenv.outcome], mu_synth):.4f}")

        # sythetic outcome
        logging.info(f"Generating synthetic residuals for synthetic outcome.")
        residuals = (df_grid[spaceenv.outcome] - mu_synth).values
        synth_residuals = utils.generate_noise_like(residuals, edge_list)
        Y_synth = mu_synth + synth_residuals
        Y_synth.name = "Y_synth"

        scale = np.std(Y_synth)

        residual_smoothness = utils.moran_I(residuals, edge_list)
        synth_residual_smoothness = utils.moran_I(synth_residuals, edge_list)
        residual_nbrs_corr = utils.get_nbrs_corr(residuals, edge_list)
        synth_residual_nbrs_corr = utils.get_nbrs_corr(synth_residuals, edge_list)

        # === Counterfactual generation ===
        logging.info(f"Generating counterfactual predictions and adding residuals")
        A = df[spaceenv.treatment]
        amin, amax = np.nanmin(A), np.nanmax(A)
        n_treatment_values = len(np.unique(A))
        n_bins = min(spaceenv.treatment_max_bins, n_treatment_values)
        avals = np.linspace(amin, amax, n_bins)

        mu_cf = []
        for a in avals:
            cfdata = df.copy()
            cfdata_grid, temp_dir = utils.create_grid_features_compact(cfdata, 
                                                                       radius=radius, 
                                                                       a=a, 
                                                                       change="center", 
                                                                       treatment=spaceenv.treatment,
                                                                       is_binary_treatment=is_binary_treatment,
                                                                       spaceenv=spaceenv,
                                                                       get_t_pct=get_t_pct,
                                                                       spline_basis=spline_basis,
                                                                       extra_colnames=extra_colnames,
                                                                       covariates=covariates)
            predicted = predictor.predict(cfdata_grid)
            mu_cf.append(predicted)
        mu_cf = pd.concat(mu_cf, axis=1)
        mu_cf.columns = [
            f"{spaceenv.outcome}_pred_{i:02d}" for i in range(len(mu_cf.columns))
        ]
        Y_cf = mu_cf + synth_residuals[:, None]
        Y_cf.columns = [f"Y_synth_{i:02d}" for i in range(len(mu_cf.columns))]
        
        spill_mu_cf = []
        for a in avals:
            cfdata = df.copy()
            cfdata_grid, temp_dir = utils.create_grid_features_compact(cfdata, 
                                                                       radius=radius, 
                                                                       a=a, 
                                                                       change="nbr", 
                                                                       treatment=spaceenv.treatment,
                                                                       is_binary_treatment=is_binary_treatment,
                                                                       spaceenv=spaceenv,
                                                                       get_t_pct=get_t_pct,
                                                                       spline_basis=spline_basis,
                                                                       extra_colnames=extra_colnames,
                                                                       covariates=covariates)
            predicted = predictor.predict(cfdata_grid)
            spill_mu_cf.append(predicted)
        spill_mu_cf = pd.concat(spill_mu_cf, axis=1)
        spill_mu_cf.columns = [
            f"spill_{spaceenv.outcome}_pred_{i:02d}" for i in range(len(spill_mu_cf.columns))
        ]
        spill_Y_cf = spill_mu_cf + synth_residuals[:, None]
        spill_Y_cf.columns = [f"spill_Y_synth_{i:02d}" for i in range(len(spill_mu_cf.columns))]

        logging.info("Plotting counterfactuals and residuals.")
        ix = np.random.choice(len(df), cfg.num_plot_samples)
        cfpred_sample = mu_cf.iloc[ix].values
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(avals, cfpred_sample.T, color="gray", alpha=0.2)
        ax.scatter(A.iloc[ix], mu.iloc[ix], color="red")

        # Draw a line for the ATE
        ax.plot(
            avals,
            mu_cf.mean(),
            color="red",
            linestyle="--",
            label="Average Treatment Effect",
            alpha=0.5,
        )
        ax.legend()

        ax.set_xlabel(spaceenv.treatment)
        ax.set_ylabel(spaceenv.outcome)
        ax.set_title("Counterfactuals")
        fig.savefig(f"{output_dir}/counterfactuals.png", dpi=300, bbox_inches="tight")

        logging.info("Plotting histogram of true and synthetic residuals.")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(residuals, bins=20, density=True, alpha=0.5, label="True")
        ax.hist(synth_residuals, bins=20, density=True, alpha=0.5, label="Synthetic")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
        ax.set_title("Residuals")
        ax.legend()
        fig.savefig(f"{output_dir}/residuals.png", dpi=300, bbox_inches="tight")
        
        cs_minimum = {}
        featimp = {}
        treat_featimp = {}

        # === Compute outcome feature importance ===
#         logging.info(f"Computing feature importance.")            
#         label = spaceenv.outcome
#         # featimp = utils.feature_importance(
#         #     data=train_data.drop(columns=[label]),
#         #     label=train_data[label],
#         #     predict_func=predictor.predict,
#         #     eval_metric=predictor.eval_metric,
#         #     **training_cfg.feat_importance,
#         # )
#         featimp = predictor.feature_importance(
#             tuning_data,
#             **training_cfg.feat_importance,
#         )


#         # convert the .importance column to dict
#         featimp = dict(featimp.importance)
#         featimp = utils.delete_grid(featimp)

#         group_featimp = utils.group_by_feat(featimp)
#         logging.info("Group Featimp: %s", group_featimp)
        
#         featimp = {}
#         for metric, subdict in group_featimp.items():
#             valid_items = {k: v for k, v in subdict.items() if not math.isnan(v)}
#             if valid_items:
#                 max_key = max(valid_items, key=valid_items.get)
#                 featimp[metric] = valid_items[max_key]
#             else:
#                 featimp[metric] = float('-inf')


#         if not is_binary_treatment and spaceenv.bsplines:
#             # this is the case when we want to merge the scores of all splines
#             # of the treat  ment into a single score. We can use max aggregation
#             tname = spaceenv.treatment
#             for c in extra_colnames:
#                 featimp[tname] = max(featimp.get(tname, 0.0), featimp.get(c, 0.0))
#                 if c in featimp:
#                     featimp.pop(c)
#         elif is_binary_treatment and spaceenv.binary_treatment_iteractions:
#             # this is the case when we want to merge interacted covariates
#             # with the treatment. We can use max aggregation strategy.
#             for c in covariates:
#                 featimp[c] = max(featimp.get(c, 0.0), featimp.get(c + "_interact", 0.0))
#                 if c + "_interact" in featimp:
#                     featimp.pop(c + "_interact")

#         treat_imp = featimp[spaceenv.treatment]
#         featimp = {c: float(featimp.get(c, 0.0)) / scale for c in covariates}
#         featimp["treatment"] = treat_imp
#         logging.info("Featimp: %s", featimp)


#         # === Fitting model to treatment variable for confounding score ===
#         logging.info(f"Fitting model to treatment variable for importance score.")
#         treat_trainer = TabularPredictor(label=spaceenv.treatment)
#         if not local_confounding:
#             cols = [cov + "_grid" for cov in covariates] + [spaceenv.treatment] + treat_nbr_cols
#         else:
#             cols = [spaceenv.treatment] + treat_nbr_cols
#         treat_tuning_data = dftrain[dftrain.index.isin(tuning_nodes)][cols]
#         treat_train_data = dftrain[dftrain.index.isin(training_nodes)][cols]
#         treat_predictor = treat_trainer.fit(
#             treat_train_data,
#             **training_cfg.fit,
#             tuning_data=treat_tuning_data,
#             use_bag_holdout=True,
#             hyperparameters=hpars,
#             feature_metadata=feature_metadata,
#         )
#         treat_predictor.refit_full()
#         results = treat_predictor.fit_summary()

#         # normalize feature importance by scale
#         tscale = np.nanstd(df[spaceenv.treatment])
#         label = spaceenv.treatment
#         # treat_featimp = utils.feature_importance(
#         #     data=treat_train_data.drop(columns=[label]),
#         #     label=treat_train_data[label],
#         #     predict_func=treat_predictor.predict,
#         #     eval_metric=treat_predictor.eval_metric,
#         #     **training_cfg.feat_importance,
#         # )
#         treat_featimp = treat_predictor.feature_importance(
#             treat_tuning_data,
#             **training_cfg.feat_importance,
#         )
        
#         treat_featimp = dict(treat_featimp.importance)
#         treat_featimp = utils.delete_grid(treat_featimp)

#         group_treat_featimp = utils.group_by_feat(treat_featimp)
#         logging.info("Group Treat Featimp: %s", group_treat_featimp)
#         treat_featimp = {}
#         for metric, subdict in group_treat_featimp.items():
#             valid_items = {k: v for k, v in subdict.items() if not math.isnan(v)}
#             if valid_items:
#                 max_key = max(valid_items, key=valid_items.get)
#                 treat_featimp[metric] = valid_items[max_key]
#             else:
#                 treat_featimp[metric] = float('-inf')

#         # do the reduction for the case of interactions
#         if is_binary_treatment and spaceenv.binary_treatment_iteractions:
#             for c in covariates:
#                 treat_featimp[c] = max(
#                     treat_featimp.get(c, 0.0), treat_featimp.get(c + "_interact", 0.0)
#                 )
#                 if c + "_interact" in treat_featimp:
#                     treat_featimp.pop(c + "_interact")

#         treat_featimp = {c: float(treat_featimp.get(c, 0.0)) / tscale for c in covariates}
#         logging.info("Treat Featimp: %s", treat_featimp)
#         # legacy confounding score by minimum
#         cs_minimum = {k: min(treat_featimp[k], featimp[k]) for k in covariates}
#         logging.info(f"Legacy conf. score by minimum:\n{cs_minimum}")

#         # Top top_k features that are important to both treatment and outcome
#         top_k = 10
#         top_feat = dict(sorted(cs_minimum.items(), key=lambda x: x[1], reverse=True)[:top_k])
#         top_featcols = list(top_feat.keys())
#         logging.info(f"Got top {len(top_feat)} features for outcome and treatment: {top_feat}.")

        # === Compute confounding scores ===
        # The strategy for confounding scores is to compute various types
        # using the baseline model.

        # For continous treatment compute the ERF and ITE scores
        # For categorical treatmetn additionally compute the ATE score
        # For both also use the minimum of the treatment and outcome model
        # As in the first version of the paper.

        # For comparability across environments, we divide the scores by the
        # variance of the synthetic outcome.

        # Obtain counterfactuals for the others
        cs_erf = {}
        cs_ite = {}
        cs_ate = {}  # will be empty if not binary

#         for i, feat in enumerate(top_featcols):
#             key_ = feat
#             value_ = [feat] + ([feat + "_interact"] if feat + "_interact" in extra_colnames else [])
            
#             if not local_confounding:
#                 leave_out_nbr_cols = []
#                 for col in covariates + extra_colnames + [spaceenv.treatment]:
#                     if col == feat or feat + "_interact" == col:
#                         continue
#                     for dr in range(-radius, radius + 1):
#                         for dc in range(-radius, radius + 1):
#                             leave_out_nbr_cols.append(f"{col}_{dr}_{dc}")
#             else:
#                 leave_out_nbr_cols = []
#                 for col in [spaceenv.treatment]:
#                     for dr in range(-radius, radius + 1):
#                         for dc in range(-radius, radius + 1):
#                             leave_out_nbr_cols.append(f"{col}_{dr}_{dc}")
                
#                 for col in covariates + extra_colnames:
#                     if col == feat or feat + "_interact" == col:
#                         continue
#                     for dr in range(0, 1):
#                         for dc in range(0, 1):
#                             leave_out_nbr_cols.append(f"{col}_{dr}_{dc}")
            
#             if not local_confounding:
#                 cols = [cov + "_grid" for cov in covariates if feat != cov] + [spaceenv.treatment + "_grid"] + [ex + "_grid" for ex in extra_colnames if feat + "_interact" != ex] + [spaceenv.outcome] + leave_out_nbr_cols
#             else:
#                 cols = [spaceenv.treatment + "_grid"] + [spaceenv.outcome] + leave_out_nbr_cols

#             leave_out_predictor = TabularPredictor(label=spaceenv.outcome)
#             leave_out_predictor = leave_out_predictor.fit(
#                 train_data[cols],
#                 **spaceenv.autogluon.leave_out_fit,
#                 tuning_data=tuning_data[cols],
#                 use_bag_holdout=True,
#                 hyperparameters=hpars,
#                 feature_metadata=feature_metadata,
#             )
#             leave_out_predictor.refit_full()
#             results = leave_out_predictor.fit_summary()

#             leave_out_mu_cf = []        
#             for a in avals:
#                 dfcols = [cov for cov in covariates if feat != cov] + [spaceenv.treatment] + [ex for ex in extra_colnames if feat + "_interact" != ex]
#                 cfdata = df[dfcols].copy()
                
#                 cfdata_grid, temp_dir = utils.create_grid_features_compact(cfdata, 
#                                                                        radius=radius, 
#                                                                        a=a, 
#                                                                        change="center", 
#                                                                        treatment=spaceenv.treatment,
#                                                                        is_binary_treatment=is_binary_treatment,
#                                                                        spaceenv=spaceenv,
#                                                                        get_t_pct=get_t_pct,
#                                                                        spline_basis=spline_basis,
#                                                                        extra_colnames=[ex for ex in extra_colnames if feat + "_interact" != ex],
#                                                                        covariates=[cov for cov in covariates if feat != cov])

#                 predicted = leave_out_predictor.predict(cfdata_grid)
#                 leave_out_mu_cf.append(predicted)
#             leave_out_mu_cf = pd.concat(leave_out_mu_cf, axis=1)

#             logging.info(f"[{i + 1} / {len(top_featcols)}]: {key_}")

#             # compute loss normalized by the variance of the outcome
#             cf_err = (leave_out_mu_cf.values - mu_cf.values) / scale
#             cs_ite[key_] = float(np.sqrt((cf_err**2).mean(0)).mean())
#             logging.info(f"ITE: {cs_ite[key_]:.3f}")

#             erf_err = (leave_out_mu_cf.values - mu_cf.values).mean(0) / scale
#             cs_erf[key_] = float(np.abs(erf_err).mean())
#             logging.info(f"ERF: {cs_erf[key_]:.3f}")

#             if n_treatment_values == 2:
#                 cs_ate[key_] = np.abs(erf_err[1] - erf_err[0])
#                 logging.info(f"ATE: {cs_ate[key_]:.3f}")

        # === Compute the spatial smoothness of each covariate
        logging.info(f"Computing spatial smoothness of each covariate.")
        moran_I_values = {}
        for c in covariates:
            moran_I_values[c] = utils.moran_I(df[c].values, edge_list)

        # === Save results ===
        logging.info(f"Saving synthetic data, graph, and metadata")
        X = df[df.columns.difference([spaceenv.outcome, spaceenv.treatment])]
        dfout = pd.concat([A, X, mu, mu_cf, Y_synth, Y_cf, spill_mu_cf, spill_Y_cf], axis=1)

        # whens saving synthetic data, respect the original data format
        if data_file.endswith("tab"):
            dfout.to_csv(f"{output_dir}/synthetic_data.tab", sep="\t", index=True)
        elif data_file.endswith("parquet"):
            dfout.to_parquet(f"{output_dir}/synthetic_data.parquet")

        # save subgraph in the right format
        if graph_file.endswith(("graphml", "graphml.gz")):
            ext = "graphml.gz" if graph_file.endswith("graphml.gz") else "graphml"
            tgt_graph_path = f"{output_dir}/graph.{ext}"
            nx.write_graphml(graph, tgt_graph_path)

        elif graph_file.endswith("tar.gz"):
            # save edges and coords
            edges = pd.DataFrame(np.array(list(graph.edges)), columns=["source", "target"])
            coords = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
            # save again as a tar.gz
            with tarfile.open(f"{output_dir}/graph.tar.gz", "w:gz") as tar:
                os.makedirs("graph", exist_ok=True)
                edges.to_parquet("graph/edges.parquet")
                coords.to_parquet("graph/coords.parquet")
                tar.add("graph/")

        metadata = {
            "base_name": f"{spaceenv.base_name}",
            "treatment": spaceenv.treatment,
            "predicted_outcome": spaceenv.outcome,
            "synthetic_outcome": "Y_synth",
            "confounding_score": utils.sort_dict(cs_minimum),
            "confounding_score_erf": utils.sort_dict(cs_erf),
            "confounding_score_ite": utils.sort_dict(cs_ite),
            "confounding_score_ate": utils.sort_dict(cs_ate),
            "spatial_scores": utils.sort_dict(moran_I_values),
            "outcome_importance": utils.sort_dict(featimp),
            "treatment_importance": utils.sort_dict(treat_featimp),
            "covariates": list(covariates),
            "treatment_values": avals.tolist(),
            "covariate_groups": covar_groups,
            "original_residual_spatial_score": float(residual_smoothness),
            "synthetic_residual_spatial_score": float(synth_residual_smoothness),
            "original_nbrs_corr": float(residual_nbrs_corr),
            "synthetic_nbrs_corr": float(synth_residual_nbrs_corr),
            "radius": radius,
        }

        # save metadata and resolved config
        with open(f"{output_dir}/metadata.yaml", "w") as f:
            yaml.dump(metadata, f, sort_keys=False)


if __name__ == "__main__":
    main()
