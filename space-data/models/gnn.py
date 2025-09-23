import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from pytorch_lightning.callbacks import LearningRateFinder
from torch.optim import Adam
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import k_hop_subgraph

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.common.utils.resource_utils import ResourceManager

def graph_data_loader(
    X, y=None, edge_list=None,
    feat_scaler: StandardScaler | None = None,
    output_scaler: StandardScaler | None = None,
    treatment_value: float | None = None,
    radius=1,
):
    if edge_list is None:
        raise ValueError("edge_list cannot be None")

    node_ids = set(X.index)
    
    # Filter edges to only valid nodes
    filtered_edge_list = [(s, t) for s, t in edge_list if s in node_ids and t in node_ids]
    if len(filtered_edge_list) == 0:
        raise ValueError("No edges left after filtering by node_ids. Check edge_list and X.index consistency.")
    
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
    y_tensor = torch.FloatTensor(output_scaler.transform(outcome))
    
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
        hidden_dim: int = 16,
        hidden_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 1e-3,
        act="relu",
        task_type: str = "regression",  # "regression" or "classification"
        trainfeat_scaler = None,
        trainoutput_scaler = None,
    ):
        super().__init__()
        self.conv1 = GCNConv(input_dim + 1, hidden_dim)
        self.convh = nn.ModuleList(
            [GCNConv(hidden_dim + 1, hidden_dim) for _ in range(hidden_layers - 1)]
        )
        
        # For classification, output_dim should be num_classes
        self.convf = GCNConv(hidden_dim + 1, output_dim)
        
        self.act = getattr(F, act)
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.task_type = task_type
        self.output_dim = output_dim
        self.trainfeat_scaler, self.trainoutput_scaler = trainfeat_scaler, trainoutput_scaler
        self.trainer = None
        self.edge_list = None
        self.radius = None

    def forward(self, batch: torch_geometric.data.Data):
        x = batch.x
        edge_index = batch.edge_index
        treatment = x[:, 0]
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        for conv in self.convh:
            x = torch.cat([treatment[:, None], x], dim=1)
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        x = torch.cat([treatment[:, None], x], dim=1)
        x = self.convf(x, edge_index)
        
        # Apply final activation based on task
        if self.task_type == "classification":
            if self.output_dim == 1:
                # Binary classification
                x = torch.sigmoid(x)
            else:
                # Multi-class classification
                x = F.log_softmax(x, dim=1)
        # For regression, return raw output (no activation)
        
        return x

    def training_step(self, batch):
        y_hat = self(batch)
        
        # Choose loss based on task type
        if self.task_type == "classification":
            if self.output_dim == 1:
                # Binary classification - use BCE loss
                loss = F.binary_cross_entropy(y_hat.squeeze(), batch.y.squeeze())
            else:
                # Multi-class classification - use NLL loss (with log_softmax)
                loss = F.nll_loss(y_hat, batch.y.long().squeeze())
        else:
            # Regression - use MSE loss
            loss = F.mse_loss(y_hat, batch.y)
            
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        
        # Choose loss based on task type
        if self.task_type == "classification":
            if self.output_dim == 1:
                loss = F.binary_cross_entropy(y_hat.squeeze(), batch.y.squeeze())
            else:
                loss = F.nll_loss(y_hat, batch.y.long().squeeze())
        else:
            loss = F.mse_loss(y_hat, batch.y)
            
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        return optimizer
    
    def predict(self, dataset):
        dataset = dataset.loc[:, ~dataset.columns.str.contains('_nbr')]
        
        loader, *_ = graph_data_loader(X=dataset, edge_list=self.edge_list, feat_scaler=self.trainfeat_scaler, output_scaler=self.trainoutput_scaler, radius=self.radius)
        preds = self.trainer.predict(self, loader)
        preds = torch.cat([pred[0:1] for pred in preds]).flatten().reshape(-1, 1)
        preds = preds.cpu().numpy()
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

        input_dim = X.shape[1] - 1
        
        trainloader, self.trainfeat_scaler, self.trainoutput_scaler = graph_data_loader(X=X, y=y, edge_list=self.edge_list, radius=self.radius)
        valloader = None
        if X_val is not None and y_val is not None:
            print("Using provided validation data")
            valloader, _, _ = graph_data_loader(
                X=X_val, y=y_val, edge_list=self.edge_list,
                feat_scaler=self.trainfeat_scaler,  # Use same scaler as training
                output_scaler=self.trainoutput_scaler,
                radius=self.radius
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
        self.model = _GCN_impl(input_dim, task_type=task_type, **params, trainfeat_scaler=self.trainfeat_scaler, trainoutput_scaler=self.trainoutput_scaler)
        self.model.edge_list = self.edge_list
        self.model.preprocess = self.preprocess
        
        self.model.radius = self.radius
        callbacks = (
            [LearningRateFinder(min_lr=1e-5, max_lr=1.0)]
        )
        self.trainer = pl.Trainer(
            accelerator="gpu",
            enable_checkpointing=False,
            logger=False,
            gradient_clip_val=10.0,
            enable_progress_bar=True,
            callbacks=callbacks,
            max_epochs=1,
            enable_model_summary=True,
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
            "hidden_dim": 16,
            "hidden_layers": 2,
            "dropout": 1e-2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
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
    
#     def score_with_y_pred_proba(
#         self,
#         y: np.ndarray,
#         y_pred_proba: np.ndarray,
#         metric: Scorer = None,
#         sample_weight: np.ndarray = None,
#         as_error: bool = False,
#     ) -> float:
#         y = (y.values).reshape(-1, 1)
#         # assert False, f"y shape: {getattr(y, 'shape', 'no shape')}, type: {type(y)}"
        
#         if metric is None:
#             metric = self.eval_metric
#         if metric.needs_pred or metric.needs_quantile:
#             y_pred = self.predict_from_proba(y_pred_proba=y_pred_proba)
#             y_pred_proba = None
#         else:
#             y_pred = None
#         return compute_metric(
#             y=y,
#             y_pred=y_pred,
#             y_pred_proba=y_pred_proba,
#             metric=metric,
#             weights=sample_weight,
#             as_error=as_error,
#             quantile_levels=self.quantile_levels,
#         )