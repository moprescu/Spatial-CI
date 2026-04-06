"""Arctic Sea Ice environment for spatial causal inference benchmarks.

Downloads Arctic Net Flux / Longwave Radiation / Sea Ice Concentration data
from Google Drive and presents it in the SpaceDataset format expected by the
benchmark pipeline.

Data: 25 km monthly grid, 1979-2021.
  - Channel 0: HFX  (net heat flux confounder)
  - Channel 1: LWDN (downward longwave radiation treatment)
  - Channel 2: SIC  (sea ice concentration outcome)

Region of interest: East Siberian Sea + Laptev Sea.
Lag-1 setup: covariates/treatment at time t, outcome at time t+1.

Temporal supervised tensors (for UNet / CVAE+UNet):
    X : (T-1, 15, H, W)  channels = [conf, lwdn, sic_t, month_1..12]
    Y : (T-1, H, W)       target  = sic_{t+1}

Counterfactual question:
    "What would SIC at t+1 have been if LWDN at t were reduced by 5%
    throughout the region, relative to observed LWDN?"
"""

import os
import tempfile

import networkx as nx
import numpy as np
import pandas as pd
import tarfile

from sci.env import SpaceDataset
from spacebench.log import LOGGER

GOOGLE_DRIVE_FILE_ID = "1lAg393qAWkpXthfAp3v1YohP4B0zNh1h"
TARGZ_FILENAME = "Arctic_Netflux_LW_SIC_causality_25km_monthly_1979_2021.tar.gz"

# Number of discrete treatment levels for ERF evaluation
N_TREATMENT_LEVELS = 10

# Counterfactual reduction factor (5 % decrease in LWDN)
CF_REDUCTION = 0.95

# JJA month indices (0-indexed: 0=Jan … 5=Jun 6=Jul 7=Aug)
JJA_MONTHS = {5, 6, 7}


def _download_from_gdrive(file_id: str, dest_path: str) -> None:
    """Download a file from Google Drive using gdown."""
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)


class ArcticEnv:
    """
    Environment for Arctic sea-ice causal inference.

    Treatment : LWDN (continuous, downward longwave radiation)
    Outcome   : SIC at lag-1 (sea ice concentration)
    Confounder: HFX (net heat flux — observed, not hidden)
    Graph     : 8-connected grid over valid pixels

    Data preparation follows the code_snippet.py protocol:
      1. Split channels, crop to region, zero-fill NaNs
      2. Add month-of-year as 12-dim one-hot spatial maps
      3. Standardize continuous channels with train-period stats
      4. Build lag-1 tensors: X(t) → Y(t+1)
      5. Build counterfactual tensors with 5% LWDN reduction
    """

    def __init__(
        self,
        name: str = "arctic_lwdn_sic_cont",
        dir: str | None = None,
        algo_rad: int = 0,
    ):
        self.name = name
        self.dir = dir or tempfile.gettempdir()

        region_bounds = (152, 197, 85, 175)
        start_month = 1

        # ----- download / extract ----------------------------------------
        data_path = os.path.join(self.dir, TARGZ_FILENAME)
        if not os.path.exists(data_path):
            os.makedirs(self.dir, exist_ok=True)
            LOGGER.info("Downloading Arctic dataset from Google Drive...")
            _download_from_gdrive(GOOGLE_DRIVE_FILE_ID, data_path)

        np_path = os.path.join(self.dir, TARGZ_FILENAME.replace(".tar.gz", ".npy"))
        if not os.path.exists(np_path):
            LOGGER.info("Extracting dataset...")
            with tarfile.open(data_path, 'r:gz') as tar:
                tar.extractall(self.dir)

        data = np.load(np_path)

        # ==================================================================
        # 1. Split channels
        # ==================================================================
        conf = data[..., 0].astype(np.float32)  # HFX  (confounder)
        lwdn = data[..., 1].astype(np.float32)  # LWDN (treatment)
        sic  = data[..., 2].astype(np.float32)  # SIC  (outcome)

        # ==================================================================
        # 2. Crop to region
        # ==================================================================
        r0, r1, c0, c1 = region_bounds
        conf = conf[:, r0:r1, c0:c1]
        lwdn = lwdn[:, r0:r1, c0:c1]
        sic  = sic[:, r0:r1, c0:c1]

        T, H, W = conf.shape
        self.H, self.W = H, W

        # valid mask from SIC
        valid_mask = np.mean(np.isfinite(sic), axis=0) > 0.05
        self.valid_mask = valid_mask

        def zero_fill(x, mask):
            y = x.copy()
            y[~np.isfinite(y)] = 0.0
            y[:, ~mask] = 0.0
            return y.astype(np.float32)

        conf = zero_fill(conf, valid_mask)
        lwdn = zero_fill(lwdn, valid_mask)
        sic  = zero_fill(sic, valid_mask)

        # ==================================================================
        # 3. Add month of year as a control (12-dim one-hot)
        # ==================================================================
        months = ((np.arange(T) + (start_month - 1)) % 12)
        month_onehot = np.zeros((T, 12), dtype=np.float32)
        month_onehot[np.arange(T), months] = 1.0

        month_maps = np.repeat(month_onehot[:, :, None, None], H, axis=2)
        month_maps = np.repeat(month_maps, W, axis=3)  # (T, 12, H, W)

        # Temporal train/val/test split (for standardization & UNet training)
        n = T - 1  # lag-1 pairs
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        train_idx = np.arange(n_train)
        val_idx = np.arange(n_train, n_train + n_val)
        test_idx = np.arange(n_train + n_val, n)
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        # Standardize continuous channels with train-period stats
        def standardize(x, mask, tidx):
            vals = x[tidx][:, mask]
            mu = vals.mean()
            sd = vals.std()
            if sd < 1e-6:
                sd = 1.0
            z = (x - mu) / sd
            z[:, ~mask] = 0.0
            return z.astype(np.float32), float(mu), float(sd)

        conf_z, conf_mu, conf_sd = standardize(conf, valid_mask, train_idx)
        lwdn_z, lwdn_mu, lwdn_sd = standardize(lwdn, valid_mask, train_idx)
        sic_z,  sic_mu,  sic_sd  = standardize(sic,  valid_mask, train_idx)

        self.lwdn_mu, self.lwdn_sd = lwdn_mu, lwdn_sd
        self.sic_mu, self.sic_sd = sic_mu, sic_sd

        # ==================================================================
        # 4. Build lag-1 supervised tensors
        # ==================================================================
        # input at time t:   conf_t, lwdn_t, sic_t, month_t  (15 ch)
        # target at time t+1: sic_{t+1}

        X_list = [
            conf_z[:-1, None, :, :],   # (T-1, 1, H, W)
            lwdn_z[:-1, None, :, :],
            sic_z[:-1,  None, :, :],
            month_maps[:-1],           # (T-1, 12, H, W)
        ]
        X = np.concatenate(X_list, axis=1).astype(np.float32)  # (T-1, 15, H, W)
        Y = sic_z[1:].astype(np.float32)                       # (T-1, H, W)

        self.X = X
        self.Y = Y
        self.channel_names = ["conf", "lwdn", "sic_t"] + [
            f"month_{m+1}" for m in range(12)
        ]

        # ==================================================================
        # 5. Counterfactual tensors
        # ==================================================================
        A_factual = lwdn[:-1].copy()          # original scale
        A_cf = CF_REDUCTION * A_factual

        A_factual_z = ((A_factual - lwdn_mu) / lwdn_sd).astype(np.float32)
        A_cf_z      = ((A_cf      - lwdn_mu) / lwdn_sd).astype(np.float32)
        A_factual_z[:, ~valid_mask] = 0.0
        A_cf_z[:, ~valid_mask] = 0.0

        X_factual = X.copy()
        X_cf      = X.copy()
        X_factual[:, 1, :, :] = A_factual_z   # channel 1 = lwdn
        X_cf[:, 1, :, :]      = A_cf_z

        self.X_factual = X_factual
        self.X_cf = X_cf

        # Counterfactual: +18 W/m² LWDN increase
        CF_PLUS_18 = 18.0
        A_cf_plus18 = A_factual + CF_PLUS_18
        A_cf_plus18_z = ((A_cf_plus18 - lwdn_mu) / lwdn_sd).astype(np.float32)
        A_cf_plus18_z[:, ~valid_mask] = 0.0
        X_cf_plus18 = X.copy()
        X_cf_plus18[:, 1, :, :] = A_cf_plus18_z
        self.X_cf_plus18 = X_cf_plus18

        # JJA mask for lag-1 pairs
        pair_months = months[:-1]
        jja_mask = np.isin(pair_months, list(JJA_MONTHS))
        self.jja_mask = jja_mask

        # ==================================================================
        # Cross-sectional per-pixel summaries (SpaceDataset compatibility)
        # ==================================================================
        # Treatment / covariates at time t, outcome at time t+1
        lwdn_pairs = lwdn[:-1]
        sic_pairs  = sic[1:]
        conf_pairs = conf[:-1]
        sic_t_pairs = sic[:-1]  # lagged outcome as covariate

        lwdn_annual = lwdn_pairs.mean(axis=0)
        sic_annual  = sic_pairs.mean(axis=0)
        conf_annual = conf_pairs.mean(axis=0)
        sic_t_annual = sic_t_pairs.mean(axis=0)

        lwdn_summer = lwdn_pairs[jja_mask].mean(axis=0)
        sic_summer  = sic_pairs[jja_mask].mean(axis=0)

        # ----- valid pixels → nodes --------------------------------------
        valid_rows, valid_cols = np.where(valid_mask)
        N = len(valid_rows)

        coord2idx = {}
        for idx, (r, c) in enumerate(zip(valid_rows, valid_cols)):
            coord2idx[(int(r), int(c))] = idx

        treatment_annual   = lwdn_annual[valid_rows, valid_cols].astype(np.float32)
        outcome_annual     = sic_annual[valid_rows, valid_cols].astype(np.float32)
        confounder_annual  = conf_annual[valid_rows, valid_cols].astype(np.float32)
        sic_t_per_pixel    = sic_t_annual[valid_rows, valid_cols].astype(np.float32)
        treatment_summer   = lwdn_summer[valid_rows, valid_cols].astype(np.float32)

        # ----- per-pixel arrays ------------------------------------------
        self.treatment = treatment_annual
        quantiles = np.linspace(0, 1, N_TREATMENT_LEVELS)
        self.treatment_values = np.quantile(treatment_annual, quantiles).astype(
            np.float32
        )
        self.outcome = outcome_annual

        self.annual_treatment    = treatment_annual
        self.summer_treatment    = treatment_summer
        self.cf_annual_treatment = (CF_REDUCTION * treatment_annual).astype(np.float32)
        self.cf_summer_treatment = (CF_REDUCTION * treatment_summer).astype(np.float32)

        # ----- 8-connected grid edges ------------------------------------
        edges = []
        for (r, c), idx in coord2idx.items():
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                nb = (r + dr, c + dc)
                if nb in coord2idx:
                    edges.append((idx, coord2idx[nb]))
        self.edge_list = edges

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(N))
        self.graph.add_edges_from(edges)

        # ----- coordinates -----------------------------------------------
        self.coordinates = np.column_stack([valid_rows, valid_cols]).astype(int)

        # ----- covariates ------------------------------------------------
        # All observed: HFX and SIC_t (no hidden confounder)
        self.node2id = {
            f"{r}_{c}": coord2idx[(r, c)] for (r, c) in coord2idx
        }
        self.covariates_df = pd.DataFrame(
            {"hfx": confounder_annual, "sic_t": sic_t_per_pixel},
            index=[f"{r}_{c}" for r, c in zip(valid_rows, valid_cols)],
        )
        self.covariate_groups = {"hfx": ["hfx"], "sic_t": ["sic_t"]}
        # No hidden confounder → make_all() yields make(None)
        self.topfeat = [None]

        # ----- spatial parameters ----------------------------------------
        self.radius = 1
        self.conf_radius = 1
        self.datatype = "grid"

        # ----- counterfactuals (no ground truth) -------------------------
        n_tv = len(self.treatment_values)
        self.counterfactuals = np.zeros((N, n_tv), dtype=np.float32)
        self.spill_counterfactuals = np.zeros((N, n_tv), dtype=np.float32)

        # ----- dummy confounding / smoothness scores ---------------------
        self.confounding_score = {
            "erf": {}, "ate": {}, "ite": {}, "importance": {},
        }
        self.smoothness_score = {}

        # ----- grid interior selection (same logic as SpaceEnv) ----------
        from sci.algorithms.utils import get_k_hop_neighbors

        rad = max(algo_rad, self.radius)
        node_list = list(self.graph.nodes())
        nbrs = {
            node: get_k_hop_neighbors(self.graph, node, rad)
            for node in node_list
        }
        nbr_counts = {node: len(neigh) for node, neigh in nbrs.items()}
        max_count = max(nbr_counts.values())
        self.max_nodes = [
            node for node, cnt in nbr_counts.items() if cnt == max_count
        ]

        self.full_graph = self.graph.copy()
        subgraph = self.graph.subgraph(self.max_nodes).copy()
        old_to_new = {old: new for new, old in enumerate(self.max_nodes)}
        self.graph = nx.relabel_nodes(subgraph, old_to_new)

        # ----- metadata --------------------------------------------------
        self.metadata = {
            "base_name": "arcticgrid",
            "treatment": "lwdn",
            "treatment_values": [str(v) for v in self.treatment_values],
            "covariates": ["hfx", "sic_t"],
            "radius": str(self.radius),
        }
        self.config = {
            "name": self.name,
            "description": "Arctic sea ice causality benchmark (LWDN -> SIC)",
        }

    # ------------------------------------------------------------------
    # SpaceEnv-compatible interface
    # ------------------------------------------------------------------

    def make(self, missing_group: str | None = None) -> SpaceDataset:
        """Generate a SpaceDataset.

        Parameters
        ----------
        missing_group : str or None
            Covariate group to hide.  ``None`` means everything is observed
            and ``missing_covariates`` is set to ``None``.
        """
        if missing_group is None:
            obs_covars = self.covariates_df.values
            miss_covars = None
            miss_smoothness = 0.0
            miss_confounding = {
                "erf": 0.0, "ate": 0.0, "ite": 0.0, "importance": 0.0,
            }
        else:
            obs_covars_cols = [
                c
                for group, cols in self.covariate_groups.items()
                if missing_group not in cols
                for c in cols
            ]
            if obs_covars_cols:
                obs_covars = self.covariates_df[obs_covars_cols].values
            else:
                obs_covars = np.zeros(
                    (len(self.treatment), 0), dtype=np.float32
                )
            miss_covars = self.covariates_df[[missing_group]].values
            miss_smoothness = self.smoothness_score.get(missing_group, 0.0)
            miss_confounding = {}
            for k in ["erf", "ate", "ite"]:
                miss_confounding[k] = self.confounding_score[k].get(
                    missing_group, 0.0
                )
            miss_confounding["importance"] = self.confounding_score[
                "importance"
            ].get(missing_group, 0.0)

        dataset = SpaceDataset(
            treatment=self.treatment,
            covariates=obs_covars,
            missing_covariates=miss_covars,
            outcome=self.outcome,
            counterfactuals=self.counterfactuals,
            spill_counterfactuals=self.spill_counterfactuals,
            edges=self.edge_list,
            coordinates=self.coordinates,
            smoothness_score=miss_smoothness,
            confounding_score=miss_confounding,
            treatment_values=self.treatment_values,
            parent_env=self.name,
            topfeat=self.topfeat,
            radius=self.radius,
            conf_radius=self.conf_radius,
            node2id=self.node2id,
            full_edge_list=self.edge_list,
            full_treatment=self.treatment,
            full_covariates=obs_covars,
            full_outcome=self.outcome,
            full_coordinates=self.coordinates,
            datatype=self.datatype,
        )

        ds = dataset[self.max_nodes]

        # Seasonal treatment arrays for counterfactual evaluation
        ds.annual_treatment    = self.annual_treatment
        ds.summer_treatment    = self.summer_treatment
        ds.cf_annual_treatment = self.cf_annual_treatment
        ds.cf_summer_treatment = self.cf_summer_treatment

        # ---- Temporal UNet tensors (full spatial maps per timestep) ----
        ds.X          = self.X            # (T-1, 15, H, W)
        ds.Y          = self.Y            # (T-1, H, W)
        ds.X_factual  = self.X_factual    # factual input
        ds.X_cf       = self.X_cf         # counterfactual input (5% LWDN reduction)
        ds.X_cf_plus18 = self.X_cf_plus18 # counterfactual input (+18 W/m² LWDN)
        ds.valid_mask = self.valid_mask    # (H, W) bool
        ds.channel_names = self.channel_names
        ds.train_idx  = self.train_idx    # temporal split indices
        ds.val_idx    = self.val_idx
        ds.test_idx   = self.test_idx
        ds.jja_mask   = self.jja_mask
        ds.lwdn_mu    = self.lwdn_mu
        ds.lwdn_sd    = self.lwdn_sd
        ds.sic_mu     = self.sic_mu
        ds.sic_sd     = self.sic_sd
        ds.grid_H     = self.H
        ds.grid_W     = self.W
        return ds

    def make_all(self):
        """Yield one SpaceDataset (no hidden confounder)."""
        for c in self.topfeat:
            yield self.make(missing_group=c)

    def has_binary_treatment(self) -> bool:
        return len(self.treatment_values) == 2

    def __repr__(self) -> str:
        return (
            f"ArcticEnv({self.name}): Arctic LWDN->SIC causality benchmark\n"
            f"  nodes (valid pixels): {len(self.treatment)}\n"
            f"  edges: {len(self.edge_list)}\n"
            f"  X shape: {self.X.shape}  (T-1, 15, H, W)\n"
            f"  Y shape: {self.Y.shape}  (T-1, H, W)\n"
            f"  treatment: continuous LWDN\n"
            f"  counterfactual: 5% LWDN reduction\n"
        )
