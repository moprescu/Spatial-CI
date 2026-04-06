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
Time-averaged per pixel to produce a cross-sectional SpaceDataset.

Counterfactual question (from code_snippet.py):
    "What would SIC at t+1 have been if LWDN at t were reduced by 5%
    throughout the region, relative to observed LWDN?"

    A_factual(s) = LWDN(s)
    A_cf(s)      = 0.95 * LWDN(s)   for all s in the region

The eval records two effects:
    cf_annual_pct  – average annual % increase in SIC from 5% LWDN reduction
    cf_summer_pct  – average summer (JJA) % increase in SIC from 5% LWDN reduction
"""

import os
import tempfile

import networkx as nx
import numpy as np
import pandas as pd
import tarfile
import io

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
    Confounder: HFX (net heat flux, masked in ``make()``)
    Graph     : 8-connected grid over valid pixels

    Counterfactual: 5 % uniform reduction of LWDN across the region.

    Attributes match the SpaceEnv interface so it plugs into ``run.py``.
    """

    def __init__(
        self,
        name: str = "arctic_lwdn_sic_cont",
        dir: str | None = None,
        algo_rad: int = 0,
    ):
        self.name = name
        self.dir = dir or tempfile.gettempdir()

        # Region: East Siberian Sea + Laptev Sea
        region_bounds = (152, 197, 85, 175)
        start_month = 1

        # ----- download --------------------------------------------------
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

        # ----- split channels (same order as code_snippet.py) ------------
        conf_raw = data[..., 0].astype(np.float32)  # HFX  (confounder)
        lwdn_raw = data[..., 1].astype(np.float32)  # LWDN (treatment)
        sic_raw = data[..., 2].astype(np.float32)   # SIC  (outcome)

        # ----- crop to region --------------------------------------------
        r0, r1, c0, c1 = region_bounds
        conf_raw = conf_raw[:, r0:r1, c0:c1]
        lwdn_raw = lwdn_raw[:, r0:r1, c0:c1]
        sic_raw = sic_raw[:, r0:r1, c0:c1]

        T, H, W = conf_raw.shape

        # ----- valid mask ------------------------------------------------
        valid_mask = np.mean(np.isfinite(sic_raw), axis=0) > 0.05

        def zero_fill(x):
            y = x.copy()
            y[~np.isfinite(y)] = 0.0
            y[:, ~valid_mask] = 0.0
            return y.astype(np.float32)

        conf_raw = zero_fill(conf_raw)
        lwdn_raw = zero_fill(lwdn_raw)
        sic_raw = zero_fill(sic_raw)

        # ----- month indices for each lag-1 pair -------------------------
        # t=0 is January 1979.  Lag-1 pairs use treatment at t, outcome
        # at t+1, so usable timesteps are t = 0 … T-2.
        months = ((np.arange(T) + (start_month - 1)) % 12)  # 0-indexed
        pair_months = months[:-1]  # month of the treatment for each pair
        jja_mask = np.isin(pair_months, list(JJA_MONTHS))

        # ----- lag-1 averages per pixel ----------------------------------
        lwdn_pairs = lwdn_raw[:-1]            # (T-1, H, W) treatment at t
        sic_pairs = sic_raw[1:]               # (T-1, H, W) outcome  at t+1
        conf_pairs = conf_raw[:-1]            # (T-1, H, W) confounder at t

        # annual average
        lwdn_annual = lwdn_pairs.mean(axis=0)
        sic_annual = sic_pairs.mean(axis=0)
        conf_annual = conf_pairs.mean(axis=0)

        # summer (JJA) average
        lwdn_summer = lwdn_pairs[jja_mask].mean(axis=0)
        sic_summer = sic_pairs[jja_mask].mean(axis=0)

        # ----- valid pixels -> nodes -------------------------------------
        valid_rows, valid_cols = np.where(valid_mask)
        N = len(valid_rows)

        coord2idx = {}
        for idx, (r, c) in enumerate(zip(valid_rows, valid_cols)):
            coord2idx[(int(r), int(c))] = idx

        # per-pixel vectors (annual)
        treatment_annual = lwdn_annual[valid_rows, valid_cols].astype(np.float32)
        outcome_annual = sic_annual[valid_rows, valid_cols].astype(np.float32)
        confounder_annual = conf_annual[valid_rows, valid_cols].astype(np.float32)

        # per-pixel vectors (summer)
        treatment_summer = lwdn_summer[valid_rows, valid_cols].astype(np.float32)

        # ----- continuous treatment (annual for training) ----------------
        self.treatment = treatment_annual
        quantiles = np.linspace(0, 1, N_TREATMENT_LEVELS)
        self.treatment_values = np.quantile(treatment_annual, quantiles).astype(
            np.float32
        )
        self.outcome = outcome_annual

        # ----- seasonal treatment arrays for counterfactual inference -----
        self.annual_treatment = treatment_annual
        self.summer_treatment = treatment_summer
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
        self.node2id = {
            f"{r}_{c}": coord2idx[(r, c)] for (r, c) in coord2idx
        }
        self.covariates_df = pd.DataFrame(
            {"hfx": confounder_annual},
            index=[f"{r}_{c}" for r, c in zip(valid_rows, valid_cols)],
        )
        self.covariate_groups = {"hfx": ["hfx"]}
        self.topfeat = ["hfx"]

        # ----- spatial parameters ----------------------------------------
        self.radius = 1
        self.conf_radius = 0
        self.datatype = "grid"

        # ----- counterfactuals (no ground truth) -------------------------
        n_tv = len(self.treatment_values)
        self.counterfactuals = np.zeros((N, n_tv), dtype=np.float32)
        self.spill_counterfactuals = np.zeros((N, n_tv), dtype=np.float32)

        # ----- dummy confounding / smoothness scores ---------------------
        self.confounding_score = {
            "erf": {"hfx": 0.0},
            "ate": {"hfx": 0.0},
            "ite": {"hfx": 0.0},
            "importance": {"hfx": 0.0},
        }
        self.smoothness_score = {"hfx": 0.0}

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

        # ----- metadata (for compatibility) ------------------------------
        self.metadata = {
            "base_name": "arcticgrid",
            "treatment": "lwdn",
            "treatment_values": [str(v) for v in self.treatment_values],
            "covariates": ["hfx"],
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
        """Generate a SpaceDataset by masking a covariate group."""
        if missing_group is None:
            missing_group = "hfx"

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

        # Seasonal treatment arrays for counterfactual inference.
        # The eval methods use these to compute annual / summer effects.
        ds.annual_treatment = self.annual_treatment
        ds.summer_treatment = self.summer_treatment
        ds.cf_annual_treatment = self.cf_annual_treatment
        ds.cf_summer_treatment = self.cf_summer_treatment
        return ds

    def make_all(self):
        """Yield one SpaceDataset per top confounder (just HFX here)."""
        for c in self.topfeat:
            yield self.make(missing_group=c)

    def has_binary_treatment(self) -> bool:
        return len(self.treatment_values) == 2

    def __repr__(self) -> str:
        return (
            f"ArcticEnv({self.name}): Arctic LWDN->SIC causality benchmark\n"
            f"  nodes (valid pixels): {len(self.treatment)}\n"
            f"  edges: {len(self.edge_list)}\n"
            f"  treatment: continuous LWDN\n"
            f"  counterfactual: 5% LWDN reduction\n"
            f"  confounder: HFX\n"
        )
