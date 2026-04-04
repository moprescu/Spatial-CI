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

There is no ground-truth ITE / ATE, so counterfactuals are set to zero.
"""

import os
import tempfile

import networkx as nx
import numpy as np
import pandas as pd

from sci.env import SpaceDataset
from spacebench.log import LOGGER

GOOGLE_DRIVE_FILE_ID = "1lAg393qAWkpXthfAp3v1YohP4B0zNh1h"
NPY_FILENAME = "Arctic_Netflux_LW_SIC_causality_25km_monthly_1979_2021.npy"


def _download_from_gdrive(file_id: str, dest_path: str) -> None:
    """Download a file from Google Drive using gdown."""
    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)


class ArcticEnv:
    """
    Environment for Arctic sea-ice causal inference.

    Treatment : LWDN (binarised at the median)
    Outcome   : SIC at lag-1
    Confounder: HFX (the single covariate, masked in ``make()``)
    Graph     : 4-connected grid over valid pixels

    Attributes match the SpaceEnv interface so it plugs into ``run.py``.
    """

    def __init__(
        self,
        name: str = "arctic_lwdn_sic_disc",
        dir: str | None = None,
        algo_rad: int = 0,
    ):
        self.name = name
        self.dir = dir or tempfile.gettempdir()

        # Region: East Siberian Sea + Laptev Sea
        region_bounds = (152, 197, 85, 175)

        # ----- download --------------------------------------------------
        data_path = os.path.join(self.dir, NPY_FILENAME)
        if not os.path.exists(data_path):
            os.makedirs(self.dir, exist_ok=True)
            LOGGER.info("Downloading Arctic dataset from Google Drive...")
            _download_from_gdrive(GOOGLE_DRIVE_FILE_ID, data_path)

        data = np.load(data_path)

        # ----- split channels --------------------------------------------
        conf_raw = data[..., 0].astype(np.float32)  # HFX
        lwdn_raw = data[..., 1].astype(np.float32)  # LWDN
        sic_raw = data[..., 2].astype(np.float32)   # SIC

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

        # ----- time-average (lag-1) per pixel ----------------------------
        lwdn_mean = lwdn_raw[:-1].mean(axis=0)  # treatment at t
        sic_mean = sic_raw[1:].mean(axis=0)      # outcome at t+1
        conf_mean = conf_raw[:-1].mean(axis=0)   # confounder at t

        # ----- valid pixels → nodes --------------------------------------
        valid_rows, valid_cols = np.where(valid_mask)
        N = len(valid_rows)

        coord2idx = {}
        for idx, (r, c) in enumerate(zip(valid_rows, valid_cols)):
            coord2idx[(int(r), int(c))] = idx

        treatment_cont = lwdn_mean[valid_rows, valid_cols]
        outcome = sic_mean[valid_rows, valid_cols]
        confounder = conf_mean[valid_rows, valid_cols]

        # binarise treatment at median
        median_trt = np.median(treatment_cont)
        self.treatment = (treatment_cont > median_trt).astype(float)
        self.treatment_values = np.array([0.0, 1.0])
        self.outcome = outcome.astype(np.float32)

        # ----- 4-connected grid edges ------------------------------------
        edges = []
        for (r, c), idx in coord2idx.items():
            for dr, dc in [(0, 1), (1, 0)]:
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
            f"{r}_{c}": coord2idx[(r, c)]
            for (r, c) in coord2idx
        }
        self.covariates_df = pd.DataFrame(
            {"hfx": confounder},
            index=[f"{r}_{c}" for r, c in zip(valid_rows, valid_cols)],
        )
        self.covariate_groups = {"hfx": ["hfx"]}
        self.topfeat = ["hfx"]

        # ----- spatial parameters ----------------------------------------
        self.radius = 1
        self.conf_radius = 0
        self.datatype = "grid"

        # ----- counterfactuals (no ground truth) -------------------------
        self.counterfactuals = np.zeros((N, 2), dtype=np.float32)
        self.spill_counterfactuals = np.zeros((N, 2), dtype=np.float32)

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
            "treatment_values": ["0", "1"],
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
            # all covariates are masked → empty observed set
            obs_covars = np.zeros((len(self.treatment), 0), dtype=np.float32)

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

        return dataset[self.max_nodes]

    def make_all(self):
        """Yield one SpaceDataset per top confounder (just HFX here)."""
        for c in self.topfeat:
            yield self.make(missing_group=c)

    def has_binary_treatment(self) -> bool:
        return len(self.treatment_values) == 2

    def __repr__(self) -> str:
        return (
            f"ArcticEnv({self.name}): Arctic LWDN→SIC causality benchmark\n"
            f"  nodes (valid pixels): {len(self.treatment)}\n"
            f"  edges: {len(self.edge_list)}\n"
            f"  treatment: binary (LWDN > median)\n"
            f"  confounder: HFX\n"
            f"  counterfactuals: zeros (no ground truth)\n"
        )
