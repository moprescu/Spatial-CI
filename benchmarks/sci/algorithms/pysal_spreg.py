import libpysal as lp
import numpy as np
from pysal.model import spreg

from sci.env import SpaceDataset
from spacebench.algorithms import SpaceAlgo
from spacebench.log import LOGGER
import scipy

# class GMError(SpaceAlgo):
#     """
#     Wrapper of PySAL GM_Error model.
#     """

#     supports_binary = True
#     supports_continuous = True

#     def fit(self, dataset: SpaceDataset):
#         noisy_covars = dataset.covariates + np.random.normal(
#             scale=1e-6, size=dataset.covariates.shape
#         )
#         x = np.concatenate([dataset.treatment[:, None], noisy_covars], axis=1)
#         self.mu_x, self.sig_x = x.mean(0), x.std(0)
#         x = (x - self.mu_x) / self.sig_x
#         self.mu_y, self.sig_y = dataset.outcome.mean(), dataset.outcome.std()
#         y = (dataset.outcome - self.mu_y) / self.sig_y

#         LOGGER.debug("Computing spatial weights")
#         # add a bit of noise to every non-diagonal element to avoid singular matrix
#         adjmat = dataset.adjacency_matrix()
#         adjmat += 1e-6
#         adjmat[np.diag_indices_from(adjmat)] = 0.0
#         w = lp.weights.util.full2W(adjmat)

#         self.model = spreg.GM_Error_Het(x=x, y=y, w=w)
#         self.t_coef = self.model.betas[1, 0] * self.sig_y / self.sig_x[0]

#     def eval(self, dataset: SpaceDataset):
#         ite = [
#             dataset.outcome + self.t_coef * (a - dataset.treatment)
#             for a in dataset.treatment_values
#         ]
#         ite = np.stack(ite, axis=1)

#         effects = {"erf": ite.mean(0), "ite": ite}

#         if dataset.has_binary_treatment():
#             effects["ate"] = self.t_coef

#         return effects



class GMLag(SpaceAlgo):
    """
    Wrapper of PySAL GM_Lag model.
    """

    supports_binary = True
    supports_continuous = True

    def __init__(self, w_lags: int = 1):
        """
        Arguments
        ----------

        w_lags : int
            Number of spatial lags to include in the model. See the GM_Lag
            documentation for more details.
        """
        super().__init__()
        self.w_lags = w_lags

    def fit(self, dataset: SpaceDataset):
        noisy_covars = dataset.covariates + np.random.normal(
            scale=1e-6, size=dataset.covariates.shape
        )
        x = np.concatenate([dataset.treatment[:, None], noisy_covars], axis=1)
        self.mu_x, self.sig_x = x.mean(0), x.std(0)
        x = (x - self.mu_x) / self.sig_x
        self.mu_y, self.sig_y = dataset.outcome.mean(), dataset.outcome.std()
        y = (dataset.outcome - self.mu_y) / self.sig_y


        LOGGER.debug("Computing spatial weights")
        # add a bit of noise to every non-diagonal element to avoid singular matrix
        adjmat = dataset.adjacency_matrix()
        adjmat += 1e-6
        adjmat[np.diag_indices_from(adjmat)] = 0.0
        w = lp.weights.util.full2W(adjmat)
        self.w = w
        self.W_full = w.full()[0]
        LOGGER.debug("Running GM_Lag model")

        self.model = spreg.GM_Lag(x=x, y=y, w=w, robust='white', w_lags=self.w_lags)
        self.t_coef = self.model.betas[1, 0] * self.sig_y / self.sig_x[0]

    def eval(self, dataset: SpaceDataset):
        n = self.W_full.shape[0]
        rho = float(self.model.rho)
        M = np.linalg.inv(np.eye(n) - rho * self.W_full)

        # ── standard ite (own treatment effect) ──────────────────────────────
        ite_list = []
        for a in dataset.treatment_values:
            delta_t = a - dataset.treatment
            own_direct = np.diag(M) * (self.t_coef * delta_t)
            ite_list.append(dataset.outcome + own_direct)
        ite = np.stack(ite_list, axis=1)

        # ── inward spillover ──────────────────────────────────────────────────
        # Set ALL other units to 1, then 0; hold i fixed at 0 (its contribution
        # is excluded by zeroing the diagonal of M before applying the shock)
        #
        # shock_1[j] = t_coef * (1 - t_j)   for all j  (everyone moves to 1)
        # shock_0[j] = t_coef * (0 - t_j)   for all j  (everyone moves to 0)
        #
        # effect on i = (M @ shock)[i], but excluding i's own shock:
        #   = (M - diag(M)) @ shock   [i]
        #   i.e. use M with diagonal zeroed out so i's own treatment change
        #   doesn't contaminate the spillover estimate

        M_offdiag = M - np.diag(np.diag(M))   # zero out diagonal

        shock_1 = self.t_coef * (1.0 - dataset.treatment)
        shock_0 = self.t_coef * (0.0 - dataset.treatment)

        spill_ite = (M_offdiag @ shock_1) - (M_offdiag @ shock_0)
        # equivalently: M_offdiag @ (shock_1 - shock_0)
        # = M_offdiag @ (t_coef * 1 - t_coef * 0) * ones
        # = t_coef * M_offdiag @ ones

        effects = {
            "erf":       ite.mean(0),
            "ite":       ite,
            "spill":     spill_ite.mean(),
            "spill_ite": spill_ite,
        }
        if dataset.has_binary_treatment():
            effects["ate"] = self.t_coef
        return effects

    def available_estimands(self):
        return ["ate", "erf", "ite"]
    
    def predict(self, dataset: SpaceDataset):
        # Standardize covariates using training stats
        x = np.concatenate([dataset.treatment[:, None], dataset.covariates], axis=1)
        x = (x - self.mu_x) / self.sig_x
        
        # Get model parameters from fitted GM_Lag
        rho = float(self.model.rho)  # spatial parameter (stored as scalar)
        beta = self.model.betas[:-1].flatten()  # coefficients excluding rho (last element)
        
        # Use stored weights matrix
        W = self.w.sparse  # scipy sparse matrix from stored weights
        
        # For spatial lag model: y = X*beta + rho*W*y + u
        # Rearranging: (I - rho*W)*y = X*beta + u  
        # So: y = (I - rho*W)^(-1) * X*beta
        
        if len(x) == W.shape[0]:
            # If predicting on same spatial structure as training data
            n = W.shape[0]
            I = scipy.sparse.eye(n, format='csr')
            A = I - rho * W
            y_hat = scipy.sparse.linalg.spsolve(A, x @ beta)
        else:
            # For new data without spatial structure, use non-spatial prediction
            # This is a limitation - true spatial prediction requires spatial weights for new locations
            y_hat = x @ beta
            # Note: This ignores spatial effects for new observations
        
        # Rescale back to original outcome units
        y_hat = y_hat * self.sig_y + self.mu_y
        
        return y_hat

class GMSpatialDurbin(SpaceAlgo):
    """
    Spatial Durbin Model (SDM) wrapper — extends GM_Lag by including
    spatially lagged covariates (WX) to better capture spillover effects.
    """
    supports_binary = True
    supports_continuous = True

    def __init__(self, w_lags: int = 1, slx_lags: int = 1):
        super().__init__()
        self.w_lags = w_lags
        self.slx_lags = slx_lags  # controls inclusion of WX terms

    def fit(self, dataset: SpaceDataset):
        noisy_covars = dataset.covariates + np.random.normal(
            scale=1e-6, size=dataset.covariates.shape
        )
        x = np.concatenate([dataset.treatment[:, None], noisy_covars], axis=1)
        self.mu_x, self.sig_x = x.mean(0), x.std(0)
        x = (x - self.mu_x) / self.sig_x
        self.mu_y, self.sig_y = dataset.outcome.mean(), dataset.outcome.std()
        y = (dataset.outcome - self.mu_y) / self.sig_y

        LOGGER.debug("Computing spatial weights")
        adjmat = dataset.adjacency_matrix()
        adjmat += 1e-6
        adjmat[np.diag_indices_from(adjmat)] = 0.0
        w = lp.weights.util.full2W(adjmat)
        w.transform = 'r'  # row-standardize so all rows sum to 1
        self.w = w
        self.W_full = w.full()[0]

        LOGGER.debug("Running Spatial Durbin Model (GM_Lag with slx_lags)")
        # slx_lags=1 tells PySAL to append WX columns → this is what makes it SDM
        self.model = spreg.GM_Lag(
            x=x, y=y, w=w,
            robust='white',
            w_lags=self.w_lags,
            slx_lags=self.slx_lags   # KEY CHANGE: adds WX terms
        )

        # --- Parse betas ---
        # GM_Lag with slx_lags=1 lays out betas as:
        #   [const, X_cols..., WX_cols..., rho]
        n_x_cols = x.shape[1]  # includes treatment + covariates

        # Direct coefficient on (standardized) treatment
        self.t_coef = self.model.betas[1, 0] * self.sig_y / self.sig_x[0]

        # Theta: WX coefficients (same width as X, starting after X block)
        # betas layout: [const(1), X(n_x_cols), WX(n_x_cols), rho(1)]
        wx_start = 1 + n_x_cols          # index where WX block begins
        wx_end   = 1 + 2 * n_x_cols      # index where WX block ends
        self.theta = self.model.betas[wx_start:wx_end, 0]  # shape (n_x_cols,)

        # Theta for treatment specifically (index 0 in X → index 0 in WX block)
        self.t_theta = self.theta[0] * self.sig_y / self.sig_x[0]

    def eval(self, dataset: SpaceDataset):
        n = self.W_full.shape[0]
        rho = float(self.model.rho)
        M = np.linalg.inv(np.eye(n) - rho * self.W_full)
        M_offdiag = M - np.diag(np.diag(M))

        # ── standard ite ─────────────────────────────────────────────────────
        ite_list = []
        for a in dataset.treatment_values:
            delta_t = a - dataset.treatment
            direct_shock = (
                self.t_coef * delta_t
                + self.t_theta * (self.W_full @ delta_t)
            )
            own_direct = np.diag(M) * direct_shock
            ite_list.append(dataset.outcome + own_direct)
        ite = np.stack(ite_list, axis=1)

        # ── inward spillover ──────────────────────────────────────────────────
        # SDM shock has two parts:
        #   beta part:  t_coef * delta_t
        #   theta part: t_theta * W @ delta_t  (neighbors' treatments affect i via WX)
        # Both are evaluated with everyone else set to 1 vs 0,
        # i's own contribution removed via M_offdiag

        for neighbor_val, sign in [(1, 1), (0, -1)]:
            delta_t = np.full(n, neighbor_val, dtype=float) - dataset.treatment
            shock = self.t_coef * delta_t + self.t_theta * (self.W_full @ delta_t)
            if neighbor_val == 1:
                spill_ite = sign * (M_offdiag @ shock)
            else:
                spill_ite += sign * (M_offdiag @ shock)

        effects = {
            "erf":       ite.mean(0),
            "ite":       ite,
            "spill":     spill_ite.mean(),
            "spill_ite": spill_ite,
        }
        if dataset.has_binary_treatment():
            effects["ate"] = self.t_coef
        return effects

    def available_estimands(self):
        return ["ate", "erf", "ite"]

    def predict(self, dataset: SpaceDataset):
        x = np.concatenate([dataset.treatment[:, None], dataset.covariates], axis=1)
        x = (x - self.mu_x) / self.sig_x

        rho  = float(self.model.rho)
        n_x_cols = x.shape[1]

        # Extract only the X betas (not WX, not rho)
        # betas layout: [const(1), X(n_x_cols), WX(n_x_cols), rho(1)]
        beta = self.model.betas[:1 + n_x_cols].flatten()  # const + X coefficients

        W = self.w.sparse

        if len(x) == W.shape[0]:
            # Build WX and augment x for the SDM prediction
            WX = W @ x  # shape (n, n_x_cols)
            x_aug = np.concatenate([x, WX], axis=1)  # const already in x

            # Full beta including WX thetas (excluding rho)
            beta_full = self.model.betas[:-1].flatten()  # all but rho

            n = W.shape[0]
            I = scipy.sparse.eye(n, format='csr')
            A = I - rho * W
            y_hat = scipy.sparse.linalg.spsolve(A, x_aug @ beta_full[1:] + beta_full[0])
        else:
            y_hat = x @ beta[1:] + beta[0]

        y_hat = y_hat * self.sig_y + self.mu_y
        return y_hat
    
if __name__ == "__main__":
    import sys
    import spacebench

    env_name = spacebench.DataMaster().list_envs()[0]
    env = spacebench.SpaceEnv(env_name)
    dataset = env.make()

    algo = GMError()
    algo.fit(dataset)
    effects1 = algo.eval(dataset)

    algo = GMLag()
    algo.fit(dataset)
    effects2 = algo.eval(dataset)

    sys.exit(0)