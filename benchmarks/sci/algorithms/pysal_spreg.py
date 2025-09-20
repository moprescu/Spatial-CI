import libpysal as lp
import numpy as np
from .gm_lag import GM_Lag

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
        LOGGER.debug("Running GM_Lag model")

        self.model = GM_Lag(x=x, y=y, w=w, robust='white', w_lags=self.w_lags)
        self.t_coef = self.model.betas[1, 0] * self.sig_y / self.sig_x[0]

    def eval(self, dataset: SpaceDataset):
        ite = [
            dataset.outcome + self.t_coef * (a - dataset.treatment)
            for a in dataset.treatment_values
        ]
        ite = np.stack(ite, axis=1)

        effects = {"erf": ite.mean(0), "ite": ite}

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