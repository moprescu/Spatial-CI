import libpysal as lp
import numpy as np
import numpy.linalg as la

import pandas as pd
from pysal.model import spreg
from typing import Optional
import torch
import torch.sparse

from sci import SpaceDataset
from spacebench.algorithms import SpaceAlgo
from spacebench.log import LOGGER

from spreg import ols as OLS
from spreg import user_output as USER
from spreg import utils as UTILS

from spreg.utils import RegressionPropsY, RegressionPropsVM, set_warn
from spreg.output import output, _summary_iteration



def spdot(a, b, array_out=True):
    """
    Matrix multiplication function for PyTorch tensors and NumPy arrays
    """
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        # Ensure both are tensors on the same device
        if not isinstance(a, torch.Tensor):
            device = b.device if isinstance(b, torch.Tensor) else 'cpu'
            a = numpy_to_torch(a, device=device)
        if not isinstance(b, torch.Tensor):
            device = a.device
            b = numpy_to_torch(b, device=device)
        
        # Ensure same device
        if a.device != b.device:
            b = b.to(a.device)
        
        # Handle sparse matrices
        if a.is_sparse:
            return torch.sparse.mm(a, b)
        elif b.is_sparse:
            return torch.sparse.mm(b.t(), a.t()).t()
        else:
            return torch.mm(a, b)
    else:
        return np.dot(a, b)



class GMError_GPU(SpaceAlgo):
    """
    Wrapper of PySAL GM_Error model with GPU acceleration support.
    """
    supports_binary = True
    supports_continuous = True
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize GMError with optional device specification.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
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
        # Convert data to tensors for GPU computation
        treatment_tensor = self._to_tensor(dataset.treatment)
        covariates_tensor = self._to_tensor(dataset.covariates)
        outcome_tensor = self._to_tensor(dataset.outcome)
        
        # Add noise to covariates using GPU
        if self.use_gpu:
            noise = torch.normal(0, 1e-6, size=covariates_tensor.shape, device=self.device)
            noisy_covars_tensor = covariates_tensor + noise
        else:
            # Fallback to numpy for CPU
            noisy_covars = dataset.covariates + np.random.normal(
                scale=1e-6, size=dataset.covariates.shape
            )
            noisy_covars_tensor = self._to_tensor(noisy_covars)
        
        # Concatenate treatment and covariates
        x_tensor = torch.cat([treatment_tensor.unsqueeze(1), noisy_covars_tensor], dim=1)
        
        # Compute standardization parameters on GPU
        self.mu_x_tensor = x_tensor.mean(0)
        self.sig_x_tensor = x_tensor.std(0)
        self.mu_y_tensor = outcome_tensor.mean()
        self.sig_y_tensor = outcome_tensor.std()
        
        # Standardize data
        x_standardized = (x_tensor - self.mu_x_tensor) / self.sig_x_tensor
        y_standardized = (outcome_tensor - self.mu_y_tensor) / self.sig_y_tensor
        
        # Convert back to numpy for PySAL compatibility
        x = self._to_numpy(x_standardized)
        y = self._to_numpy(y_standardized)
        
        # Store standardization parameters as numpy for compatibility
        self.mu_x = self._to_numpy(self.mu_x_tensor)
        self.sig_x = self._to_numpy(self.sig_x_tensor)
        self.mu_y = self._to_numpy(self.mu_y_tensor)
        self.sig_y = self._to_numpy(self.sig_y_tensor)
        
        LOGGER.debug("Computing spatial weights")
        
        # Adjacency matrix computation - can be accelerated on GPU for large matrices
        if self.use_gpu and dataset.adjacency_matrix().shape[0] > 1000:  # Only use GPU for large matrices
            adjmat_tensor = self._to_tensor(dataset.adjacency_matrix())
            
            # Add noise and zero diagonal on GPU
            adjmat_tensor += 1e-6
            diag_mask = torch.eye(adjmat_tensor.shape[0], device=self.device, dtype=torch.bool)
            adjmat_tensor[diag_mask] = 0.0
            
            adjmat = self._to_numpy(adjmat_tensor)
        else:
            # Use original CPU computation for smaller matrices
            adjmat = dataset.adjacency_matrix()
            adjmat += 1e-6
            adjmat[np.diag_indices_from(adjmat)] = 0.0
        
        w = lp.weights.util.full2W(adjmat)
        LOGGER.debug("PySAL GM_Error_Het")
        # PySAL GM_Error_Het doesn't support GPU directly, so we use CPU here
        self.model = GM_Error_Het(x=x, y=y, w=w, max_iter=1)
        
        # Store treatment coefficient
        self.t_coef = self.model.betas[1, 0] * self.sig_y / self.sig_x[0]
    
    def eval(self, dataset: SpaceDataset) -> dict:
        """
        Evaluate the model with GPU acceleration for large datasets.
        """
        if self.use_gpu:
            # GPU-accelerated evaluation
            outcome_tensor = self._to_tensor(dataset.outcome)
            treatment_tensor = self._to_tensor(dataset.treatment)
            
            # Compute ITE for each treatment value
            ite_list = []
            for a in dataset.treatment_values:
                a_tensor = torch.full_like(treatment_tensor, a)
                ite_a = outcome_tensor + self.t_coef * (a_tensor - treatment_tensor)
                ite_list.append(ite_a)
            
            ite_tensor = torch.stack(ite_list, dim=1)
            erf_tensor = ite_tensor.mean(0)
            
            # Convert back to numpy
            ite = self._to_numpy(ite_tensor)
            erf = self._to_numpy(erf_tensor)
            
        else:
            # Original CPU computation
            ite = [
                dataset.outcome + self.t_coef * (a - dataset.treatment)
                for a in dataset.treatment_values
            ]
            ite = np.stack(ite, axis=1)
            erf = ite.mean(0)
        
        effects = {"erf": erf, "ite": ite}
        
        if dataset.has_binary_treatment():
            effects["ate"] = self.t_coef
            
        return effects
    
    def to(self, device: str):
        """Move model to specified device."""
        self.device = torch.device(device)
        self.use_gpu = self.device.type == 'cuda'
        
        # Move tensors to new device if they exist
        if hasattr(self, 'mu_x_tensor'):
            self.mu_x_tensor = self.mu_x_tensor.to(self.device)
            self.sig_x_tensor = self.sig_x_tensor.to(self.device)
            self.mu_y_tensor = self.mu_y_tensor.to(self.device)
            self.sig_y_tensor = self.sig_y_tensor.to(self.device)
        
        return self

    
def torch_to_numpy(tensor):
    """Convert PyTorch tensor to NumPy array, handling CUDA tensors"""
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()
    return tensor

def numpy_to_torch(array, device=None):
    """Convert NumPy array to PyTorch tensor"""
    if not isinstance(array, torch.Tensor):
        tensor = torch.from_numpy(np.asarray(array)).float()
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    return array

def scipy_to_torch_sparse(sparse_matrix, device=None):
    """Convert scipy sparse matrix to PyTorch sparse tensor"""
    if hasattr(sparse_matrix, 'tocoo'):
        coo = sparse_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.from_numpy(coo.data).float()
        size = coo.shape
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size, dtype=torch.float32)
        if device is not None:
            sparse_tensor = sparse_tensor.to(device)
        return sparse_tensor
    else:
        raise ValueError("Input must be a scipy sparse matrix")

def torch_sparse_to_scipy(sparse_tensor):
    """Convert PyTorch sparse tensor to scipy sparse matrix"""
    if sparse_tensor.is_cuda:
        sparse_tensor = sparse_tensor.cpu()
    
    coo = sparse_tensor.coalesce()
    indices = coo.indices().numpy()
    values = coo.values().numpy()
    shape = coo.shape
    
    from scipy import sparse as SP
    return SP.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()


class BaseGM_Error_Het:
    """
    GMM method for a spatial error model with heteroskedasticity (note: no
    consistency checks, diagnostics or constant added); based on
    Arraiz et al. (2010), following Anselin (2011).

    GPU-accelerated version using PyTorch CUDA tensors with original function calls.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : Sparse matrix
                   Spatial weights sparse matrix
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from
                   Arraiz et al. (2010). Note: epsilon provides an additional
                   stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from Arraiz et al. (2010). Note: max_iter provides
                   an additional stop condition.
    step1c       : boolean
                   If True, then include Step 1c from Arraiz et al. (2010).
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
    use_gpu      : boolean
                   If True, converts arrays to PyTorch CUDA tensors for GPU acceleration.
                   
    Attributes
    ----------
    betas        : array or tensor
                   kx1 array of estimated coefficients
    u            : array or tensor
                   nx1 array of residuals
    e_filtered   : array or tensor
                   nx1 array of spatially filtered residuals
    predy        : array or tensor
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array or tensor
                   nx1 array for dependent variable
    x            : array or tensor
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from Arraiz et al. (2010).
    iteration    : integer
                   Number of iterations of steps 2a and 2b from Arraiz et al. (2010).
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array or tensor
                   Variance covariance matrix (kxk)
    xtx          : float
                   X'X
    use_gpu      : boolean
                   Whether GPU acceleration is being used
    device       : torch.device
                   Device where tensors are stored (CPU or CUDA)

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> import libpysal
    >>> import spreg
    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = 'r'
    >>> # GPU version
    >>> reg = BaseGM_Error_Het(y, X, w.sparse, step1c=True, use_gpu=True)
    >>> # Convert results back to CPU for printing if needed
    >>> betas_result = torch_to_numpy(reg.betas) if reg.use_gpu else reg.betas
    >>> vm_diag = torch_to_numpy(torch.sqrt(torch.diag(reg.vm))) if reg.use_gpu else np.sqrt(np.diag(reg.vm))
    >>> print(np.around(np.hstack((betas_result, vm_diag.reshape(4,1))), 4))
    """

    def __init__(self, y, x, w, max_iter=1, epsilon=0.00001, step1c=False, hard_bound=False, use_gpu=True):

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.step1c = step1c
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Convert inputs to PyTorch tensors if using GPU
        if self.use_gpu:
            y = numpy_to_torch(y, device=self.device)
            x = numpy_to_torch(x, device=self.device)
            # Convert sparse matrix to PyTorch sparse tensor
            if hasattr(w, 'tocsr') or hasattr(w, 'toarray'):
                w = scipy_to_torch_sparse(w, device=self.device)
        
        # 1a. OLS --> \tilde{betas}
        ols = BaseOLS(y=y, x=x)
        self.x, self.y, self.n, self.k, self.xtx = ols.x, ols.y, ols.n, ols.k, ols.xtx
        wA1 = get_A1_het(w)

        # 1b. GMM --> \tilde{\lambda1}
        moments = UTILS._moments2eqs(wA1, w, ols.u)
        lambda1 = UTILS.optim_moments(moments)

        if step1c:
            # 1c. GMM --> \tilde{\lambda2}
            sigma = get_psi_sigma(w, ols.u, lambda1)
            vc1 = get_vc_het(w, wA1, sigma)
            lambda2 = UTILS.optim_moments(moments, vc1)
        else:
            lambda2 = lambda1

        lambda_old = lambda2

        self.iteration, eps = 0, 1
        while self.iteration < max_iter and eps > epsilon:
            # 2a. reg -->\hat{betas}
            xs = UTILS.get_spFilter(w, lambda_old, self.x)
            ys = UTILS.get_spFilter(w, lambda_old, self.y)
            ols_s = BaseOLS(y=ys, x=xs)
            self.predy = spdot(self.x, ols_s.betas)
            self.u = self.y - self.predy

            # 2b. GMM --> \hat{\lambda}
            sigma_i = get_psi_sigma(w, self.u, lambda_old)
            vc_i = get_vc_het(w, wA1, sigma_i)
            moments_i = UTILS._moments2eqs(wA1, w, self.u)
            lambda3 = UTILS.optim_moments(moments_i, vc_i)
            eps = abs(lambda3 - lambda_old)
            lambda_old = lambda3
            self.iteration += 1

        self.iter_stop = UTILS.iter_msg(self.iteration, max_iter)
        if hard_bound:
            if abs(lambda3) >= 0.99:
                raise Exception("Spatial error parameter was outside the bounds of -0.99 and 0.99")
        else:
            if abs(lambda3) >= 0.99:
                self.set_warn("Spatial error parameter was outside the bounds of -0.99 and 0.99")

        sigma = get_psi_sigma(w, self.u, lambda3)
        vc3 = get_vc_het(w, wA1, sigma)
        self.vm = get_vm_het(moments_i[0], lambda3, self, w, vc3)
        
        # Use PyTorch's vstack if arrays are on GPU, otherwise use NumPy
        if self.use_gpu:
            if isinstance(ols_s.betas, torch.Tensor):
                lambda3_tensor = torch.tensor([[lambda3]], device=self.device, dtype=torch.float32)
                self.betas = torch.vstack((ols_s.betas, lambda3_tensor))
            else:
                ols_betas_tensor = numpy_to_torch(ols_s.betas, device=self.device)
                lambda3_tensor = torch.tensor([[lambda3]], device=self.device, dtype=torch.float32)
                self.betas = torch.vstack((ols_betas_tensor, lambda3_tensor))
            
            # Handle sparse matrix multiplication for filtered residuals
            if w.is_sparse:
                w_u = torch.sparse.mm(w, self.u.unsqueeze(-1) if self.u.dim() == 1 else self.u)
                self.e_filtered = self.u - lambda3 * w_u.squeeze(-1)
            else:
                self.e_filtered = self.u - lambda3 * torch.mm(w, self.u)
        else:
            self.betas = np.vstack((ols_s.betas, lambda3))
            if hasattr(w, 'multiply') or hasattr(w, 'dot'):
                # Scipy sparse matrix
                self.e_filtered = self.u - lambda3 * w @ self.u
            else:
                # Dense array
                self.e_filtered = self.u - lambda3 * np.dot(w, self.u)
            
        self._cache = {}




class GM_Error_Het(BaseGM_Error_Het):
    """
    GMM method for a spatial error model with heteroskedasticity, with results
    and diagnostics; based on :cite:`Arraiz2010`, following
    :cite:`Anselin2011`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : pysal W object
                   Spatial weights object
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX-Error type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
    max_iter     : int
                   Maximum number of iterations of steps 2a and 2b from :cite:`Arraiz2010`.
                   Note: epsilon provides an additional
                   stop condition.
    epsilon      : float
                   Minimum change in lambda required to stop iterations of
                   steps 2a and 2b from :cite:`Arraiz2010`. Note: max_iter provides
                   an additional stop condition.
    step1c       : boolean
                   If True, then include Step 1c from :cite:`Arraiz2010`.
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    latex        : boolean
                   Specifies if summary is to be printed in latex format
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
    Attributes
    ----------
    output       : dataframe
                   regression results pandas dataframe
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    iter_stop    : string
                   Stop criterion reached during iteration of steps 2a and 2b
                   from :cite:`Arraiz2010`.
    iteration    : integer
                   Number of iterations of steps 2a and 2b from :cite:`Arraiz2010`.
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    vm           : array
                   Variance covariance matrix (kxk)
    std_err      : array
                   1xk array of standard errors of the betas
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    xtx          : float
                   :math:`X'X`
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    title        : string
                   Name of the regression method used

    Examples
    --------
    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import libpysal

    Open data on Columbus neighborhood crime (49 areas) using libpysal.io.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')

    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) and CRIME (crime) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this class adds a vector of ones to the
    independent variables passed in.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``columbus.shp``.

    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, his allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are all set with the preliminaries, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> from spreg import GM_Error_Het
    >>> reg = GM_Error_Het(y, X, w=w, step1c=True, name_y='home value', name_x=['income', 'crime'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. This class offers an error model that explicitly accounts
    for heteroskedasticity and that unlike the models from
    ``spreg.error_sp``, it allows for inference on the spatial
    parameter.

    >>> print(reg.name_x)
    ['CONSTANT', 'income', 'crime', 'lambda']

    Hence, we find the same number of betas as of standard errors,
    which we calculate taking the square root of the diagonal of the
    variance-covariance matrix:

    >>> print(np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4))
    [[47.9963 11.479 ]
     [ 0.7105  0.3681]
     [-0.5588  0.1616]
     [ 0.4118  0.168 ]]

    Alternatively, we can have a summary of the output by typing:
    print(reg.summary)

    """

    def __init__(
        self,
        y,
        x,
        w,
        slx_lags=0,
        slx_vars="All",
        max_iter=1,
        epsilon=0.00001,
        step1c=False,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        latex=False,
        hard_bound=False,
    ):

        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        x_constant, name_x, warn = USER.check_constant(x, name_x)
        name_x = USER.set_name_x(name_x, x_constant)  # initialize in case None, includes constant
        set_warn(self, warn)
        self.title = "GM SPATIALLY WEIGHTED LEAST SQUARES (HET)"

        if slx_lags >0:
            #lag_x = get_lags(w, x_constant[:, 1:], slx_lags)
            #x_constant = np.hstack((x_constant, lag_x))
#            name_x += USER.set_name_spatial_lags(name_x, slx_lags)
            #name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags) # exclude constant

            x_constant,name_x = USER.flex_wx(w,x=x_constant,name_x=name_x,constant=True,
                                             slx_lags=slx_lags,slx_vars=slx_vars)

            self.title += " WITH SLX (SLX-Error)"

        # OLD
        #if slx_lags >0:
            #lag_x = get_lags(w, x_constant[:, 1:], slx_lags)
            #x_constant = np.hstack((x_constant, lag_x))
#            name_x += USER.set_name_spatial_lags(name_x, slx_lags)
            #name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)  # no constant
            #self.title += " WITH SLX (SLX-Error)"

        BaseGM_Error_Het.__init__(
            self,
            y=y,
            x=x_constant,
            w=w.sparse,
            max_iter=max_iter,
            step1c=step1c,
            epsilon=epsilon,
            hard_bound = hard_bound
        )


        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
#        self.name_x = USER.set_name_x(name_x, x_constant)
        self.name_x = name_x  # constant already included
        self.name_x.append("lambda")
        self.name_w = USER.set_name_w(name_w, w)
        self.output = pd.DataFrame(self.name_x, columns=['var_names'])
        self.output['var_type'] = ['o'] + ['x'] * (len(self.name_x)-2) + ['lambda']
        self.output['regime'], self.output['equation'] = (0, 0)
        self.other_top = _summary_iteration(self)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)

        
class BaseOLS:
    """
    PyTorch CUDA-compatible Ordinary least squares (OLS) implementation
    
    This version automatically detects whether input arrays are NumPy or PyTorch
    and uses the appropriate backend for computations.
    
    Parameters
    ----------
    y            : array (NumPy or PyTorch tensor)
                   nx1 array for dependent variable
    x            : array (NumPy or PyTorch tensor)
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    robust       : string
                   If 'white', then a White consistent estimator of the
                   variance-covariance matrix is given.  If 'hac', then a
                   HAC consistent estimator of the variance-covariance
                   matrix is given. Default set to None.
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n.
    
    Attributes
    ----------
    betas        : array or tensor
                   kx1 array of estimated coefficients
    u            : array or tensor
                   nx1 array of residuals
    predy        : array or tensor
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array or tensor
                   nx1 array for dependent variable
    x            : array or tensor
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array or tensor
                   Variance covariance matrix (kxk)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    xtx          : tensor or array
                   X'X
    xtxi         : tensor or array
                   (X'X)^-1
    using_torch  : boolean
                   Whether PyTorch tensors are being used
    device       : torch.device
                   Device where tensors are stored
    """
    
    def __init__(self, y, x, robust=None, gwk=None, sig2n_k=True):
        # Detect if we're using PyTorch tensors
        
        self.using_torch = isinstance(y, torch.Tensor) or isinstance(x, torch.Tensor)
        if self.using_torch:
            # Ensure both arrays are PyTorch tensors on same device
            if not isinstance(x, torch.Tensor):
                device = y.device if isinstance(y, torch.Tensor) else 'cpu'
                x = numpy_to_torch(x, device=device)
            if not isinstance(y, torch.Tensor):
                device = x.device
                y = numpy_to_torch(y, device=device)

            # Ensure same device
            if x.device != y.device:
                y = y.to(x.device)

            self.device = x.device
            self.array_lib = torch
            self.linalg = torch.linalg
        else:
            self.device = None
            self.array_lib = np
            self.linalg = np.linalg
            
        self.x = x
        self.y = y
        
        # Core OLS computations
        self.xtx = spdot(self.x.t() if self.using_torch else self.x.T, self.x)
        xty = spdot(self.x.t() if self.using_torch else self.x.T, y)
        
        try:
            if self.using_torch:
                self.xtxi = torch.linalg.inv(self.xtx)
            else:
                self.xtxi = self.linalg.inv(self.xtx)
        except (torch.linalg.LinAlgError if self.using_torch else np.linalg.LinAlgError):
            # Fallback to pseudo-inverse for singular matrices
            if self.using_torch:
                self.xtxi = torch.linalg.pinv(self.xtx)
            else:
                self.xtxi = self.linalg.pinv(self.xtx)
            
        self.betas = spdot(self.xtxi, xty)
        self.predy = spdot(self.x, self.betas)
        self.u = y - self.predy
        
        # Store dimensions
        self.n, self.k = self.x.shape
        
        # Set sigma squared
        if sig2n_k:
            self.sig2 = self.sig2n_k
        else:
            self.sig2 = self.sig2n
            
        # Handle robust variance estimation
        if robust is not None:
            # Basic variance-covariance matrix
            self.vm = self.sig2 * self.xtxi
        else:
            self.vm = self.sig2 * self.xtxi

    @property
    def mean_y(self):
        """Mean of dependent variable"""
        if self.using_torch:
            return float(torch.mean(self.y).item())
        else:
            return float(np.mean(self.y))
    
    @property
    def std_y(self):
        """Standard deviation of dependent variable"""
        if self.using_torch:
            return float(torch.std(self.y).item())
        else:
            return float(np.std(self.y))
    
    @property
    def utu(self):
        """Sum of squared residuals"""
        if self.using_torch:
            return float(torch.sum(self.u ** 2).item())
        else:
            return float(np.sum(self.u ** 2))
    
    @property
    def sig2n(self):
        """Sigma squared (computed with n in the denominator)"""
        return self.utu / self.n
    
    @property
    def sig2n_k(self):
        """Sigma squared (computed with n-k in the denominator)"""
        return self.utu / (self.n - self.k)


def get_psi_sigma(w, u, lamb):
    """
    PyTorch CUDA compatible version of get_psi_sigma
    
    Computes the Sigma matrix needed to compute Psi

    Parameters
    ----------
    w           : Sparse tensor or matrix
                  Spatial weights sparse matrix/tensor
    u           : tensor or array
                  nx1 vector of residuals
    lamb        : float
                  Lambda

    Returns
    -------
    E           : sparse tensor or matrix
                  Diagonal matrix with squared filtered residuals
    """
    
    if isinstance(w, torch.Tensor) or isinstance(u, torch.Tensor):
        # Working with PyTorch tensors
        if not isinstance(w, torch.Tensor):
            device = u.device if isinstance(u, torch.Tensor) else 'cpu'
            w = scipy_to_torch_sparse(w, device=device) if hasattr(w, 'tocoo') else numpy_to_torch(w, device=device)
        if not isinstance(u, torch.Tensor):
            u = numpy_to_torch(u, device=w.device)
        
        # Ensure same device
        if hasattr(w, 'device') and w.device != u.device:
            u = u.to(w.device)
        
        # Compute filtered residuals
        if w.is_sparse:
            wu = torch.sparse.mm(w, u.unsqueeze(-1) if u.dim() == 1 else u).squeeze(-1)
        else:
            wu = torch.mm(w, u.unsqueeze(-1) if u.dim() == 1 else u).squeeze(-1)
        
        e = (u - lamb * wu) ** 2
        
        # Create diagonal sparse matrix
        n = e.shape[0]
        indices = torch.stack([torch.arange(n, device=e.device), torch.arange(n, device=e.device)])
        E = torch.sparse_coo_tensor(indices, e.flatten(), (n, n), dtype=torch.float32, device=e.device)
        return E.coalesce()
        
    else:
        # Working with NumPy/scipy - original implementation
        from scipy import sparse as SP
        e = (u - lamb * (w @ u)) ** 2
        E = SP.dia_matrix((e.flat, 0), shape=(w.shape[0], w.shape[0]))
        return E.tocsr()


def get_vc_het(w, wA1, E):
    """
    PyTorch CUDA compatible version of get_vc_het
    
    Computes the VC matrix Psi based on lambda as in Arraiz et al

    Parameters
    ----------
    w           : Sparse tensor or matrix
                  Spatial weights sparse matrix/tensor
    wA1         : Sparse tensor or matrix  
                  wA1 matrix
    E           : sparse tensor or matrix
                  Sigma (diagonal matrix)

    Returns
    -------
    Psi         : tensor or array
                  2x2 array with estimator of the variance-covariance matrix
    """
    
    if isinstance(w, torch.Tensor) or isinstance(E, torch.Tensor):
        # Working with PyTorch tensors
        device = None
        if isinstance(w, torch.Tensor):
            device = w.device
        elif isinstance(E, torch.Tensor):
            device = E.device
        
        # Convert all to tensors on same device
        if not isinstance(w, torch.Tensor):
            w = scipy_to_torch_sparse(w, device=device) if hasattr(w, 'tocoo') else numpy_to_torch(w, device=device)
        if not isinstance(wA1, torch.Tensor):
            wA1 = scipy_to_torch_sparse(wA1, device=device) if hasattr(wA1, 'tocoo') else numpy_to_torch(wA1, device=device)
        if not isinstance(E, torch.Tensor):
            E = scipy_to_torch_sparse(E, device=device) if hasattr(E, 'tocoo') else numpy_to_torch(E, device=device)
        
        # Sparse matrix operations
        if wA1.is_sparse and E.is_sparse:
            aPatE = 2 * torch.sparse.mm(wA1, E.to_dense()).to_sparse()
        else:
            aPatE = 2 * torch.mm(wA1.to_dense() if wA1.is_sparse else wA1, 
                                E.to_dense() if E.is_sparse else E)
        
        # Compute w + w.T
        if w.is_sparse:
            wPwt = w + w.t()
            if E.is_sparse:
                wPwtE = torch.sparse.mm(wPwt, E.to_dense()).to_sparse()
            else:
                wPwtE = torch.mm(wPwt.to_dense(), E)
        else:
            wPwt = w + w.t()
            wPwtE = torch.mm(wPwt, E.to_dense() if E.is_sparse else E)
        
        # Compute psi components
        if aPatE.is_sparse:
            psi11 = torch.sparse.mm(aPatE, aPatE.to_dense())
            psi12 = torch.sparse.mm(aPatE, wPwtE.to_dense() if wPwtE.is_sparse else wPwtE)
        else:
            psi11 = torch.mm(aPatE, aPatE)
            psi12 = torch.mm(aPatE, wPwtE.to_dense() if wPwtE.is_sparse else wPwtE)
        
        if wPwtE.is_sparse:
            psi22 = torch.sparse.mm(wPwtE, wPwtE.to_dense())
        else:
            psi22 = torch.mm(wPwtE, wPwtE)
        
        # Extract diagonal elements and sum
        psi11_diag = torch.diag(psi11.to_dense() if hasattr(psi11, 'is_sparse') and psi11.is_sparse else psi11)
        psi12_diag = torch.diag(psi12.to_dense() if hasattr(psi12, 'is_sparse') and psi12.is_sparse else psi12)
        psi22_diag = torch.diag(psi22.to_dense() if hasattr(psi22, 'is_sparse') and psi22.is_sparse else psi22)
        
        psi = [torch.sum(psi11_diag).item(), torch.sum(psi12_diag).item(), torch.sum(psi22_diag).item()]
        
        result = torch.tensor([[psi[0], psi[1]], [psi[1], psi[2]]], device=device, dtype=torch.float32)
        return result / (2.0 * w.shape[0])
        
    else:
        # Working with NumPy/scipy - original implementation
        aPatE = 2 * wA1 @ E
        wPwtE = (w + w.T) @ E

        psi11 = aPatE @ aPatE
        psi12 = aPatE @ wPwtE
        psi22 = wPwtE @ wPwtE
        psi = list(map(np.sum, [psi11.diagonal(), psi12.diagonal(), psi22.diagonal()]))
        return np.array([[psi[0], psi[1]], [psi[1], psi[2]]]) / (2.0 * w.shape[0])


def get_vm_het(G, lamb, reg, w, psi):
    """
    PyTorch CUDA compatible version of get_vm_het
    
    Computes the variance-covariance matrix Omega as in Arraiz et al

    Parameters
    ----------
    G           : tensor or array
                  G from moments equations
    lamb        : float
                  Final lambda from spHetErr estimation
    reg         : regression object
                  output instance from a regression model
    w           : Sparse tensor or matrix
                  Spatial weights sparse matrix/tensor
    psi         : tensor or array
                  2x2 array with the variance-covariance matrix of the moment equations

    Returns
    -------
    vm          : tensor or array
                  (k+1)x(k+1) array with the variance-covariance matrix of the parameters
    """
    
    if reg.using_torch if hasattr(reg, 'using_torch') else False:
        # Working with PyTorch tensors
        device = reg.device if hasattr(reg, 'device') else 'cpu'
        
        # Convert inputs to tensors
        if not isinstance(G, torch.Tensor):
            G = numpy_to_torch(G, device=device)
        if not isinstance(psi, torch.Tensor):
            psi = numpy_to_torch(psi, device=device)
        if not isinstance(w, torch.Tensor):
            w = scipy_to_torch_sparse(w, device=device) if hasattr(w, 'tocoo') else numpy_to_torch(w, device=device)
        
        # Ensure same device
        G = G.to(device)
        psi = psi.to(device)
        
        lamb_tensor = torch.tensor([[lamb]], device=device, dtype=torch.float32)
        J_vec = torch.tensor([[1], [2 * lamb]], device=device, dtype=torch.float32)
        J = torch.mm(G, J_vec)
        
        # Get spatial filter
        Zs = get_spFilter_torch(w, lamb, reg.x)
        
        # Compute ZstEZs
        E_sigma = get_psi_sigma(w, reg.u, lamb)
        if E_sigma.is_sparse:
            ZstE = torch.sparse.mm(Zs.t(), E_sigma.to_dense())
        else:
            ZstE = torch.mm(Zs.t(), E_sigma)
        ZstEZs = torch.mm(ZstE, Zs)
        
        # Compute ZsZsi
        ZstZs = torch.mm(Zs.t(), Zs)
        ZsZsi = torch.linalg.inv(ZstZs)
        
        # Compute omega components
        omega11 = w.shape[0] * torch.mm(torch.mm(ZsZsi, ZstEZs), ZsZsi)
        
        psi_inv = torch.linalg.inv(psi)
        JtPsiInv = torch.mm(J.t(), psi_inv)
        omega22 = torch.linalg.inv(torch.mm(JtPsiInv, J))
        
        # Construct final variance matrix
        zero = torch.zeros((reg.k, 1), dtype=torch.float32, device=device)
        zero_t = torch.zeros((1, reg.k), dtype=torch.float32, device=device)
        
        top = torch.hstack((omega11, zero))
        bottom = torch.hstack((zero_t, omega22))
        vm = torch.vstack((top, bottom)) / w.shape[0]
        
        return vm
        
    else:
        # Working with NumPy/scipy - original implementation
        import scipy.linalg as la
        
        J = np.dot(G, np.array([[1], [2 * lamb]]))
        Zs = get_spFilter_numpy(w, lamb, reg.x)
        ZstEZs = spdot((Zs.T @ get_psi_sigma(w, reg.u, lamb)), Zs)
        ZsZsi = la.inv(spdot(Zs.T, Zs))
        omega11 = w.shape[0] * np.dot(np.dot(ZsZsi, ZstEZs), ZsZsi)
        omega22 = la.inv(np.dot(np.dot(J.T, la.inv(psi)), J))
        zero = np.zeros((reg.k, 1), float)
        vm = (
            np.vstack((np.hstack((omega11, zero)), np.hstack((zero.T, omega22))))
            / w.shape[0]
        )
        return vm

    
def get_A1_het(S, chunk_size=1000):
    """
    Dense CUDA implementation for computing S.T @ S - diag(S.T @ S)
    Much simpler and often faster than sparse operations.
    
    Parameters
    ----------
    S           : torch.Tensor (dense)
                  Spatial weights matrix on CUDA device
    chunk_size  : int
                  Size of chunks to process if matrix is too large for full computation
    """
    if not isinstance(S, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")
    
    device = S.device
    n = S.shape[0]
    dtype = S.dtype
    
    print(f"Matrix info: n={n}, device={device}, dtype={dtype}")
    
    # Clear cache at start
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    try:
        # Try full computation first
        print("Attempting full matrix multiplication...")
        StS = torch.mm(S.t(), S)
        
        # Remove diagonal elements
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        A1 = StS * mask.to(dtype)
        
        print("Full computation successful!")
        return A1
        
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "insufficient resources" in error_msg:
            print(f"Full computation failed due to memory: {e}")
            print(f"Falling back to chunked computation with chunk_size={chunk_size}")
            
            # Clear cache and try chunked approach
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return _chunked_dense_computation(S, chunk_size)
        else:
            raise e


def _chunked_dense_computation(S, chunk_size):
    """
    Chunked dense computation when full matrix doesn't fit in memory
    """
    device = S.device
    n = S.shape[0]
    dtype = S.dtype
    
    # Initialize result tensor
    A1 = torch.zeros(n, n, dtype=dtype, device=device)
    
    total_chunks = (n + chunk_size - 1) // chunk_size
    processed_chunks = 0
    
    print(f"Processing {total_chunks}x{total_chunks} chunks...")
    
    # Compute S.T @ S in chunks
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        
        # Extract chunk of rows from S.T (columns from S)
        S_chunk_cols = S[:, i_start:i_end]  # Shape: (n, chunk_size)
        
        for j_start in range(0, n, chunk_size):
            j_end = min(j_start + chunk_size, n)
            
            # Extract chunk of columns from S.T (rows from S) 
            S_chunk_rows = S[j_start:j_end, :]  # Shape: (chunk_size, n)
            
            # Compute chunk: S_chunk_rows @ S_chunk_cols
            # This gives us StS[j_start:j_end, i_start:i_end]
            try:
                chunk_result = torch.mm(S_chunk_rows, S_chunk_cols)
                
                # Remove diagonal elements within this chunk
                if i_start == j_start:  # Diagonal chunk
                    chunk_i_size = i_end - i_start
                    chunk_j_size = j_end - j_start
                    diag_size = min(chunk_i_size, chunk_j_size)
                    
                    # Create mask for diagonal elements
                    diag_mask = torch.eye(diag_size, dtype=torch.bool, device=device)
                    if chunk_i_size != chunk_j_size:
                        # Handle rectangular chunks
                        full_mask = torch.zeros(chunk_j_size, chunk_i_size, dtype=torch.bool, device=device)
                        full_mask[:diag_size, :diag_size] = diag_mask
                        diag_mask = full_mask
                    
                    chunk_result = chunk_result * (~diag_mask).to(dtype)
                
                # Store result
                A1[j_start:j_end, i_start:i_end] = chunk_result
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Memory error in chunk ({j_start}:{j_end}, {i_start}:{i_end})")
                    # Could implement further subdivision here if needed
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            processed_chunks += 1
            if processed_chunks % 100 == 0:
                print(f"Processed {processed_chunks}/{total_chunks**2} chunks")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    return A1






def get_A1_het_scipy_impl(S):
    """
    Original scipy implementation for backward compatibility
    """
    try:
        from scipy import sparse as SP
    except ImportError:
        raise ImportError("SciPy is required for sparse matrix operations")
    
    StS = S.T @ S
    d = SP.dia_matrix(([StS.diagonal()], [0]), shape=S.shape).tocsr()
    return StS - d