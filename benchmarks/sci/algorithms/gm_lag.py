"""
Spatial Two Stages Least Squares with PyTorch CUDA
"""

import numpy as np
import torch
import numpy.linalg as la
from spreg import user_output as USER
from spreg.utils import RegressionPropsY, RegressionPropsVM, set_warn
from spreg import robust as ROBUST

import pandas as pd
from spreg.output import output, _spat_diag_out, _spat_pseudo_r2, _summary_impacts
from itertools import compress
import scipy.sparse as SPf
import copy

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def to_torch(arr, device=device):
    """Convert numpy array to torch tensor on specified device"""
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr.astype(np.float32)).to(device)
    elif torch.is_tensor(arr):
        return arr.to(device)
    else:
        return torch.tensor(arr, dtype=torch.float32, device=device)

def to_numpy(tensor):
    """Convert torch tensor back to numpy array"""
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    else:
        return tensor

def lag_spatial(w, y):
    """
    Spatial lag operator using PyTorch.
    """
    # Convert inputs to torch tensors
    if hasattr(w, 'sparse'):
        # Handle PySAL weights object
        w_sparse = w.sparse
        # Convert scipy sparse matrix to torch sparse tensor
        w_coo = w_sparse.tocoo()
        indices = torch.LongTensor([w_coo.row, w_coo.col]).to(device)
        values = torch.FloatTensor(w_coo.data).to(device)
        w_torch = torch.sparse_coo_tensor(indices, values, w_coo.shape).to(device)
    else:
        # Assume w is already a matrix
        w_torch = to_torch(w)
    
    y_torch = to_torch(y)
    
    # Perform sparse matrix multiplication
    if w_torch.is_sparse:
        result = torch.sparse.mm(w_torch, y_torch)
    else:
        result = torch.mm(w_torch, y_torch)
    
    return to_numpy(result)

def power_expansion(
    w, data, scalar, post_multiply=False, threshold=0.0000000001, max_iterations=None
):
    """
    Compute the inverse using power expansion with PyTorch.
    """
    # Convert to torch tensors
    if hasattr(w, 'sparse'):
        w_sparse = w.sparse
        w_coo = w_sparse.tocoo()
        indices = torch.LongTensor([w_coo.row, w_coo.col]).to(device)
        values = torch.FloatTensor(w_coo.data).to(device)
        ws = torch.sparse_coo_tensor(indices, values, w_coo.shape).to(device)
    else:
        ws = to_torch(w)
    
    data_torch = to_torch(data)
    scalar_torch = torch.tensor(scalar, dtype=torch.float32, device=device)
    
    if post_multiply:
        data_torch = data_torch.T
    
    running_total = data_torch.clone()
    increment = data_torch.clone()
    count = 1
    test = 10000000
    if max_iterations == None:
        max_iterations = 10000000
        
    while test > threshold and count <= max_iterations:
        if post_multiply:
            if ws.is_sparse:
                increment = torch.sparse.mm(increment, ws) * scalar_torch
            else:
                increment = torch.mm(increment, ws) * scalar_torch
        else:
            if ws.is_sparse:
                increment = torch.sparse.mm(ws, increment) * scalar_torch
            else:
                increment = torch.mm(ws, increment) * scalar_torch
        
        running_total += increment
        test_old = test
        test = float(torch.norm(increment).cpu())
        
        if test > test_old:
            raise Exception(
                "power expansion will not converge, check model specification and that weight are less than 1"
            )
        count += 1
    
    return to_numpy(running_total)

def inverse_prod(
    w,
    data,
    scalar,
    post_multiply=False,
    inv_method="power_exp",
    threshold=0.0000000001,
    max_iterations=None,
):
    """
    Inverse product computation with PyTorch.
    """
    if inv_method == "power_exp":
        inv_prod = power_expansion(
            w,
            data,
            scalar,
            post_multiply=post_multiply,
            threshold=threshold,
            max_iterations=max_iterations,
        )
    elif inv_method == "true_inv":
        # Convert to torch for matrix operations
        try:
            if hasattr(w, 'n'):
                n = w.n
                w_full = w.full()[0]
            else:
                n = w.shape[0]
                w_full = w
            
            eye = torch.eye(n, dtype=torch.float32, device=device)
            w_torch = to_torch(w_full)
            scalar_torch = torch.tensor(scalar, dtype=torch.float32, device=device)
            
            matrix = torch.linalg.inv(eye - scalar_torch * w_torch)
            data_torch = to_torch(data)
            
            if post_multiply:
                inv_prod = torch.mm(data_torch.T, matrix)
            else:
                inv_prod = torch.mm(matrix, data_torch)
            
            inv_prod = to_numpy(inv_prod)
        except:
            # Fallback to numpy if torch fails
            try:
                matrix = la.inv(np.eye(w.n) - (scalar * w.full()[0]))
            except:
                matrix = la.inv(np.eye(w.shape[0]) - (scalar * w))
            if post_multiply:
                inv_prod = np.matmul(data.T, matrix)
            else:
                inv_prod = np.matmul(matrix, data)
    else:
        raise Exception("Invalid method selected for inversion.")
    return inv_prod

def sp_att(w, y, predy, w_y, rho, hard_bound=False):
    """Spatial attributes calculation with PyTorch."""
    # Convert to numpy for this function as it interfaces with existing code
    y_np = to_numpy(y) if torch.is_tensor(y) else y
    predy_np = to_numpy(predy) if torch.is_tensor(predy) else predy
    w_y_np = to_numpy(w_y) if torch.is_tensor(w_y) else w_y
    
    xb = predy_np - rho * w_y_np
    if np.abs(rho) < 1:
        predy_sp = inverse_prod(w, xb, rho)
        warn = None
        resid_sp = y_np - predy_sp
    else:
        if hard_bound:
            raise Exception(
                "Spatial autoregressive parameter is outside the maximum/minimum bounds."
            )
        else:
            warn = "*** WARNING: Estimate for spatial lag coefficient is outside the boundary (-1, 1). ***"
            predy_sp = np.zeros(y_np.shape, float)
            resid_sp = np.zeros(y_np.shape, float)
    
    return predy_sp, resid_sp, warn

def spdot(a, b, array_out=True):
    """
    Matrix multiplication with PyTorch support.
    """
    # Convert to torch if numpy arrays
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        a_torch = to_torch(a)
        b_torch = to_torch(b)
        result = torch.mm(a_torch, b_torch)
        if array_out:
            return to_numpy(result)
        else:
            return result
    elif (
        type(a).__name__ == "csr_matrix"
        or type(b).__name__ == "csr_matrix"
        or type(a).__name__ == "csc_matrix"
        or type(b).__name__ == "csc_matrix"
    ):
        # Handle sparse matrices - convert to torch sparse if possible
        ab = a @ b
        if array_out:
            if type(ab).__name__ == "csc_matrix" or type(ab).__name__ == "csr_matrix":
                ab = ab.toarray()
        return ab
    else:
        # Handle torch tensors
        if torch.is_tensor(a) and torch.is_tensor(b):
            if a.is_sparse or b.is_sparse:
                if a.is_sparse:
                    result = torch.sparse.mm(a, b)
                else:
                    result = torch.sparse.mm(b.t(), a.t()).t()
            else:
                result = torch.mm(a, b)
            
            if array_out:
                return to_numpy(result)
            else:
                return result
        else:
            raise Exception(
                "Invalid format for 'spdot' argument: %s and %s"
                % (type(a).__name__, type(b).__name__)
            )

def sphstack(a, b, array_out=False):
    """
    Horizontal stacking with PyTorch support.
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if array_out:
            return np.hstack((a, b))
        else:
            a_torch = to_torch(a)
            b_torch = to_torch(b)
            return torch.cat((a_torch, b_torch), dim=1)
    elif torch.is_tensor(a) and torch.is_tensor(b):
        result = torch.cat((a, b), dim=1)
        if array_out:
            return to_numpy(result)
        else:
            return result
    elif type(a).__name__ == "csr_matrix" or type(b).__name__ == "csr_matrix":
        ab = SP.hstack((a, b), format="csr")
        if array_out:
            if type(ab).__name__ == "csr_matrix":
                ab = ab.toarray()
        return ab
    else:
        raise Exception(
            "Invalid format for 'sphstack' argument: %s and %s"
            % (type(a).__name__, type(b).__name__)
        )

def get_lags(w, x, w_lags):
    """
    Calculates spatial lags using PyTorch.
    """
    lag = lag_spatial(w, x)
    spat_lags = lag
    for i in range(w_lags - 1):
        lag = lag_spatial(w, lag)
        spat_lags = sphstack(spat_lags, lag, array_out=True)
    return spat_lags

def get_lags_split(w, x, max_lags, split_at):
    """
    Calculates spatial lags split into groups using PyTorch.
    """
    rs_l = lag = lag_spatial(w, x)
    rs_h = None
    if 0 < split_at < max_lags:
        for _ in range(split_at-1):
            lag = lag_spatial(w, lag)
            rs_l = sphstack(rs_l, lag, array_out=True)

        for i in range(max_lags - split_at):
            lag = lag_spatial(w, lag)
            rs_h = sphstack(rs_h, lag, array_out=True) if i > 0 else lag
    else:
        raise ValueError("max_lags must be greater than split_at and split_at must be greater than 0")

    return rs_l, rs_h

def set_endog(y, x, w, yend, q, w_lags, lag_q, slx_lags=0, slx_vars="all"):
    # Create spatial lag of y
    yl = lag_spatial(w, y)
    # spatial and non-spatial instruments
    if issubclass(type(yend), np.ndarray):
        if slx_lags > 0:
            lag_x, lag_xq = get_lags_split(w, x, slx_lags+1, slx_lags)
        else:
            lag_xq = x
        if lag_q:
            lag_vars = sphstack(lag_xq, q, array_out=True)
        else:
            lag_vars = lag_xq
        spatial_inst = get_lags(w, lag_vars, w_lags)
        q = sphstack(q, spatial_inst, array_out=True)
        yend = sphstack(yend, yl, array_out=True)
    elif yend == None:  # spatial instruments only
        if slx_lags > 0:
            lag_x, lag_xq = get_lags_split(w, x, slx_lags+w_lags, slx_lags)
        else:
            lag_xq = get_lags(w, x, w_lags)
        q = lag_xq
        yend = yl
    else:
        raise Exception("invalid value passed to yend")
    if slx_lags == 0:
        return yend, q
    else:  # adjust returned lag_x here using slx_vars
        if (isinstance(slx_vars,list)):     # slx_vars has True,False
            if len(slx_vars) != x.shape[1] :
                raise Exception("slx_vars incompatible with x column dimensions")
            else:  # use slx_vars to extract proper columns
                vv = slx_vars * slx_lags
                lag_x = lag_x[:,vv]
            return yend, q, lag_x
        else:  # slx_vars is "All"
            return yend, q, lag_x

class BaseTSLS(RegressionPropsY, RegressionPropsVM):
    """
    Two stage least squares with PyTorch CUDA support.
    """

    def __init__(
        self, y, x, yend, q=None, h=None, robust=None, gwk=None, sig2n_k=False
    ):

        if issubclass(type(q), np.ndarray) and issubclass(type(h), np.ndarray):
            raise Exception("Please do not provide 'q' and 'h' together")
        if q is None and h is None:
            raise Exception("Please provide either 'q' or 'h'")

        self.y = y
        self.n = y.shape[0]
        self.x = x

        self.kstar = yend.shape[1]
        # including exogenous and endogenous variables
        z = sphstack(self.x, yend, array_out=True)
        if type(h).__name__ not in ["ndarray", "csr_matrix"]:
            # including exogenous variables and instrument
            h = sphstack(self.x, q, array_out=True)
        self.z = z
        self.h = h
        self.q = q
        self.yend = yend
        # k = number of exogenous variables and endogenous variables
        self.k = z.shape[1]
        
        # Convert to torch for computations
        h_torch = to_torch(h)
        z_torch = to_torch(z)
        y_torch = to_torch(y)
        
        hth_torch = torch.mm(h_torch.t(), h_torch)
        hth = to_numpy(hth_torch)

        try:
            hthi_torch = torch.linalg.inv(hth_torch)
            hthi = to_numpy(hthi_torch)
        except:
            raise Exception("H'H singular - no inverse")
        
        zth_torch = torch.mm(z_torch.t(), h_torch)
        hty_torch = torch.mm(h_torch.t(), y_torch)
        factor_1_torch = torch.mm(zth_torch, hthi_torch)
        factor_2_torch = torch.mm(factor_1_torch, zth_torch.t())

        try:
            varb_torch = torch.linalg.inv(factor_2_torch)
            varb = to_numpy(varb_torch)
        except:
            raise Exception("Singular matrix Z'H(H'H)^-1H'Z - endogenous variable(s) may be part of X")
        
        factor_3_torch = torch.mm(varb_torch, factor_1_torch)
        betas_torch = torch.mm(factor_3_torch, hty_torch)
        betas = to_numpy(betas_torch)
        
        self.betas = betas
        self.varb = varb
        self.zthhthi = to_numpy(factor_1_torch)

        # predicted values
        self.predy = to_numpy(torch.mm(z_torch, betas_torch))

        # residuals
        u = y - self.predy
        self.u = u

        # attributes used in property
        self.hth = hth  # Required for condition index
        self.hthi = hthi  # Used in error models
        self.htz = to_numpy(zth_torch.t())

        if robust:
            self.vm = ROBUST.robust_vm(reg=self, gwk=gwk, sig2n_k=sig2n_k)

        if sig2n_k:
            self.sig2 = self.sig2n_k
        else:
            self.sig2 = self.sig2n

    @property
    def pfora1a2(self):
        if "pfora1a2" not in self._cache:
            self._cache["pfora1a2"] = self.n * np.dot(self.zthhthi.T, self.varb)
        return self._cache["pfora1a2"]

    @property
    def vm(self):
        try:
            return self._cache["vm"]
        except AttributeError:
            self._cache = {}
            self._cache["vm"] = np.dot(self.sig2, self.varb)
        except KeyError:
            self._cache["vm"] = np.dot(self.sig2, self.varb)
        return self._cache["vm"]

    @vm.setter
    def vm(self, val):
        try:
            self._cache["vm"] = val
        except AttributeError:
            self._cache = {}
            self._cache["vm"] = val
        except KeyError:
            self._cache["vm"] = val

class BaseGM_Lag(BaseTSLS):
    """
    Spatial two stage least squares with PyTorch CUDA support.
    """

    def __init__(
            self,
            y,
            x,
            yend=None,
            q=None,
            w=None,
            w_lags=1,
            slx_lags=0,
            slx_vars="All",
            lag_q=True,
            robust=None,
            gwk=None,
            sig2n_k=False,
    ):
        if slx_lags > 0:
            yend2, q2, wx = set_endog(y, x[:, 1:], w, yend, q, w_lags, lag_q, slx_lags, slx_vars)
            x = np.hstack((x, wx))
        else:
            yend2, q2 = set_endog(y, x[:, 1:], w, yend, q, w_lags, lag_q)

        BaseTSLS.__init__(
            self, y=y, x=x, yend=yend2, q=q2, robust=robust, gwk=gwk, sig2n_k=sig2n_k
        )

class GM_Lag(BaseGM_Lag):
    """
    Spatial two stage least squares with PyTorch CUDA acceleration and full compatibility.
    
    This implementation maintains the exact same interface as the original while using
    PyTorch tensors on CUDA for accelerated computation.
    """

    def __init__(
            self,
            y,
            x,
            yend=None,
            q=None,
            w=None,
            w_lags=1,
            lag_q=True,
            slx_lags=0,
            slx_vars="All",
            regimes = None,
            robust=None,
            gwk=None,
            sig2n_k=False,
            spat_diag=True,
            spat_impacts="simple",
            vm=False,
            name_y=None,
            name_x=None,
            name_yend=None,
            name_q=None,
            name_w=None,
            name_gwk=None,
            name_ds=None,
            latex=False,
            hard_bound=False,
            **kwargs,
    ):
        if regimes is not None:
            from spreg.twosls_sp_regimes import GM_Lag_Regimes
            self.__class__ = GM_Lag_Regimes
            self.__init__(
                y=y,
                x=x,
                regimes=regimes,
                yend=yend,
                q=q,
                w=w,
                w_lags=w_lags,
                slx_lags=slx_lags,
                lag_q=lag_q,
                robust=robust,
                gwk=gwk,
                sig2n_k=sig2n_k,
                spat_diag=spat_diag,
                spat_impacts=spat_impacts,
                vm=vm,
                name_y=name_y,
                name_x=name_x,
                name_yend=name_yend,
                name_q=name_q,
                name_w=name_w,
                name_gwk=name_gwk,
                name_ds=name_ds,
                latex=latex,
                hard_bound=hard_bound,
                **kwargs,
            )
        else:            
            n = USER.check_arrays(x, yend, q)
            y, name_y = USER.check_y(y, n, name_y)
            w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
            USER.check_robust(robust, gwk)
            yend, q, name_yend, name_q = USER.check_endog([yend, q], [name_yend, name_q])
            spat_diag, warn = USER.check_spat_diag(spat_diag=spat_diag, w=w, robust=robust, slx_lags=slx_lags)
            set_warn(self, warn)
            x_constant, name_x, warn = USER.check_constant(x, name_x)
            set_warn(self, warn)
            name_x = USER.set_name_x(name_x, x_constant)

            # kx and wkx are used to replace complex calculation for output
            if slx_lags > 0:  # adjust for flexwx
                if (isinstance(slx_vars,list)):     # slx_vars has True,False
                    if len(slx_vars) != x.shape[1] :
                        raise Exception("slx_vars incompatible with x column dimensions")
                    else:  # use slx_vars to extract proper columns
                        workname = name_x[1:]
                        kx = len(workname)
                        vv = list(compress(workname,slx_vars))
                        name_x += USER.set_name_spatial_lags(vv, slx_lags)
                        wkx = slx_vars.count(True)
                else:
                    kx = len(name_x) - 1
                    wkx = kx
                    name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)

            BaseGM_Lag.__init__(
                self,
                y=y,
                x=x_constant,
                w=w,
                yend=yend,
                q=q,
                w_lags=w_lags,
                slx_lags=slx_lags,
                slx_vars=slx_vars,
                robust=robust,
                gwk=gwk,
                lag_q=lag_q,
                sig2n_k=sig2n_k,
            )

            self.rho = self.betas[-1]
            self.predy_e, self.e_pred, warn = sp_att(
                w, self.y, self.predy, self.yend[:, -1].reshape(self.n, 1), self.rho, hard_bound=hard_bound
            )
            set_warn(self, warn)
            self.title = "SPATIAL TWO STAGE LEAST SQUARES (PYTORCH CUDA)"
            if slx_lags > 0:
                self.title += " WITH SLX (SPATIAL DURBIN MODEL)"
            self.name_ds = USER.set_name_ds(name_ds)
            self.name_y = USER.set_name_y(name_y)
            self.name_x = name_x
            self.name_yend = USER.set_name_yend(name_yend, yend)
            self.name_yend.append(USER.set_name_yend_sp(self.name_y))
            self.name_z = self.name_x + self.name_yend
            self.name_q = USER.set_name_q(name_q, q)

            if slx_lags > 0:
                self.name_x0 = []
                self.name_x0.append(self.name_x[0])  # constant
                if (isinstance(slx_vars,list)):   # boolean list passed
                    self.name_x0.extend(list(compress(self.name_x[1:],[not i for i in slx_vars])))
                    self.name_x0.extend(self.name_x[-wkx:])
                else:
                    okx = int((self.k - self.kstar - 1) / (slx_lags + 1))
                    self.name_x0.extend(self.name_x[-okx:])

                self.name_q.extend(USER.set_name_q_sp(self.name_x0, w_lags, self.name_q, lag_q))
                var_types = ['o'] + ['x']*kx + ['wx'] * wkx * slx_lags + ['yend'] * (len(self.name_yend) - 1) + ['rho']
            else:
                self.name_q.extend(USER.set_name_q_sp(self.name_x, w_lags, self.name_q, lag_q))
                var_types = ['o'] + ['x'] * (len(self.name_x)-1) + ['yend'] * (len(self.name_yend) - 1) + ['rho']

            self.name_h = USER.set_name_h(self.name_x, self.name_q)
            self.robust = USER.set_robust(robust)
            self.name_w = USER.set_name_w(name_w, w)
            self.name_gwk = USER.set_name_w(name_gwk, gwk)
            self.slx_lags = slx_lags
            self.slx_vars = slx_vars

            self.output = pd.DataFrame(self.name_x + self.name_yend, columns=['var_names'])
            self.output['var_type'] = var_types
            self.output['regime'], self.output['equation'] = (0, 0)
            self.other_top = _spat_pseudo_r2(self)
            diag_out = None

            if spat_diag:
                diag_out = _spat_diag_out(self, w, 'yend')
            if spat_impacts:
                self.sp_multipliers, impacts_str = _summary_impacts(self, w, spat_impacts, slx_lags,slx_vars)
                try:
                    diag_out += impacts_str
                except TypeError:
                    diag_out = impacts_str
            output(reg=self, vm=vm, robust=robust, other_end=diag_out, latex=latex)


def _test():
    import doctest
    import numpy as np
    start_suppress = np.get_printoptions()["suppress"]
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == "__main__":
    _test()

    import numpy as np
    import libpysal

    db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
    y_var = "CRIME"
    y = np.array([db.by_col(y_var)]).reshape(49, 1)
    x_var = ["INC"]
    x = np.array([db.by_col(name) for name in x_var]).T
    yd_var = ["HOVAL"]
    yd = np.array([db.by_col(name) for name in yd_var]).T
    q_var = ["DISCBD"]
    q = np.array([db.by_col(name) for name in q_var]).T
    w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    w.transform = "r"
    
    print("Testing PyTorch CUDA GM_Lag...")
    model = GM_Lag(
        y,
        x,
        yd,
        q,
        w=w,
        spat_diag=True,
        name_y=y_var,
        name_x=x_var,
        name_yend=yd_var,
        name_q=q_var,
        name_ds="columbus",
        name_w="columbus.gal",
    )
    print(model.output)
    print(model.summary)
    print(f"Computation completed on: {device}")