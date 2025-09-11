"""
Spatial Two Stages Least Squares
"""

import numpy as np
import numpy.linalg as la
from spreg import user_output as USER
from spreg.utils import RegressionPropsY, RegressionPropsVM, set_warn, sp_att
from spreg import robust as ROBUST

import pandas as pd
from spreg.output import output, _spat_diag_out, _spat_pseudo_r2, _summary_impacts
from itertools import compress
import scipy.sparse as SP
import copy

def lag_spatial(w, y):
    """
    Spatial lag operator.

    If w is row standardized, returns the average of each observation's neighbors;
    if not, returns the weighted sum of each observation's neighbors.

    Parameters
    ----------
    w                   : W
                          libpysal spatial weightsobject
    y                   : array
                          numpy array with dimensionality conforming to w (see examples)

    Returns
    -------
    wy                  : array
                          array of numeric values for the spatial lag

    Examples
    --------
    Setup a 9x9 binary spatial weights matrix and vector of data; compute the
    spatial lag of the vector.

    >>> import libpysal
    >>> import numpy as np
    >>> w = libpysal.weights.lat2W(3, 3)
    >>> y = np.arange(9)
    >>> yl = libpysal.weights.lag_spatial(w, y)
    >>> yl
    array([ 4.,  6.,  6., 10., 16., 14., 10., 18., 12.])

    Row standardize the weights matrix and recompute the spatial lag

    >>> w.transform = 'r'
    >>> yl = libpysal.weights.lag_spatial(w, y)
    >>> yl
    array([2.        , 2.        , 3.        , 3.33333333, 4.        ,
           4.66666667, 5.        , 6.        , 6.        ])

    Explicitly define data vector as 9x1 and recompute the spatial lag

    >>> y.shape = (9, 1)
    >>> yl = libpysal.weights.lag_spatial(w, y)
    >>> yl
    array([[2.        ],
           [2.        ],
           [3.        ],
           [3.33333333],
           [4.        ],
           [4.66666667],
           [5.        ],
           [6.        ],
           [6.        ]])

    Take the spatial lag of a 9x2 data matrix

    >>> yr = np.arange(8, -1, -1)
    >>> yr.shape = (9, 1)
    >>> x = np.hstack((y, yr))
    >>> yl = libpysal.weights.lag_spatial(w, x)
    >>> yl
    array([[2.        , 6.        ],
           [2.        , 6.        ],
           [3.        , 5.        ],
           [3.33333333, 4.66666667],
           [4.        , 4.        ],
           [4.66666667, 3.33333333],
           [5.        , 3.        ],
           [6.        , 2.        ],
           [6.        , 2.        ]])
    """
    return w.sparse @ y


def power_expansion(
    w, data, scalar, post_multiply=False, threshold=0.0000000001, max_iterations=None
):
    r"""
    Compute the inverse of a matrix using the power expansion (Leontief
    expansion).  General form is:

        .. math:: 
            x &= (I - \rho W)^{-1}v = [I + \rho W + \rho^2 WW + \dots]v \\
              &= v + \rho Wv + \rho^2 WWv + \dots

    Examples
    --------
    Tests for this function are in inverse_prod()

    """
    try:
        ws = w.sparse
    except:
        ws = w
    if post_multiply:
        data = data.T
    running_total = copy.copy(data)
    increment = copy.copy(data)
    count = 1
    test = 10000000
    if max_iterations == None:
        max_iterations = 10000000
    while test > threshold and count <= max_iterations:
        if post_multiply:
            increment = increment @ ws * scalar
        else:
            increment = ws @ increment * scalar
        running_total += increment
        test_old = test
        test = la.norm(increment)
        if test > test_old:
            raise Exception(
                "power expansion will not converge, check model specification and that weight are less than 1"
            )
        count += 1
    return running_total


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

    Parameters
    ----------

    w               : Pysal W object
                      nxn Pysal spatial weights object

    data            : Numpy array
                      nx1 vector of data

    scalar          : float
                      Scalar value (typically rho or lambda)

    post_multiply   : boolean
                      If True then post-multiplies the data vector by the
                      inverse of the spatial filter, if false then
                      pre-multiplies.
    inv_method      : string
                      If "true_inv" uses the true inverse of W (slow);
                      If "power_exp" uses the power expansion method (default)

    threshold       : float
                      Test value to stop the iterations. Test is against
                      sqrt(increment' * increment), where increment is a
                      vector representing the contribution from each
                      iteration.

    max_iterations  : integer
                      Maximum number of iterations for the expansion.

    Examples
    --------

    >>> import numpy, libpysal
    >>> import numpy.linalg as la
    >>> from spreg import inverse_prod
    >>> np.random.seed(10)
    >>> w = libpysal.weights.util.lat2W(5, 5)
    >>> w.transform = 'r'
    >>> data = np.random.randn(w.n)
    >>> data.shape = (w.n, 1)
    >>> rho = 0.4
    >>> inv_pow = inverse_prod(w, data, rho, inv_method="power_exp")

    # true matrix inverse

    >>> inv_reg = inverse_prod(w, data, rho, inv_method="true_inv")
    >>> np.allclose(inv_pow, inv_reg, atol=0.0001)
    True
    >>> # test the transpose version
    >>> inv_pow = inverse_prod(w, data, rho, inv_method="power_exp", post_multiply=True)
    >>> inv_reg = inverse_prod(w, data, rho, inv_method="true_inv", post_multiply=True)
    >>> np.allclose(inv_pow, inv_reg, atol=0.0001)
    True

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
        try:
            matrix = la.inv(np.eye(w.n) - (scalar * w.full()[0]))
        except:
            matrix = la.inv(np.eye(w.shape[0]) - (scalar * w))
        if post_multiply:
#            inv_prod = spdot(data.T, matrix)
            inv_prod = np.matmul(data.T,matrix)   # inverse matrix is dense, wrong type in spdot
        else:
#            inv_prod = spdot(matrix, data)
            inv_prod = np.matmul(matrix,data)
    else:
        raise Exception("Invalid method selected for inversion.")
    return inv_prod


def sp_att(w, y, predy, w_y, rho, hard_bound=False):
    xb = predy - rho * w_y
    if np.abs(rho) < 1:
        predy_sp = inverse_prod(w, xb, rho)
        warn = None
        # Note 1: Here if omitting pseudo-R2; If not, see Note 2.
        resid_sp = y - predy_sp
    else:
        if hard_bound:
            raise Exception(
                "Spatial autoregressive parameter is outside the maximum/minimum bounds."
            )
        else:
            # warn = "Warning: Estimate for rho is outside the boundary (-1, 1). Computation of true inverse of W was required (slow)."
            # predy_sp = inverse_prod(w, xb, rho, inv_method="true_inv")
            warn = "*** WARNING: Estimate for spatial lag coefficient is outside the boundary (-1, 1). ***"
            predy_sp = np.zeros(y.shape, float)
            resid_sp = np.zeros(y.shape, float)
    # resid_sp = y - predy_sp #Note 2: Here if computing true inverse; If not,
    # see Note 1.
    return predy_sp, resid_sp, warn

def spdot(a, b, array_out=True):
    """
    Matrix multiplication function to deal with sparse and dense objects

    Parameters
    ----------
    a           : array
                  first multiplication factor. Can either be sparse or dense.
    b           : array
                  second multiplication factor. Can either be sparse or dense.
    array_out   : boolean
                  If True (default) the output object is always a np.array

    Returns
    -------
    ab : array
         product of a times b. Sparse if a and b are sparse. Dense otherwise.
    """
    if type(a).__name__ == "ndarray" and type(b).__name__ == "ndarray":
        ab = np.dot(a, b)
    elif (
        type(a).__name__ == "csr_matrix"
        or type(b).__name__ == "csr_matrix"
        or type(a).__name__ == "csc_matrix"
        or type(b).__name__ == "csc_matrix"
    ):
        ab = a @ b
        if array_out:
            if type(ab).__name__ == "csc_matrix" or type(ab).__name__ == "csr_matrix":
                ab = ab.toarray()
    else:
        raise Exception(
            "Invalid format for 'spdot' argument: %s and %s"
            % (type(a).__name__, type(b).__name__)
        )
    return ab


def sphstack(a, b, array_out=False):
    """
    Horizontal stacking of vectors (or matrices) to deal with sparse and dense objects

    Parameters
    ----------
    a           : array or sparse matrix
                  First object.
    b           : array or sparse matrix
                  Object to be stacked next to a
    array_out   : boolean
                  If True the output object is a np.array; if False (default)
                  the output object is an np.array if both inputs are
                  arrays or CSR matrix if at least one input is a CSR matrix

    Returns
    -------
    ab          : array or sparse matrix
                  Horizontally stacked objects
    """
    if type(a).__name__ == "ndarray" and type(b).__name__ == "ndarray":
        ab = np.hstack((a, b))
    elif type(a).__name__ == "csr_matrix" or type(b).__name__ == "csr_matrix":
        ab = SP.hstack((a, b), format="csr")
        if array_out:
            if type(ab).__name__ == "csr_matrix":
                ab = ab.toarray()
    else:
        raise Exception(
            "Invalid format for 'sphstack' argument: %s and %s"
            % (type(a).__name__, type(b).__name__)
        )
    return ab

def get_lags(w, x, w_lags):
    """
    Calculates a given order of spatial lags and all the smaller orders

    Parameters
    ----------
    w       : weight
              PySAL weights instance
    x       : array
              nxk arrays with the variables to be lagged
    w_lags  : integer
              Maximum order of spatial lag

    Returns
    --------
    rs      : array
              nxk*(w_lags) array with spatially lagged variables

    """
    lag = lag_spatial(w, x)
    spat_lags = lag
    for i in range(w_lags - 1):
        lag = lag_spatial(w, lag)
        spat_lags = sphstack(spat_lags, lag)
    return spat_lags


def get_lags_split(w, x, max_lags, split_at):
    """
    Calculates a given order of spatial lags and all the smaller orders,
    separated into two groups (up to split_at and above)

    Parameters
    ----------
    w       : weight
              PySAL weights instance
    x       : array
              nxk arrays with the variables to be lagged
    max_lags  : integer
              Maximum order of spatial lag
    split_at: integer
              Separates the resulting lags into two cc: up to split_at and above

    Returns
    --------
    rs_l,rs_h: tuple of arrays
               rs_l: nxk*(split_at) array with spatially lagged variables up to split_at
               rs_h: nxk*(w_lags-split_at) array with spatially lagged variables above split_at

    """
    rs_l = lag = lag_spatial(w, x)
    rs_h = None
    if 0 < split_at < max_lags:
        for _ in range(split_at-1):
            lag = lag_spatial(w, lag)
            rs_l = sphstack(rs_l, lag)

        for i in range(max_lags - split_at):
            lag = lag_spatial(w, lag)
            rs_h = sphstack(rs_h, lag) if i > 0 else lag
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
            lag_vars = sphstack(lag_xq, q)
        else:
            lag_vars = lag_xq
        spatial_inst = get_lags(w, lag_vars, w_lags)
        q = sphstack(q, spatial_inst)
        yend = sphstack(yend, yl)
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
    else:  # ajdust returned lag_x here using slx_vars
        if (isinstance(slx_vars,list)):     # slx_vars has True,False
            if len(slx_vars) != x.shape[1] :
                raise Exception("slx_vars incompatible with x column dimensions")
            else:  # use slx_vars to extract proper columns
                vv = slx_vars @ slx_lags
                lag_x = lag_x[:,vv]
            return yend, q, lag_x
        else:  # slx_vars is "All"
            return yend, q, lag_x


class BaseTSLS(RegressionPropsY, RegressionPropsVM):

    """
    Two stage least squares (2SLS) (note: no consistency checks,
    diagnostics or constant added)

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note:
                   this should not contain any variables from x); cannot be
                   used in combination with h
    h            : array
                   Two dimensional array with n rows and one column for each
                   exogenous variable to use as instruments (note: this
                   can contain variables from x); cannot be used in
                   combination with q
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
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    kstar        : integer
                   Number of endogenous variables.
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    hth          : float
                   :math:`H'H`
    hthi         : float
                   :math:`(H'H)^{-1}`
    varb         : array
                   :math:`(Z'H (H'H)^{-1} H'Z)^{-1}`
    zthhthi      : array
                   :math:`Z'H(H'H)^{-1}`
    pfora1a2     : array
                   :math:`n(zthhthi)'varb`


    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> import spreg
    >>> db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),'r')
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> reg = spreg.twosls.BaseTSLS(y, X, yd, q=q)
    >>> print(reg.betas.T)
    [[88.46579584  0.5200379  -1.58216593]]
    >>> reg = spreg.twosls.BaseTSLS(y, X, yd, q=q, robust="white")

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
        z = sphstack(self.x, yend)
        if type(h).__name__ not in ["ndarray", "csr_matrix"]:
            # including exogenous variables and instrument
            h = sphstack(self.x, q)
        self.z = z
        self.h = h
        self.q = q
        self.yend = yend
        # k = number of exogenous variables and endogenous variables
        self.k = z.shape[1]
        hth = spdot(h.T, h)

        try:
            hthi = la.inv(hth)
        except:
            raise Exception("H'H singular - no inverse")
        
        zth = spdot(z.T, h)
        hty = spdot(h.T, y)
        factor_1 = np.dot(zth, hthi)
        factor_2 = np.dot(factor_1, zth.T)
        # this one needs to be in cache to be used in AK

        try:
            varb = la.inv(factor_2)
        except:
            raise Exception("Singular matrix Z'H(H'H)^-1H'Z - endogenous variable(s) may be part of X")
        
        factor_3 = np.dot(varb, factor_1)
        betas = np.dot(factor_3, hty)
        self.betas = betas
        self.varb = varb
        self.zthhthi = factor_1

        # predicted values
        self.predy = spdot(z, betas)

        # residuals
        u = y - self.predy
        self.u = u

        # attributes used in property
        self.hth = hth  # Required for condition index
        self.hthi = hthi  # Used in error models
        self.htz = zth.T

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
    Spatial two stage least squares (S2SLS) (note: no consistency checks,
    diagnostics or constant added); Anselin (1988) [Anselin1988]_

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable; assumes the constant is
                   in column 0.
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note:
                   this should not contain any variables from x); cannot be
                   used in combination with h
    w            : Pysal weights matrix
                   Spatial weights matrix
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional
                   instruments (q).
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the Spatial Durbin type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
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
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    kstar        : integer
                   Number of endogenous variables.
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    hth          : float
                   H'H
    hthi         : float
                   (H'H)^-1
    varb         : array
                   (Z'H (H'H)^-1 H'Z)^-1
    zthhthi      : array
                   Z'H(H'H)^-1
    pfora1a2     : array
                   n(zthhthi)'varb

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> import spreg
    >>> np.set_printoptions(suppress=True) #prevent scientific format
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = 'r'
    >>> db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),'r')
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> # no non-spatial endogenous variables
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> reg = spreg.twosls_sp.BaseGM_Lag(y, X, w=w, w_lags=2)
    >>> reg.betas
    array([[45.30170561],
           [ 0.62088862],
           [-0.48072345],
           [ 0.02836221]])
    >>> spreg.se_betas(reg)
    array([17.91278862,  0.52486082,  0.1822815 ,  0.31740089])
    >>> reg = spreg.twosls_sp.BaseGM_Lag(y, X, w=w, w_lags=2, robust='white')
    >>> reg.betas
    array([[45.30170561],
           [ 0.62088862],
           [-0.48072345],
           [ 0.02836221]])
    >>> spreg.se_betas(reg)
    array([20.47077481,  0.50613931,  0.20138425,  0.38028295])
    >>> # instrument for HOVAL with DISCBD
    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))
    >>> yd = np.array(db.by_col("CRIME"))
    >>> yd = np.reshape(yd, (49,1))
    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))
    >>> X = np.hstack((np.ones(y.shape),X))
    >>> reg = spreg.twosls_sp.BaseGM_Lag(y, X, w=w, yend=yd, q=q, w_lags=2)
    >>> reg.betas
    array([[100.79359082],
           [ -0.50215501],
           [ -1.14881711],
           [ -0.38235022]])
    >>> spreg.se_betas(reg)
    array([53.0829123 ,  1.02511494,  0.57589064,  0.59891744])

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
            yend2, q2, wx = set_endog(y, x[:, 1:], w, yend, q, w_lags, lag_q, slx_lags,slx_vars)
            x = np.hstack((x, wx))
        else:
            yend2, q2 = set_endog(y, x[:, 1:], w, yend, q, w_lags, lag_q)

        

        BaseTSLS.__init__(
            self, y=y, x=x, yend=yend2, q=q2, robust=robust, gwk=gwk, sig2n_k=sig2n_k
        )


class GM_Lag(BaseGM_Lag):
    """
    Spatial two stage least squares (S2SLS) with results and diagnostics;
    Anselin (1988) :cite:`Anselin1988`

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note:
                   this should not contain any variables from x); cannot be
                   used in combination with h
    w            : pysal W object
                   Spatial weights object
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional
                   instruments (q).
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the Spatial Durbin type.
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged
    regimes      : list or pandas.Series
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
                   For other regimes-specific arguments, see GM_Lag_Regimes
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
    spat_diag    : boolean
                   If True, then compute Anselin-Kelejian test and Common Factor Hypothesis test (if applicable)
    spat_impacts : string or list
                   Include average direct impact (ADI), average indirect impact (AII),
                    and average total impact (ATI) in summary results.
                    Options are 'simple', 'full', 'power', 'all' or None.
                    See sputils.spmultiplier for more information.
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    latex        : boolean
                   Specifies if summary is to be printed in latex format
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the bounds of -1 and 1.
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
    e_pred       : array
                   nx1 array of residuals (using reduced form)
    predy        : array
                   nx1 array of predicted y values
    predy_e      : array
                   nx1 array of predicted y values (using reduced form)
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    kstar        : integer
                   Number of endogenous variables.
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    robust       : string
                   Adjustment for robust standard errors
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    ak_test      : tuple
                   Anselin-Kelejian test; tuple contains the pair (statistic,
                   p-value)
    cfh_test     : tuple
                   Common Factor Hypothesis test; tuple contains the pair (statistic,
                   p-value). Only when it applies (see specific documentation).
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_z       : list of strings
                   Names of exogenous and endogenous variables for use in
                   output
    name_q       : list of strings
                   Names of external instruments
    name_h       : list of strings
                   Names of all instruments used in ouput
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    title        : string
                   Name of the regression method used
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    hth          : float
                   :math:`H'H`
    hthi         : float
                   :math:`(H'H)^{-1}`
    varb         : array
                   :math:`(Z'H (H'H)^{-1} H'Z)^{-1}`
    zthhthi      : array
                   :math:`Z'H(H'H)^{-1}`
    pfora1a2     : array
                   n(zthhthi)'varb
    sp_multipliers: dict
                   Dictionary of spatial multipliers (if spat_impacts is not None)

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis. Since we will need some tests for our
    model, we also import the diagnostics module.

    >>> import numpy as np
    >>> import libpysal
    >>> import spreg

    Open data on Columbus neighborhood crime (49 areas) using libpysal.io.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),'r')

    Extract the HOVAL column (home value) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) and CRIME (crime rates) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in, but this can be overridden by passing
    constant=False.

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
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    This class runs a lag model, which means that includes the spatial lag of
    the dependent variable on the right-hand side of the equation. If we want
    to have the names of the variables printed in the
    output summary, we will have to pass them in as well, although this is
    optional. The default most basic model to be run would be:

    >>> from spreg import GM_Lag
    >>> np.set_printoptions(suppress=True) #prevent scientific format
    >>> reg=GM_Lag(y, X, w=w, w_lags=2, name_x=['inc', 'crime'], name_y='hoval', name_ds='columbus')
    >>> reg.betas
    array([[45.30170561],
           [ 0.62088862],
           [-0.48072345],
           [ 0.02836221]])

    Once the model is run, we can obtain the standard error of the coefficient
    estimates by calling the diagnostics module:

    >>> spreg.se_betas(reg)
    array([17.91278862,  0.52486082,  0.1822815 ,  0.31740089])

    But we can also run models that incorporates corrected standard errors
    following the White procedure. For that, we will have to include the
    optional parameter ``robust='white'``:

    >>> reg=GM_Lag(y, X, w=w, w_lags=2, robust='white', name_x=['inc', 'crime'], name_y='hoval', name_ds='columbus')
    >>> reg.betas
    array([[45.30170561],
           [ 0.62088862],
           [-0.48072345],
           [ 0.02836221]])

    And we can access the standard errors from the model object:

    >>> reg.std_err
    array([20.47077481,  0.50613931,  0.20138425,  0.38028295])

    The class is flexible enough to accomodate a spatial lag model that,
    besides the spatial lag of the dependent variable, includes other
    non-spatial endogenous regressors. As an example, we will assume that
    CRIME is actually endogenous and we decide to instrument for it with
    DISCBD (distance to the CBD). We reload the X including INC only and
    define CRIME as endogenous and DISCBD as instrument:

    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))
    >>> yd = np.array(db.by_col("CRIME"))
    >>> yd = np.reshape(yd, (49,1))
    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))

    And we can run the model again:

    >>> reg=GM_Lag(y, X, w=w, yend=yd, q=q, w_lags=2, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')
    >>> reg.betas
    array([[100.79359082],
           [ -0.50215501],
           [ -1.14881711],
           [ -0.38235022]])

    Once the model is run, we can obtain the standard error of the coefficient
    estimates by calling the diagnostics module:

    >>> spreg.se_betas(reg)
    array([53.0829123 ,  1.02511494,  0.57589064,  0.59891744])

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
            name_x = USER.set_name_x(name_x, x_constant)  # need to check for None and set defaults

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
                    name_x += USER.set_name_spatial_lags(name_x[1:], slx_lags)  # exclude constant
                    

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
            self.title = "SPATIAL TWO STAGE LEAST SQUARES"
            if slx_lags > 0:
                self.title += " WITH SLX (SPATIAL DURBIN MODEL)"
            self.name_ds = USER.set_name_ds(name_ds)
            self.name_y = USER.set_name_y(name_y)
            #        self.name_x = USER.set_name_x(name_x, x_constant)   # name_x contains SLX terms for slx_lags > 0
            self.name_x = name_x  # already contains constant in new setup
            self.name_yend = USER.set_name_yend(name_yend, yend)
            self.name_yend.append(USER.set_name_yend_sp(self.name_y))
            self.name_z = self.name_x + self.name_yend
            self.name_q = USER.set_name_q(name_q, q)

            if slx_lags > 0:  # need to remove all but last SLX variables from name_x
                self.name_x0 = []
                self.name_x0.append(self.name_x[0])  # constant
                if (isinstance(slx_vars,list)):   # boolean list passed
                    # x variables that were not lagged
                    self.name_x0.extend(list(compress(self.name_x[1:],[not i for i in slx_vars])))
                    # last wkx variables
                    self.name_x0.extend(self.name_x[-wkx:])


                else:
                    okx = int((self.k - self.kstar - 1) / (slx_lags + 1))  # number of original exogenous vars

                    self.name_x0.extend(self.name_x[-okx:])

                self.name_q.extend(USER.set_name_q_sp(self.name_x0, w_lags, self.name_q, lag_q))

                #var_types = ['x'] * (kx + 1) + ['wx'] * kx * slx_lags + ['yend'] * (len(self.name_yend) - 1) + ['rho']
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