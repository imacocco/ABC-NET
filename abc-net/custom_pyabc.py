# Custom routines for PyABC package

from typing import Callable, Union

import numpy as np
import pandas as pd
import scipy.stats as st

from pyabc.parameters import Parameter
from pyabc.transition.base import Transition
from pyabc.transition.exceptions import NotEnoughParticles
from pyabc.transition.util import smart_cov
from pyabc.random_variables import RV

BandwidthSelector = Callable[[int, int], float]

def scott_rule_of_thumb(n_samples, dimension):
    """
    Scott's rule of thumb.

    .. math::

       \\left ( \\frac{1}{n} \\right ) ^{\\frac{1}{d+4}}

    (see also scipy.stats.kde.gaussian_kde.scotts_factor)
    """
    return n_samples ** (-1.0 / (dimension + 4))


def silverman_rule_of_thumb(n_samples, dimension):
    """
    Silverman's rule of thumb.

    .. math::

       \\left ( \\frac{4}{n (d+2)} \\right ) ^ {\\frac{1}{d + 4}}

    (see also scipy.stats.kde.gaussian_kde.silverman_factor)
    """
    return (4 / n_samples / (dimension + 2)) ** (1 / (dimension + 4))


class my_MultivariateNormalTransition(Transition):
    """
    Transition via a multivariate Gaussian KDE estimate, where the sum of proposed jump adds to 0
    and all returned values must be positive

    Parameters
    ----------

    scaling: float
        Scaling is a factor which additionally multiplies the
        covariance with. Since Silverman and Scott usually have too large
        bandwidths, it should make most sense to have 0 < scaling <= 1

    bandwidth_selector: optional
        Defaults to `silverman_rule_of_thumb`.
        The bandwidth selector is a function of the form
        f(n_samples: float, dimension: int),
        where n_samples denotes the (effective) samples size (and is therefore)
        a float and dimension is the parameter dimension.

    """

    def __init__(
        self,
        scaling: float = 1,
        bandwidth_selector: BandwidthSelector = silverman_rule_of_thumb,
        verbose = False
    ):
        self.scaling: float = scaling
        self.bandwidth_selector: BandwidthSelector = bandwidth_selector
        # base population as an array
        self._X_arr: Union[np.ndarray, None] = None
        # perturbation covariance matrix
        self.cov: Union[np.ndarray, None] = None
        # normal perturbation distribution
        self.normal = None
        # cache a range array
        self._range = None
        
        self.dims = None
        self.max_iter = 10000

    def fit(self, X: pd.DataFrame, w: np.ndarray = None) -> None:
        if len(X) == 0:
            raise NotEnoughParticles("Fitting not possible.")
        self._X_arr = X.values
        self.dims = self._X_arr.shape[1]
        if w is None:
            w = np.ones(len(X))/len(X)
        sample_cov = smart_cov(self._X_arr, w)
        dim = sample_cov.shape[0]
        eff_sample_size = 1 / (w**2).sum()
        bw_factor = self.bandwidth_selector(eff_sample_size, dim)

        self.cov = sample_cov * bw_factor**2 * self.scaling
        self.normal = st.multivariate_normal(cov=self.cov, allow_singular=True)

        # cache range array
        self._range = np.arange(len(self.X))

    def rvs(self, size: int = None) -> Union[Parameter, pd.DataFrame]:
        if size is None:
            return self.rvs_single()
        
        sample_ind = np.random.choice(self._range, size=size, p=self.w, replace=True)
        sample = self.X.iloc[sample_ind]
        acc_steps = np.zeros((size,self.dims))
        for i,s in enumerate(sample.to_numpy()):
            # print('s: ',i,' vals: ', s)
            for j in range(self.max_iter):
                step = np.random.multivariate_normal(np.zeros(self.cov.shape[0]), self.cov)
                # select one of the coordinate and set it to -sum(other_coordinates)
                step[np.random.randint(self.dims)] -= step.sum()
                # check all of returned parameters are positive
                if np.all(s + step > 0):
                    # print('necessary iters: ',j)
                    acc_steps[i] = step
                    break
        return sample + acc_steps

    def rvs_single(self) -> Parameter:
        sample_ind = np.random.choice(self._range, p=self.w, replace=True)
        sample = self.X.iloc[sample_ind]
        # print(sample)
        for _ in range(self.max_iter):
            step = np.random.multivariate_normal(np.zeros(self.cov.shape[0]), self.cov)
            step[np.random.randint(self.dims)] -= step.sum()
            if np.all(sample + step > 0):
                # print(step,step.sum(),sum(sample+step))
                return Parameter(sample + step)
        # if no valid perturbation was found, return old value?        
        return Parameter(sample)

    def pdf(self,x: Union[Parameter, pd.Series, pd.DataFrame]) -> Union[float, np.ndarray]:
        # convert to numpy array in correct order
        if isinstance(x, (Parameter, pd.Series)):
            x = np.array([x[key] for key in self.X.columns])
        else:
            x = x[self.X.columns].to_numpy()

        # compute density
        if len(x.shape) == 1:
            return self._pdf_single(x)
        else:
            return np.array([self._pdf_single(xi) for xi in x])

        # alternative (higher memory consumption but broadcast)
        # x = np.atleast_3d(x)  # n_sample x n_par x 1
        # dens = self.normal.pdf(np.swapaxes(x - self._X_arr.T, 1, 2)) * self.w
        # dens = np.atleast_2d(dens).sum(axis=1).squeeze()

    def _pdf_single(self, x: np.ndarray):
        return float((self.normal.pdf(x - self._X_arr) * self.w).sum())

    
class my_second_MultivariateNormalTransition(Transition):
    """
    Transition via a multivariate Gaussian KDE estimate, where all but 1 coordinates are
    constrained to have a 0 total increment and the final points need to bepositive.

    Parameters
    ----------

    scaling: float
        Scaling is a factor which additionally multiplies the
        covariance with. Since Silverman and Scott usually have too large
        bandwidths, it should make most sense to have 0 < scaling <= 1

    bandwidth_selector: optional
        Defaults to `silverman_rule_of_thumb`.
        The bandwidth selector is a function of the form
        f(n_samples: float, dimension: int),
        where n_samples denotes the (effective) samples size (and is therefore)
        a float and dimension is the parameter dimension.

    """

    def __init__(
        self,
        scaling: float = 1,
        bandwidth_selector: BandwidthSelector = silverman_rule_of_thumb,
        verbose = False
    ):
        self.scaling: float = scaling
        self.bandwidth_selector: BandwidthSelector = bandwidth_selector
        # base population as an array
        self._X_arr: Union[np.ndarray, None] = None
        # perturbation covariance matrix
        self.cov: Union[np.ndarray, None] = None
        # normal perturbation distribution
        self.normal = None
        # cache a range array
        self._range = None
        
        self.dims = None
        self.max_iter = 10000

    def fit(self, X: pd.DataFrame, w: np.ndarray = None) -> None:
        if len(X) == 0:
            raise NotEnoughParticles("Fitting not possible.")
        self._X_arr = X.values
        self.dims = self._X_arr.shape[1]
        if w is None:
            w = np.ones(len(X))/len(X)
        sample_cov = smart_cov(self._X_arr, w)
        dim = sample_cov.shape[0]
        eff_sample_size = 1 / (w**2).sum()
        bw_factor = self.bandwidth_selector(eff_sample_size, dim)

        self.cov = sample_cov * bw_factor**2 * self.scaling
        self.normal = st.multivariate_normal(cov=self.cov, allow_singular=True)

        # cache range array
        self._range = np.arange(len(self.X))

    def rvs(self, size: int = None) -> Union[Parameter, pd.DataFrame]:
        if size is None:
            return self.rvs_single()
        
        sample_ind = np.random.choice(self._range, size=size, p=self.w, replace=True)
        sample = self.X.iloc[sample_ind]
        acc_steps = np.zeros((size,self.dims))
        for i,s in enumerate(sample.to_numpy()):
            # print('s: ',i,' vals: ', s)
            for j in range(self.max_iter):
                step = np.random.multivariate_normal(np.zeros(self.cov.shape[0]), self.cov)
                step[np.random.randint(self.dims-1)+1] -= step[1:].sum()    # excluding beta from the "normalization"
                if np.all(s + step > 0):
                    # print('necessary iters: ',j)
                    acc_steps[i] = step
                    break
        return sample + acc_steps

    def rvs_single(self) -> Parameter:
        sample_ind = np.random.choice(self._range, p=self.w, replace=True)
        sample = self.X.iloc[sample_ind]
        # print(sample)
        for _ in range(self.max_iter):
            step = np.random.multivariate_normal(np.zeros(self.cov.shape[0]), self.cov)
            step[np.random.randint(self.dims-1)+1] -= step[1:].sum()
            if np.all(sample + step > 0):
                return Parameter(sample + step)
        # if no valid perturbation was found, return old value?        
        return Parameter(sample)

    def pdf(self,x: Union[Parameter, pd.Series, pd.DataFrame]) -> Union[float, np.ndarray]:
        # convert to numpy array in correct order
        if isinstance(x, (Parameter, pd.Series)):
            x = np.array([x[key] for key in self.X.columns])
        else:
            x = x[self.X.columns].to_numpy()

        # compute density
        if len(x.shape) == 1:
            return self._pdf_single(x)
        else:
            return np.array([self._pdf_single(xi) for xi in x])

        # alternative (higher memory consumption but broadcast)
        # x = np.atleast_3d(x)  # n_sample x n_par x 1
        # dens = self.normal.pdf(np.swapaxes(x - self._X_arr.T, 1, 2)) * self.w
        # dens = np.atleast_2d(dens).sum(axis=1).squeeze()

    def _pdf_single(self, x: np.ndarray):
        return float((self.normal.pdf(x - self._X_arr) * self.w).sum())    
    
    
from pyabc import DistributionBase
from scipy.stats import beta
    
# customized Dirichlet distributions, in order to deal with contraints and other out-of-constraint parameters    
class Dirichlet_marg(DistributionBase):
    
    def __init__(self, alpha,corr: float=1.):
        '''Creates Dirichlet distribution with parameter `alpha`.'''
        # from math import gamma
        # from operator import mul
        # self._alpha = np.array(alpha)
        # self._coef = gamma(np.sum(self._alpha)) / \
        #              np.multiply.reduce([gamma(a) for a in self._alpha])
        self.alpha = alpha        
        self.dim = len(alpha)
        alpha_0 = sum(alpha)
        self.p1 = beta(alpha[0]/corr,alpha_0-alpha[0]/corr)
        self.p2 = beta(alpha[1]/corr,alpha_0-alpha[1]/corr)
        self.p3 = beta(alpha[2]/corr,alpha_0-alpha[2]/corr)
        self.p4 = beta(alpha[3]/corr,alpha_0-alpha[3]/corr)
        self.p5 = beta(alpha[4]/corr,alpha_0-alpha[4]/corr)
        self.p6 = beta(alpha[5]/corr,alpha_0-alpha[5]/corr)
        self.p7 = beta(alpha[6]/corr,alpha_0-alpha[6]/corr)
        self.p8 = beta(alpha[7]/corr,alpha_0-alpha[7]/corr)
        
    def pdf(self, x):
        return self.p1.pdf(x['p1'])*self.p2.pdf(x['p2'])*self.p3.pdf(x['p3'])*self.p4.pdf(x['p4'])\
                *self.p5.pdf(x['p5'])*self.p6.pdf(x['p6'])*self.p7.pdf(x['p7'])*self.p8.pdf(x['p8'])
        
    def rvs(self):
        p1 = self.p1.rvs()
        p2 = self.p2.rvs()
        p3 = self.p3.rvs()
        p4 = self.p4.rvs()
        p5 = self.p5.rvs()
        p6 = self.p6.rvs()
        p7 = self.p7.rvs()
        p8 = self.p8.rvs()
        s = p1+p2+p3+p4+p5+p6+p7+p8
        p1/=s
        p2/=s
        p3/=s
        p4/=s
        p5/=s
        p6/=s
        p7/=s
        p8/=s
        return Parameter(p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6, p7=p7, p8=p8)
     
class Dirichlet_plus_one(DistributionBase):
    
    def __init__(self, alpha, corr: float=1., beta_lim: tuple=(5000,20000)):
        '''Creates Dirichlet distribution with parameter `alpha`.'''
        # from math import gamma
        # from operator import mul
        # self._alpha = np.array(alpha)
        # self._coef = gamma(np.sum(self._alpha)) / \
        #              np.multiply.reduce([gamma(a) for a in self._alpha])
        self.alpha = alpha        
        self.dim = 5#len(alpha)+1
        alpha_0 = sum(alpha)
        self.p1 = beta(alpha[0]/corr,alpha_0-alpha[0]/corr)
        self.p2 = beta(alpha[1]/corr,alpha_0-alpha[1]/corr)
        self.p3 = beta(alpha[2]/corr,alpha_0-alpha[2]/corr)
        self.p4 = beta(alpha[3]/corr,alpha_0-alpha[3]/corr)
        self.beta = RV("uniform", beta_lim[0], beta_lim[1])
        
    def pdf(self, x):
        return self.p1.pdf(x['p1'])*self.p2.pdf(x['p2'])*self.p3.pdf(x['p3'])*self.p4.pdf(x['p4'])*self.beta.pdf(x['beta'])
        
    def rvs(self):
        
        p1 = self.p1.rvs()
        p2 = self.p2.rvs()
        p3 = self.p3.rvs()
        p4 = self.p4.rvs()
        beta = self.beta.rvs()
        
        s = p1+p2+p3+p4
        p1/=s
        p2/=s
        p3/=s
        p4/=s

        return Parameter(beta=beta, p1=p1, p2=p2, p3=p3, p4=p4)




# not working!!!
# each parameter must be a scalar
    
class Dirichlet(DistributionBase):
    def __init__(self, alpha):
        '''Creates Dirichlet distribution with parameter `alpha`.'''
        # from math import gamma
        # from operator import mul
        # self._alpha = np.array(alpha)
        # self._coef = gamma(np.sum(self._alpha)) / \
        #              np.multiply.reduce([gamma(a) for a in self._alpha])
        self.p1 = dirichlet(alpha=alpha)
        self.alpha = alpha
        self.dim = len(alpha)
        
    def pdf(self, x):
        #print(x)
        #x = np.array(x).reshape(self.dim)
        return self.p1.pdf(list(x.values())[0])
        
    def rvs(self):
        return pyabc.Parameter(p1=self.p1.rvs()[0])