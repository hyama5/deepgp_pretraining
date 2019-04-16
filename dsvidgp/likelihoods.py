"""Likelihood functions for output features."""

import torch
import math
import numpy as np

from .parameter_transform import positive
from .quadrature import GaussHermiteQuadrature
from .type_config import NUMPY_DTYPE


class Gaussian(torch.nn.Module):
    """Gaussian type likelihood.

    The relationship between output variable y
    and latent function f is given by
        p(y|f) = N(y; f, v)
    where parameter v is variance of noise.
    """

    def __init__(self, output_dim, initial_variance=1e-3,
                 share_parameter=False):
        """Gaussian type likelihood.

        Parameters:
            output_dim: scalar integer
            initial_variance: scalar float
            share_dimension: bool
                If true, use the same parameter for all dimension
        """
        super().__init__()

        self.share_parameter = share_parameter
        self.output_dim = output_dim

        if self.share_parameter:
            variance_init_val = positive.backward(
                initial_variance) * np.ones(1)
        else:
            variance_init_val = positive.backward(
                initial_variance) * np.ones(output_dim)

        self.variance_ = torch.nn.Parameter(
            torch.from_numpy(variance_init_val.astype(NUMPY_DTYPE)))

    def predictive_expectation(self, y, f_dstr):
        r"""Calculate predictive expcextation of log likelihoods.

        Predictive expcextation of log likelihoods given by
            \int q(f) log p(y|f) df
            = \int q(f) (-N/2 log(2 \pi) - N/2 log(v) \\
                            -1/2 (y-f)^T (y-f) / v) df
            = -N/2 log(2 \pi) - N/2 log(v) \\
                -1/2 (y-f_mean)^T (y-f_mean) / v \\
                -1/2 Tr[diag(f_dstr['var'])] / v

        In multidimentional cases, we sum up log likelihoods of all dims.
        """
        if self.share_parameter:
            variance = positive.forward_tensor(
                self.variance_).expand(self.output_dim)
        else:
            variance = positive.forward_tensor(self.variance_)

        size_B = y.size(0)
        size_D = y.size(1)

        diff = y - f_dstr['mean']
        pe1 = -0.5 * size_B * size_D * np.log(2.0 * np.pi)
        pe2 = -0.5 * size_B * torch.sum(torch.log(variance))
        pe3 = -0.5 * torch.sum(diff * diff / variance)
        pe4 = -0.5 * torch.sum(f_dstr['var'] / variance)

        return pe2 + pe3 + pe4 + pe1

    def predict(self, f_dstr):
        r"""Calculate predictive expcextation of output variable.

        The predictive distribtuion is given by
            q(y) = \int q(f) p(y | f) df
        """
        variance = positive.forward_tensor(self.variance_)

        y_var = f_dstr['var'] + variance.expand_as(f_dstr['var'])
        return {'mean': f_dstr['mean'], 'var': y_var}


def inverse_probit(x):
    """Inverse of probit function."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))) \
        * (1.0 - 2.0e-3) + 1.0e-3


class Bernoulli(torch.nn.Module):
    r"""Bernoulli type likelihood used for binary classification.

    The relationship between output variable y \in {+1, -1}
    and latent function f is given by
        p(y|f) = p^{y==+1} (1-p)^{y==-1}
        p = invprobit(f)
    """

    def __init__(self, num_hermgauss=20):
        r"""Bernoulli type likelihood used for binary classification.

        Parameters:
            num_hermgauss: integer,
                Number of samples used for Gaussian-Hermite approximation.
        """
        super().__init__()

        self._quadrature = GaussHermiteQuadrature(num_hermgauss)

    def predictive_expectation(self, y, f_dstr):
        r"""Calculate predictive expcextation of log likelihoods.

        Predictive expcextation of log likelihoods is given by
            \int q(f) log p(y|f) df
            = \int q(f) {(y==+1) log(p) + (y==-1) log(1-p)} df
            = \int q(f) log(invprobit(y * f)) df
        Note: invprobit(f) == 1 - invprobit(1 - f)

        The integral is computed by Gaussian-Hermite quadrature.
        """
        def func(f_grid, y=y):
            return torch.log(inverse_probit(f_grid * y[None, :, :]))

        pe = torch.sum(self._quadrature(
            f_dstr['mean'], torch.sqrt(f_dstr['var']), func))

        return pe

    def predict(self, f_dstr):
        r"""Calculate predictive expcextation of output variable.

        The predictive distribtuion is given by
            q(y) = \int q(f) p(y | f) df
        """
        prob_positive = inverse_probit(
            f_dstr['mean'] / torch.sqrt(1.0 + f_dstr['var']))

        return {'prob_positive': prob_positive}
