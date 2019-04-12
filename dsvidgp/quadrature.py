"""Gauss-Hermite Quadrature."""

import numpy as np
import torch
import math

from .type_config import NUMPY_DTYPE


class GaussHermiteQuadrature(torch.nn.Module):
    r"""Gauss-Hermite Quadrature for numerical integral.

    The integral
        I = \int N(x; \mu, \sigma^2) f(x) dx
          = \int \sqrt(2) N(t; 0, 1) f(t) dt
        t = \sqrt(2) \sigma x + \mu
    is approximated by
        I = 1/sqrt(pi) \sum_i^N gh_w_i * f(t_i)
        t_i = \sqrt(2) \sigma gh_x_i + \mu
    where
        gh_w_i is Hermite polynomial
    """

    def __init__(self, num_hermgauss=20):
        """Gauss-Hermite Quadrature for numerical integral.

        Parameters:
            num_hermgauss: integer, approximation points
        """
        super().__init__()

        gh_x, gh_w = np.polynomial.hermite.hermgauss(num_hermgauss)
        self.gh_x = torch.nn.Parameter(
            torch.from_numpy(gh_x[:, None, None].astype(NUMPY_DTYPE)),
            requires_grad=False)
        self.gh_w = torch.nn.Parameter(
            torch.from_numpy(gh_w[:, None, None].astype(NUMPY_DTYPE)),
            requires_grad=False)

    def forward(self, mu, sigma, func):
        """Perform Gaussian-Hermite integral."""
        t = math.sqrt(2.0) * sigma[None, :, :] * self.gh_x + mu[None, :, :]
        gh_result = torch.sum(self.gh_w * func(t), dim=0) / math.sqrt(math.pi)
        return gh_result
