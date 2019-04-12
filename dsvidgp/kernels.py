
"""Definitions of kernel functions."""


import numpy as np
import math
import torch
from .parameter_transform import positive
from .type_config import NUMPY_DTYPE

INITIAL_FLOOR = 1e-4
JITTER = 1e-6
ARCCOS_EPS = 1e-4


class RBF(torch.nn.Module):
    r"""Radius Basis Function Kernel.

    k(x, x') = m \exp[(x - x')^T L^{-1} (x - x')] + f
    L = diag(l) = diag[l_1^2, l_2^2, ... l_D^2]
    where
        - l: lengthscale
        - m: magnitude
        - f: flooring value
    ----
    Parameters:
        input_dim: scalar integer
        initial_lengthscale: scalar float
    """

    def __init__(self, input_dim, initial_lengthscale=1.0):
        """RBF kernel.

        Parameters:
            input_dim: scalar integer
            initial_lengthscale: scalar float

        """
        super().__init__()

        lengthscale_init_val = positive.backward(
            initial_lengthscale) * np.ones(input_dim)
        magnitude_init_val = positive.backward(np.array([1.0]))
        floor_init_val = positive.backward(np.array([INITIAL_FLOOR]))

        self.lengthscale_ = torch.nn.Parameter(
            torch.from_numpy(lengthscale_init_val.astype(NUMPY_DTYPE)))
        self.magnitude_ = torch.nn.Parameter(
            torch.from_numpy(magnitude_init_val.astype(NUMPY_DTYPE)))
        self.floor_ = torch.nn.Parameter(
            torch.from_numpy(floor_init_val.astype(NUMPY_DTYPE)))

        self._jitter = JITTER

    def K(self, x1, x2=None):
        """Calculate Gram matrix.

        When x2 is not assigned, K(x1, x1) is computed.

        Parameters:
            x1: N1 x D matrix (N1 samples of D dimentional vector)
            x2: N2 x D matrix (N2 samples of D dimentional vector)
        Returns:
            K: N1 x N2 gram matrix.
        """
        lengthscale = positive.forward_tensor(self.lengthscale_)
        magnitude = positive.forward_tensor(self.magnitude_)
        floor = positive.forward_tensor(self.floor_)

        if x2 is None:
            x2 = x1
            sym = True
        else:
            sym = False

        beta = 1.0 / lengthscale ** 2.0
        xx1 = torch.sum((beta * x1 ** 2.0), dim=1, keepdim=True)
        xx2 = torch.sum((beta * x2 ** 2.0), dim=1, keepdim=True).t()
        x1x2 = torch.mm(beta * x1, x2.t())

        D = xx1 + xx2 - 2.0 * x1x2

        if sym:
            K = magnitude * torch.exp(-0.5 * D) + (floor + self._jitter) * \
                torch.eye(D.size(0), dtype=D.dtype, device=D.device)
        else:
            K = magnitude * torch.exp(-0.5 * D)

        return K

    def K_diag(self, x1):
        """Calculate diagonal Gram matrix.

        Diagonal components of K(x1, x1).

        Parameters:
            x1: N x D matrix (N samples of D dimentional vector)
        Returns:
            K: N-length vector.
        """
        magnitude = positive.forward_tensor(self.magnitude_)
        floor = positive.forward_tensor(self.floor_)

        return magnitude * torch.ones_like(x1[:, 0]) + floor + self._jitter


class ArcCos(torch.nn.Module):
    r"""Arc-Cosine kernel.

    A kernel function derived from neural network with an infinite number
    of hidden units and ReLU activation function.

    [1] Cho, Youngmin, and Lawrence K. Saul.
    "Kernel methods for deep learning." Proc. NIPS 2009.


    k_0(x, x') = s_{w,0}^2  + s_{b,0}^2 x^T L x'
    k_{p+1}(x, x') = \sigma_{w,p+1}^2  + \sigma_{b,p+1}^2 \sqrt(k_p(x, x))
                \sqrt(k_p(x', x')) (sin \theta + (\pi - \theta) \cos \theta)
    \theta_p = \arccos \frac{k_p(x, x')}{\sqrt(k_p(x, x)) \sqrt(k_p(x', x'))}

    L = diag(l) = diag[l_1^2, l_2^2, ... l_D^2]
    where
        - L: relevance
        - s_w: variance of neural network weight prior
        - s_b: variance of neural network bias prior
    """

    def __init__(self, input_dim, num_layers=1, initial_relevance=1.0,
                 initial_sigma_w=None, initial_sigma_b=0.01, normalize=True):
        """Arc-Cosine kernel.

        ----
        Parameters:
            input_dim: scalar integer
            num_layers: layers of neural network
            initial_relevance: scalar float,
                which is to be devided by input_dim
            initial_sigma_w: scalar float,
                variance of neural network weight prior
            initial_sigma_b: scalar float,
                variance of neural network bias prior
            normalize: bool
                force the diagonal value of Gram matrix unity
        """
        super().__init__()

        assert num_layers > 0, "Num layers mush be positive integer."

        if initial_sigma_w is None:
            # use He initialization
            initial_sigma_w = [2.0 / input_dim] + [1.0] * num_layers

        relevance_init_val = positive.backward(
            initial_relevance) * np.ones(input_dim)
        sigam_w_init_val = positive.backward(
            initial_sigma_w * np.ones(num_layers + 1))
        sigam_b_init_val = positive.backward(
            initial_sigma_b * np.ones(num_layers + 1))

        magnitude_init_val = positive.backward(np.array([1.0]))
        floor_init_val = positive.backward(np.array([INITIAL_FLOOR]))

        self.relevance_ = torch.nn.Parameter(
            torch.from_numpy(relevance_init_val.astype(NUMPY_DTYPE)))
        self.sigma_w_ = torch.nn.Parameter(
            torch.from_numpy(sigam_w_init_val.astype(NUMPY_DTYPE)))
        self.sigma_b_ = torch.nn.Parameter(
            torch.from_numpy(sigam_b_init_val.astype(NUMPY_DTYPE)))

        self.magnitude_ = torch.nn.Parameter(
            torch.from_numpy(magnitude_init_val.astype(NUMPY_DTYPE)))
        self.floor_ = torch.nn.Parameter(
            torch.from_numpy(floor_init_val.astype(NUMPY_DTYPE)))

        self._num_layers = num_layers
        self._normalize = normalize

        self._jitter = JITTER

    def K(self, x1, x2=None):
        """Calculate Gram matrix.

        When x2 is not assigned, K(x1, x1) is computed.

        Parameters:
            x1: N1 x D matrix (N1 samples of D dimentional vector)
            x2: N2 x D matrix (N2 samples of D dimentional vector)
        Returns:
            K: N1 x N2 gram matrix.
        """
        relevance = positive.forward_tensor(self.relevance_)
        sigma_w = positive.forward_tensor(self.sigma_w_)
        sigma_b = positive.forward_tensor(self.sigma_b_)

        magnitude = positive.forward_tensor(self.magnitude_)
        floor = positive.forward_tensor(self.floor_)

        if x2 is None:
            x2 = x1
            sym = True
        else:
            sym = False

        x1x2 = torch.mm(relevance * x1, x2.t())
        xx1 = torch.sum(relevance * (x1 ** 2.0), dim=1, keepdim=True)
        xx2 = torch.sum(relevance * (x2 ** 2.0), dim=1, keepdim=True).t()

        Ki0 = sigma_b[0] + sigma_w[0] * x1x2
        k1diag_sqrt = torch.sqrt(sigma_b[0] + sigma_w[0] * xx1)
        k2diag_sqrt = torch.sqrt(sigma_b[0] + sigma_w[0] * xx2)

        Ki = Ki0

        for i in range(self._num_layers):
            sb = sigma_b[i + 1]
            sw = sigma_w[i + 1]

            dd = torch.mm(k1diag_sqrt, k2diag_sqrt)
            div = Ki / dd
            # to resrtict "div" in [-1, 1]
            div = ARCCOS_EPS + (1 - 2.0 * ARCCOS_EPS) * div
            theta = torch.acos(div)
            Ki = sb + sw / (2.0 * np.pi) * dd * \
                (torch.sin(theta) + (math.pi - theta) * torch.cos(theta))

            k1diag_sqrt = torch.sqrt(sb + sw / 2.0 * k1diag_sqrt ** 2.0)
            k2diag_sqrt = torch.sqrt(sb + sw / 2.0 * k2diag_sqrt ** 2.0)

        if self._normalize:
            Ki = Ki / torch.mm(k1diag_sqrt, k2diag_sqrt)

        if sym:
            K = magnitude * Ki + (floor + self._jitter) * \
                torch.eye(Ki.size(0), dtype=Ki.dtype, device=Ki.device)
        else:
            K = magnitude * Ki

        return K

    def K_diag(self, x1):
        """Calculate diagonal Gram matrix.

        Diagonal components of K(x1, x1).

        Parameters:
            x1: N x D matrix (N samples of D dimentional vector)
        Returns:
            K: N-length vector.
        """
        magnitude = positive.forward_tensor(self.magnitude_)
        floor = positive.forward_tensor(self.floor_)

        if self._normalize:
            return magnitude * torch.ones_like(x1[:, 0]) + floor + self._jitter

        else:
            relevance = positive.forward_tensor(self.relevance_)
            sigma_w = positive.forward_tensor(self.sigma_w_)
            sigma_b = positive.forward_tensor(self.sigma_b_)

            xx1 = torch.sum(relevance * (x1 ** 2.0), dim=1)
            k1diag = sigma_b[0] + sigma_w[0] * xx1
            for i in range(self._num_layers):
                sb = sigma_b[i + 1]
                sw = sigma_w[i + 1]

                k1diag = sb + sw / 2.0 * k1diag

            return magnitude * k1diag + floor + self._jitter
