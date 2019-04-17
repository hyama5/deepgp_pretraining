"""Layers as a component of deep models."""
import torch
import numpy as np

from .parameter_transform import positive
from . import torch_matrix_utils as tmu
from .type_config import NUMPY_DTYPE

VAR_FLOOR = 1e-4


class SVGPLayer(torch.nn.Module):
    r"""Class of Gaussian process layer with stochastic variational inference.

    [1] Hensman, James, Nicol\`o Fusi, and Neil D. Lawrence.
    "Gaussian processes for Big data," Proc. UAI, 2013.
    """

    def __init__(self, kernel, input_dim, output_dim, num_inducings=None,
                 initial_inducing_input=None,
                 initial_q_S_value=1.0, fix_inducing=False):
        """Class of Gaussian process layer with SVI.

        Parameters:
            kernel: Kernel function instance
            input_dim: integer, the dimensionality of input
            output_dim: integer, the dimensionality of output
            num_inducing: the number of inducing inputs
                Not required if initial_inducing_input is given.
            initial_inducing_input: (num_inducing) x (input_dim) matrix
                Initial value of inducing input
            initial_q_S_value: scalar float
                Initial variance of the variational distribution
                of output variables. In the bottom and middle layers,
                the values should be small to get training fast.
            fix_inducing: bool
                Fix inducing input when this value is True.
                If the inducing input is not fixed in the middle
                and top layers, deep GP models often over-fit.
        """
        super().__init__()

        if initial_inducing_input is None:
            if num_inducings is None:
                msg = 'Neither num_inducings nor initial_inducing_input '\
                      'is not given.'
                raise RuntimeError(msg)
            else:
                initial_inducing_input = np.random.randn(
                    num_inducings, input_dim).astype(NUMPY_DTYPE)
        else:
            num_inducings = len(initial_inducing_input)

            if input_dim != initial_inducing_input.shape[1]:
                msg = 'Inconsistency between input_dim and '\
                      'initial_inducing_input'
                raise RuntimeError(msg)

        self.kernel = kernel
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducings = num_inducings

        q_diag_S_init_val = positive.backward(
            initial_q_S_value * np.ones((num_inducings, output_dim)))
        q_mu_init_val = np.zeros((num_inducings, output_dim))

        self.q_diag_S_ = torch.nn.Parameter(
            torch.from_numpy(q_diag_S_init_val.astype(NUMPY_DTYPE)))
        self.q_mu = torch.nn.Parameter(
            torch.from_numpy(q_mu_init_val.astype(NUMPY_DTYPE)))

        self.z = torch.nn.Parameter(
            torch.from_numpy(initial_inducing_input.astype(NUMPY_DTYPE)),
            requires_grad=(not fix_inducing))

    def fix_inducing_(self, fix_inducing=True):
        """Change flag to fix inducing input z."""
        self.z.requires_grad_(not fix_inducing)

    def forward(self, x):
        """Infer or sample output.

        When validation mode, this method returns the predictive
        distribution. When training mode, this method return the
        sample from the distribution.
        """
        if self.training:
            return self.sample(x)
        else:
            return self.pred_mean_and_var(x)['mean']

    def pred_mean_and_var(self, x):
        r"""Infer predictive distribution of the layer.

        Predict q(f|x) = \int q(f|x,u) q(u) du
        where q(u) =  N(u, q_mu, q_S)

        q(f|x,u) &= N(f; mu_1, Sigma_1)
        mu_1 = K(x, z) K(z, z)^{-1} u
        Sigma_1  = K(x, x) - K(x, z) K(z, z)^{-1} K(z, x)

        Then,
        q(f|x,u) &= N(f, mu_f, Sigma_f)
        mu_f = K(x, z) K(z, z)^{-1} q_mu
        Sigma_f = K(x, x) - K(x, z) K(z, z)^{-1} K(z, x) \\
                    + K(x, z) K(z, z)^{-1} q_S K(z, z)^{-1} K(z, x)


        ----
        Parameters:
            kernel: kernel function
            x: input tensor

        ----
        Outputs:
            dict:
                'mean': (N x D) tensor
                'var': (N x D) tensor
                where N = x.size(0) and D is output dim.
        """
        K_uf = self.kernel.K(self.z, x)
        cho_K_u = tmu.cholesky(self.kernel.K(self.z))
        diag_K_f = self.kernel.K_diag(x)

        q_diag_S = positive.forward_tensor(self.q_diag_S_)

        pred_mu = tmu.cho_solve_AXB(K_uf.t(), cho_K_u, self.q_mu)

        MM = tmu.cho_solve(cho_K_u, K_uf)  # K_M^{-1} K_uf
        m2 = torch.sum(K_uf * MM, dim=0)  # diag(K_uf^T  K_M^{-1} K_uf)

        s1 = diag_K_f[:, None].expand(-1, self.output_dim)
        s2 = m2[:, None].expand(-1, self.output_dim)
        s3 = torch.mm((MM * MM).t(), q_diag_S)

        s = s1 - s2 + s3
        pred_var = torch.clamp(s, min=VAR_FLOOR)

        return {'mean': pred_mu, 'var': pred_var}

    def sample(self, x):
        """Sample by inferring predictive distributions."""
        pred = self.pred_mean_and_var(x)
        r = torch.randn_like(pred['mean'])
        return pred['mean'] + r * torch.sqrt(pred['var'])

    def kl_divergence(self):
        r"""Calculate Kullback-Leibler divergence of inducing outputs.

        KL(q(u)||p(u) = \int q(f|x,u) q(u) du
        where q(u) =  N(u, q_mu, q_S)

        Let K_u = K(z, z),
        KL = \int q(u) (log(q(u) - log p(u)) du
           = \int q(u) (-1/2 log|S| -1/2(u - q_mu)^T S^{-1} (u - q_mu) \\
                        +1/2 log|K_u| + 1/2 u^T K_u^{-1} u)
           = -1/2 log|S| -1/2 (0 + Tr(I_M))
                +1/2 log|K_u| + 1/2 (q_mu^T K_u^{-1} q_mu + Tr(K_u^{-1} S)) \\
           = -1/2 log|S| -1/2 M
                +1/2 log|K_u| + 1/2 q_mu^T K_u^{-1} q_mu
                + 1/2 Tr(K_u^{-1} S)) \\

        In multidimensional case, we sum up KLDs of all dims.

        Parameters:
            kernel: kernel function
            x: input tensor

        Outputs;
            KL divergence scalar value

        """
        size_M = self.q_mu.size(0)
        size_D = self.q_mu.size(1)

        cho_K_u = tmu.cholesky(self.kernel.K(self.z))
        inv_K_u = tmu.cho_inv(cho_K_u)

        q_diag_S = positive.forward_tensor(self.q_diag_S_)

        kl1 = -0.5 * torch.sum(torch.log(q_diag_S))
        kl2 = -0.5 * size_M * size_D
        kl3 = 0.5 * size_D * tmu.cho_log_det(cho_K_u)
        # 0.5 * \sum_d Tr[K_M^{-1} S_d] where S_d is diagonal
        kl4 = 0.5 * torch.sum(q_diag_S.t() * torch.diag(inv_K_u))
        kl5 = 0.5 * \
            torch.trace(tmu.cho_solve_AXB(self.q_mu.t(), cho_K_u, self.q_mu))

        kl = kl1 + kl2 + kl3 + kl4 + kl5

        return kl

    def extra_repr(self):
        """Extra information for repr()."""
        return 'input_dim={}, output_dim={}, num_inducings={}'.format(
            self.input_dim, self.output_dim, self.num_inducings,
        )


class NNLayer(torch.nn.Module):
    """1-hidden layer used for pre-training of deep GP.

    This performs the following mapping.
    [input] -> Linear -> ReLU -> [hidden units] ->
        Dropout -> Linear -> BatchNormalization -> [output]
    """

    def __init__(self, input_dim, output_dim, num_hidden_units,
                 dropout_zero_rate=0.2, batch_norm=True):
        """1-hidden layer used for pre-training of deep GP.

        Parameters:
            input_dim: integer, the dimensionality of input
            output_dim: integer, the dimensionality of output
            num_hidden_units: integer, the dimensionality of hidden units
            dropout_zero_rate: float, dropout rate for hidden units
            batch_norm: bool, to use batch normalization
        """
        super().__init__()

        modules = list()

        modules.append(torch.nn.Linear(in_features=input_dim,
                                       out_features=num_hidden_units,
                                       bias=True))

        modules.append(torch.nn.ReLU())
        if dropout_zero_rate > 0:
            modules.append(torch.nn.Dropout(p=dropout_zero_rate))

        modules.append(torch.nn.Linear(in_features=num_hidden_units,
                                       out_features=output_dim,
                                       bias=False))
        if batch_norm:
            modules.append(torch.nn.BatchNorm1d(output_dim))

        self.seq = torch.nn.Sequential(*modules)

        self.reset_parameters()

    def forward(self, x):
        """Perform forward propagation."""
        return self.seq(x)

    def reset_parameters(self):
        """Reset DNN parameters.

        # He initialization
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

                if m.bias is not None:
                    m.bias.data.zero_()
