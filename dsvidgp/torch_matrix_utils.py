"""Utility functions of linear algebra."""

import torch


def cholesky(X):
    r"""Cholesky decomposition of X.

    >>> cholesky(X)
    return L
    torch.mm(L, L.t()) = X
    """
    if torch.__version__ >= '1.0.0':
        return torch.cholesky(X)
    else:
        return torch.potrf(X, upper=False)


def cho_log_det(cho_C):
    """Compute tensor log determinant of $C$ from cholesky factor.

    ----
    Parameters:
        cho_C: lower triangular tensor where cho_C cho_C^T = C
    ----
    Outputs:
        C^{-1}
    """
    return 2.0 * torch.sum(torch.log(torch.diagonal(cho_C)))


def cho_inv(cho_C):
    """Compute tensor $C^{-1}$ from cholesky factor.

    ----
    Parameters:
        cho_C: lower triangular tensor where cho_C cho_C^T = C
    ----
    Outputs:
        C^{-1}
    ----
    Note:
        Gradient of potri is not supperted yet in pytorch 0.4.1
    """
    I = torch.eye(cho_C.size(0), dtype=cho_C.dtype, device=cho_C.device)
    return cho_solve(cho_C, I)


def cho_solve(cho_C, b):
    """Compute tensor $C^{-1} b$ from cholesky factor.

    ----
    Parameters:
        cho_C: (N x N) lower triangular tensor where cho_C cho_C^T = C
        b: (N x L) tensor
    ----
    Outputs:
        C^{-1} b
    ----
    Note:
        Gradient of potrs is not supperted yet in pytorch 0.4.1
        # return torch.potrs(b, cho_C, upper=False)
    """
    tmp, _ = torch.trtrs(b, cho_C, upper=False)
    tmp2, _ = torch.trtrs(tmp, cho_C.t(), upper=True)
    return tmp2


def cho_solve_AXB(a, cho_C, b):
    """Compute tensor $a C^{-1} b$ from cholesky factor.

    ----
    Parameters:
        a: (M x N) tensor
        cho_C: (N x N) lower triangular tensor where cho_C cho_C^T = C
        b: (N x L) tensor
    ----
    Outputs:
        a C^{-1} b
    """
    left, _ = torch.trtrs(a.t(), cho_C, upper=False)
    right, _ = torch.trtrs(b, cho_C, upper=False)

    return torch.mm(left.t(), right)


def one_hot(class_vec, num_classes, dtype=None, device=None):
    """Create one-hot encoded tensor.

    ----
    Paramaters:
        class_vec: (N x 1) LongTensor
        num_classes: C
        dtype: data type for Tensor
        device: device for Tensor
    """
    I = torch.eye(num_classes, dtype=dtype, device=device)
    return I[class_vec.squeeze()]
