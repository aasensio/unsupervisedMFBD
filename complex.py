import torch

def complex_multiply_astar_b(a, b):
    """
    Compute the product of the complex conjugate of a and the b in torch format
    [..., 0] -> real part
    [..., 1] -> imaginary part

    out = a* x b

    Parameters
    ----------
    a : torch complex array
        a
    b : torch complex array
        b
    """
    tmp1 = torch.unsqueeze(a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1], -1)
    tmp2 = torch.unsqueeze(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], -1)

    return torch.cat([tmp1, tmp2], -1)


def complex_division(a, b):
    """
    Compute the division of two complex numbers

    out = a / b

    Parameters
    ----------
    a : torch complex array
        a
    b : torch complex array
        b
    """
    denominator = torch.unsqueeze(b[..., 0] * b[..., 0] + b[..., 1] * b[..., 1], -1)

    tmp1 = torch.unsqueeze(a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1], -1)
    tmp2 = torch.unsqueeze(a[..., 1] * b[..., 0] - a[..., 0] * b[..., 1], -1)

    return torch.cat([tmp1 / denominator, tmp2 / denominator], -1)