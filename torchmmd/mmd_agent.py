import collections

import torch


ReplayElement = collections.namedtuple("shape_type", ["name", "shape", "type"])


def gaussian_rbf_kernel(d: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
    """

    :param d: shape: (batch_size, num_samples, num_samples)
    """
    b, n, n = d.shape
    sigmas_tensor = torch.as_tensor(sigmas, dtype=torch.float32, device=d.device)
    k = sigmas_tensor.size(0)
    h = 1 / sigmas_tensor.view(-1, 1)
    s = h @ d.view(1, -1)
    assert s.shape == (k, b*n*n)
    return torch.exp(-s).sum(dim=0).view(d.shape)


def huber_loss(u: torch.Tensor, kappa: float = 1) -> torch.Tensor:
    if kappa == 0:
        return u.abs()
    huber_loss_case_one = u.abs().le(kappa).float() * 0.5 * u ** 2
    huber_loss_case_two = u.abs().gt(kappa).float() * kappa * (u.abs() - 0.5 * kappa)
    return huber_loss_case_one + huber_loss_case_two
