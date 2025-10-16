import torch
def gaussian_rbf(distances):
    return torch.exp(-(distances.pow(2)))


def quadratic_rbf(distances):
    phi = distances.pow(2)
    return phi


def inverse_quadratic_rbf(distances):
    phi = torch.ones_like(distances) / (torch.ones_like(distances) + distances.pow(2))
    return phi


def multiquadric_rbf(distances):
    phi = (torch.ones_like(distances) + distances.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric_rbf(distances):
    phi = torch.ones_like(distances) / (torch.ones_like(distances) + distances.pow(2)).pow(0.5)
    return phi


def spline_rbf(distances):
    phi = distances.pow(2) * torch.log(distances + torch.ones_like(distances))
    return phi


def poisson_one_rbf(distances):
    phi = (distances - torch.ones_like(distances)) * torch.exp(-distances)
    return phi


def poisson_two_rbf(distances):
    phi = ((distances - 2 * torch.ones_like(distances)) / 2 * torch.ones_like(distances)) * distances * torch.exp(-distances)
    return phi


def matern32_rbf(distances):
    phi = (torch.ones_like(distances) + 3 ** 0.5 * distances) * torch.exp(-(3 ** 0.5) * distances)
    return phi


def matern52_rbf(distances):
    phi = (torch.ones_like(distances) + 5 ** 0.5 * distances + (5 / 3) * distances.pow(2)) * torch.exp(-(5 ** 0.5) * distances)
    return phi


rbf_kernels = {
    "gaussian": gaussian_rbf,
    "quadratic": quadratic_rbf,
    "inverse quadratic": inverse_quadratic_rbf,
    "multiquadric": multiquadric_rbf,
    "inverse multiquadric": inverse_multiquadric_rbf,
    "spline": spline_rbf,
    "poisson one": poisson_one_rbf,
    "poisson two": poisson_two_rbf,
    "matern32": matern32_rbf,
    "matern52": matern52_rbf,
}