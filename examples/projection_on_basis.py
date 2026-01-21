"""
Study function that are function of the distance between particules
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from better_kan import create_dataset
from better_kan.functions import Grid, Splines, ChebyshevPolynomial


def compute_act(model, x_test):
    min_vals = torch.min(x_test, dim=0).values
    max_vals = torch.max(x_test, dim=0).values
    ranges = [torch.linspace(min_vals[d], max_vals[d], 150) for d in range(1)]
    x_in = torch.stack(ranges, dim=1)
    acts_vals = model.activations_eval(x_in).cpu().detach().numpy()
    x_ranges = x_in.cpu().detach().numpy()
    return x_ranges[:, 0], acts_vals[:, 0, 0], x_in


fcts = [lambda x: torch.sin(x).unsqueeze(-1), lambda x: torch.exp(x).unsqueeze(-1), lambda x: torch.log(torch.abs(x)).unsqueeze(-1)]

fig, axs = plt.subplots(ncols=len(fcts))

# grid = Grid(1, 15)
# orig_model = Splines(1, 1, grid, k=3)
orig_model = ChebyshevPolynomial(1, 1, poly_order=15)
x, _ = orig_model.collocations_points()
for k, fct in enumerate(fcts):
    proj_weights = orig_model.project_on_basis(fct)
    orig_model.weights = nn.Parameter(proj_weights)
    ranges, act_vals_before, x_layers = compute_act(orig_model, x)

    axs[k].plot(ranges, act_vals_before, "-")
    axs[k].plot(ranges, fct(x_layers).numpy().squeeze(), "--")

plt.show()
