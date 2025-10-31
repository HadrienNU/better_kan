"""
In this example, we investigate the loss increase after grid update

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from math import isqrt

from better_kan import create_dataset, train
from better_kan.functions import Grid, Splines

loss_fn = lambda x, y: torch.mean((x - y) ** 2).cpu().detach().numpy()


def compute_act(model, x_test, layers_range=None):

    # Get range from test_data
    if layers_range is None:
        min_vals = torch.min(x_test, dim=0).values
        max_vals = torch.max(x_test, dim=0).values
        ranges = [torch.linspace(min_vals[d], max_vals[d], 150) for d in range(1)]
        x_in = torch.stack(ranges, dim=1)
    else:
        x_in = layers_range
    acts_vals = model.activations_eval(x_in).cpu().detach().numpy()
    x_ranges = x_in.cpu().detach().numpy()
    return x_ranges[:, 0], acts_vals[:, 0, 0], x_in


torch.manual_seed(7)

# model = build_KAN(Splines, [7, 2, 1], grid_size=5, fast_version=True)
# f = lambda x: 1 / (1 + torch.exp(-(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2 + x[:, [2]] ** 2 + x[:, [3]] ** 2 + x[:, [4]] ** 2 - (x[:, [5]] - x[:, [6]]).abs() ** 0.5)))
# dataset = create_dataset(f, n_var=7)
# grid_upds = [6,10,15]
# use_data_for_update = True

# Alternative dataset
# f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
# dataset = create_dataset(f, n_var=2)
# orig_model = build_KAN(Splines, [2, 1], grid_size=5)
# grid_upds = [10, 20, 40]
# use_data_for_update = True

# Even smaller dataset
f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]))
dataset = create_dataset(f, n_var=1)

grid = Grid(1, 5)
orig_model = Splines(1, 1, grid, k=3)

use_data_for_update = True
train_losses = []
test_losses = []

print(sum(p.numel() for p in orig_model.parameters() if p.requires_grad), [(k, p.shape) for k, p in orig_model.named_parameters()])

orig_model.update_grid(dataset["train_input"])
results = train(orig_model, dataset, opt="LBFGS", steps=25, lamb=0.0, lr=1e-2)  # To have a minimum of coherence with data


# Compute initial values of the loss
train_losses.append(loss_fn(orig_model.forward(dataset["train_input"]), dataset["train_label"]))
test_losses.append(loss_fn(orig_model.forward(dataset["test_input"]), dataset["test_label"]))

ranges, act_vals_before, x_layers = compute_act(orig_model, dataset["test_input"], layers_range=None)


fig, axs = plt.subplots(ncols=2)
axs[0].scatter(dataset["test_input"], dataset["test_label"])
axs[0].plot(ranges, act_vals_before, "-")

k = 0
axs[0].scatter(orig_model.grid.grid[:, 0].numpy(), np.zeros_like(orig_model.grid.grid[:, 0].numpy()) + k)

grid_upds = [(5, True), (10, True), (10, False), (20, True), (20, False)]
for g, use_data_for_update in grid_upds:
    model = deepcopy(orig_model)
    if use_data_for_update:
        model.update_grid(dataset["train_input"], grid_size=g)
    else:
        model.update_grid(None, grid_size=g)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), [(k, p.shape) for k, p in model.named_parameters()])
    _, act_vals_after, x_layers = compute_act(model, None, layers_range=x_layers)

    train_losses.append(loss_fn(model.forward(dataset["train_input"]), dataset["train_label"]))
    test_losses.append(loss_fn(model.forward(dataset["test_input"]), dataset["test_label"]))

    k += 0.5
    axs[0].plot(ranges, act_vals_after, "-", label=f"{g}, {use_data_for_update}")
    axs[0].scatter(model.grid.grid[:, 0].numpy(), np.zeros_like(model.grid.grid[:, 0].numpy()) + k)


axs[0].legend()
axs[1].plot(train_losses)
axs[1].plot(test_losses)
axs[1].legend(["train", "test"])
axs[1].set_ylabel("RMSE")
axs[1].set_yscale("log")
axs[1].grid(which="major")
axs[1].grid(which="minor", linestyle=":")
axs[1].set_xticks(range(1, 1 + len(grid_upds)), labels=[str(e) for e in grid_upds])
plt.show()
