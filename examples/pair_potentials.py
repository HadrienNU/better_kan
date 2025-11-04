"""
Study function that are function of the distance between particules
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from better_kan import build_KAN, create_dataset, train, plot
from better_kan.functions import Splines

torch.manual_seed(7)


# Return distance matrices in a vector form
def distance_matrix(x):
    num_vectors = x.shape[-1]
    mat = torch.cdist(x.unsqueeze(-1), x.unsqueeze(-1))
    rows, cols = torch.triu_indices(num_vectors, num_vectors, offset=1)
    return mat[:, rows, cols]


def smooth_max_dist(x):
    alpha = 2.0
    dists = distance_matrix(x)
    # Option 1
    # return (dists * torch.exp(alpha * dists)).sum(dim=1) / torch.exp(alpha * dists).sum(dim=1)
    # option 2
    return torch.logsumexp(alpha * dists, dim=1) / alpha


dataset = create_dataset(smooth_max_dist, n_var=10)
print(dataset["train_input"].shape, dataset["train_label"].shape)

print(dataset["train_label"])
# grids = np.array([5, 10, 20, 50, 100])

# train_losses = []
# test_losses = []
# steps = 25
# k = 3
# n_params = []

# fig_res, axs_res = plt.subplots()
# x_sorted = np.sort(dataset["test_label"].cpu())
# axs_res.plot(x_sorted, x_sorted, "-", color="red")

# model = build_KAN(Splines, [2, 5, 1], grid_size=grids[0], k=k)
# for i in range(grids.shape[0]):
#     model.update_grid(None, grid_size=grids[i])
#     results = train(model, dataset, opt="LBFGS", steps=steps, stop_grid_update_step=50, update_grid=None, lamb=0.0)
#     train_losses += results["train_loss"]
#     test_losses += results["test_loss"]
#     n_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
#     with torch.no_grad():
#         axs_res.scatter(dataset["test_label"].cpu(), model(dataset["test_input"]).cpu())
# print("N _params", n_params)
# fig, axs = plt.subplots(ncols=2)

# axs[0].plot(train_losses)
# axs[0].plot(test_losses)
# axs[0].legend(["train", "test"])
# axs[0].set_ylabel("RMSE")
# axs[0].set_xlabel("step")
# axs[0].set_yscale("log")

# n_params = 3 * grids
# train_vs_G = train_losses[(steps - 1) :: steps]
# test_vs_G = test_losses[(steps - 1) :: steps]
# axs[1].plot(n_params, train_vs_G, marker="o")
# axs[1].plot(n_params, test_vs_G, marker="o")
# axs[1].plot(n_params, 100 * n_params ** (-4.0), ls="--", color="black")
# axs[1].set_xscale("log")
# axs[1].set_yscale("log")
# axs[1].legend(["train", "test", r"$N^{-4}$"])
# axs[1].set_xlabel("number of params")
# axs[1].set_ylabel("RMSE")

# plot(model, title="KAN_after training", tick=False)


# plt.show()
