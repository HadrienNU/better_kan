"""
Study function that are function of the distance between particules
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from better_kan import build_KAN, plot, train
from better_kan.functions import Splines, ChebyshevPolynomial
from better_kan.equivariance import S, V, Trivial, parametrize_kan_equivariance
from tqdm import tqdm

# torch.manual_seed(7)


# Return distance matrices in a vector form
def distance_matrix(x):
    num_vectors = x.shape[-1]
    mat = torch.cdist(x.unsqueeze(-1), x.unsqueeze(-1))
    rows, cols = torch.triu_indices(num_vectors, num_vectors, offset=1)
    return mat[:, rows, cols]


def create_dataset(f, n_var=2, ranges=[-1, 1], train_num=1000, test_num=1000, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)

    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:, i] = (
            torch.rand(
                train_num,
            )
            * (ranges[i, 1] - ranges[i, 0])
            + ranges[i, 0]
        )
        test_input[:, i] = (
            torch.rand(
                test_num,
            )
            * (ranges[i, 1] - ranges[i, 0])
            + ranges[i, 0]
        )

    train_dists = distance_matrix(train_input)
    test_dists = distance_matrix(test_input)
    train_label = f(train_dists)
    test_label = f(test_dists)

    if train_label.ndim == 1:
        train_label.unsqueeze(-1)
        test_label.unsqueeze(-1)

    dataset = {}
    dataset["train_input"] = train_dists
    dataset["test_input"] = test_dists

    dataset["train_label"] = train_label
    dataset["test_label"] = test_label

    return dataset


def smooth_max_dist(dists):
    alpha = 2.0
    # Option 1
    # return ((dists * torch.exp(alpha * dists)).sum(dim=1) / torch.exp(alpha * dists).sum(dim=1))
    # option 2
    return torch.logsumexp(alpha * dists, dim=1).unsqueeze(-1) / alpha


dataset = create_dataset(smooth_max_dist, n_var=8)
print(dataset["train_input"].shape, dataset["train_label"].shape)
n_in_var = dataset["train_input"].shape[-1]

# print(dataset["train_label"])
grids = np.array([10, 20, 50, 100])

train_losses = []
test_losses = []
steps = 50
n_params = []

fig_res, axs_res = plt.subplots()
x_sorted = np.sort(dataset["test_label"].cpu())
axs_res.plot(x_sorted, x_sorted, "-", color="red")
axs_res.set_xlabel("x_test")
axs_res.set_ylabel("f(x_test)")

model = build_KAN(Splines, [n_in_var, 1, 1], grid_size=grids[0], k=3, add_batch_norm=False)
# model = build_KAN(ChebyshevPolynomial, [n_in_var, 1, 1], poly_order=15, add_batch_norm=False)
model.reset_parameters(init_type="noise", scale=10.0)
model.forward(dataset["train_input"])

# Implement permutation invariance
G = S(n_in_var)  # Permutation invariance of all input in this case
rep = V(G)
G2 = S(model.layers[1].in_features)  # Permutation invariance of all input in this case
rep2 = V(G2)
rep_trivial = V(Trivial(model.layers[1].out_features))

parametrize_kan_equivariance(model, [rep, rep2, rep_trivial])
model.update_grid(dataset["train_input"])

plot(model, dataset["test_input"], title="KAN_before training", tick=True, scale=2.0, norm_alpha=True)


with torch.no_grad():
    axs_res.scatter(dataset["test_label"].cpu(), model(dataset["test_input"]).cpu())


for i in range(grids.shape[0]):
    model.update_grid(None, grid_size=grids[i])
    results = train(model, dataset, opt="LBFGS", steps=steps, lamb=0.0, lr=1.0e-3, batch=-1)
    train_losses += results["train_loss"]
    test_losses += results["test_loss"]
    n_params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
    with torch.no_grad():
        axs_res.scatter(dataset["test_label"].cpu(), model(dataset["test_input"]).cpu())
print("N _params", n_params)


fig, axs = plt.subplots(ncols=2)

axs[0].plot(train_losses)
axs[0].plot(test_losses)
axs[0].legend(["train", "test"])
axs[0].set_ylabel("RMSE")
axs[0].set_xlabel("step")
axs[0].set_yscale("log")

n_params = np.asarray(n_params)
train_vs_G = train_losses[(steps - 1) :: steps]
test_vs_G = test_losses[(steps - 1) :: steps]
axs[1].plot(n_params, train_vs_G, marker="o")
axs[1].plot(n_params, test_vs_G, marker="o")
axs[1].plot(n_params, 100 * n_params ** (-3.0), ls="--", color="black")
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].legend(["train", "test", r"$N^{-3}$"])
axs[1].set_xlabel("number of params")
axs[1].set_ylabel("RMSE")

plot(model, dataset["test_input"], title="KAN_after training", tick=True, scale=2.0, norm_alpha=True)


plt.show()
