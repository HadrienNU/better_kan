import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from efficient_kan_rbf import *


f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
dataset = create_dataset(f, n_var=2)
print(dataset["train_input"].shape, dataset["train_label"].shape)


grids = np.array([5, 10, 20, 50, 100])

train_losses = []
test_losses = []
steps = 50
k = 3

for i in range(grids.shape[0]):
    if i == 0:
        model = build_splines_layers([2, 1, 1], grid_size=grids[i], k=k)
    if i != 0:
        model = build_splines_layers([2, 1, 1], grid_size=grids[i], k=k).initialize_from_another_model(model, dataset["train_input"])
    results = train(model, dataset, opt="LBFGS", steps=steps, stop_grid_update_step=30)
    train_losses += results["train_loss"]
    test_losses += results["test_loss"]
plt.figure()
plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(["train", "test"])
plt.ylabel("RMSE")
plt.xlabel("step")
plt.yscale("log")

plt.figure()
n_params = 3 * grids
train_vs_G = train_losses[(steps - 1) :: steps]
test_vs_G = test_losses[(steps - 1) :: steps]
plt.plot(n_params, train_vs_G, marker="o")
plt.plot(n_params, test_vs_G, marker="o")
plt.plot(n_params, 100 * n_params ** (-4.0), ls="--", color="black")
plt.xscale("log")
plt.yscale("log")
plt.legend(["train", "test", r"$N^{-4}$"])
plt.xlabel("number of params")
plt.ylabel("RMSE")

plot(model, title="KAN_after training", tick=False)
plt.show()
