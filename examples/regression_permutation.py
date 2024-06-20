import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from better_kan import build_rbf_layers, create_dataset, plot, train

# Build model with some permutation invariant input

# For now bigger model fail quickly
# model = build_rbf_layers([7, 15, 4, 1], grid_size=5, permutation_invariants=[0, 1, 1, 1, 1, 2, 2])  #


# f = lambda x: 1 / (1 + torch.exp(-(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2 + x[:, [2]] ** 2 + x[:, [3]] ** 2 + x[:, [4]] ** 2 - (x[:, [5]] + x[:, [6]]) ** 0.5)))
# dataset = create_dataset(f, n_var=7)

model = build_rbf_layers([3, 7, 4, 1], grid_size=5, permutation_invariants=[0, 1, 1])  #


f = lambda x: 1.0 / (1.0 + torch.exp(-(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2 + x[:, [2]] ** 2)))
dataset = create_dataset(f, n_var=3)

print(dataset["train_input"].shape, dataset["train_label"].shape)


results = train(model, dataset, opt="Adam", steps=5000, update_grid=True, stop_grid_update_step=4500, grid_update_num=100, lamb=0.01, lr=1e-2)

plt.figure()

plt.plot(results["train_loss"])
plt.plot(results["test_loss"])
plt.plot(results["reg"])
plt.legend(["train", "test", "reg"])
plt.ylabel("RMSE")
plt.xlabel("step")
plt.yscale("log")


plot(model, title="KAN_after training", tick=False)

plt.figure()

fitted_value = model(dataset["test_input"])
with torch.no_grad():
    x_sorted = np.sort(dataset["test_label"])
    plt.plot(x_sorted, x_sorted, "-", color="red")
    plt.scatter(dataset["test_label"], model(dataset["test_input"]))
    plt.figure()
    plt.hexbin(dataset["test_label"], model(dataset["test_input"]), gridsize=25)
plt.xlabel("True Value")
plt.ylabel("Estimated Value")

plt.show()
