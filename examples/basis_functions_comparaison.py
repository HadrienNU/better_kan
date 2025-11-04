"""
Compare results from various basis functions
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from better_kan import build_KAN, create_dataset, plot, train
from better_kan.functions import ChebyshevPolynomial, HermitePolynomial, Splines, GridReLU, RBFFunction

f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
dataset = create_dataset(f, n_var=2)
print(dataset["train_input"].shape, dataset["train_label"].shape)


fig, axs = plt.subplots(ncols=2)
x_sorted = np.sort(dataset["test_label"])
axs[1].plot(x_sorted, x_sorted, "-", color="red")

# Grid based functions
for name, fct in zip(["Splines", "GridRelu", "RBFFunction"], [Splines, GridReLU, RBFFunction]):

    model = build_KAN(fct, [2, 5, 1], grid_size=5, fast_version=True)
    for layer in model.layers:
        layer.reset_parameters(init_type="noise", scale=0.1)
    model.update_grid(dataset["train_input"])

    results = train(model, dataset, opt="LBFGS", steps=200, lamb=0.00)

    axs[0].plot(results["train_loss"], label=name + "_train")
    axs[0].plot(results["test_loss"], label=name + "_test")

    axs[1].scatter(dataset["test_label"].detach().numpy(), model(dataset["test_input"]).detach().numpy(), label=name)

# Polynomial Layers
for name, fct in zip(["Chebyshev", "Hermite"], [ChebyshevPolynomial, HermitePolynomial]):

    model = build_KAN(fct, [2, 5, 1], poly_order=3, fast_version=True)
    for layer in model.layers:
        layer.reset_parameters(init_type="noise", scale=0.1)
    model.update_grid(dataset["train_input"])

    results = train(model, dataset, opt="LBFGS", steps=200, lamb=0.00)

    axs[0].plot(results["train_loss"], label=name + "_train")
    axs[0].plot(results["test_loss"], label=name + "_test")

    axs[1].scatter(dataset["test_label"].detach().numpy(), model(dataset["test_input"]).detach().numpy(), label=name)


axs[0].set_ylabel("RMSE")
axs[0].set_xlabel("step")
axs[0].set_yscale("log")
axs[0].legend()
axs[0].grid()
axs[1].legend()
plt.show()
