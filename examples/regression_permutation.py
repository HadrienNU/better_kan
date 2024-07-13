import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from better_kan import KAN, build_splines_layers, create_dataset, plot, train


# Build model with some permutation invariant input
model = KAN(build_splines_layers([7, 4, 1], grid_size=5, permutation_invariants=[0, 1, 1, 1, 1, 2, 2], sbasis_trainable=True, sb_trainable=False))  #


f = lambda x: 1 / (1 + torch.exp(-(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2 + x[:, [2]] ** 2 + x[:, [3]] ** 2 + x[:, [4]] ** 2 - (x[:, [5]] - x[:, [6]]).abs() ** 0.5)))
dataset = create_dataset(f, n_var=7)

print(dataset["train_input"].shape, dataset["train_label"].shape)
model(dataset["train_input"], update_grid=True)

print(model)
print(model.layers[0].parametrizations.weights.original0.shape)
print(model.layers[1].parametrizations.weights.original0.shape)


results = train(model, dataset, opt="Adam", steps=1500, update_grid=True, stop_grid_update_step=1450, grid_update_freq=50, lamb=0.01, lr=1e-2)

plt.figure()

plt.plot(results["train_loss"])
plt.plot(results["test_loss"])
plt.plot(results["reg"])
plt.legend(["train", "test", "reg"])
plt.ylabel("RMSE")
plt.xlabel("step")
plt.yscale("log")


plot(model, title="KAN_after training", tick=False, norm_alpha=True)

new_model = model.prune(mode="auto")

new_model(dataset["train_input"])
plot(new_model, title="KAN after pruning", tick=False)


fig, axs = plt.subplots(2, 2)

fitted_value = model(dataset["test_input"])
with torch.no_grad():
    x_sorted = np.sort(dataset["test_label"])
    axs[0, 0].set_title("After training")
    axs[0, 0].plot(x_sorted, x_sorted, "-", color="red")
    axs[0, 0].scatter(dataset["test_label"], model(dataset["test_input"]))
    axs[0, 1].set_title("After training")
    axs[0, 1].hexbin(dataset["test_label"], model(dataset["test_input"]), gridsize=25)

    axs[1, 0].set_title("After pruning")
    axs[1, 0].plot(x_sorted, x_sorted, "-", color="red")
    axs[1, 0].scatter(dataset["test_label"], model(dataset["test_input"]))
    axs[1, 1].set_title("After pruning")
    axs[1, 1].hexbin(dataset["test_label"], model(dataset["test_input"]), gridsize=25)
plt.xlabel("True Value")
plt.ylabel("Estimated Value")


plt.show()
