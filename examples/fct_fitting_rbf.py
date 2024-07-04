import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from better_kan import create_dataset, KAN, build_rbf_layers, train, plot, update_plot


plt.ion()  # turning interactive mode on

f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
dataset = create_dataset(f, n_var=2)
print(dataset["train_input"].shape, dataset["train_label"].shape)


grids = np.array([5, 10, 20, 50, 100])

train_losses = []
test_losses = []
steps = 75


model = KAN(build_rbf_layers([2, 1, 1], grid_size=grids[0], optimize_grid=False))
model.forward(dataset["test_input"])
inserts_axes, act_lines = plot(model, title="during training", tick=False)
plt.pause(0.25)  # Allow enough time for plot to draw

for i in range(grids.shape[0]):
    if i != 0:
        model = KAN(build_rbf_layers([2, 1, 1], grid_size=grids[i], optimize_grid=False)).initialize_from_another_model(model)
    results = train(model, dataset, opt="LBFGS", steps=steps, update_grid=True, lamb=0.0)
    train_losses += results["train_loss"]
    test_losses += results["test_loss"]
    update_plot(model, inserts_axes, act_lines)
    plt.pause(0.25)  # Allow enough time for plot to redraw

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

plt.ioff()  # turning interactive mode off to have blocking plt.show()

plot(model, title="KAN_after training", tick=False)
plt.show()
