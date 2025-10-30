import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from better_kan import build_KAN, create_dataset, plot, train
from better_kan.functions import RBFFunction

datasets = []

n_peak = 5
n_num_per_peak = 100
n_sample = n_peak * n_num_per_peak

x_grid = torch.linspace(-1, 1, steps=n_sample)

x_centers = 2 / n_peak * (np.arange(n_peak) - n_peak / 2 + 0.5)

x_sample = torch.stack([torch.linspace(-1 / n_peak, 1 / n_peak, steps=n_num_per_peak) + center for center in x_centers]).reshape(-1)


y = 0.0
for center in x_centers:
    y += torch.exp(-((x_grid - center) ** 2) * 300)

y_sample = 0.0
for center in x_centers:
    y_sample += torch.exp(-((x_sample - center) ** 2) * 300)

plt.subplots(1, 5, figsize=(15, 2), num="Input data")
plt.subplots_adjust(wspace=0, hspace=0)

for i in range(1, 6):
    plt.subplot(1, 5, i)
    group_id = i - 1
    plt.plot(x_grid.detach().numpy(), y.detach().numpy(), color="black", alpha=0.1)
    plt.scatter(
        x_sample[group_id * n_num_per_peak : (group_id + 1) * n_num_per_peak].detach().numpy(),
        y_sample[group_id * n_num_per_peak : (group_id + 1) * n_num_per_peak].detach().numpy(),
        color="black",
        s=2,
    )
    plt.xlim(-1, 1)
    plt.ylim(-1, 2)

model = build_KAN(RBFFunction, [1, 1], grid_size=200, fast_version=False)


ys = []
for group_id in range(n_peak):
    dataset = {}
    dataset["train_input"] = x_sample[group_id * n_num_per_peak : (group_id + 1) * n_num_per_peak][:, None]
    dataset["train_label"] = y_sample[group_id * n_num_per_peak : (group_id + 1) * n_num_per_peak][:, None]
    dataset["test_input"] = x_sample[group_id * n_num_per_peak : (group_id + 1) * n_num_per_peak][:, None]
    dataset["test_label"] = y_sample[group_id * n_num_per_peak : (group_id + 1) * n_num_per_peak][:, None]
    train(model, dataset, opt="LBFGS", steps=100, update_grid=False)
    y_pred = model(x_grid[:, None])
    ys.append(y_pred.detach().numpy()[:, 0])
plt.subplots(1, 5, figsize=(15, 2), num="Fitted")
plt.subplots_adjust(wspace=0, hspace=0)

for i in range(1, 6):
    plt.subplot(1, 5, i)
    group_id = i - 1
    plt.plot(x_grid.detach().numpy(), y.detach().numpy(), color="black", alpha=0.1)
    plt.plot(x_grid.detach().numpy(), ys[i - 1], color="black")
    plt.xlim(-1, 1)
    plt.ylim(-1, 2)
plt.show()
