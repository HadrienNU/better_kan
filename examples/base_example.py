import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from better_kan import build_rbf_layers, create_dataset, plot, train


model = build_rbf_layers([2, 5, 1], grid_size=5)


f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
dataset = create_dataset(f, n_var=2)
print(dataset["train_input"].shape, dataset["train_label"].shape)


model(dataset["train_input"])
plot(model, title="KAN_initialisation", tick=False)


results = train(model, dataset, opt="LBFGS", steps=60, update_grid=True, lamb=0.05)

plt.figure()

plt.plot(results["train_loss"])
plt.plot(results["test_loss"])
plt.plot(results["reg"])
plt.legend(["train", "test", "reg"])
plt.ylabel("RMSE")
plt.xlabel("step")
plt.yscale("log")


plot(model, title="KAN_after training", tick=False)
plt.show()
