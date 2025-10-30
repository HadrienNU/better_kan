import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


from better_kan import build_KAN, create_dataset, plot, train
from better_kan.functions import Splines, SymbolicFunction


model = build_KAN(Splines, [2, 5, 1], grid_size=5, fast_version=False)


f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
dataset = create_dataset(f, n_var=2)
print(dataset["train_input"].shape, dataset["train_label"].shape)


model.update_grid(dataset["train_input"])
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


new_model = model.prune(mode="highest", active_neurons_id=[1])

new_model(dataset["test_input"])


plot(new_model, title="KAN after pruning", tick=False)


symbolic_kan = build_KAN(SymbolicFunction, [2, new_model.layers[0].out_features, 1])
# Replace both layers with symbolic expression
symbolic_kan.layers[1].functions[0].project_on_basis(new_model.layers[1].functions[0], [["exp"]])
symbolic_kan.layers[0].functions[0].project_on_basis(new_model.layers[0].functions[0], [["auto", "auto"]])
symbolic_kan.forward(dataset["test_input"])
plot(symbolic_kan, title="Symbolic KAN", tick=False)
plt.show()
