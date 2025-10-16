import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from better_kan import KAN, build_splines_layers, create_dataset, plot, train
from better_kan.symbolic import Symbolic_KANLayer

model = KAN(build_splines_layers([2, 5, 1], grid_size=5, fast_version=False))


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


new_model = model.prune(mode="auto")

new_model(dataset["test_input"])


plot(new_model, title="KAN after pruning", tick=False)


# Replace both layer with symbolic expression
symbolic_layers = [Symbolic_KANLayer(la.in_features, la.out_features) for la in new_model.layers]
symbolic_layers[1].set_from_another_layer(new_model.layers[1], [["exp"]])
symbolic_layers[0].set_from_another_layer(new_model.layers[0], [["auto", "auto"]])
symbolic_kan = KAN(symbolic_layers)
symbolic_kan.forward(dataset["test_input"])
plot(symbolic_kan, title="Symbolic KAN", tick=False)
plt.show()
