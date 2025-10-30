import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from better_kan import KAN, build_rbf_layers, build_splines_layers, create_dataset, plot, train
from better_kan import build_KAN, create_dataset, plot, train
from better_kan.functions import Splines, RBF

layers = build_rbf_layers([2, 3], grid_size=4)

mlp_layer = nn.Sequential(nn.Linear(3, 4), nn.ELU())
mlp_layer.in_features = 3  # The added layer should have an in_features and out_features attribute
mlp_layer.out_features = 4
layers.append(mlp_layer)

layers.extend(build_splines_layers([4, 1], grid_size=3))

model = KAN(layers)

dataset = create_dataset(lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2), n_var=2)
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
