import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from better_kan import KAN, RBFKANLayer, build_rbf_layers, create_dataset, plot, train

features_layers = build_rbf_layers([4, 6], grid_size=5, fast_version=False, rbf_kernel="gaussian")
# features_layers.append(RBFKANLayer(6, 1, grid_size=5, fast_version=False, rbf_kernel="gaussian", pooling_op="logsumexp"))
model = KAN(features_layers)

f = lambda x: torch.cdist(x.unsqueeze(2), x.unsqueeze(2), 1).reshape(-1, 6)  # softmax_{i,j} |x_i-x_j|
# f = lambda x: torch.logsumexp(torch.cdist(x.unsqueeze(2), x.unsqueeze(2), 1), (1, 2)).unsqueeze(1)  # softmax_{i,j} |x_i-x_j|
dataset = create_dataset(f, n_var=4)
print(dataset["train_input"].shape, dataset["train_label"].shape)


model.update_grid(dataset["train_input"])

results = train(model, dataset, opt="LBFGS", steps=60, update_grid=True, lamb=0.05)


fig, axs = plt.subplots(2)

axs[0].plot(results["train_loss"])
axs[0].plot(results["test_loss"])
axs[0].plot(results["reg"])
axs[0].legend(["train", "test", "reg"])
axs[0].set_ylabel("RMSE")
axs[0].set_xlabel("step")
axs[0].set_yscale("log")

with torch.no_grad():
    x_sorted = np.sort(dataset["test_label"].cpu())
    axs[1].plot(x_sorted, x_sorted, "-", color="red")
    axs[1].scatter(dataset["test_label"].cpu(), model(dataset["test_input"]).cpu())


plot(model, title="KAN_after training", tick=False)


new_model = model.prune(mode="auto")

new_model(dataset["train_input"])


plot(new_model, title="KAN after pruning", tick=False)

plt.show()
