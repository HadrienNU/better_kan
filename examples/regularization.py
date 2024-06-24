import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker

from better_kan import KAN, build_chebyshev_layers, create_dataset, train


f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
dataset = create_dataset(f, n_var=2)


l1_mesh = torch.logspace(-5, 1, steps=10)
entropy_mesh = torch.logspace(-5, 1, steps=10)

results = np.zeros((10, 10, 3))

for n, l1 in enumerate(l1_mesh):
    for m, entropy in enumerate(entropy_mesh):

        model = KAN(build_chebyshev_layers([2, 5, 1], grid_size=30))

        res = train(model, dataset, opt="LBFGS", steps=100, update_grid=True, lamb=1.0, lamb_entropy=entropy, lamb_l1=l1)

        for i, name in enumerate(["train_loss", "test_loss", "reg"]):
            results[n, m, i] = res[name][-1]

X_plot = l1_mesh.cpu().detach().numpy()
Y_plot = entropy_mesh.cpu().detach().numpy()

fig, axs = plt.subplots(1, 3)

axs[0].set_title("Train loss")
h = axs[0].contourf(X_plot, Y_plot, results[:, :, 0], locator=ticker.LogLocator())
plt.colorbar(h, ax=axs[0])

axs[1].set_title("Test loss")
h = axs[1].contourf(X_plot, Y_plot, results[:, :, 1], locator=ticker.LogLocator())
plt.colorbar(h, ax=axs[1])

axs[2].set_title("Regularization")
h = axs[2].contourf(X_plot, Y_plot, results[:, :, 2], locator=ticker.LogLocator())
plt.colorbar(h, ax=axs[2])

plt.show()
