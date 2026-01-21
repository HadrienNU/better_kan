import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from better_kan.functions import Grid
from better_kan.functions import ActivationFunction, ChebyshevPolynomial, HermitePolynomial, Splines, GridReLU, RBFFunction


n_instance = 100

# Juste prendre une fonction et l'initialiser de plein de manière et tracer les résultats
# fcts_list = [(ActivationFunction, {"base_activation": nn.ReLU})]

fcts_list = [
    # (ActivationFunction, {"base_activation": nn.ReLU}),
    (ChebyshevPolynomial, {"poly_order": 4}),
    (HermitePolynomial, {"poly_order": 5}),
    (Splines, {"grid": Grid(n_instance, 15, order=3)}),
    (GridReLU, {"grid": Grid(n_instance, 10, order=2)}),
    (RBFFunction, {"grid": Grid(n_instance, 15, order=0), "rbf_kernel": "gaussian"}),
    (RBFFunction, {"grid": Grid(n_instance, 10, order=0), "rbf_kernel": "inverse quadratic"}),
]
# ()"uniform", {"a": -1.0, "b": 1.0}),
init_list = [("noise", {"scale": 1.0}), ("noise", {"scale": 10.0})]


fig, axs = plt.subplots(nrows=len(fcts_list), ncols=len(init_list), squeeze=False)
_, axs_cumul = plt.subplots(ncols=len(fcts_list), sharey=True)

ranges = [torch.linspace(-1.0, 1.0, 150) for _ in range(n_instance)]
x_in = torch.stack(ranges, dim=1)
x_ranges = x_in.cpu().detach().numpy()

for m, (fct_class, fct_kwargs) in enumerate(fcts_list):
    fct = fct_class(in_features=n_instance, out_features=1, **fct_kwargs)
    for n, (init_type, init_kwargs) in enumerate(init_list):
        fct.reset_parameters(init_type, **init_kwargs)
        acts_vals = fct.activations_eval(x_in).cpu().detach().numpy()
        for d in range(n_instance):
            axs[m, n].plot(x_ranges[:, d], acts_vals[:, 0, d])
        axs[m, n].plot(x_ranges[:, 0], acts_vals[:, 0, :].mean(axis=-1), "--", lw=1.5)

        axs[m, n].set_title(fct_class.__name__ + "_" + init_type)
        mean_val = acts_vals[:, 0, :].mean(axis=-1)
        # std_vals = acts_vals[:, 0, :].std(axis=-1)
        axs_cumul[m].fill_between(x_ranges[:, 0], acts_vals[:, 0, :].min(axis=-1), acts_vals[:, 0, :].max(axis=-1), alpha=0.5)
        axs_cumul[m].plot(x_ranges[:, 0], mean_val, "--", lw=1.5, label=init_type)
        axs_cumul[m].set_title(fct_class.__name__)
axs_cumul[-1].legend()
plt.show()
