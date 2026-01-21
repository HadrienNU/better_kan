import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from better_kan import build_KAN, create_dataset, plot, train
from better_kan.functions import Splines

model = build_KAN(Splines, [2, 5, 1], grid_size=5, fast_version=False)
for layer in model.layers:
    layer.reset_parameters(init_type="noise", scale=10.0)

f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
dataset = create_dataset(f, n_var=2, train_num=12)

print(dataset["train_input"].shape, dataset["train_label"].shape)

output = model(dataset["train_input"])
print(output)

plot(model, dataset["test_input"], title="KAN_initialisation", tick=False)

plt.show()
