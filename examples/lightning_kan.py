import torch
from torch import optim
import lightning
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from better_kan import KAN, build_rbf_layers, plot
from better_kan.lightning import create_lightning_dataset


# define the LightningModule
class KAN_Lightning(lightning.LightningModule):
    def __init__(
        self,
        kan,
        lamb=0.01,
        lamb_l1=1.0,
        lamb_entropy=1.0,
        update_grid=True,
        grid_update_num=10,
        stop_grid_update_step=50,
    ):
        super().__init__()
        self.kan = kan
        self.lamb = lamb
        self.lamb_l1 = lamb_l1
        self.lamb_entropy = lamb_entropy
        self.update_grid = update_grid
        self.stop_grid_update_step = stop_grid_update_step

        self.grid_update_freq = int(stop_grid_update_step / grid_update_num)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        x, y = batch
        x = x.view(-1, self.kan.width[0])  # Assure input has the correct size

        pred = self.kan.forward(x, update_grid=(batch_idx % self.grid_update_freq == 0 and batch_idx < self.stop_grid_update_step and self.update_grid))
        train_loss = torch.mean((pred - y) ** 2)
        reg_ = self.kan.regularization_loss()
        loss = train_loss + self.lamb * reg_

        self.log("train_loss", loss)
        self.log("regularization", reg_)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.1)
        # optimizer = optim.LBFGS(self.parameters(), lr=1.0)
        return optimizer


f = lambda x: 1 / (1 + torch.exp(-(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2 + x[:, [2]] ** 2 + x[:, [3]] ** 2 + x[:, [4]] ** 2)))
dataset = create_lightning_dataset(f, n_var=5)

data_loader = DataLoader(dataset, batch_size=1000)
model = KAN_Lightning(KAN(build_rbf_layers([5, 3, 3, 1], grid_size=5)))
print(model)

# Train model ==============================

# define trainer
trainer = lightning.Trainer(max_epochs=5, enable_checkpointing=False)
trainer.fit(model, data_loader)

plot(model.kan)

plt.show()
