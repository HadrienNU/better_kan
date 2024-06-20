from torch import optim
import torch
import lightning
import numpy as np
import matplotlib.pyplot as plt


class MetricsCallback(lightning.Callback):
    """Lightning callback which saves logged metrics into a dictionary.
    The metrics are recorded at the end of each validation epoch.
    """

    def __init__(self):
        super().__init__()
        self.metrics = {"epoch": []}

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if not trainer.sanity_checking:
            self.metrics["epoch"].append(trainer.current_epoch)
            for key, val in metrics.items():
                val = val.item()
                if key in self.metrics:
                    self.metrics[key].append(val)
                else:
                    self.metrics[key] = [val]


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
        self.grid_update_num = grid_update_num
        self.stop_grid_update_step = stop_grid_update_step

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.kan.forward(x, update_grid=(batch_idx % self.grid_update_freq == 0 and batch_idx < self.stop_grid_update_step and self.update_grid))
        train_loss = torch.mean((pred - y) ** 2)
        reg_ = self.kan.regularization_loss()
        loss = train_loss + self.lamb * reg_

        self.log("train_loss", loss)
        self.log("regularization", reg_)
        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=0.1)
        optimizer = optim.LBFGS(self.parameters(), lr=1.0)
        return optimizer


def plot_metrics(
    metrics,
    keys=["train_loss_epoch", "valid_loss"],
    x=None,  # 'epoch'
    labels=None,  # ['Train','Valid'],
    linestyles=None,  # ['-','--']
    colors=None,  # ['fessa0','fessa1']
    xlabel="Epoch",
    ylabel="Loss",
    title="Learning curves",
    yscale=None,
    ax=None,
):
    """Plot logged metrics."""

    # Setup axis
    return_axs = False
    if ax is None:
        return_axs = True
        _, ax = plt.subplots(figsize=(5, 4), dpi=100)

    # Plot metrics
    auto_x = True if x is None else False
    for i, key in enumerate(keys):
        y = metrics[key]
        lstyle = linestyles[i] if linestyles is not None else None
        label = labels[i] if labels is not None else key
        color = colors[i] if colors is not None else None
        if auto_x:
            x = np.arange(len(y))
        ax.plot(x, y, linestyle=lstyle, label=label, color=color)

    # Plot settings
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if yscale is not None:
        ax.set_yscale(yscale)

    ax.legend(ncol=1, frameon=False)

    if return_axs:
        return ax
    else:
        return None
