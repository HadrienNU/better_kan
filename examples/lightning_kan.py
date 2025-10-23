import torch
from torch import optim
import lightning
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from better_kan import KAN, build_rbf_layers, plot
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_lightning_dataset(f, n_var=2, ranges=[-1, 1], train_num=1000, normalize_input=False, normalize_label=False, seed=0):
    """
    create dataset

    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        seed : int
            random seed. Default: 0.

    Returns:
    --------
        dataset : dic
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']

    Example
    -------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    torch.Size([100, 2])
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)

    train_input = torch.zeros(train_num, n_var)
    for i in range(n_var):
        train_input[:, i] = (
            torch.rand(
                train_num,
            )
            * (ranges[i, 1] - ranges[i, 0])
            + ranges[i, 0]
        )

    train_label = f(train_input)

    def normalize(data, mean, std):
        return (data - mean) / std

    if normalize_input is True:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)

    if normalize_label is True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)

    return SimpleDataset(train_input, train_label)


# define the LightningModule
class KAN_Lightning(lightning.LightningModule):
    def __init__(
        self,
        kan,
        lamb=0.01,
        lamb_l1=1.0,
        lamb_entropy=1.0,
        update_grid=True,
        grid_update_freq=10,
        stop_grid_update_step=50,
    ):
        super().__init__()
        self.kan = kan
        self.lamb = lamb
        self.lamb_l1 = lamb_l1
        self.lamb_entropy = lamb_entropy
        self.update_grid = update_grid
        self.stop_grid_update_step = stop_grid_update_step

        self.grid_update_freq = grid_update_freq

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        x, y = batch
        x = x.view(-1, self.kan.width[0])  # Assure input has the correct size

        if batch_idx % self.grid_update_freq == 0 and batch_idx < self.stop_grid_update_step and self.update_grid:
            self.kan.update_grid()
        pred = self.kan.forward(x)
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
