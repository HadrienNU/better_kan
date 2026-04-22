# ML Potential with Dense Graph Layers

Machine Learning Potential implementation using dense graph layers for learning atomic energies and forces from pairwise distances.

## Features

- **Dense Graph Layers**: Equivariant graph convolutions for atomic systems
- **Variable Size Systems**: Proper masking for systems with different numbers of atoms
- **Atom Type Support**: Multiple element types with learnable embeddings
- **Energy + Force Training**: Force computation via automatic differentiation
- **KAN-style Model**: Optional KAN-inspired activation functions

## Dataset Format

The npz file should contain:

```
distances:    List of N x N distance matrices
atom_types:    List of atom type indices (atomic numbers)
energies:      List of total energies (one per system)
positions:     List of N x 3 position arrays (required for force computation)
forces:        List of N x 3 force arrays (optional, for supervision)
```

## Usage

### 1. Explore Dataset

```bash
python train.py --dataset path/to/data.npz --explore
```

### 2. Train Model

```bash
# Basic training (Adam optimizer, energy only)
python train.py --dataset path/to/data.npz --output_dir results

# Energy + Force training
python train.py --dataset path/to/data.npz --lambda_force 0.1 --output_dir results

# With KAN-style model
python train.py --dataset path/to/data.npz --use_kan --output_dir results

# With LBFGS optimizer (better for small datasets)
python train.py --dataset path/to/data.npz --optimizer lbfgs --output_dir results

# Custom architecture
python train.py --dataset path/to/data.npz \
    --hidden_dim 128 \
    --n_edge_layers 3 \
    --n_node_layers 2 \
    --use_self_loops \
    --output_dir results
```

### 3. Evaluate Model

```bash
python evaluate.py --dataset path/to/data.npz \
    --checkpoint results/best_model.pt \
    --output_dir evaluation
```

### 4. Visualize Results

```bash
# Training curves
python visualize.py --history_path results/training_history.npz \
    --plot_curves --output_dir visualizations

# Summary dashboard
python visualize.py \
    --history_path results/training_history.npz \
    --metrics_path evaluation/metrics.npz \
    --predictions_path evaluation/predictions.npz \
    --plot_dashboard --output_dir visualizations
```

## Model Architecture

### MLPPotential

```
Input:
  - Distance matrices [B, N, N]
  - Atom types [B, N]
  - Mask [B, N]

1. Atom Embedding → [B, N, D_atom]

2. Edge Features:
   - Concat [distance, emb_i, emb_j] → [B, N, N, edge_dim]

3. Edge Processing:
   - Linear projection
   - DenseGraphConvEdgeToEdge × n_edge_layers
   - Activation (SiLU)

4. Edge-to-Node Aggregation:
   - DenseGraphConvEdgeToNode
   - Activation

5. Node Processing:
   - DenseGraphConvNodeToNode × n_node_layers
   - Activation

6. Energy Prediction:
   - Global pooling (masked mean)
   - MLP → scalar energy per system

7. Force Computation (optional):
   - Autograd from energy w.r.t. positions
   - Forces = -∇E
```

### MLPPotentialWithKAN

Similar to MLPPotential but with learnable activation scalars for each channel:
```python
edge_features = edge_features * sigmoid(activation_scalars)
node_features = node_features * sigmoid(activation_scalars)
```

## Key Components

### Dense Graph Layers

- **DenseGraphConvEdgeToEdge**: Edge-to-edge convolution with equivariant basis
- **DenseGraphConvEdgeToNode**: Aggregate edge features to nodes
- **DenseGraphConvNodeToNode**: Node-to-node convolution

All layers support:
- Masking for variable-sized systems
- Optional self-loops
- Batch processing

### Equivariance Properties

- **Permutation equivariance**: Automatic via dense graph layers
- **Energy invariance**: Global pooling
- **Force equivariance**: Gradients of invariant energy

## Hyperparameters

### Model
- `hidden_dim`: Hidden dimension (default: 64)
- `n_edge_layers`: Number of edge layers (default: 2)
- `n_node_layers`: Number of node layers (default: 1)
- `atom_embedding_dim`: Atom embedding dimension (default: 16)
- `use_self_loops`: Include self-loops in edge layers
- `use_kan`: Use KAN-style model

### Training
- `batch_size`: Batch size (default: 32)
- `epochs`: Number of epochs (default: 200)
- `learning_rate`: Learning rate (default: 0.001)
- `optimizer`: adam or lbfgs (default: adam)
- `weight_decay`: L2 regularization (default: 0.0)
- `lambda_energy`: Energy loss weight (default: 1.0)
- `lambda_force`: Force loss weight (default: 0.1)

## Output Files

Training produces:
- `best_model.pt`: Best model checkpoint
- `final_model.pt`: Final model checkpoint
- `checkpoint_epoch_*.pt`: Periodic checkpoints
- `training_history.npz`: Loss history

Evaluation produces:
- `metrics.npz`: Evaluation metrics
- `predictions.npz`: Predictions vs true values
- `energy_predictions.png`: Energy scatter plot
- `energy_error_histogram.png`: Energy error distribution
- `force_predictions.png`: Force component scatter plots
- `force_error_histogram.png`: Force error distribution

## File Structure

```
experiments/mlip/
├── data_loader.py          # Dataset class and preprocessing
├── models/
│   ├── __init__.py
│   └── mlp_model.py        # MLPPotential models
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── visualize.py            # Visualization tools
├── configs/
│   └── default.yaml         # Default configuration
├── pair_potentials.py      # Reference implementation
└── README.md               # This file
```

## Examples

### Basic Example

```python
from data_loader import AtomicDataset, collate_fn
from models.mlp_model import MLPPotential
from torch.utils.data import DataLoader

# Load dataset
dataset = AtomicDataset("data.npz")

# Create dataloaders
train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Create model
model = MLPPotential(
    n_atom_types=dataset.n_atom_types,
    hidden_dim=64,
    n_edge_layers=2,
    n_node_layers=1,
)

# Forward pass
batch = next(iter(train_loader))
output = model(
    batch["distances"],
    batch["atom_types"],
    batch["mask"],
    return_forces=True,
    positions=batch["positions"],
)

energy = output["energy"]
forces = output["forces"]
```

### Training Loop Example

```python
import torch.nn.functional as F

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        output = model(
            batch["distances"],
            batch["atom_types"],
            batch["mask"],
            return_forces=True,
            positions=batch["positions"],
        )

        loss_energy = F.mse_loss(output["energy"], batch["energy"])
        loss_force = F.mse_loss(output["forces"], batch["forces"])
        loss = loss_energy + 0.1 * loss_force

        loss.backward()
        optimizer.step()
```

## Technical Notes

### Masking

All operations respect the mask for variable-sized systems:
- Edge features: `mask_edge = mask.unsqueeze(1) * mask.unsqueeze(2)`
- Node features: `mask.unsqueeze(-1)`
- Pooling: Masked mean instead of standard mean

### Force Computation

Forces are computed via automatic differentiation:
```python
positions.requires_grad_(True)
energy = model(...)
grad_energy = torch.autograd.grad(energy.sum(), positions, create_graph=True)[0]
forces = -grad_energy
```

### Memory Efficiency

For large systems:
- Reduce `hidden_dim`
- Reduce batch size
- Use gradient checkpointing (future feature)
- Consider sparse graph layers for very large N

## References

- Dense graph layers are based on equivariant graph neural networks
- KAN-style activations inspired by Kolmogorov-Arnold Networks
- Force computation via autograd follows standard MLIP practice

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `hidden_dim`
- Reduce `max_atoms` by preprocessing data

### Poor Energy Convergence
- Increase `n_edge_layers`
- Increase `hidden_dim`
- Try different learning rates
- Use LBFGS optimizer for small datasets

### Poor Force Convergence
- Increase `lambda_force`
- Ensure positions are provided and have gradients enabled
- Check force data normalization

### NaN Loss
- Check for zero distances in dataset
- Verify atom type mapping
- Reduce learning rate
- Add gradient clipping
