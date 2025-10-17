"""
Contain a Grid class
"""


import torch
import torch.nn as nn

class Grid(nn.Module):
    def __init__( self, in_features, size, order, grid_range= (-1,1), grid_alpha=0.02):
        super().__init__()

        self.grid_size=size
        self.order = order # Define the grid order
        self.grid_range=grid_range

        h = (grid_range[1] - grid_range[0]) / self.grid_size
        grid = (torch.arange(-self.order, self.grid_size + self.order + 1) * h + grid_range[0]).expand(in_features, -1).transpose(0, 1).contiguous()

        self.grid_alpha = grid_alpha

        self.grid = nn.Parameter(grid, requires_grad=False)

    def _initialize(self):

        self.grid=


    def update(self, x,  grid_size=-1, margin=0.01):
        torch._assert(x.dim() == 2 and x.size(1) == self.in_features, "Input dimension does not match layer size")
        batch = x.size(0)
        basis_values = self.basis(x)
        unreduced_basis_output = torch.sum(basis_values.unsqueeze(1) * self.weights.unsqueeze(0), dim=2)  # (batch, out, in)
        unreduced_basis_output = unreduced_basis_output.transpose(1, 2)  # (batch, in, out)
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        if grid_size > 0:
            self.grid_size=grid_size
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size, dtype=torch.int64, device=x.device)]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = torch.arange(self.grid_size, dtype=x.dtype, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin

        grid = self.grid_alpha * grid_uniform + (1 - self.grid_alpha) * grid_adaptive
        assign_parameters(self, "grid", grid)

        new_basis_values = self.basis(x)

        

        assign_parameters(self, "weights", self.curve2coeff(x, unreduced_basis_output))



class ParametrizedGrid(nn.Module):


    def __init__( self, in_features, size, order, grid_range= (-1,1), grid_alpha=0.02):
        super().__init__()

        self.grid_size=size
        self.order = order # Define the grid order
        self.grid_range=grid_range # TODO: make it 2*n_in

        h = (grid_range[1] - grid_range[0]) / self.grid_size
        grid = (torch.arange(-self.order, self.grid_size + self.order + 1) * h + grid_range[0]).expand(in_features, -1).transpose(0, 1).contiguous()

        self.grid_alpha = grid_alpha

        self.s = nn.Parameter(grid)

    @property # Convert Parameters s  to grid
    def grid(self):
        return self.grid_range[0]*(self.grid_range[1]-self.grid_range[0])*sum(softmax(self.s))
    

    def right_inverse(self,grid):
        """
        From a grid, give the s values
        """