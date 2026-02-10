import torch
import torch.nn as nn

import hidenn_playground.shape_functions as shape_functions


class IsoparametricMapping1D(nn.Module):
    def __init__(self, shape_function: shape_functions.ShapeFunction):
        super().__init__()
        self.sf = shape_function

    def map(self, xi, x_nodes):
        """
        Maps reference coordinate to physical position based on the elements positions

        Args:
            xi: The reference coordinate (N_e, N_q, dim)
            x_nodes: The nodal points (N_e, N_nodes, dim)

        Returns:
            The positions interpolated in the physical space: (N_e, N_q, dim)
        """
        # N_e x N_q x N_nodes
        N = self.sf.N(xi)
        return torch.einsum("eq...,eqn...->eq...", x_nodes, N)

    def inverse_map(self, x, x_nodes):
        """
        Maps physical position to reference coordinate for linear simplex elements.

        Args:
            x:        (N_e, N_q, dim)      physical coordinates
            x_nodes:  (N_e, N_nodes, dim)  nodal coordinates

        Returns:
            xi:       (N_e, N_q, dim)      reference coordinates

        Note:
            This linear mapping only works for linear shape functions.
        """
        # Base node (node 0)
        # (N_e, dim)
        x0 = x_nodes[:, 0, :]

        # Build Jacobian matrix J = [x1-x0, x2-x0, ...]
        # (N_e, dim, dim)
        J = x_nodes[:, 1:, :] - x0[:, None, :]

        # Invert Jacobian
        # (N_e, dim, dim)
        J_inv = torch.linalg.inv(J)

        # Compute x - x0
        # (N_e, N_q, dim)
        dx = x - x0[:, None, :]

        # Apply inverse mapping
        # xi = J_inv @ dx
        xi = torch.einsum("eij...,eqj...->eqi", J_inv, dx)

        return xi

    def element_size(self, x_nodes):
        """
        Computes the size of the physical elements.

        Args:
            x_nodes: The nodal points in physical space (N_e, N_nodes, dim)

        Returns:
            The size of the element.
        """
        return x_nodes[:, 1, :] - x_nodes[:, 0, :]
