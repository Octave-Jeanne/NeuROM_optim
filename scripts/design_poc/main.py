from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import library modules
from hidenn_playground import elements, quadratures, shape_functions, geometry

torch.set_default_dtype(torch.float32)


# =========================================================
# Mesh
# =========================================================
class Mesh1D(nn.Module):
    def __init__(self, nodes, connectivity):
        super().__init__()

        self.register_buffer("nodes_positions", nodes)
        self.register_buffer("conn", connectivity)

        self.n_nodes = nodes.shape[0]
        self.n_elements = connectivity.shape[0]

        element_ids = torch.arange(self.conn.size(0))
        element_nodes_ids = self.conn[element_ids, :].T
        element_nodes_ids = element_nodes_ids.t()[:, :, None]

        self.register_buffer("element_nodes_ids", element_nodes_ids)

        element_nodes_positions = torch.gather(
            self.nodes_positions[:, None, :].repeat(1, 2, 1),
            0,
            element_nodes_ids.repeat(1, 1, 1),
        )

        self.register_buffer(
            "element_nodes_positions",
            element_nodes_positions.to(self.nodes_positions.dtype),
        )


# =========================================================
# Field (DOFs + BCs)
# =========================================================
class ScalarField1D(nn.Module):
    def __init__(self, mesh, dirichlet_nodes):
        super().__init__()

        n_nodes = mesh.n_nodes

        values = 0.5 * torch.ones(n_nodes, 1)
        dofs_free = torch.ones(n_nodes, dtype=torch.bool)
        dofs_free[dirichlet_nodes] = False

        self.register_buffer("dofs_free", dofs_free)
        self.values_free = nn.Parameter(values[dofs_free])
        self.register_buffer("values_imposed", torch.zeros((~dofs_free).sum(), 1))

    def full_values(self):
        full = torch.zeros(
            self.dofs_free.shape[0],
            1,
            device=self.values_free.device,
            dtype=self.values_free.dtype,
        )
        full[self.dofs_free] = self.values_free
        full[~self.dofs_free] = self.values_imposed
        return full


# =========================================================
# Evaluator
# =========================================================
class ElementEvaluator1D(nn.Module):
    def __init__(self, mesh, field, sf, quad, mapping):
        super().__init__()
        self.mesh = mesh
        self.field = field
        self.sf = sf
        self.quad = quad
        self.mapping = mapping

    def evaluate(self):
        # (N_q, N_nodes)
        x_q_barycentric = self.quad.points()
        # (N_e, N_nodes)
        x_el_nodes = self.mesh.element_nodes_positions
        # (N_q, dim )
        xi = elements.barycentric_to_reference(
            x_lambda=x_q_barycentric, element=self.quad.reference_element
        )
        # (N_e, N_q, dim)
        xi_g = xi.unsqueeze(0).expand(self.mesh.n_elements, -1, -1)

        # (N_e, N_q, dim)
        x_g = self.mapping.map(xi_g, x_el_nodes)
        # Required for autograd
        x_g.requires_grad_(True)

        # (N_e, N_q, dim)
        xi_q = self.mapping.inverse_map(x_g, x_el_nodes)

        # (N_e, N_q, N_nodes)
        N = self.sf.N(xi_q)

        # Compute weighted measure
        w = self.quad.weights()
        dx = self.mapping.element_size(x_el_nodes)
        measure = dx * w

        # Compute nodal values per element
        nodes_values = torch.gather(
            self.field.full_values()[:, None, :].repeat(1, 2, 1),
            0,
            self.mesh.element_nodes_ids.repeat(1, 1, 1),
        )
        nodes_values = nodes_values.to(N.dtype)

        # Interpolate field
        # (N_e, N_q, dim)
        u_q = torch.einsum("en...,eqn...->eq...", nodes_values, N)
        return x_g, u_q, measure

    def evaluate_at(self, x):
        device = self.mesh.nodes_positions.device
        x = x.unsqueeze(1).unsqueeze(2)
        # List elements to which `x` belongs to.
        ids = []
        for x_i in x:
            for e, conn in enumerate(self.mesh.conn):
                x_first = self.mesh.nodes_positions[conn[0]]
                x_second = self.mesh.nodes_positions[conn[1]]
                if x_i >= x_first and x_i <= x_second:
                    ids.append(e)
                    break

        element_ids = torch.tensor(ids, device=device)

        element_nodes_ids = self.mesh.conn[element_ids, :]
        # (N_e, N_q, dim)
        x_nodes = self.mesh.element_nodes_positions[element_ids]

        xi = self.mapping.inverse_map(x, x_nodes)
        N = self.sf.N(xi)
        u_full = self.field.full_values()
        nodes_values = u_full[element_nodes_ids]
        nodes_values = nodes_values.to(N.dtype)
        u_q = torch.einsum("en...,eqn...->eq...", nodes_values, N)
        return u_q.detach()


# =========================================================
# Physics
# =========================================================
class PoissonPhysics:
    def __init__(self, f):
        self.f = f

    def integrand(self, x, u):
        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        return 0.5 * du_dx**2 - self.f(x) * u


# =========================================================
# Integrator
# =========================================================
class Integrator:
    def integrate(self, integrand, measure):
        """
        Args:
            integrand: The field to integrate (N_e, N_q)
            measure: The measure to weight the integrand (N_e, N_q)
        """
        return torch.einsum("eq...,eq...->", integrand, measure)


# =========================================================
# Force function
# =========================================================
def f(x):
    return 1000.0


# ============================================================
# FEM Model
# ============================================================
class FEMModel(nn.Module):
    """
    Thin orchestration module.

    Responsibilities:
    - Own all FEM submodules so .to(device/dtype) works globally
    - Provide forward() for training / inference
    - Act as checkpoint root

    Non-responsibilities:
    - No FEM math implementation
    - No assembly logic
    - No physics implementation
    """

    def __init__(
        self,
        mesh,
        field,
        evaluator,
        physics,
        integrator,
    ):
        super().__init__()

        # Core pipeline
        self.mesh = mesh
        self.field = field
        self.evaluator = evaluator
        self.physics = physics
        self.integrator = integrator

    # -----------------------------------------------------
    # Main Forward Pass (Energy / Residual evaluation)
    # -----------------------------------------------------
    def forward(self):
        """
        Returns:
            scalar loss / energy
        """
        x_q, u_q, measure = self.evaluator.evaluate()
        integrand = self.physics.integrand(x_q, u_q)
        loss = self.integrator.integrate(integrand, measure)

        return loss


# =========================================================
# Main
# =========================================================
def main():

    N = 40
    nodes = torch.linspace(0, 6.28, N)[:, None]
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T

    mesh = Mesh1D(nodes, elements)

    sf = shape_functions.LinearSegment()
    # quad = quadratures.MidPoint1D()
    quad = quadratures.TwoPoints1D()
    mapping = geometry.IsoparametricMapping1D(sf)

    field = ScalarField1D(mesh, dirichlet_nodes=[0, N - 1])

    evaluator = ElementEvaluator1D(mesh, field, sf, quad, mapping)

    physics = PoissonPhysics(f)
    integrator = Integrator()

    model = FEMModel(
        mesh=mesh,
        field=field,
        evaluator=evaluator,
        physics=physics,
        integrator=integrator,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=10)
    loss_history = []

    plot_loss = True
    plot_test = True

    print("* Training")
    n_epochs = 7000
    for i in range(n_epochs):
        loss = model()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        print(f"{i=} loss={loss.item():.3e}", end="\r")

    print("\n* Evaluation")
    # At quadrature points
    x_q, u_q, _ = model.evaluator.evaluate()

    # At test points
    x_test = torch.linspace(0, 6, 30)
    u_test = model.evaluator.evaluate_at(x_test).squeeze()
    if plot_loss:
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()

    if plot_test:
        plt.figure()
        plt.plot(
            x_q.flatten().detach(), u_q.flatten().detach(), "+", label="Gauss points"
        )
        plt.plot(x_test, u_test, "o", label="Test points")
        plt.xlabel("x [mm]")
        plt.ylabel("u(x) [mm]")
        plt.title("Displacement Field")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
