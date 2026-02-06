from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)


# =========================================================
# Reference Element
# =========================================================
class ReferenceElement(ABC):
    pass


class LineReferenceElement(ReferenceElement):
    def __init__(self):
        self.simplex = torch.tensor([-1.0, 1.0])
        self.dim = 1
        self.n_nodes = self.simplex.shape[0]
        self.measure = torch.abs(self.simplex[1] - self.simplex[0]).item()


# =========================================================
# Shape Functions
# =========================================================
class ShapeFunction(ABC):
    def __init__(self, reference_element: ReferenceElement):
        self.reference_element = reference_element

    @abstractmethod
    def N(self, xi):
        pass


class LinearLineShapeFunction(ShapeFunction):
    def __init__(self):
        super().__init__(LineReferenceElement())

    def N(self, xi):
        return torch.stack([-0.5 * (xi - 1.0), 0.5 * (xi + 1.0)], dim=1)


class SecondOrderLineShapeFunction(ShapeFunction):
    def __init__(self):
        super().__init__(LineReferenceElement())

    def N(self, xi):
        return torch.stack(
            [0.5 * xi * (xi - 1.0), 1.0 - xi**2, 0.5 * xi * (xi + 1.0)], dim=1
        )


# =========================================================
# Quadratures
# =========================================================
class QuadratureRule(ABC):
    def __init__(self, reference_element):
        self.reference_element = reference_element()

    @abstractmethod
    def points(self, device):
        pass

    @abstractmethod
    def weights(self, device):
        pass


class MidPointQuadrature1D(QuadratureRule):
    def __init__(self):
        super().__init__(LineReferenceElement)

    def points(self, device):
        return torch.tensor([0.0, 0.0], device=device)

    def weights(self, device):
        w = self.reference_element.measure
        return torch.tensor([w], device=device)


class TwoPointsQuadrature1D(QuadratureRule):
    def __init__(self):
        super().__init__(LineReferenceElement)

    def points(self, device):
        l1 = 0.5 * (1.0 - 1 / 3**0.5)
        l2 = 0.5 * (1.0 + 1 / 3**0.5)
        return torch.tensor([-l2 + l1, -l1 + l2], device=device)

    def weights(self, device):
        w = 0.5 * self.reference_element.measure
        return torch.tensor([w, w], device=device)


# =========================================================
# Geometry Mapping
# =========================================================
class IsoparametricMapping1D:
    def __init__(self, shape_function):
        self.sf = shape_function

    def map(self, element_nodes_positions, xi):
        N = self.sf.N(xi)
        return torch.einsum("enx,en->ex", element_nodes_positions, N)

    def inverse_map(self, x, element_nodes_positions):
        # Construct M^{-1}
        det_J = element_nodes_positions[:, 0, 0] - element_nodes_positions[:, 1, 0]
        inverse_mapping = torch.ones(
            [int(element_nodes_positions.shape[0]), 2, 2],
            dtype=x.dtype,
            device=x.device,
        )
        inverse_mapping[:, 0, 1] = -element_nodes_positions[:, 1, 0]
        inverse_mapping[:, 1, 1] = element_nodes_positions[:, 0, 0]
        inverse_mapping[:, 1, 0] = -1 * inverse_mapping[:, 1, 0]
        inverse_mapping[:, :, :] /= det_J[:, None, None]
        x_extended = torch.stack((x, torch.ones_like(x)), dim=1)

        # Get barycentric coordinates on reference element
        xi_barycentric = torch.einsum(
            "eij,ej...->ei", inverse_mapping, x_extended.squeeze(1)
        )

        # Convert to coordinate on the reference element
        x_ref = self.sf.reference_element.simplex.repeat(xi_barycentric.shape[0], 1)
        xi = torch.einsum("ei,ei->e", xi_barycentric, x_ref)

        return xi

    def element_length(self, x_nodes):
        return x_nodes[:, 1, 0] - x_nodes[:, 0, 0]


# =========================================================
# Mesh
# =========================================================
class Mesh1D:
    def __init__(self, nodes, connectivity):
        self.nodes_positions = nodes
        self.conn = connectivity
        self.n_nodes = nodes.shape[0]
        self.n_elements = connectivity.shape[0]

        # Individual element index
        element_ids = torch.arange(self.conn.size(0))

        # Pairs of nodes indices per element
        element_nodes_ids = self.conn[element_ids, :].T
        self.element_nodes_ids = (
            torch.as_tensor(element_nodes_ids)
            .to(self.nodes_positions.device)
            .t()[:, :, None]
        )

        # Pairs of nodes positions
        element_nodes_positions = torch.gather(
            self.nodes_positions[:, None, :].repeat(1, 2, 1),
            0,
            self.element_nodes_ids.repeat(1, 1, 1),
        )
        self.element_nodes_positions = element_nodes_positions.to(
            self.nodes_positions.dtype
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
        full = torch.zeros(self.dofs_free.shape[0], 1, device=self.values_free.device)
        full[self.dofs_free] = self.values_free
        full[~self.dofs_free] = self.values_imposed
        return full


# =========================================================
# Evaluator (Discretization Engine)
# =========================================================
class ElementEvaluator1D:
    def __init__(self, mesh, field, sf, quad, mapping):
        self.mesh = mesh
        self.field = field
        self.sf = sf
        self.quad = quad
        self.mapping = mapping

    def evaluate(self):
        device = self.mesh.nodes_positions.device

        # Get reference position of quadrature points
        xi_g_barycentric = (
            self.quad.points(device).repeat(self.mesh.n_elements, 1).squeeze(-1)
        )

        # Convert to coordinate on the reference element
        x_ref = self.sf.reference_element.simplex.repeat(xi_g_barycentric.shape[0], 1)
        xi_g = torch.einsum("ei,ei->e", xi_g_barycentric, x_ref)

        # Get physical positions of quadrature points on mesh
        x_g = self.mapping.map(self.mesh.element_nodes_positions, xi_g)
        x_g.requires_grad_(True)
        # Get coordinates of those points back in reference coordinates
        xi_q = self.mapping.inverse_map(x_g, self.mesh.element_nodes_positions)

        # Get shape function coordinate representation in reference element
        N = self.sf.N(xi_q)

        # Get quadrature weights
        w = self.quad.weights(device)

        # Evaluate measure
        measure = (
            self.mapping.element_length(self.mesh.nodes_positions[self.mesh.conn])[
                :, None
            ]
            * w[None, :]
        )

        # Assemble values at nodes
        nodes_values = torch.gather(
            self.field.full_values()[:, None, :].repeat(1, 2, 1),
            0,
            self.mesh.element_nodes_ids.repeat(1, 1, 1),
        )
        nodes_values = nodes_values.to(N.dtype)

        # Interpolate
        u_q = torch.einsum("gij,gi->gj", nodes_values, N)

        return x_g, u_q, measure

    def evaluate_at(self, x):
        device = self.mesh.nodes_positions.device

        # Get elements in which x is located
        ids = []
        for x_i in x:
            for e, conn in enumerate(self.mesh.conn):
                x_first = self.mesh.nodes_positions[conn[0]]
                x_second = self.mesh.nodes_positions[conn[1]]
                if x_i >= x_first and x_i <= x_second:
                    ids.append(e)
                    break
        element_ids = torch.tensor(ids)

        # Pairs of nodes indices per element
        element_nodes_ids = self.mesh.conn[element_ids, :].T
        element_nodes_ids = (
            torch.as_tensor(element_nodes_ids).to(device).t()[:, :, None].squeeze(-1)
        )

        # Pairs of nodes positions
        element_nodes_positions = self.mesh.element_nodes_positions[element_ids]

        # We want to compute: u(x) = u_i^0 * N_i^0(\xi) + u_i^1 * N_i^1(\xi)
        # Where i is the index of element to which u x belongs
        # and \xi is the reference coordinate
        # Get reference coordinate of x in pair of nodes
        xi = self.mapping.inverse_map(x, element_nodes_positions)

        # Get shape function coordinate representation in reference element
        N = self.sf.N(xi)

        self.field.eval()
        u_full = self.field.full_values()

        # Assemble values at nodes
        nodes_values = u_full[element_nodes_ids]
        nodes_values = nodes_values.to(N.dtype)

        # Interpolate
        u_q = torch.einsum("gij,gi->gj", nodes_values, N)
        return u_q.detach()


# =========================================================
# Physics (Integrand ONLY)
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
        return torch.sum(integrand * measure)


# =========================================================
# Force function
# =========================================================
def f(x):
    return 1000.0 + 0 * x  # same as original


# =========================================================
# Main
# =========================================================
def main():
    """
    Proof of concept of design for HideNN-FEM

    This file presents the same interpolation than in the [noteBook from Alexandre](https://alexandredabyseesaram.github.io/Simplified_1D_NeuROM/demos/1D_element_based.html).
    The design is at follow:
    * Interface for a *reference element*, ReferenceElement.
      It exposes:
      * A dimension
      * Which simplex it represents (segment, triangle, tetrahedron, ...)
      * The simplex measure (length, area, volume, ...)
      * The number of nodes of the simplex
    * Interface for a *shape function*, ShapeFunction.
      It exposes:
        * The element over which it operates.
        * A function N() which acts over a reference coordinate (local to the element).
    * Interface for a *quadrature rule*, QuadratureRule.
      It exposes:
      * The element over which it operates.
      * The points() which it defines (should be in barycentric coordinates to easily generalize to triangle and tetrahedron)
      * The weights() associated which those points.
    """
    # ---------------- Mesh ----------------
    N = 40
    nodes = torch.linspace(0, 6.28, N)[:, None]
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T

    mesh = Mesh1D(nodes, elements)

    # ---------------- FEM building blocks ----------------
    sf = LinearLineShapeFunction()
    quad = TwoPointsQuadrature1D()
    mapping = IsoparametricMapping1D(sf)

    # ---------------- Field ----------------
    field = ScalarField1D(mesh, dirichlet_nodes=[0, N - 1])

    # ---------------- Evaluator ----------------
    evaluator = ElementEvaluator1D(mesh, field, sf, quad, mapping)

    # ---------------- Physics + Integrator ----------------
    physics = PoissonPhysics(f)
    integrator = Integrator()

    # ---------------- Training ----------------
    optimizer = torch.optim.Adam(field.parameters(), lr=1)
    loss_history = []

    # ---------------- Plots ----------------
    plot_loss = True
    plot_test = True

    print("* Training")
    n_epochs = 7000
    for i in range(n_epochs):
        x_q, u_q, measure = evaluator.evaluate()
        integrand = physics.integrand(x_q, u_q)
        loss = integrator.integrate(integrand, measure)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        print(f"{i=} loss={loss.item():.3e}", end="\r")

    # ---------------- Plot Loss ----------------
    if plot_loss:
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()

    # ---------------- Evaluation ----------------
    print("\n* Evaluation")

    x_test = torch.linspace(0, 6, 30)
    u_test = evaluator.evaluate_at(x_test)

    # ---------------- Plot solution ----------------
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
