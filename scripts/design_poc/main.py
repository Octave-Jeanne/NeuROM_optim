from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)


# =========================================================
# Reference Element
# =========================================================
class ReferenceElement(nn.Module, ABC):
    def __init__(self):
        super().__init__()


class LineReferenceElement(ReferenceElement):
    def __init__(self):
        super().__init__()

        simplex = torch.tensor([-1.0, 1.0])
        self.register_buffer("simplex", simplex)

        self.dim = 1
        self.n_nodes = simplex.shape[0]

        measure = torch.abs(simplex[1] - simplex[0])
        self.register_buffer("measure", measure)


# =========================================================
# Shape Functions
# =========================================================
class ShapeFunction(nn.Module, ABC):
    def __init__(self, reference_element: ReferenceElement):
        super().__init__()
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
class QuadratureRule(nn.Module, ABC):
    def __init__(self, reference_element):
        super().__init__()
        self.reference_element = reference_element()


class MidPointQuadrature1D(QuadratureRule):
    def __init__(self):
        super().__init__(LineReferenceElement)

        points = torch.tensor([0.0])
        weights = self.reference_element.measure.clone()

        self.register_buffer("points_ref", points)
        self.register_buffer("weights_ref", weights[None])

    def points(self):
        return self.points_ref

    def weights(self):
        return self.weights_ref


class TwoPointsQuadrature1D(QuadratureRule):
    def __init__(self):
        super().__init__(LineReferenceElement)

        l1 = 0.5 * (1.0 - 1 / 3**0.5)
        l2 = 0.5 * (1.0 + 1 / 3**0.5)

        points = torch.tensor([-l2 + l1, -l1 + l2])
        w = 0.5 * self.reference_element.measure
        weights = torch.tensor([w, w])

        self.register_buffer("points_ref", points)
        self.register_buffer("weights_ref", weights)

    def points(self):
        return self.points_ref

    def weights(self):
        return self.weights_ref


# =========================================================
# Geometry Mapping
# =========================================================
class IsoparametricMapping1D(nn.Module):
    def __init__(self, shape_function):
        super().__init__()
        self.sf = shape_function

    def map(self, element_nodes_positions, xi):
        N = self.sf.N(xi)
        return torch.einsum("enx,en->ex", element_nodes_positions, N)

    def inverse_map(self, x, element_nodes_positions):
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

        xi_barycentric = torch.einsum(
            "eij,ej...->ei", inverse_mapping, x_extended.squeeze(1)
        )

        x_ref = self.sf.reference_element.simplex.repeat(xi_barycentric.shape[0], 1)
        xi = torch.einsum("ei,ei->e", xi_barycentric, x_ref)

        return xi

    def element_length(self, x_nodes):
        return x_nodes[:, 1, 0] - x_nodes[:, 0, 0]


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
        xi_g_barycentric = (
            self.quad.points().repeat(self.mesh.n_elements, 1).squeeze(-1)
        )

        x_ref = self.sf.reference_element.simplex.repeat(xi_g_barycentric.shape[0], 1)
        xi_g = torch.einsum("ei,ei->e", xi_g_barycentric, x_ref)

        x_g = self.mapping.map(self.mesh.element_nodes_positions, xi_g)
        x_g.requires_grad_(True)

        xi_q = self.mapping.inverse_map(x_g, self.mesh.element_nodes_positions)
        N = self.sf.N(xi_q)
        w = self.quad.weights()
        measure = (
            self.mapping.element_length(self.mesh.nodes_positions[self.mesh.conn])[
                :, None
            ]
            * w[None, :]
        )

        nodes_values = torch.gather(
            self.field.full_values()[:, None, :].repeat(1, 2, 1),
            0,
            self.mesh.element_nodes_ids.repeat(1, 1, 1),
        )
        nodes_values = nodes_values.to(N.dtype)

        u_q = torch.einsum("gij,gi->gj", nodes_values, N)

        return x_g, u_q, measure

    def evaluate_at(self, x):
        device = self.mesh.nodes_positions.device

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
        element_nodes_positions = self.mesh.element_nodes_positions[element_ids]

        xi = self.mapping.inverse_map(x, element_nodes_positions)
        N = self.sf.N(xi)

        u_full = self.field.full_values()
        nodes_values = u_full[element_nodes_ids]
        nodes_values = nodes_values.to(N.dtype)

        u_q = torch.einsum("gij,gi->gj", nodes_values, N)
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
        return torch.sum(integrand * measure)


# =========================================================
# Force function
# =========================================================
def f(x):
    return 1000.0 + 0 * x


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

    sf = LinearLineShapeFunction()
    quad = TwoPointsQuadrature1D()
    mapping = IsoparametricMapping1D(sf)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=1)
    loss_history = []

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

    if plot_loss:
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()

    print("\n* Evaluation")

    x_test = torch.linspace(0, 6, 30)
    u_test = evaluator.evaluate_at(x_test)

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
