import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

# =========================================================
# Reference Element
# =========================================================
class LineReferenceElement:
    dim = 1
    n_nodes = 2


# =========================================================
# Shape Functions
# =========================================================
class LinearLineShapeFunction:
    def N(self, xi):
        # xi shape: (n_q,)
        return torch.stack([1 - xi, xi], dim=-1)

    def dN_dxi(self, xi):
        # xi shape: (E,Q)
        return torch.stack([
            -torch.ones_like(xi),
            torch.ones_like(xi)
        ], dim=-1)


# =========================================================
# Quadrature
# =========================================================
class MidPointQuadrature1D:
    def points(self, device):
        return torch.tensor([0.5], device=device)

    def weights(self, device):
        return torch.tensor([1.0], device=device)


# =========================================================
# Geometry Mapping
# =========================================================
class IsoparametricMapping1D:
    def __init__(self, shape_function):
        self.sf = shape_function

    def map(self, X_nodes, xi):
        # X_nodes: (E, 2, 1)
        N = self.sf.N(xi)  # (Q,2)
        return torch.einsum("eqi,eni->eqn", N, X_nodes).squeeze(-1)

    def element_length(self, X_nodes):
        return X_nodes[:, 1, 0] - X_nodes[:, 0, 0]


# =========================================================
# Mesh
# =========================================================
class Mesh1D:
    def __init__(self, nodes, connectivity):
        self.nodes = nodes
        self.conn = connectivity
        self.n_elements = connectivity.shape[0]


# =========================================================
# Field (DOFs + BCs)
# =========================================================
class ScalarField1D(nn.Module):
    def __init__(self, mesh, dirichlet_nodes):
        super().__init__()
        self.mesh = mesh
        n_nodes = mesh.nodes.shape[0]

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
        device = self.mesh.nodes.device

        xi_base = self.quad.points(device)
        xi = xi_base.unsqueeze(0).repeat(self.mesh.n_elements, 1)
        xi.requires_grad_(True)
        w = self.quad.weights(device)          # (Q,)

        conn = self.mesh.conn
        X = self.mesh.nodes[conn]              # (E, 2, 1)
        U_nodes = self.field.full_values()[conn]  # (E, 2, 1)

        # Physical coordinates of quadrature points
        x_q = self.mapping.map(X, xi)  # (E, Q)

        # Interpolated field at quadrature points
        N = self.sf.N(xi)               # (Q,2)
        u_q = torch.einsum("eqi,eni->eqn", N, U_nodes).squeeze(-1)

        # Gradient per element 
        # Shape function derivatives
        dN_dxi = self.sf.dN_dxi(xi)   # (E,Q,2)

        # Element Jacobian
        J = self.mapping.element_length(X)[:, None]   # (E,1)

        # du/dxi
        du_dxi = torch.einsum("eqi,eni->eqn", dN_dxi, U_nodes).squeeze(-1)

        # du/dx
        grad_u = du_dxi / J

        # Measure = detJ * weight
        measure = self.mapping.element_length(X)[:, None] * w[None, :]
        #import pdb ; breakpoint()
        return x_q, u_q, grad_u, measure


# =========================================================
# Physics (Integrand ONLY)
# =========================================================
class PoissonPhysics:
    def __init__(self, f):
        self.f = f

    def integrand(self, u, grad_u, x):
        return 0.5 * grad_u**2 - self.f(x) * u


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
    return 1000.0 + 0*x  # same as original


# =========================================================
# Main
# =========================================================
def main():
    # ---------------- Mesh ----------------
    N = 40
    nodes = torch.linspace(0, 6.28, N)[:, None]
    elements = torch.vstack([
        torch.arange(0, N - 1),
        torch.arange(1, N)
    ]).T

    mesh = Mesh1D(nodes, elements)

    # ---------------- FEM building blocks ----------------
    sf = LinearLineShapeFunction()
    quad = MidPointQuadrature1D()
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

    print("* Training")
    for i in range(7000):
        x_q, u_q, grad_u_q, measure = evaluator.evaluate()
        integrand = physics.integrand(u_q, grad_u_q, x_q)
        loss = integrator.integrate(integrand, measure)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        print(f"{i=} loss={loss.item():.3e}", end="\r")

    # ---------------- Plot Loss ----------------
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    # ---------------- Evaluation ----------------
    field.eval()
    U_full = field.full_values()

    # Evaluate solution at test points
    x_test = torch.linspace(0, 6, 30)
    u_test = []

    for x in x_test:
        for e, conn in enumerate(mesh.conn):
            x1 = mesh.nodes[conn[0]]
            x2 = mesh.nodes[conn[1]]
            if x >= x1 and x <= x2:
                xi = (x - x1) / (x2 - x1)
                N_local = torch.tensor([1 - xi, xi])
                u_nodes = U_full[conn].squeeze()
                u_test.append((N_local @ u_nodes).item())
                break

    u_test = torch.tensor(u_test)

    # Gauss points solution
    x_q, u_q, _, _ = evaluator.evaluate()

    # ---------------- Plot solution ----------------
    plt.figure()
    plt.plot(x_q.flatten().detach(), u_q.flatten().detach(), "+", label="Gauss points")
    plt.plot(x_test, u_test, "o", label="Test points")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Displacement Field")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
