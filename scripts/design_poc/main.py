from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

# =========================================================
# Reference Element
# =========================================================
@dataclass
class ReferenceElement(ABC):
    pass

@dataclass
class LineReferenceElement(ReferenceElement):
    dim = 1
    n_nodes = 2
    simplex = torch.tensor([-1.,1.])


# =========================================================
# Shape Functions
# =========================================================
class ShapeFunction(ABC):
    @abstractmethod
    def N(self, xi):
        pass

class LinearLineShapeFunction(ShapeFunction):
    def N(self, xi):
        # xi shape: (n_q,)
        return torch.stack([-0.5*(xi+1.), 0.5*(xi+1.)], dim=1)

# =========================================================
# Quadratures
# =========================================================
class QuadratureRule(ABC):
    def __init__(self, reference_element):
        self.reference_element = reference_element

    @abstractmethod
    def points(self, device):
        pass

    @abstractmethod
    def weights(self, device ):
       pass  

class MidPointQuadrature1D(QuadratureRule):
    def __init__(self, reference_element = LineReferenceElement):
        super().__init__(reference_element)

    def points(self, device ):
        return torch.tensor([0.], device=device)

    def weights(self, device):
        w = self.reference_element.simplex[1] - self.reference_element.simplex[0]
        return torch.tensor([w], device=device)

class TwoPointsQuadrature1D(LineReferenceElement):
    def __init__(self, reference_element = LineReferenceElement):
        super().__init__(reference_element)

    def points(self, device ):
        l1=0.5*(1. - 1/3**0.5)
        l2=0.5*(1. + 1/3**0.5)
        return torch.tensor([-l2+ l1, -l1 + l2], device=device)

    def weights(self, device ):
        w = 0.5*(self.reference_element.simplex[1] - self.reference_element.simplex[0])
        return torch.tensor([w,w], device=device) 

# =========================================================
# Geometry Mapping
# =========================================================
class IsoparametricMapping1D:
    def __init__(self, shape_function):
        self.sf = shape_function

    def map(self, nodes_positions, xi):
        N = self.sf.N(xi)
        return torch.einsum("enx,enx->e", nodes_positions,N).squeeze(-1)

    def inverse_map(self, x, nodes_coord):
        inverse_mapping          = torch.ones([int(nodes_coord.shape[0]), 2, 2], dtype=x.dtype, device=x.device)
        det_J                    = nodes_coord[:,0,0] - nodes_coord[:,1,0]
        inverse_mapping[:,0,1]   = -nodes_coord[:,1,0]
        inverse_mapping[:,1,1]   = nodes_coord[:,0,0]
        inverse_mapping[:,1,0]   = -1*inverse_mapping[:,1,0]
        inverse_mapping[:,:,:]  /= det_J[:,None,None]
        x_extended = torch.stack((x, torch.ones_like(x)),dim=1)

        return torch.einsum('eij,ej...->ei',inverse_mapping,x_extended.squeeze(1))

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

        #Individual element index
        self.elements_ids = torch.arange(self.conn.size(0))
        
        #Pairs of nodes indices per element
        element_nodes_ids  = self.conn[self.elements_ids,:].T
        self.element_nodes_ids = torch.as_tensor(element_nodes_ids).to(self.nodes_positions.device).t()[:,:,None]
        
        #Pairs of nodes positions
        element_nodes_positions = torch.gather(self.nodes_positions[:,None,:].repeat(1,2,1),0, self.element_nodes_ids.repeat(1,1,1))
        self.element_nodes_positions = element_nodes_positions.to(self.nodes_positions.dtype)

        def sub_mesh_at(self,x):
            import math
            def append_if_not_close(lst, value, rel_tol=1e-9, abs_tol=0.0):
                if not any(math.isclose(value, x, rel_tol=rel_tol, abs_tol=abs_tol) for x in lst):
                    lst.append(value)

            connectivity = []
            nodes = []
            n_idx = 0
            for x_i in x:
                for k in self.elements_ids:
                    x_first = self.element_nodes_positions[k,0]
                    x_second = self.element_nodes_positions[k,1]
                    if x_i >= x_first and x_i <= x_second:
                        #Add node if not already existing 
                        append_if_not_close(nodes, x_first)
                        append_if_not_close(nodes, x_second)
                        connectivity.append(n_idx, n_idx+1)
                        n_idx+=1
                        break

            nodes = torch.FloatTensor(nodes)
            connectivity = torch.FloatTensor(connectivity)

            return Mesh1D(nodes=nodes, connectivity=connectivity)

# =========================================================
# Field (DOFs + BCs)
# =========================================================
class ScalarField1D(nn.Module):
    def __init__(self, mesh, dirichlet_nodes):
        super().__init__()
        self.mesh = mesh
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

        #Get reference position of quadrature points
        xi_g = self.quad.points(device).repeat(self.mesh.n_elements,1)
        import pdb ; breakpoint()
      
        #Get physical positions of quadrature points on mesh 
        x_g = self.mapping.map( self.mesh.element_nodes_positions, xi_g)

        #Get coordinates of those points back in reference coordinates
        x_q = self.mapping.inverse_map( x_g, self.mesh.element_nodes_positions)
        #Get shape function coordinate representation in reference element
        N = self.sf.N(x_q)

        #Get quadrature weights
        w = self.quad.weights(device) 

        #Evaluate measure
        measure = self.mapping.element_length(self.mesh.nodes_positions[self.mesh.conn])[:, None] * w[None, :]

        #Assemble values at nodes
        nodes_values = torch.gather(self.field.full_values()[:,None,:].repeat(1,2,1),0, self.mesh.element_nodes_ids.repeat(1,1,1))
        nodes_values = nodes_values.to(N.dtype)

        #Interpolate
        u_q = torch.einsum('gij,gi->gj', nodes_values, x_q)

        return x_g, u_q, measure

    def evaluate_at(self, x):
        device = self.mesh.nodes.device

        #Get elements to which x belongs
        sub_mesh = self.mesh.sub_mesh_at(x)

        #Get barycentric coordinates of points on sub-mesh
        #Only done on elements to which x belongs to
        x_q = self.mapping.inverse_map( x_g, self.mesh.element_nodes_positions)

        #Get corresponding shape functions
        N = self.sf.N(x_q).repeat(self.elements.shape[0],1)

        #Assemble values at nodes
        nodes_values   = torch.gather(self.field.full_values()[:,None,:].repeat(1,2,1),0, self.mesh.ids.repeat(1,1,1))
        nodes_values   = nodes_values.to(N.dtype)

        #Interpolate
        u_q = torch.einsum('gi...,gi->g',nodes_values,N)

        return u_q


# =========================================================
# Physics (Integrand ONLY)
# =========================================================
class PoissonPhysics:
    def __init__(self, f):
        self.f = f

    def integrand(self, x, u):
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
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
        x_q, u_q, measure = evaluator.evaluate()
        integrand = physics.integrand(x_q, u_q)
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
    exit()
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
